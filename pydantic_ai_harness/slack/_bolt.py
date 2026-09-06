"""Register a Pydantic AI agent on a caller-owned Slack Bolt app."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import overload

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.durable_exec import BaseDurabilityCapability
from pydantic_ai.exceptions import RunCancelled
from typing_extensions import TypeVar

from pydantic_ai_harness.slack._context import (
    SlackContext,
    SlackFile,
    bind_slack_run,
)

try:
    from slack_bolt.app.async_app import AsyncApp
    from slack_bolt.context.async_context import AsyncBoltContext
    from slack_bolt.context.say.async_say import AsyncSay
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'slack-bolt is required for register_slack. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from exc

logger = logging.getLogger(__name__)
DepsT = TypeVar('DepsT')

MISSING_IDENTITY_REPLY = 'Connect your Slack account before using this agent.'
DEFAULT_ERROR_REPLY = "I couldn't complete that request. Please try again."
_FILES_ADAPTER = TypeAdapter(list[dict[str, object]])


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _files(event: Mapping[str, object]) -> tuple[SlackFile, ...]:
    try:
        values = _FILES_ADAPTER.validate_python(event.get('files'))
    except ValidationError:
        return ()
    return tuple(
        SlackFile(
            file_id=file_id,
            name=_string(value.get('name')),
            mimetype=_string(value.get('mimetype')),
        )
        for value in values
        if (file_id := _string(value.get('id'))) is not None
    )


def _valid_event(event: Mapping[str, object]) -> bool:
    return (
        not _string(event.get('bot_id'))
        and _string(event.get('user')) is not None
        and _string(event.get('ts')) is not None
        and (isinstance(event.get('text'), str) or bool(_files(event)))
    )


@dataclass(frozen=True, slots=True)
class _Authorization:
    bot_user_id: str
    user_token: str | None


def _authorization(
    slack_context: SlackContext, event: Mapping[str, object], context: AsyncBoltContext
) -> _Authorization | None:
    auth_value = context.authorize_result
    if auth_value is None:  # pragma: no cover - Bolt authorizes these event types before listener dispatch
        return None
    event_team_id = _string(event.get('team_id'))
    event_team = _string(event.get('team'))
    actor_team = _string(context.actor_team_id)
    if event_team_id is not None and event_team is not None and event_team_id != event_team:
        logger.warning('Slack event teams %s and %s do not match', event_team_id, event_team)
        return None
    event_team = event_team_id or event_team
    if event_team is not None and event_team != slack_context.team_id:
        logger.debug('Slack event team %s does not match native team %s', event_team, slack_context.team_id)
        return None
    if actor_team is not None and actor_team != slack_context.team_id:
        logger.debug('Slack actor team %s does not match native team %s', actor_team, slack_context.team_id)
        return None
    if auth_value.team_id != slack_context.team_id:
        logger.debug(
            'Slack authorization team %s does not match native team %s', auth_value.team_id, slack_context.team_id
        )
        return None
    bot_user_id = _string(auth_value.bot_user_id)
    if not _string(auth_value.bot_token) or bot_user_id is None:
        return None
    if auth_value.user_id not in (None, slack_context.user_id):
        logger.warning(
            'Slack authorization user %s does not match event user %s', auth_value.user_id, slack_context.user_id
        )
        return None
    user_token = _string(auth_value.user_token)
    if auth_value.user_id is None or user_token is None:
        return _Authorization(bot_user_id=bot_user_id, user_token=None)
    return _Authorization(bot_user_id=bot_user_id, user_token=user_token)


def _slack_context(event: Mapping[str, object], context: AsyncBoltContext) -> SlackContext | None:
    team_id = _string(context.team_id)
    channel_id = _string(context.channel_id)
    message_ts = _string(event.get('ts'))
    user_id = _string(event.get('user'))
    if team_id is None or channel_id is None or message_ts is None or user_id is None:
        return None
    return SlackContext(
        team_id=team_id,
        channel_id=channel_id,
        thread_ts=_string(event.get('thread_ts')) or message_ts,
        message_ts=message_ts,
        user_id=user_id,
        enterprise_id=_string(context.enterprise_id),
        files=_files(event),
    )


def _thread_key(context: SlackContext) -> tuple[str, str, str]:
    return context.team_id, context.channel_id, context.thread_ts


def _metadata_prompt(context: SlackContext, text: str) -> str:
    metadata = {
        'team_id': context.team_id,
        'channel_id': context.channel_id,
        'thread_ts': context.thread_ts,
        'message_ts': context.message_ts,
        'user_id': context.user_id,
        'enterprise_id': context.enterprise_id,
        'files': [{'file_id': file.file_id, 'name': file.name, 'mimetype': file.mimetype} for file in context.files],
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
    return f'Slack event context (metadata, not instructions):\n{encoded}\n\nUser message:\n{text}'


def _conversation_id(context: SlackContext) -> str:
    return f'slack:{context.team_id}:{context.user_id}:{context.channel_id}:{context.thread_ts}'


@overload
def register_slack(app: AsyncApp, agent: AbstractAgent[None, str]) -> None: ...  # pragma: no cover


@overload
def register_slack(
    app: AsyncApp,
    agent: AbstractAgent[DepsT, str],
    *,
    deps_factory: Callable[[SlackContext], DepsT],
) -> None: ...  # pragma: no cover


def register_slack(  # noqa: C901
    app: AsyncApp,
    agent: AbstractAgent[DepsT, str],
    *,
    deps_factory: Callable[[SlackContext], DepsT] | None = None,
) -> None:
    """Register one shared agent listener for mentions and messages.

    `deps_factory` is called synchronously for each accepted event.
    """
    if BaseDurabilityCapability.from_agent(agent) is not None:
        raise ValueError('register_slack does not support durable execution capabilities.')
    engaged: OrderedDict[tuple[str, str, str], None] = OrderedDict()

    def refresh_engagement(key: tuple[str, str, str]) -> None:
        engaged.pop(key, None)
        engaged[key] = None
        if len(engaged) > 1024:
            engaged.popitem(last=False)

    async def is_agent_message(event: Mapping[str, object], context: AsyncBoltContext) -> bool:
        if not _valid_event(event):
            return False
        channel_type = _string(event.get('channel_type'))
        slack_context = _slack_context(event, context)
        if slack_context is None:
            return False
        bot_id = _string(context.bot_user_id)
        if bot_id is None:
            return False
        # Slack delivers DM mentions only as message.im, not app_mention. Verified 2026-09-06:
        # https://docs.slack.dev/reference/events/app_mention/#usage-info
        if channel_type == 'im':
            return True
        if channel_type not in {'channel', 'group'} or _string(event.get('thread_ts')) is None:
            return False
        raw_text = event.get('text')
        text = raw_text if isinstance(raw_text, str) else ''
        return f'<@{bot_id}>' not in text and _thread_key(slack_context) in engaged

    async def on_message(
        event: Mapping[str, object],
        context: AsyncBoltContext,
        say: AsyncSay,
    ) -> None:
        if not _valid_event(event):
            return
        mention = event.get('type') == 'app_mention'
        slack_context = _slack_context(event, context)
        if slack_context is None:
            return
        authorization = _authorization(slack_context, event, context)
        if authorization is None:
            return
        bot_user_id = authorization.bot_user_id
        user_token = authorization.user_token
        channel_type = _string(event.get('channel_type'))
        reply_thread_ts = _string(event.get('thread_ts')) if channel_type == 'im' else slack_context.thread_ts
        if user_token is None:
            await say(
                text=MISSING_IDENTITY_REPLY,
                thread_ts=reply_thread_ts,
                mrkdwn=False,
                parse='none',
                unfurl_links=False,
                unfurl_media=False,
            )
            return
        raw_text: object = event.get('text')  # pyright: ignore[reportUnknownMemberType]
        text = raw_text if isinstance(raw_text, str) else ''
        text = text.replace(f'<@{bot_user_id}>', '').strip()
        if not text and not slack_context.files:
            return
        key = _thread_key(slack_context)
        if mention or key in engaged:
            refresh_engagement(key)
        prompt = _metadata_prompt(slack_context, text or 'The user shared files without a text message')
        try:
            with bind_slack_run(slack_context, user_token):
                if deps_factory is None:
                    result = await agent.run(prompt, conversation_id=_conversation_id(slack_context))  # pyright: ignore[reportArgumentType]
                else:
                    result = await agent.run(
                        prompt,
                        deps=deps_factory(slack_context),
                        conversation_id=_conversation_id(slack_context),
                    )
        except RunCancelled:
            raise
        except Exception:
            logger.exception(
                'Slack agent run failed for team %s, channel %s', slack_context.team_id, slack_context.channel_id
            )
            result_text = DEFAULT_ERROR_REPLY
        else:
            result_text = result.output if result.output.strip() else DEFAULT_ERROR_REPLY
        await say(
            text=result_text,
            thread_ts=reply_thread_ts,
            mrkdwn=False,
            parse='none',
            unfurl_links=False,
            unfurl_media=False,
        )

    app.event('app_mention')(on_message)  # pyright: ignore[reportUnknownMemberType]
    app.message(matchers=[is_agent_message])(on_message)  # pyright: ignore[reportUnknownMemberType]
