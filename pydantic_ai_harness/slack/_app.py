"""A ready-made Slack app that serves a Pydantic AI agent over Socket Mode."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Collection, Mapping
from typing import Protocol
from weakref import WeakValueDictionary

import anyio
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.agent import AbstractAgent

from pydantic_ai_harness.slack._interactions import PROMPT_ACTION_PREFIX, SlackInteractions
from pydantic_ai_harness.slack._store import ConversationStore, InMemoryConversationStore
from pydantic_ai_harness.slack._thread import SlackThread
from pydantic_ai_harness.slack._toolset import MAX_MESSAGE_CHARS

try:
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_bolt.app.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'slack-bolt is required for SlackAgent. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error

logger = logging.getLogger(__name__)

_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_ACTION_LIST_ADAPTER = TypeAdapter(list[dict[str, object]])


class _Ack(Protocol):
    async def __call__(self) -> object: ...  # pragma: no cover


def _string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def _mapping(value: object) -> dict[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _prompt_click_fields(body: Mapping[str, object]) -> tuple[str, str, str] | None:
    try:
        actions = _ACTION_LIST_ADAPTER.validate_python(body.get('actions'))
    except ValidationError:
        return None
    user = _mapping(body.get('user'))
    if not actions or user is None:
        return None
    action = actions[0]
    block_id = _string(action, 'block_id')
    value = _string(action, 'value')
    user_id = _string(user, 'id')
    if block_id is None or value is None or user_id is None:
        return None
    return block_id, value, user_id


DEFAULT_ERROR_REPLY = 'Something went wrong handling that message. The details are in the logs.'


class SlackAgent:
    """Serve one Pydantic AI agent as a Slack bot.

    Wires the pieces in this package to a Socket Mode Slack app: DMs and channel
    mentions start a run, history is kept per thread, prompts posted by
    `ask_user` and by approval handling resolve through the same registry, and
    the agent's text output is posted back into the thread.

    The agent's `deps` type must be
    [`SlackThread`][pydantic_ai_harness.slack.SlackThread].

    Socket Mode opens an outbound connection, so no public HTTPS endpoint is
    needed. For OAuth distribution across workspaces, build your own `AsyncApp`
    in HTTP mode and call [`handle_message`][pydantic_ai_harness.slack.SlackAgent.handle_message]
    from your own listeners instead.

    Attributes:
        app: The underlying Bolt app. Register extra listeners on it if you need them.
    """

    def __init__(
        self,
        agent: AbstractAgent[SlackThread, str],
        *,
        store: ConversationStore | None = None,
        interactions: SlackInteractions | None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        allowed_user_ids: Collection[str] | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None:
        """Configure the app without connecting.

        Args:
            agent: A text-output agent whose deps type is `SlackThread`.
            store: Where thread history lives. Defaults to process memory, which
                is lost on restart.
            interactions: Prompt registry shared with the chat toolset and any
                approval handler. Pass the same instance you gave those.
            bot_token: Bot token (`xoxb-`). Defaults to `SLACK_BOT_TOKEN`.
            app_token: App-level token (`xapp-`) with `connections:write`.
                Defaults to `SLACK_APP_TOKEN`.
            allowed_user_ids: Slack user ids allowed to start runs. Defaults to
                `SLACK_ALLOWED_USER_IDS`, comma separated. When neither is set,
                anyone who can reach the bot can spend its tokens and invoke its
                tools, and a warning says so at startup.
            error_reply: Posted in the thread when a run raises.

        Raises:
            ValueError: If a token is missing, or `allowed_user_ids` is a string
                rather than a collection of ids.
        """
        self._agent = agent
        self._store = store if store is not None else InMemoryConversationStore()
        self._interactions = interactions
        self._error_reply = error_reply

        bot = bot_token or os.environ.get('SLACK_BOT_TOKEN')
        app = app_token or os.environ.get('SLACK_APP_TOKEN')
        if not bot:
            raise ValueError('bot_token is required; pass it or set SLACK_BOT_TOKEN')
        if not app:
            raise ValueError('app_token is required; pass it or set SLACK_APP_TOKEN')
        self._app_token = app

        if isinstance(allowed_user_ids, str):
            raise ValueError('allowed_user_ids must be a collection of user ids, not a string')
        if allowed_user_ids is None:
            configured = os.environ.get('SLACK_ALLOWED_USER_IDS', '')
            allowed_user_ids = [entry.strip() for entry in configured.split(',') if entry.strip()]
        self._allowed_user_ids = frozenset(allowed_user_ids)
        if not self._allowed_user_ids:
            logger.warning(
                'No allowed user ids configured, so anyone who can reach this bot can run the agent. '
                'Pass allowed_user_ids or set SLACK_ALLOWED_USER_IDS to restrict it.'
            )

        # Weak, so a thread's lock is collected once no turn holds it. A busy
        # workspace would otherwise accumulate one lock per thread forever.
        self._locks: WeakValueDictionary[str, anyio.Lock] = WeakValueDictionary()
        self.app = AsyncApp(token=bot)
        self._register_listeners()

    def _register_listeners(self) -> None:
        # Bolt ships no type information for its decorators, so registration is
        # done by explicit call and the untyped members are silenced one by one.
        # Everything the listeners hand on is narrowed before it leaves here.
        self.app.event('app_mention')(self._on_mention)  # pyright: ignore[reportUnknownMemberType]
        self.app.event('message')(self._on_direct_message)  # pyright: ignore[reportUnknownMemberType]
        prompt_clicks = re.compile(f'^{re.escape(PROMPT_ACTION_PREFIX)}')
        self.app.action(prompt_clicks)(self._on_prompt_click)  # pyright: ignore[reportUnknownMemberType]

    async def _on_mention(
        self, event: Mapping[str, object], client: AsyncWebClient, context: Mapping[str, object]
    ) -> None:
        await self.handle_message(
            event,
            client,
            bot_user_id=_string(context, 'bot_user_id'),
            team_id=_string(context, 'team_id'),
        )

    async def _on_direct_message(
        self, event: Mapping[str, object], client: AsyncWebClient, context: Mapping[str, object]
    ) -> None:
        # Channel messages reach this listener too, and `app_mention` already
        # covers those. Only a direct or group DM starts a run without a mention.
        if event.get('channel_type') not in ('im', 'mpim'):
            return
        await self.handle_message(
            event,
            client,
            bot_user_id=_string(context, 'bot_user_id'),
            team_id=_string(context, 'team_id'),
        )

    async def _on_prompt_click(self, ack: _Ack, body: Mapping[str, object]) -> None:
        await ack()
        fields = _prompt_click_fields(body)
        if fields is None or not self.resolve_prompt(block_id=fields[0], value=fields[1], user_id=fields[2]):
            # Slack shows nothing for a click that changes nothing, so the
            # operator needs a record of prompts that expired or were clicked by
            # someone who could not answer them.
            logger.info('Ignoring a Slack prompt click that resolved nothing')

    def resolve_prompt(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Route a button click from a prompt back to the run waiting on it.

        Call it from your own action handler when you build the Bolt app
        yourself. Returns `False` when the click changed nothing, which covers an
        expired prompt, a repeat click, and a person not allowed to answer.
        """
        if self._interactions is None:
            return False
        return self._interactions.resolve(block_id=block_id, value=value, user_id=user_id)

    async def handle_message(
        self,
        event: Mapping[str, object],
        client: AsyncWebClient,
        *,
        bot_user_id: str | None = None,
        team_id: str | None = None,
    ) -> None:
        """Run the agent for one inbound Slack message and post its reply.

        Call this from your own Bolt listeners when you build the app yourself.
        Messages the bot itself sent, messages with a subtype, and messages from
        users outside the allowlist are ignored.

        `team_id` names the workspace and keeps history separate across a
        multi-workspace install. Bolt puts it on the listener's `context`, which
        is more reliable than the event body.
        """
        if event.get('bot_id') is not None or event.get('subtype') is not None:
            return
        user_id = _string(event, 'user')
        channel_id = _string(event, 'channel')
        text = _string(event, 'text')
        timestamp = _string(event, 'ts')
        if user_id is None or channel_id is None or text is None or timestamp is None:
            return
        if user_id == bot_user_id:
            return
        if self._allowed_user_ids and user_id not in self._allowed_user_ids:
            logger.info('Ignoring Slack message from %s, who is not on the allowlist', user_id)
            return

        # Strip only this bot's own mention. Removing every mention would delete
        # the names of people the message refers to, which the agent needs.
        prompt = text.replace(f'<@{bot_user_id}>', '').strip() if bot_user_id else text.strip()
        if not prompt:
            return

        thread_ts = _string(event, 'thread_ts')
        thread = SlackThread(
            client=client,
            channel_id=channel_id,
            thread_ts=thread_ts or timestamp,
            user_id=user_id,
            team_id=team_id or _string(event, 'team'),
        )

        # One run at a time per thread, so a follow-up queues behind the run it is
        # adding to rather than racing it for the same history.
        lock = self._locks.setdefault(thread.key, anyio.Lock())
        async with lock:
            await self._run_turn(thread, prompt)

    async def _run_turn(self, thread: SlackThread, prompt: str) -> None:
        try:
            history = await self._store.load(thread.key)
            result = await self._agent.run(
                prompt,
                deps=thread,
                conversation_id=thread.key,
                message_history=list(history) or None,
            )
            if result.output.strip():
                await self._post(thread, result.output)
            # Saved only once the reply is out. Saving first would leave the next
            # turn building on an answer nobody in the thread ever saw.
            await self._store.save(thread.key, result.all_messages())
        except Exception:
            logger.exception('Slack agent run failed in %s', thread.key)
            try:
                await self._post(thread, self._error_reply)
            except Exception:
                logger.exception('Could not post the error reply in %s', thread.key)

    async def _post(self, thread: SlackThread, text: str) -> None:
        for start in range(0, len(text), MAX_MESSAGE_CHARS):
            await thread.client.chat_postMessage(  # pyright: ignore[reportUnknownMemberType]
                channel=thread.channel_id,
                thread_ts=thread.thread_ts,
                text=text[start : start + MAX_MESSAGE_CHARS],
            )

    async def start(self) -> None:
        """Connect over Socket Mode and serve until cancelled."""
        await AsyncSocketModeHandler(self.app, self._app_token).start_async()

    def run(self) -> None:
        """Start the app and block. The synchronous entry point for a script."""
        anyio.run(self.start)
