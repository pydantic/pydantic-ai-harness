"""A ready-made Slack app that serves a Pydantic AI agent, over Socket Mode or HTTP."""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from time import monotonic
from typing import Generic, Protocol, overload
from weakref import WeakValueDictionary

import anyio
from pydantic import AliasPath, BaseModel, Field, TypeAdapter, ValidationError
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from typing_extensions import TypeVar

from pydantic_ai_harness.slack._access import SlackAccess
from pydantic_ai_harness.slack._capability import Slack, SlackMCPAuthenticationError
from pydantic_ai_harness.slack._client import SlackClient
from pydantic_ai_harness.slack._context import (
    SlackContext,
    SlackContextEntity,
    SlackFile,
    SlackMessageContext,
    bind_slack_context,
)
from pydantic_ai_harness.slack._interactions import PROMPT_ACTION_PREFIX
from pydantic_ai_harness.slack._store import ConversationStore, InMemoryConversationStore
from pydantic_ai_harness.slack._thread import SlackThread

try:
    import slack_bolt.adapter.asgi.async_handler as slack_asgi
    import slack_bolt.adapter.socket_mode.async_handler as slack_socket_mode
    import slack_bolt.app.async_app as slack_async_app
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'slack-bolt is required for SlackApp. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error

logger = logging.getLogger(__name__)

BotDepsT = TypeVar('BotDepsT', default=None)
"""The deps type the served agent takes. Its own variable because `BotDepsT` is
contravariant, and `SlackApp` builds deps as well as passing them."""


class _Ack(Protocol):
    async def __call__(self) -> object: ...  # pragma: no cover


def _string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


class _ActiveMessageContext(BaseModel):
    channel_id: str
    message_ts: str


class _ActiveContextEntity(BaseModel):
    type: str
    value: str | _ActiveMessageContext
    team_id: str | None = None


class _AttachedFile(BaseModel):
    id: str
    name: str | None = None
    mimetype: str | None = None


_ACTIVE_CONTEXT_ENTITY_ADAPTER = TypeAdapter(_ActiveContextEntity)
_OBJECT_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_OBJECT_LIST_ADAPTER = TypeAdapter(list[object])
_ATTACHED_FILES_ADAPTER = TypeAdapter(list[_AttachedFile])


def _active_entities(event: Mapping[str, object]) -> tuple[SlackContextEntity, ...]:
    try:
        app_context = _OBJECT_MAPPING_ADAPTER.validate_python(event.get('app_context'))
        raw_entities = _OBJECT_LIST_ADAPTER.validate_python(app_context.get('entities'))
    except ValidationError:
        return ()
    entities: list[SlackContextEntity] = []
    for raw_entity in raw_entities:
        try:
            entity = _ACTIVE_CONTEXT_ENTITY_ADAPTER.validate_python(raw_entity)
        except ValidationError:
            continue
        value = (
            SlackMessageContext(channel_id=entity.value.channel_id, message_ts=entity.value.message_ts)
            if isinstance(entity.value, _ActiveMessageContext)
            else entity.value
        )
        entities.append(SlackContextEntity(entity_type=entity.type, value=value, team_id=entity.team_id))
    return tuple(entities)


def _attached_files(event: Mapping[str, object]) -> tuple[SlackFile, ...]:
    try:
        files = _ATTACHED_FILES_ADAPTER.validate_python(event.get('files'))
    except ValidationError:
        return ()
    return tuple(SlackFile(file_id=file.id, name=file.name, mimetype=file.mimetype) for file in files)


class _PromptClick(BaseModel):
    """The three fields that identify a button click, pulled out of Slack's payload.

    Slack nests them, and everything else in the payload is ignored. Declaring
    the shape means a payload that is not a prompt click fails validation rather
    than being routed with a field missing.
    """

    block_id: str = Field(validation_alias=AliasPath('actions', 0, 'block_id'))
    value: str = Field(validation_alias=AliasPath('actions', 0, 'value'))
    user_id: str = Field(validation_alias=AliasPath('user', 'id'))


_CLICK_ADAPTER = TypeAdapter(_PromptClick)


def _prompt_click(body: Mapping[str, object]) -> _PromptClick | None:
    """The click this payload describes, or `None` when it does not describe one."""
    try:
        return _CLICK_ADAPTER.validate_python(body)
    except ValidationError:
        return None


DEFAULT_ERROR_REPLY = 'Something went wrong handling that message. The details are in the logs.'
MISSING_MCP_AUTH_REPLY = (
    'Slack workspace access is not connected for your user. '
    'Ask the app owner for the Slack authorization link, then retry.'
)
# Slack-owned message and retry contracts were verified 2026-09-05 against:
# https://docs.slack.dev/reference/methods/chat.postMessage/
# https://docs.slack.dev/apis/events-api/#retries
# Re-check both before changing delivery limits or deduplication retention.
MAX_DELIVERY_CHARS = 3500
"""Local plain-text chunk size, kept below Slack's recommended 4,000-character limit."""

MAX_MARKDOWN_CHARS = 12_000
"""Slack's documented `markdown_text` limit."""

_EVENT_DEDUPLICATION_SECONDS = 600.0
"""Local retention beyond Slack's standard final retry at five minutes."""

_REMEMBERED_THREADS = 1000
_REMEMBERED_MESSAGES_PER_TEAM = 30_000
"""Local memory bound matching Slack's documented hourly event cap per workspace."""

_USER_MESSAGE_SUBTYPES = {None, 'file_share', 'thread_broadcast', 'me_message'}


class SlackApp(Generic[BotDepsT]):
    """Serve one Pydantic AI agent as a Slack bot.

    DMs and channel mentions start a run, history is kept per thread, and the
    agent's text output is posted back into the thread.

    Any agent works, whatever its `deps` type. The thread a run is answering is
    bound around the run rather than passed as `deps`, so
    [`Slack`][pydantic_ai_harness.slack.Slack] posts to the right place
    and the agent keeps the deps it already had.

    Button clicks route themselves: the app finds the `Slack` on the agent and
    hands approval clicks to it with nothing wired up by hand.

    Slack can reach the bot two ways, and only the last line differs:

    ```python {test="skip"}
    bot = SlackApp(agent)

    bot.run()                                   # Socket Mode, no public URL
    slack_asgi = bot.http_app()                 # Events API, complete ASGI app
    ```

    Socket Mode needs `SLACK_APP_TOKEN` and opens an outbound connection. The
    Events API needs `SLACK_SIGNING_SECRET` and a public HTTPS endpoint. This
    adapter expects a long-running process for both transports; FaaS deployments
    need Bolt's lazy-listener execution strategy.

    Attributes:
        app: The underlying Bolt app. Register extra listeners on it if you need
            them, or hand it to any of Bolt's other adapters.
    """

    @overload
    def __init__(  # pragma: no cover - static overload
        self,
        agent: AbstractAgent[None, str],
        *,
        deps: None = None,
        deps_factory: None = None,
        store: ConversationStore | None = None,
        app: None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        signing_secret: str | None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - static overload
        self,
        agent: AbstractAgent[None, str],
        *,
        deps: None = None,
        deps_factory: None = None,
        store: ConversationStore | None = None,
        app: slack_async_app.AsyncApp,
        bot_token: None = None,
        app_token: str | None = None,
        signing_secret: None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - static overload
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        deps: BotDepsT,
        deps_factory: None = None,
        store: ConversationStore | None = None,
        app: None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        signing_secret: str | None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - static overload
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        deps: BotDepsT,
        deps_factory: None = None,
        store: ConversationStore | None = None,
        app: slack_async_app.AsyncApp,
        bot_token: None = None,
        app_token: str | None = None,
        signing_secret: None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - static overload
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        deps: None = None,
        deps_factory: Callable[[SlackThread], BotDepsT],
        store: ConversationStore | None = None,
        app: None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        signing_secret: str | None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - static overload
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        deps: None = None,
        deps_factory: Callable[[SlackThread], BotDepsT],
        store: ConversationStore | None = None,
        app: slack_async_app.AsyncApp,
        bot_token: None = None,
        app_token: str | None = None,
        signing_secret: None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None: ...

    def __init__(  # pyright: ignore[reportInconsistentOverload] - overloads enforce the generic dependency contract
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        deps: BotDepsT = None,
        deps_factory: Callable[[SlackThread], BotDepsT] | None = None,
        store: ConversationStore | None = None,
        app: slack_async_app.AsyncApp | None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        signing_secret: str | None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None:
        """Configure the app without connecting.

        Args:
            agent: Any text-output agent. Its deps type does not matter.
            deps: Passed to the agent as `deps` for every thread. Agents that
                take no deps can leave it unset.
            deps_factory: Builds the deps for one run from the
                [`SlackThread`][pydantic_ai_harness.slack.SlackThread] it is
                answering, for deps that differ per channel or per person. Kept
                separate from `deps` rather than accepting either in one argument,
                so a deps object that happens to be callable is not mistaken for a
                factory.
            store: Where thread history lives. Defaults to process memory, which
                is lost on restart.
            app: Caller-configured Bolt app. Pass this for OAuth installations;
                Bolt then supplies the bot client and invoking user's token.
            bot_token: Bot token (`xoxb-`). Defaults to `SLACK_BOT_TOKEN`.
            app_token: App-level token (`xapp-`) with `connections:write`.
                Defaults to `SLACK_APP_TOKEN`. Only [`run`][pydantic_ai_harness.slack.SlackApp.run]
                needs it, so an Events API deployment can leave it unset.
            signing_secret: Secret Slack signs its HTTP requests with. Defaults to
                `SLACK_SIGNING_SECRET`. Only
                [`http_app`][pydantic_ai_harness.slack.SlackApp.http_app] needs
                it, so a Socket Mode deployment can leave it unset.
            access: Who may invoke the agent. Defaults to the comma-separated
                `SLACK_ALLOWED_USER_IDS`. Use `SlackAccess.workspace()` to opt
                into workspace-wide access.
            install_url: Public per-user OAuth URL shown when Slack MCP access
                is missing. Defaults to `SLACK_INSTALL_URL`.
            error_reply: Posted in the thread when a run raises.

        Raises:
            ValueError: If `bot_token` is missing, both `deps` and `deps_factory`
                were given, or no access policy was configured.
        """
        self._agent = agent
        if deps is not None and deps_factory is not None:
            raise ValueError('Pass deps or deps_factory, not both.')
        self._deps = deps
        self._deps_factory = deps_factory
        self._store = store if store is not None else InMemoryConversationStore()
        self._error_reply = error_reply
        self._install_url = install_url or os.environ.get('SLACK_INSTALL_URL')
        self._slack = _find_slack(agent.root_capability)
        self._caller_owned_app = app is not None

        if app is not None and (bot_token is not None or signing_secret is not None):
            raise ValueError('bot_token and signing_secret configure a new Bolt app and cannot be passed with app')

        bot = bot_token or os.environ.get('SLACK_BOT_TOKEN')
        if app is None and not bot:
            raise ValueError('bot_token is required when app is not passed; pass it or set SLACK_BOT_TOKEN')
        # Each transport needs one of these and neither needs both, so they are
        # checked where they are used rather than here.
        self._app_token = app_token or os.environ.get('SLACK_APP_TOKEN')
        self._signing_secret = signing_secret or os.environ.get('SLACK_SIGNING_SECRET')

        if access is None:
            configured = os.environ.get('SLACK_ALLOWED_USER_IDS', '')
            user_ids = [entry.strip() for entry in configured.split(',') if entry.strip()]
            if not user_ids:
                raise ValueError(
                    'Slack agent access is not configured. Set SLACK_ALLOWED_USER_IDS, pass '
                    'access=SlackAccess.users(...), or explicitly use SlackAccess.workspace().'
                )
            access = SlackAccess.users(*user_ids)
        self._access = access
        self._allow_fixed_mcp_fallback = (
            not self._caller_owned_app and access.allowed_user_ids is not None and len(access.allowed_user_ids) == 1
        )

        env_mcp_token = os.environ.get('SLACK_MCP_TOKEN')
        if (
            self._slack is not None
            and self._slack.tools.selected
            and not self._caller_owned_app
            and self._slack.mcp_token is None
            and env_mcp_token
            and (access.allowed_user_ids is None or len(access.allowed_user_ids) != 1)
        ):
            raise ValueError(
                'SLACK_MCP_TOKEN is a fixed user identity and requires exactly one allowed Slack user. '
                'Use per-user OAuth for multiple users, or pass mcp_token explicitly to opt into a shared identity.'
            )

        # Weak, so a thread's lock is collected once no turn holds it. A busy
        # workspace would otherwise accumulate one lock per thread forever.
        self._locks: WeakValueDictionary[str, anyio.Lock] = WeakValueDictionary()
        # Slack retries an event it thinks was not delivered, and a retried
        # mention would start a second run with the same side effects.
        self._handled_events: dict[str, OrderedDict[str, float]] = {}
        self._engaged_threads: OrderedDict[str, None] = OrderedDict()
        self._active_runs: dict[str, anyio.CancelScope] = {}
        self.app = app if app is not None else slack_async_app.AsyncApp(token=bot, signing_secret=self._signing_secret)
        self._register_listeners()

    @property
    def access(self) -> SlackAccess:
        """The policy that decides who may invoke this Slack app."""
        return self._access

    def _deps_for(self, thread: SlackThread) -> BotDepsT:
        """The deps this run gets, built for this thread when a factory was given."""
        if self._deps_factory is not None:
            return self._deps_factory(thread)
        return self._deps

    def _register_listeners(self) -> None:
        # Bolt ships no type information for its decorators, so registration is
        # done by explicit call and the untyped members are silenced one by one.
        # Everything the listeners hand on is narrowed before it leaves here.
        self.app.event('app_mention')(self._on_event)  # pyright: ignore[reportUnknownMemberType]
        self.app.event('message')(self._on_direct_message)  # pyright: ignore[reportUnknownMemberType]
        self.app.event('app_context_changed')(self._on_context_changed)  # pyright: ignore[reportUnknownMemberType]
        self.app.event('agent_session_stopped')(self._on_session_stopped)  # pyright: ignore[reportUnknownMemberType]
        prompt_clicks = re.compile(f'^{re.escape(PROMPT_ACTION_PREFIX)}')
        self.app.action(prompt_clicks)(self._on_prompt_click)  # pyright: ignore[reportUnknownMemberType]

    async def _on_event(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
    ) -> None:
        """Pull the envelope fields off a Bolt listener's arguments and run a turn."""
        _ = body
        await self._handle_message(
            event,
            client,
            bot_user_id=_string(context, 'bot_user_id'),
            team_id=_string(context, 'team_id'),
            enterprise_id=_string(context, 'enterprise_id'),
            user_token=_string(context, 'user_token'),
        )

    async def _on_direct_message(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
    ) -> None:
        # A DM always invokes the agent. A channel reply does so only when stored
        # history proves the agent already joined that thread after one mention.
        if event.get('channel_type') == 'im':
            await self._on_event(event, client, context, body)
            return
        channel_id = _string(event, 'channel')
        thread_ts = _string(event, 'thread_ts')
        if event.get('channel_type') == 'mpim':
            bot_user_id = _string(context, 'bot_user_id')
            text = _string(event, 'text') or ''
            if thread_ts is None and bot_user_id is not None and f'<@{bot_user_id}>' in text:
                await self._on_event(event, client, context, body)
                return
        if channel_id is None or thread_ts is None:
            return
        thread = SlackThread(channel_id=channel_id, thread_ts=thread_ts, team_id=_string(context, 'team_id'))
        if thread.key in self._engaged_threads or await self._store.load(thread.key):
            await self._on_event(event, client, context, body)

    async def _on_context_changed(self, event: Mapping[str, object]) -> None:
        """Acknowledge context updates; Slack attaches the current value to later DM events."""
        _ = event

    async def _on_session_stopped(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        context: Mapping[str, object],
    ) -> None:
        """Cancel the run addressed by Slack's native Agent stop button."""
        channel_id = _string(event, 'channel')
        thread_ts = _string(event, 'thread_ts')
        if channel_id is None or thread_ts is None:
            return
        thread = SlackThread(channel_id=channel_id, thread_ts=thread_ts, team_id=_string(context, 'team_id'))
        scope = self._active_runs.get(thread.key)
        stop_user_id = _string(event, 'user')
        if scope is not None and stop_user_id is not None and self._access.allows(stop_user_id):
            scope.cancel()
            with anyio.CancelScope(shield=True):
                await self._post(client, thread, 'Stopped.')

    async def _on_prompt_click(self, ack: _Ack, body: Mapping[str, object]) -> None:
        await ack()
        click = _prompt_click(body)
        if click is None or not self._resolve_prompt(block_id=click.block_id, value=click.value, user_id=click.user_id):
            # Slack shows nothing for a click that changes nothing, so the
            # operator needs a record of prompts that expired or were clicked by
            # someone who could not answer them.
            logger.info('Ignoring a Slack prompt click that resolved nothing')

    def _resolve_prompt(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Route a button click from the registered Bolt listener to a suspended run."""
        if self._slack is None:
            return False
        return self._slack._resolve_prompt(  # pyright: ignore[reportPrivateUsage] - same package adapter
            block_id=block_id, value=value, user_id=user_id
        )

    async def _handle_message(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        *,
        bot_user_id: str | None = None,
        team_id: str | None = None,
        enterprise_id: str | None = None,
        user_token: str | None = None,
    ) -> None:
        """Run the agent for one inbound Slack message and post its reply.

        Messages the bot itself sent, non-message subtypes such as edits, and
        messages from users outside the allowlist are ignored. User-authored
        file shares, thread broadcasts, and `/me` messages are accepted.

        `team_id` names the workspace and keeps history separate across a
        multi-workspace install. Bolt puts it on the listener's `context`, which
        is more reliable than the event body.

        Slack's channel and message timestamp identify redeliveries, so the same
        message does not start a second run with the same side effects.
        """
        if event.get('bot_id') is not None or event.get('subtype') not in _USER_MESSAGE_SUBTYPES:
            return
        user_id = _string(event, 'user')
        channel_id = _string(event, 'channel')
        text = _string(event, 'text')
        timestamp = _string(event, 'ts')
        if user_id is None or channel_id is None or text is None or timestamp is None:
            return
        team_key = team_id or _string(event, 'team') or ''
        identifier = f'{channel_id}:{timestamp}'
        if self._already_handled(team_key, identifier):
            logger.info('Ignoring a repeat delivery of Slack message %s:%s', channel_id, timestamp)
            return
        if user_id == bot_user_id:
            return
        if not self._access.allows(user_id):
            logger.info('Ignoring Slack message from %s, who is not on the allowlist', user_id)
            return

        # Strip only this bot's own mention. Removing every mention would delete
        # the names of people the message refers to, which the agent needs.
        files = _attached_files(event)
        prompt = text.replace(f'<@{bot_user_id}>', '').strip() if bot_user_id else text.strip()
        if not prompt and files:
            names = ', '.join(f'`{file.file_id}` ({file.name or "unnamed file"})' for file in files)
            prompt = f'The user shared these Slack files without accompanying text: {names}.'
        if not prompt:
            return

        thread_ts = _string(event, 'thread_ts')
        thread = SlackThread(
            channel_id=channel_id,
            thread_ts=thread_ts or timestamp,
            user_id=user_id,
            team_id=team_id or _string(event, 'team'),
        )
        slack_context = SlackContext(
            channel_id=channel_id,
            thread_ts=thread.thread_ts or timestamp,
            message_ts=timestamp,
            user_id=user_id,
            team_id=thread.team_id,
            enterprise_id=enterprise_id,
            active_entities=_active_entities(event),
            files=files,
        )
        self._remember_engagement(thread.key)

        # One run at a time per thread, so a follow-up queues behind the run it is
        # adding to rather than racing it for the same history.
        lock = self._locks.setdefault(thread.key, anyio.Lock())
        async with lock:
            await self._run_turn(client, thread, slack_context, prompt, user_token)

    def _already_handled(self, team_key: str, identifier: str) -> bool:
        """True when this message identity was seen. Record it otherwise."""
        now = monotonic()
        expires_before = now - _EVENT_DEDUPLICATION_SECONDS
        for stale_team, stale_events in tuple(self._handled_events.items()):
            if (  # pragma: no branch - time-based retention is an internal memory policy
                stale_team != team_key and next(reversed(stale_events.values())) <= expires_before
            ):
                del self._handled_events[stale_team]  # pragma: no cover
        team_events = self._handled_events.setdefault(team_key, OrderedDict())
        while team_events:
            _, seen_at = next(iter(team_events.items()))
            if seen_at > expires_before:
                break
            team_events.popitem(last=False)  # pragma: no cover - time-based retention
        if identifier in team_events:
            return True
        team_events[identifier] = now
        while len(team_events) > _REMEMBERED_MESSAGES_PER_TEAM:  # pragma: no cover - internal memory bound
            team_events.popitem(last=False)
        return False

    def _remember_engagement(self, thread_key: str) -> None:
        """Remember that this app owns replies in a thread, with bounded state."""
        self._engaged_threads[thread_key] = None
        self._engaged_threads.move_to_end(thread_key)
        if len(self._engaged_threads) > _REMEMBERED_THREADS:  # pragma: no cover - internal memory bound
            self._engaged_threads.popitem(last=False)

    async def _run_turn(
        self,
        client: SlackClient,
        thread: SlackThread,
        slack_context: SlackContext,
        prompt: str,
        user_token: str | None,
    ) -> None:
        await self._set_status(client, thread, 'processing')
        with anyio.CancelScope() as scope:
            self._active_runs[thread.key] = scope
            await self._run_turn_in_scope(client, thread, slack_context, prompt, user_token)

    async def _run_turn_in_scope(
        self,
        client: SlackClient,
        thread: SlackThread,
        slack_context: SlackContext,
        prompt: str,
        user_token: str | None,
    ) -> None:
        try:
            history = await self._store.load(thread.key)
            # Bound rather than passed as deps, so the agent's own deps are
            # untouched and `Slack` still knows where it is talking.
            with bind_slack_context(
                slack_context,
                client,
                user_token=user_token,
                allow_fixed_mcp_fallback=self._allow_fixed_mcp_fallback,
            ):
                result = await self._agent.run(
                    prompt,
                    deps=self._deps_for(thread),
                    conversation_id=thread.key,
                    message_history=list(history) or None,
                )
            if result.output.strip():
                await self._post(client, thread, result.output)
            # Saved only once the reply is out. Saving first would leave the next
            # turn building on an answer nobody in the thread ever saw.
            await self._store.save(thread.key, result.all_messages())
        except SlackMCPAuthenticationError:
            logger.exception('Slack MCP authentication is missing in %s', thread.key)
            reply = (
                f'Slack workspace access is not connected for your user. Authorize it at {self._install_url}, then retry.'
                if self._install_url is not None
                else MISSING_MCP_AUTH_REPLY
            )
            await self._post(client, thread, reply)
        except Exception:
            logger.exception('Slack agent run failed in %s', thread.key)
            try:
                await self._post(client, thread, self._error_reply)
            except Exception:
                logger.exception('Could not post the error reply in %s', thread.key)
        finally:
            self._active_runs.pop(thread.key, None)
            with anyio.CancelScope(shield=True):
                await self._set_status(client, thread, 'active')

    async def _set_status(self, client: SlackClient, thread: SlackThread, status: str) -> None:
        """Update Slack's visible agent status without making delivery depend on it."""
        if thread.thread_ts is None:  # pragma: no cover - inbound messages always create a thread root
            return
        try:
            await client.agents_sessions_setStatus(
                channel_id=thread.channel_id,
                thread_ts=thread.thread_ts,
                status=status,
            )
        except Exception:
            logger.debug('Could not update the Slack agent status in %s', thread.key, exc_info=True)

    async def _post(self, client: SlackClient, thread: SlackThread, text: str) -> None:
        safe_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if len(safe_text) <= MAX_MARKDOWN_CHARS:
            await client.chat_postMessage(
                channel=thread.channel_id,
                thread_ts=thread.thread_ts,
                markdown_text=safe_text,
            )
            return
        for start in range(0, len(text), MAX_DELIVERY_CHARS):
            await client.chat_postMessage(
                channel=thread.channel_id,
                thread_ts=thread.thread_ts,
                text=text[start : start + MAX_DELIVERY_CHARS],
                mrkdwn=False,
            )

    def http_app(self, path: str = '/slack/events') -> slack_asgi.AsyncSlackRequestHandler:
        """An ASGI app serving Slack's Events API, to mount on your own server.

        Mount it at `path`, and give Slack that same URL as the request URL. Bolt
        verifies the signature on every request and answers Slack's setup
        challenge, so there is nothing to write.

        Serve the returned handler as the ASGI application. It routes the event
        endpoint at `path` plus Bolt's OAuth install and redirect endpoints when
        the caller-configured app enables OAuth. If it is combined with another
        ASGI application, mount it at `/` after more specific routes so Bolt can
        still see the full request paths. Mounting it at `/slack/events` strips
        that prefix and makes Bolt return 404.

        ```python {test="skip"}
        slack_asgi = bot.http_app()
        # uvicorn my_module:slack_asgi
        ```

        Bolt acknowledges the request before running the listener in its normal
        long-running-server mode.

        Raises:
            ValueError: If no signing secret was configured.
        """
        if not self._signing_secret and not self._caller_owned_app:
            raise ValueError(
                'signing_secret is required to serve the Events API; pass it or set SLACK_SIGNING_SECRET. '
                'Socket Mode needs no signing secret -- use run() instead if that is what you want.'
            )
        return slack_asgi.AsyncSlackRequestHandler(self.app, path=path)

    async def start(self) -> None:
        """Connect over Socket Mode and serve until cancelled.

        Raises:
            ValueError: If no app token was configured.
        """
        if not self._app_token:
            raise ValueError(
                'app_token is required for Socket Mode; pass it or set SLACK_APP_TOKEN. '
                'The Events API needs no app token -- use http_app() instead if that is what you want.'
            )
        await slack_socket_mode.AsyncSocketModeHandler(self.app, self._app_token).start_async()

    def run(self) -> None:
        """Start Socket Mode and block. The synchronous entry point for a script."""
        anyio.run(self.start)


def _find_slack(root: AbstractCapability[BotDepsT]) -> Slack[BotDepsT] | None:
    """The agent's `Slack`, so clicks and the bot token can reach it.

    Walks the whole capability tree rather than the top level, so a `Slack`
    inside a combined or wrapped capability is still found.
    """
    found: list[Slack[BotDepsT]] = []

    def visit(capability: AbstractCapability[BotDepsT]) -> AbstractCapability[BotDepsT]:
        if isinstance(capability, Slack):
            found.append(capability)
        return capability

    root.visit_and_replace(visit)
    if len(found) > 1:
        raise ValueError(
            'SlackApp requires exactly one Slack capability. Combine its tool selection and configuration in one Slack.'
        )
    return found[0] if found else None
