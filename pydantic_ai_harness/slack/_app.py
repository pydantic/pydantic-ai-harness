"""Host a text-output Pydantic AI agent in Slack."""

from __future__ import annotations

import inspect
import logging
import os
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Generic, Protocol

import anyio
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage
from pydantic_ai.result import StreamedRunResult
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_bolt.adapter.asgi.utils import scope_type
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.context.async_context import AsyncBoltContext
from slack_bolt.context.say.async_say import AsyncSay
from slack_bolt.context.say_stream.async_say_stream import AsyncSayStream
from slack_bolt.context.set_status.async_set_status import AsyncSetStatus
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from typing_extensions import TypeIs, TypeVar

logger = logging.getLogger(__name__)

AgentDepsT = TypeVar('AgentDepsT')
ValueT = TypeVar('ValueT')

_FILES_ADAPTER = TypeAdapter(list[dict[str, object]])
_MAX_RECENT_EVENTS = 10_000
_MAX_CONCURRENT_RUNS = 100
_STREAM_FLUSH_SECONDS = 0.5


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackContext:
    """Where a Slack message came from, and the tokens `SlackApp` holds for that workspace.

    `deps_factory` receives one of these per message. `user_token` is set only when the app was
    installed with user scopes.
    """

    team_id: str
    channel_id: str
    thread_ts: str
    message_ts: str
    user_id: str
    bot_token: str | None = field(default=None, repr=False)
    user_token: str | None = field(default=None, repr=False)


class SlackHistory(Protocol):
    """Store model-message history by Slack thread."""

    async def load(self, thread_key: str) -> Sequence[ModelMessage]:
        """Load the messages saved for a Slack thread."""
        ...

    async def save(self, thread_key: str, messages: Sequence[ModelMessage]) -> None:
        """Replace the messages saved for a Slack thread."""
        ...

    async def delete(self, thread_key: str) -> None:
        """Delete the messages saved for a Slack thread."""
        ...


@dataclass(slots=True)
class _HistoryEntry:
    written_at: float
    messages: list[ModelMessage]


class InMemorySlackHistory:
    """Keep a bounded, expiring set of Slack thread histories in memory."""

    def __init__(self, *, ttl_seconds: float = 86400, max_threads: int = 1000) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_threads = max_threads
        self._entries: OrderedDict[str, _HistoryEntry] = OrderedDict()
        self._lock = anyio.Lock()

    async def load(self, thread_key: str) -> Sequence[ModelMessage]:
        """Load a copy of a thread's messages, removing it first if it expired."""
        async with self._lock:
            entry = self._entries.get(thread_key)
            if entry is None:
                return ()
            if time.monotonic() - entry.written_at >= self._ttl_seconds:
                del self._entries[thread_key]
                return ()
            return list(entry.messages)

    async def save(self, thread_key: str, messages: Sequence[ModelMessage]) -> None:
        """Replace a thread's messages and evict the oldest thread if needed."""
        async with self._lock:
            self._entries.pop(thread_key, None)
            self._entries[thread_key] = _HistoryEntry(time.monotonic(), list(messages))
            while len(self._entries) > self._max_threads:
                self._entries.popitem(last=False)

    async def delete(self, thread_key: str) -> None:
        """Delete a thread's messages if present."""
        async with self._lock:
            self._entries.pop(thread_key, None)


@dataclass(slots=True)
class _ThreadLock:
    lock: anyio.Lock
    users: int = 1


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _file_names(event: Mapping[str, object]) -> list[str]:
    try:
        values = _FILES_ADAPTER.validate_python(event.get('files'))
    except ValidationError:
        return []
    return [name for value in values if (name := _string(value.get('name')) or _string(value.get('id'))) is not None]


def _is_awaitable(value: ValueT | Awaitable[ValueT]) -> TypeIs[Awaitable[ValueT]]:
    return inspect.isawaitable(value)


class SlackApp(Generic[AgentDepsT]):
    """Expose a text-output Pydantic AI agent through Slack Events API or Socket Mode."""

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, str],
        *,
        bot_token: str | None = None,
        signing_secret: str | None = None,
        app_token: str | None = None,
        oauth_settings: AsyncOAuthSettings | None = None,
        deps_factory: Callable[[SlackContext], AgentDepsT | Awaitable[AgentDepsT]] | None = None,
        history: SlackHistory | None = None,
        error_reply: str = 'Sorry, something went wrong. Please try again.',
        client: AsyncWebClient | None = None,
    ) -> None:
        """Configure Slack hosting without opening a connection or making an API call."""
        resolved_signing_secret = signing_secret or os.environ.get('SLACK_SIGNING_SECRET')
        if not resolved_signing_secret:
            raise ValueError('SlackApp requires signing_secret or the SLACK_SIGNING_SECRET environment variable.')
        resolved_bot_token = bot_token or os.environ.get('SLACK_BOT_TOKEN')
        if not resolved_bot_token and oauth_settings is None:
            raise ValueError('SlackApp requires bot_token or the SLACK_BOT_TOKEN environment variable.')
        if agent.output_type is not str:
            raise TypeError('SlackApp posts text replies and requires an agent whose output_type is exactly str.')

        if client is not None and oauth_settings is None:
            client.token = resolved_bot_token

        self._agent = agent
        self._app_token = app_token
        self._deps_factory = deps_factory
        self._history = history if history is not None else InMemorySlackHistory()
        self._error_reply = error_reply
        self._recent_events: OrderedDict[str, None] = OrderedDict()
        self._thread_locks: dict[str, _ThreadLock] = {}
        self._thread_locks_guard = anyio.Lock()
        self._run_semaphore = anyio.Semaphore(_MAX_CONCURRENT_RUNS)
        self._bolt = AsyncApp(
            token=resolved_bot_token if oauth_settings is None else None,
            signing_secret=resolved_signing_secret,
            oauth_settings=oauth_settings,
            client=client,
        )
        self._handler = AsyncSlackRequestHandler(self._bolt, path='/slack/events')

        # Installed Bolt's AsyncioListenerRunner.run awaits automatic event acknowledgement at
        # slack_bolt/listener/asyncio_runner.py:103-106, then schedules the listener coroutine with
        # asyncio.ensure_future at line 137 when process_before_response is false. AsyncApp's default is false,
        # so every operation in this listener runs after Bolt has produced the HTTP acknowledgement.
        self._bolt.event('app_mention')(self._on_message)  # pyright: ignore[reportUnknownMemberType]
        self._bolt.message(matchers=[self._matches_message])(  # pyright: ignore[reportUnknownMemberType]
            self._on_message
        )

    @property
    def bolt(self) -> AsyncApp:
        """Return the underlying Bolt app so callers can register additional listeners."""
        return self._bolt

    async def __call__(
        self,
        scope: scope_type,
        receive: Callable[[], Awaitable[dict[str, object]]],
        send: Callable[[dict[str, object]], Awaitable[None]],
    ) -> None:
        """Handle Slack Events API and OAuth requests as an ASGI application."""
        await self._handler(scope, receive, send)

    async def serve(self) -> None:
        """Serve this app over Slack Socket Mode until cancelled."""
        app_token = self._app_token or os.environ.get('SLACK_APP_TOKEN')
        if not app_token:
            raise ValueError('SlackApp.serve() requires app_token or the SLACK_APP_TOKEN environment variable.')
        handler = AsyncSocketModeHandler(self._bolt, app_token)
        try:
            await handler.start_async()
        finally:
            with anyio.CancelScope(shield=True):
                await handler.close_async()

    async def _matches_message(self, event: Mapping[str, object]) -> bool:
        if not self._valid_event(event):
            return False
        channel_type = _string(event.get('channel_type'))
        if channel_type == 'im':
            return True
        return channel_type in {'channel', 'group'} and _string(event.get('thread_ts')) is not None

    async def _on_message(
        self,
        event: Mapping[str, object],
        context: AsyncBoltContext,
        say: AsyncSay,
        request: AsyncBoltRequest,
    ) -> None:
        if not self._valid_event(event):
            return
        bot_user_id = _string(context.bot_user_id)
        user_id = _string(event.get('user'))
        if bot_user_id is None or user_id == bot_user_id:
            return
        slack_context = self._slack_context(event, context)
        if slack_context is None:
            return

        text_value = event.get('text')
        text = text_value if isinstance(text_value, str) else ''
        text = text.replace(f'<@{bot_user_id}>', '').strip()
        file_names = _file_names(event)
        if not text and not file_names:
            return

        event_id = _string(request.body.get('event_id'))
        if event_id is not None and not self._claim_event(event_id):
            return

        is_mention = event.get('type') == 'app_mention'
        channel_type = _string(event.get('channel_type'))
        thread_key = self._thread_key(slack_context)

        async with self._ordered_thread(thread_key):
            try:
                messages = await self._history.load(thread_key)
                if not is_mention and channel_type != 'im' and not messages:
                    return
                async with self._run_semaphore:
                    await self._run(
                        slack_context,
                        text,
                        file_names,
                        messages,
                        say,
                        context.say_stream,
                        context.set_status,
                    )
            except RunCancelled:
                raise
            except Exception:
                logger.exception(
                    'Slack agent run failed for team %s, channel %s',
                    slack_context.team_id,
                    slack_context.channel_id,
                )
                try:
                    await say(text=self._error_reply, thread_ts=slack_context.thread_ts)
                except Exception:
                    logger.exception(
                        'Posting the Slack error reply failed for team %s, channel %s',
                        slack_context.team_id,
                        slack_context.channel_id,
                    )

    @staticmethod
    def _valid_event(event: Mapping[str, object]) -> bool:
        return (
            _string(event.get('bot_id')) is None
            and _string(event.get('subtype')) is None
            and _string(event.get('user')) is not None
            and _string(event.get('ts')) is not None
        )

    @staticmethod
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
            bot_token=_string(context.bot_token),
            user_token=_string(context.user_token),
        )

    @staticmethod
    def _thread_key(context: SlackContext) -> str:
        return f'{context.team_id}:{context.channel_id}:{context.thread_ts}'

    def _claim_event(self, event_id: str) -> bool:
        # Slack retries a delivery it did not get a 200 for; the retry carries the same event_id.
        if event_id in self._recent_events:
            return False
        self._recent_events[event_id] = None
        if len(self._recent_events) > _MAX_RECENT_EVENTS:
            self._recent_events.popitem(last=False)
        return True

    @asynccontextmanager
    async def _ordered_thread(self, thread_key: str) -> AsyncGenerator[None]:
        async with self._thread_locks_guard:
            entry = self._thread_locks.get(thread_key)
            if entry is None:
                entry = _ThreadLock(anyio.Lock())
                self._thread_locks[thread_key] = entry
            else:
                entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            async with self._thread_locks_guard:
                entry.users -= 1
                if entry.users == 0:
                    del self._thread_locks[thread_key]

    async def _run(
        self,
        context: SlackContext,
        text: str,
        file_names: Sequence[str],
        messages: Sequence[ModelMessage],
        say: AsyncSay,
        say_stream: AsyncSayStream | None,
        set_status: AsyncSetStatus | None,
    ) -> None:
        if set_status is not None:
            try:
                await set_status(status='Thinking...')
            except SlackApiError:
                logger.exception(
                    'Setting Slack thread status failed for team %s, channel %s', context.team_id, context.channel_id
                )

        prompt = self._prompt(text, file_names)
        metadata = {
            'team_id': context.team_id,
            'channel_id': context.channel_id,
            'thread_ts': context.thread_ts,
            'message_ts': context.message_ts,
            'user_id': context.user_id,
        }
        deps_factory = self._deps_factory
        if deps_factory is None:
            run = self._agent.run_stream(  # pyright: ignore[reportArgumentType]
                prompt, message_history=messages or None, metadata=metadata
            )
        else:
            deps_or_awaitable = deps_factory(context)
            if _is_awaitable(deps_or_awaitable):
                deps = await deps_or_awaitable
            else:
                deps = deps_or_awaitable
            run = self._agent.run_stream(
                prompt,
                message_history=messages or None,
                deps=deps,
                metadata=metadata,
            )
        async with run as result:
            await self._deliver(result, say, say_stream, context.thread_ts)
            await self._history.save(self._thread_key(context), result.all_messages())

    @staticmethod
    def _prompt(text: str, file_names: Sequence[str]) -> str:
        if not file_names:
            return text
        attachment_text = f'Attached Slack files: {", ".join(file_names)}'
        return f'{text}\n\n{attachment_text}' if text else attachment_text

    @staticmethod
    async def _deliver(
        result: StreamedRunResult[AgentDepsT, str],
        say: AsyncSay,
        say_stream: AsyncSayStream | None,
        thread_ts: str,
    ) -> str:
        # Core groups deltas that arrive within the debounce window, so each Slack call carries about half a
        # second of text instead of one call per token.
        if say_stream is None:
            text = ''.join([chunk async for chunk in result.stream_text(delta=True, debounce_by=None)])
            await say(text=text, thread_ts=thread_ts)
            return text

        streamer = await say_stream(buffer_size=1, thread_ts=thread_ts)
        parts: list[str] = []
        async for chunk in result.stream_text(delta=True, debounce_by=_STREAM_FLUSH_SECONDS):
            parts.append(chunk)
            await streamer.append(markdown_text=chunk)  # pyright: ignore[reportUnknownMemberType]
        await streamer.stop()  # pyright: ignore[reportUnknownMemberType]
        return ''.join(parts)
