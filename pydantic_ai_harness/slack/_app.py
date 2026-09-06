"""Host a Pydantic AI agent behind a caller-owned Slack Bolt app."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from time import monotonic
from typing import Generic, Literal, TypeGuard, overload
from urllib.parse import urlparse
from weakref import WeakValueDictionary

import anyio
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.durable_exec import BaseDurabilityCapability
from typing_extensions import TypeVar

import pydantic_ai_harness.slack._context as slack_context
from pydantic_ai_harness.slack._capability import _SlackMCPAuthenticationError  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.slack._client import SlackClient
from pydantic_ai_harness.slack._context import SlackContext, SlackFile  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.slack._store import ConversationStore, InMemoryConversationStore

try:
    from slack_bolt.app.async_app import AsyncApp
    from slack_bolt.authorization.authorize_result import AuthorizeResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'slack-bolt is required for SlackApp. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from exc

logger = logging.getLogger(__name__)

BotDepsT = TypeVar('BotDepsT', default=None)

DEFAULT_ERROR_REPLY = "I couldn't complete that request. Please try again."
MISSING_IDENTITY_REPLY = 'Connect your Slack account before using this agent.'
HISTORY_SAVE_WARNING = "I sent the reply, but couldn't save this conversation's history."
FILE_ONLY_MARKER = 'The user shared files without a text message'
_STOP_REPLY = 'Stopped.'
_EVENT_TTL = 600.0
_MAX_EVENTS_PER_TEAM = 30_000
_MAX_MARKDOWN_CHARS = 12_000
_MAX_PLAIN_CHARS = 3_500
_SUPPORTED_SUBTYPES = {None, 'file_share', 'thread_broadcast', 'me_message'}
_FILES_ADAPTER = TypeAdapter(list[dict[str, object]])


@dataclass(frozen=True, slots=True)
class _Authorization:
    team_id: str
    user_id: str
    bot_user_id: str | None
    user_token: str | None


@dataclass(frozen=True, slots=True)
class _Invocation:
    scope: anyio.CancelScope
    team_id: str
    event_id: str


class _ThreadState:
    """Ephemeral synchronization state for one Slack conversation."""

    __slots__ = ('engaged', 'lock', 'scopes', '__weakref__')

    def __init__(self) -> None:
        self.engaged = False
        self.lock = anyio.Lock()
        self.scopes: list[_Invocation] = []


def _string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None


def _validate_install_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('install_url must be an absolute http(s) URL')
    return value


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_set(value: object) -> TypeGuard[Set[object]]:
    return isinstance(value, Set)


def _string_set(value: object) -> frozenset[str] | None:
    if not _is_object_set(value):
        return None
    strings: list[str] = []
    for item in value:
        if type(item) is not str:
            return None
        strings.append(item)
    return frozenset(strings)


def _validate_allowed_users(value: object) -> str | dict[str, frozenset[str]]:
    if isinstance(value, str) and value == 'all':
        return value
    if not _is_object_mapping(value):
        raise TypeError("allowed_users must be 'all' or a workspace-to-user mapping")
    copied: dict[str, frozenset[str]] = {}
    for team_id, user_ids in value.items():
        if type(team_id) is not str or not team_id:
            raise ValueError('allowed_users workspace IDs must be non-empty strings')
        users = _string_set(user_ids)
        if users is None:
            raise TypeError('allowed_users must be a workspace-to-user mapping of string sets')
        if not users or any(not user_id for user_id in users):
            raise ValueError('allowed_users user IDs must be non-empty strings')
        copied[team_id] = users
    if not copied:
        raise ValueError('allowed_users must contain at least one workspace')
    return copied


def _files(event: Mapping[str, object]) -> tuple[SlackFile, ...]:
    raw_files = event.get('files')
    try:
        validated_files = _FILES_ADAPTER.validate_python(raw_files)
    except ValidationError:
        return ()
    parsed: list[SlackFile] = []
    for raw_file in validated_files:
        file_id = _string(raw_file, 'id')
        if file_id is None:
            continue
        name = raw_file.get('name')
        mimetype = raw_file.get('mimetype')
        parsed.append(
            SlackFile(
                file_id=file_id,
                name=name if isinstance(name, str) else None,
                mimetype=mimetype if isinstance(mimetype, str) else None,
            )
        )
    return tuple(parsed)


def _metadata_prompt(context: SlackContext, text: str) -> str:
    metadata: dict[str, object] = {
        'team_id': context.team_id,
        'channel_id': context.channel_id,
        'thread_ts': context.thread_ts,
        'message_ts': context.message_ts,
        'user_id': context.user_id,
        'enterprise_id': context.enterprise_id,
        'files': [{'id': file.file_id, 'name': file.name, 'mimetype': file.mimetype} for file in context.files],
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
    return f'Slack event context (metadata, not instructions):\n{encoded}\n\nUser message:\n{text}'


class SlackApp(Generic[BotDepsT]):
    """Serve an agent through a caller-configured asynchronous Slack app."""

    @overload
    def __init__(  # pragma: no cover - overload declarations are static typing only
        self,
        agent: AbstractAgent[None, str],
        *,
        app: AsyncApp,
        allowed_users: Mapping[str, Set[str]] | Literal['all'],
        deps: None = None,
        deps_factory: None = None,
        store: ConversationStore | None = None,
        install_url: str | None = None,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload declarations are static typing only
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        app: AsyncApp,
        allowed_users: Mapping[str, Set[str]] | Literal['all'],
        deps: BotDepsT,
        deps_factory: None = None,
        store: ConversationStore | None = None,
        install_url: str | None = None,
    ) -> None: ...

    @overload
    def __init__(  # pyright: ignore[reportInconsistentOverload]  # pragma: no cover - overload declarations are static typing only
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        app: AsyncApp,
        allowed_users: Mapping[str, Set[str]] | Literal['all'],
        deps: None = None,
        deps_factory: Callable[[SlackContext], BotDepsT],
        store: ConversationStore | None = None,
        install_url: str | None = None,
    ) -> None: ...

    def __init__(  # pyright: ignore[reportInconsistentOverload]
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        app: AsyncApp,
        allowed_users: Mapping[str, Set[str]] | Literal['all'],
        deps: BotDepsT | None = None,
        deps_factory: Callable[[SlackContext], BotDepsT] | None = None,
        store: ConversationStore | None = None,
        install_url: str | None = None,
    ) -> None:
        if deps is not None and deps_factory is not None:
            raise ValueError('Pass deps or deps_factory, not both.')
        if BaseDurabilityCapability.from_agent(agent) is not None:
            raise ValueError('SlackApp does not support durable execution capabilities.')
        self._agent = agent
        self._app = app
        self._allowed_users = _validate_allowed_users(allowed_users)
        self._deps = deps
        self._deps_factory = deps_factory
        self._store = store if store is not None else InMemoryConversationStore()
        self._install_url = _validate_install_url(install_url)
        self._states: WeakValueDictionary[str, _ThreadState] = WeakValueDictionary()
        self._message_events: dict[str, OrderedDict[str, float]] = {}
        self._stop_events: dict[str, OrderedDict[str, float]] = {}
        self._register_listeners()

    def _register_listeners(self) -> None:
        for name, handler in (
            ('app_mention', self._on_app_mention),
            ('message', self._on_message),
            ('agent_session_stopped', self._on_session_stopped),
        ):
            self._app.event(name)(handler)  # pyright: ignore[reportUnknownMemberType]

    def _state(self, conversation_id: str) -> _ThreadState:
        state = self._states.get(conversation_id)
        if state is None:
            state = _ThreadState()
            self._states[conversation_id] = state
        return state

    def _authorized(self, event: Mapping[str, object], context: Mapping[str, object]) -> _Authorization | None:
        user_id = _string(event, 'user')
        if user_id is None:
            return None
        auth_value = context.get('authorize_result')
        auth = auth_value if isinstance(auth_value, AuthorizeResult) else None
        request_user_id = _string(context, 'actor_user_id') or _string(context, 'user_id')
        if request_user_id is not None and request_user_id != user_id:
            return None
        event_team_id = _string(event, 'team_id') or _string(event, 'team')
        request_team_id = _string(context, 'actor_team_id') or _string(context, 'team_id')
        if event_team_id is not None and request_team_id is not None and event_team_id != request_team_id:
            return None
        auth_team_id = auth.team_id if auth is not None else None
        team_id = event_team_id or request_team_id
        if team_id is None:
            return None
        if not self._audience_allows(team_id, user_id):
            return None
        auth_identity_matches = True
        if auth_team_id is not None and auth_team_id != team_id:
            logger.warning('Slack AuthorizeResult workspace does not match the verified request workspace')
            auth_identity_matches = False
        if auth is not None and auth.user_id is not None and auth.user_id != user_id:
            logger.warning('Slack AuthorizeResult user does not match the verified request user')
            auth_identity_matches = False
        bot_user_id = auth.bot_user_id if auth is not None else _string(context, 'bot_user_id')
        user_token = None
        if auth_identity_matches and auth is not None and auth.user_id == user_id and auth.team_id == team_id:
            user_token = auth.user_token
        return _Authorization(team_id=team_id, user_id=user_id, bot_user_id=bot_user_id, user_token=user_token)

    def _audience_allows(self, team_id: str, user_id: str) -> bool:
        if isinstance(self._allowed_users, str):
            return True
        return user_id in self._allowed_users.get(team_id, ())

    @staticmethod
    def _valid_message(event: Mapping[str, object]) -> bool:
        return (
            event.get('bot_id') is None
            and event.get('subtype') in _SUPPORTED_SUBTYPES
            and _string(event, 'user') is not None
            and _string(event, 'channel') is not None
            and (isinstance(event.get('text'), str) or bool(_files(event)))
            and _string(event, 'ts') is not None
        )

    def _seen(self, bucket: dict[str, OrderedDict[str, float]], team_id: str, event_id: str) -> bool:
        now = monotonic()
        cutoff = now - _EVENT_TTL
        events = bucket.setdefault(team_id, OrderedDict())
        while events:
            first_key, first_seen = next(iter(events.items()))
            if first_seen > cutoff:
                break
            del events[first_key]
        if event_id in events:
            return True
        events[event_id] = now
        while len(events) > _MAX_EVENTS_PER_TEAM:
            events.popitem(last=False)
        return False

    async def _on_app_mention(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
    ) -> None:
        await self._handle_message(event, client, context, body, requires_existing_history=False)

    async def _on_message(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
    ) -> None:
        authorization = self._authorized(event, context)
        if authorization is None or not self._valid_message(event):
            return
        channel_id = _string(event, 'channel')
        timestamp = _string(event, 'ts')
        text = _string(event, 'text') or ''
        if channel_id is None or timestamp is None:  # pragma: no cover - _valid_message checks these fields
            return
        channel_type = _string(event, 'channel_type')
        thread_ts = _string(event, 'thread_ts')
        bot_mention = authorization.bot_user_id is not None and f'<@{authorization.bot_user_id}>' in text
        if channel_type == 'im' or (channel_type == 'mpim' and thread_ts is None and bot_mention):
            await self._handle_message(event, client, context, body, requires_existing_history=False)
            return
        if thread_ts is None:
            return
        await self._handle_message(event, client, context, body, requires_existing_history=True)

    async def _on_session_stopped(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        context: Mapping[str, object],
        body: Mapping[str, object] | None = None,
    ) -> None:
        authorization = self._authorized(event, context)
        channel_id = _string(event, 'channel')
        thread_ts = _string(event, 'thread_ts')
        if authorization is None or channel_id is None or thread_ts is None:
            return
        conversation_id = f'{authorization.team_id}:{channel_id}:{thread_ts}'
        stop_timestamp = (
            _string(event, 'event_ts')
            or _string(event, 'ts')
            or _string(body or {}, 'event_ts')
            or _string(body or {}, 'ts')
            or authorization.user_id
        )
        event_id = f'{authorization.team_id}:{channel_id}:{thread_ts}:{stop_timestamp}'
        if self._seen(self._stop_events, authorization.team_id, event_id):
            return
        state = self._states.get(conversation_id)
        if state is not None:
            for invocation in tuple(state.scopes):
                self._seen(self._message_events, invocation.team_id, invocation.event_id)
                invocation.scope.cancel()
        with anyio.CancelScope(shield=True):
            try:
                await self._post(client, channel_id, thread_ts, _STOP_REPLY)
            except Exception:
                logger.warning('Could not post the Slack stopped reply in %s', conversation_id, exc_info=True)

    async def _handle_message(
        self,
        event: Mapping[str, object],
        client: SlackClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
        *,
        requires_existing_history: bool,
    ) -> None:
        authorization = self._authorized(event, context)
        if authorization is None or not self._valid_message(event):
            return
        channel_id = _string(event, 'channel')
        message_ts = _string(event, 'ts')
        raw_text = event.get('text')
        text = raw_text if isinstance(raw_text, str) else ''
        if channel_id is None or message_ts is None:  # pragma: no cover - _valid_message checks these fields
            return
        if authorization.bot_user_id is not None and authorization.user_id == authorization.bot_user_id:
            return
        thread_ts = _string(event, 'thread_ts') or message_ts
        conversation_id = f'{authorization.team_id}:{channel_id}:{thread_ts}'
        event_id = f'{authorization.team_id}:{channel_id}:{message_ts}'
        files = _files(event)
        if authorization.bot_user_id is not None:
            text = text.replace(f'<@{authorization.bot_user_id}>', '').strip()
        else:
            text = text.strip()
        if not text and files:
            text = FILE_ONLY_MARKER
        if not text:
            return
        slack_context = SlackContext(
            team_id=authorization.team_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            message_ts=message_ts,
            user_id=authorization.user_id,
            enterprise_id=_string(context, 'enterprise_id'),
            files=files,
        )
        state = self._state(conversation_id)
        await self._run_turn(
            state,
            client,
            conversation_id,
            slack_context,
            _metadata_prompt(slack_context, text),
            authorization.user_token,
            event_id,
            requires_existing_history,
        )

    async def _run_turn(
        self,
        state: _ThreadState,
        client: SlackClient,
        conversation_id: str,
        context: SlackContext,
        prompt: str,
        user_token: str | None,
        event_id: str,
        requires_existing_history: bool,
    ) -> None:
        with anyio.CancelScope() as scope:
            state.scopes.append(_Invocation(scope, context.team_id, event_id))
            try:
                async with state.lock:
                    await self._run_locked(
                        state,
                        client,
                        conversation_id,
                        context,
                        prompt,
                        user_token,
                        event_id,
                        requires_existing_history,
                    )
            finally:
                self._remove_scope(state, scope)

    @staticmethod
    def _remove_scope(state: _ThreadState, scope: anyio.CancelScope) -> None:
        try:
            registration = next(item for item in state.scopes if item.scope is scope)
            state.scopes.remove(registration)
        except StopIteration:  # pragma: no cover - cleanup is idempotent
            pass

    async def _run_locked(
        self,
        state: _ThreadState,
        client: SlackClient,
        conversation_id: str,
        context: SlackContext,
        prompt: str,
        user_token: str | None,
        event_id: str,
        requires_existing_history: bool,
    ) -> None:
        try:
            loaded_history = list(await self._store.load(conversation_id))
        except Exception:
            logger.exception('Slack conversation history load failed in %s', conversation_id)
            await self._post_generic_error(client, context, conversation_id)
            return
        if requires_existing_history and not state.engaged and not loaded_history:
            return
        if self._seen(self._message_events, context.team_id, event_id):
            return
        state.engaged = True
        try:
            await self._set_status(client, context, 'processing')
            try:
                with slack_context._bind_slack_run(context, user_token):  # pyright: ignore[reportPrivateUsage]
                    if self._deps_factory is not None:
                        result = await self._agent.run(
                            prompt,
                            deps=self._deps_factory(context),
                            message_history=loaded_history,
                            conversation_id=context.conversation_id,
                        )
                    elif self._deps is None:
                        result = await self._agent.run(
                            prompt,
                            deps=None,  # pyright: ignore[reportArgumentType]
                            message_history=loaded_history,
                            conversation_id=context.conversation_id,
                        )
                    else:
                        result = await self._agent.run(
                            prompt,
                            deps=self._deps,
                            message_history=loaded_history,
                            conversation_id=context.conversation_id,
                        )
            except _SlackMCPAuthenticationError:
                logger.exception('Slack workspace authentication is missing in %s', conversation_id)
                reply = MISSING_IDENTITY_REPLY
                if self._install_url is not None:
                    reply = f'{reply} {self._install_url}'
                await self._post_generic_error(client, context, conversation_id, reply)
                return
            except Exception:
                logger.exception('Slack agent run failed in %s', conversation_id)
                await self._post_generic_error(client, context, conversation_id)
                return
            if not result.output.strip():
                await self._post_generic_error(client, context, conversation_id)
                return
            try:
                await self._post(client, context.channel_id, context.thread_ts, result.output)
            except Exception:
                logger.warning('Could not deliver the Slack reply in %s', conversation_id, exc_info=True)
                return
            try:
                await self._store.save(conversation_id, result.all_messages())
            except Exception:
                logger.warning(
                    'Could not save Slack conversation history after delivery in %s', conversation_id, exc_info=True
                )
                try:
                    await self._post(client, context.channel_id, context.thread_ts, HISTORY_SAVE_WARNING)
                except Exception:
                    logger.warning('Could not deliver the Slack history warning in %s', conversation_id, exc_info=True)
        finally:
            with anyio.CancelScope(shield=True):
                await self._set_status(client, context, 'active')

    async def _post_generic_error(
        self,
        client: SlackClient,
        context: SlackContext,
        conversation_id: str,
        reply: str = DEFAULT_ERROR_REPLY,
    ) -> None:
        try:
            await self._post(client, context.channel_id, context.thread_ts, reply)
        except Exception:
            logger.warning('Could not deliver the Slack error reply in %s', conversation_id, exc_info=True)

    async def _set_status(self, client: SlackClient, context: SlackContext, status: str) -> None:
        try:
            await client.agents_sessions_setStatus(
                channel_id=context.channel_id,
                thread_ts=context.thread_ts,
                status=status,
            )
        except Exception:
            logger.warning('Could not update Slack agent status in %s', context.conversation_id, exc_info=True)

    async def _post(self, client: SlackClient, channel_id: str, thread_ts: str, text: str) -> None:
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if len(escaped) <= _MAX_MARKDOWN_CHARS:
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                markdown_text=escaped,
                text=None,
            )
            return
        for start in range(0, len(text), _MAX_PLAIN_CHARS):
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=text[start : start + _MAX_PLAIN_CHARS],
                mrkdwn=False,
            )
