"""Discord Gateway and REST adapter for the Channels API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
from collections.abc import AsyncGenerator, Collection
from types import TracebackType
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import anyio
import httpx
from pydantic import TypeAdapter, ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from ._host import ChannelEvent

# Discord protocol assumptions, verified 2026-09-05:
# - Gateway v10 heartbeats, resume state, Identify limits, and session-invalidating close codes:
#   https://docs.discord.com/developers/events/gateway
#   https://docs.discord.com/developers/topics/opcodes-and-status-codes
# - Bot authentication, message length, nonce idempotency, and REST rate-limit retry fields:
#   https://docs.discord.com/developers/reference
#   https://docs.discord.com/developers/resources/message#create-message
#   https://docs.discord.com/developers/topics/rate-limits
# - Default and reply message types are the user-authored text events admitted here:
#   https://docs.discord.com/developers/resources/message#message-object-message-types
# Re-check these pages before changing Gateway opcodes, intents, message types, authentication, or retry policy.
_API_BASE_URL = 'https://discord.com/api/v10'
_GATEWAY_URL = 'wss://gateway.discord.gg/?v=10&encoding=json'
_DEFAULT_INTENTS = (1 << 9) | (1 << 12)
_MESSAGE_CONTENT_INTENT = 1 << 15
_USER_MESSAGE_TYPES = {0, 19}
_MAX_REPLY_LENGTH = 20_000
_MAX_RETRY_AFTER = 60.0
_FATAL_CLOSE_CODES = {4004, 4010, 4011, 4012, 4013, 4014}
_SESSION_INVALIDATING_CLOSE_CODES = {1000, 1001, 4003, 4007, 4009}
_HELLO_TIMEOUT = 30.0
_MAX_CONSECUTIVE_IDENTIFY_ATTEMPTS = 100
_JSON_OBJECT = TypeAdapter(dict[str, object])
_JSON_OBJECTS = TypeAdapter(list[dict[str, object]])
_GATEWAY_LOGGER = logging.Logger('pydantic_ai_harness.channels.discord.gateway', level=logging.CRITICAL + 1)


class DiscordChannelError(RuntimeError):
    """A Discord Gateway or REST operation failed."""


class DiscordChannel:
    """Translate Discord messages and replies at the provider-neutral Channels boundary.

    The async event iterator reconnects and resumes Gateway sessions when Discord permits it.
    Durable event claiming remains the caller's responsibility; repeated Discord deliveries
    retain the same `ChannelEvent.event_id`.
    """

    def __init__(
        self,
        token: str,
        *,
        allowed_user_ids: Collection[str] | None,
        allowed_guild_ids: Collection[str] | None = (),
        require_mention: bool = True,
        intents: int | None = None,
        api_base_url: str = _API_BASE_URL,
        gateway_url: str = _GATEWAY_URL,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure Discord authentication, admission policy, and transport endpoints.

        Passing `None` for an allowlist explicitly admits every ID in that category. Guild
        messages are denied by default, while direct messages follow `allowed_user_ids`.
        """
        if not token:
            raise ValueError('Discord bot token must not be empty')
        self._token = token
        self._allowed_user_ids = None if allowed_user_ids is None else frozenset(allowed_user_ids)
        self._allowed_guild_ids = None if allowed_guild_ids is None else frozenset(allowed_guild_ids)
        self._require_mention = require_mention
        self._intents = (
            _DEFAULT_INTENTS | (_MESSAGE_CONTENT_INTENT if not require_mention else 0) if intents is None else intents
        )
        self._api_base_url = api_base_url.rstrip('/')
        self._gateway_url = gateway_url
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._sequence: int | None = None
        self._session_id: str | None = None
        self._resume_gateway_url: str | None = None
        self._bot_user_id: str | None = None
        self._heartbeat_acknowledged = True
        self._last_identify_at: float | None = None
        self._identify_attempts = 0
        self._rest_lock = anyio.Lock()
        self._rest_blocked_until = 0.0
        self._events_active = False
        self._gateway_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> DiscordChannel:
        """Enter the adapter lifecycle."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the adapter lifecycle and close owned resources."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close active Gateway and adapter-owned HTTP resources."""
        gateway_task = self._gateway_task
        if gateway_task is not None:
            gateway_task.cancel()
        with anyio.CancelScope(shield=True):
            if gateway_task is not None:
                await asyncio.gather(gateway_task, return_exceptions=True)
                if self._gateway_task is gateway_task:
                    self._gateway_task = None
                    self._clear_session()
                    self._events_active = False
            if self._owns_http_client and self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None

    async def events(self) -> AsyncGenerator[ChannelEvent, None]:
        """Yield admitted text messages while maintaining the Discord Gateway session."""
        if self._events_active:
            raise DiscordChannelError('DiscordChannel supports one active event iterator')
        self._events_active = True
        queue: asyncio.Queue[ChannelEvent] = asyncio.Queue(maxsize=100)
        gateway_task = asyncio.create_task(self._gateway_loop(queue))
        self._gateway_task = gateway_task
        next_event: asyncio.Task[ChannelEvent] | None = None
        try:
            while True:
                next_event = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait((gateway_task, next_event), return_when=asyncio.FIRST_COMPLETED)
                if gateway_task in done:
                    next_event.cancel()
                    await asyncio.gather(next_event, return_exceptions=True)
                    next_event = None
                    await gateway_task
                    raise DiscordChannelError('Discord Gateway event loop stopped unexpectedly')  # pragma: no cover
                event = next_event.result()
                next_event = None
                yield event
        finally:
            if next_event is not None:
                next_event.cancel()
            gateway_task.cancel()
            with anyio.CancelScope(shield=True):
                if next_event is not None:
                    await asyncio.gather(next_event, return_exceptions=True)
                await asyncio.gather(gateway_task, return_exceptions=True)
                if self._gateway_task is gateway_task:
                    self._gateway_task = None
                    self._clear_session()
                    self._events_active = False

    async def _gateway_loop(self, queue: asyncio.Queue[ChannelEvent]) -> None:
        reconnect_delay = 1.0
        while True:
            established = asyncio.Event()
            using_resume_gateway = self._resume_gateway_url is not None
            try:
                gateway_url = self._gateway_connection_url()
                async with connect(
                    gateway_url, ping_interval=None, close_timeout=1, max_size=None, logger=_GATEWAY_LOGGER
                ) as websocket:
                    await self._read_connection(websocket, queue, established)
            except ConnectionClosed as exc:
                close = exc.rcvd or exc.sent
                close_code = close.code if close is not None else None
                if close_code in _FATAL_CLOSE_CODES:
                    raise DiscordChannelError(
                        f'Discord Gateway rejected the connection (close code {close_code})'
                    ) from exc
                if close_code in _SESSION_INVALIDATING_CLOSE_CODES:
                    self._clear_session()
            except (OSError, TimeoutError):
                if using_resume_gateway:
                    self._clear_session()
            except InvalidStatus as exc:
                status_code = exc.response.status_code
                if using_resume_gateway:
                    self._clear_session()
                elif status_code < 500:
                    raise DiscordChannelError(f'Discord Gateway handshake failed with HTTP {status_code}') from exc
            if established.is_set():
                reconnect_delay = 1.0
            await anyio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)

    async def reply(self, event: ChannelEvent, text: str) -> None:
        """Post a reply to the source Discord message, honoring REST rate limits."""
        if not text:
            raise DiscordChannelError('Discord replies must not be empty')
        if len(text) > _MAX_REPLY_LENGTH:
            raise DiscordChannelError('Discord replies must not exceed 20,000 characters')
        chunks = [text[index : index + 2000] for index in range(0, len(text), 2000)]
        reply_digest = hashlib.blake2s(text.encode()).digest()
        async with self._rest_lock:
            for index, chunk in enumerate(chunks):
                await self._send_chunk(event, chunk, index, reply_digest)

    def _gateway_connection_url(self) -> str:
        base = self._resume_gateway_url or self._gateway_url
        parts = urlsplit(base)
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key not in {'v', 'encoding', 'compress'}
        ]
        query.extend((('v', '10'), ('encoding', 'json')))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    async def _read_connection(
        self,
        websocket: ClientConnection,
        queue: asyncio.Queue[ChannelEvent],
        established: asyncio.Event,
    ) -> None:
        heartbeat_task = await self._gateway_handshake(websocket)
        if heartbeat_task is None:
            return
        try:
            while True:
                raw_message = await websocket.recv()
                payload = self._parse_payload(raw_message)
                if payload is None:
                    continue
                opcode = self._integer(payload.get('op'))
                if opcode == 10:
                    self._hello_interval(payload, repeated=True)
                elif opcode == 0:
                    if not await self._handle_dispatch(websocket, queue, established, payload):
                        return
                elif opcode == 1:
                    await self._send_heartbeat(websocket, expect_ack=False)
                elif opcode == 7:
                    await websocket.close(code=4000)
                elif opcode == 9:
                    if payload.get('d') is not True:
                        self._clear_session()
                    await websocket.close(code=4000)
                elif opcode == 11:
                    self._heartbeat_acknowledged = True
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def _gateway_handshake(self, websocket: ClientConnection) -> asyncio.Task[None] | None:
        with anyio.fail_after(_HELLO_TIMEOUT):
            while True:
                payload = self._parse_payload(await websocket.recv())
                if payload is None:
                    continue
                opcode = self._integer(payload.get('op'))
                if opcode == 10:
                    interval = self._hello_interval(payload, repeated=False)
                    self._heartbeat_acknowledged = True
                    await self._authenticate(websocket)
                    return asyncio.create_task(self._heartbeat(websocket, interval))
                if opcode == 7:
                    await websocket.close(code=4000)
                    return None
                if opcode == 9:
                    if payload.get('d') is not True:
                        self._clear_session()
                    await websocket.close(code=4000)
                    return None

    async def _handle_dispatch(
        self,
        websocket: ClientConnection,
        queue: asyncio.Queue[ChannelEvent],
        established: asyncio.Event,
        payload: dict[str, object],
    ) -> bool:
        sequence = self._integer(payload.get('s'))
        event = self._dispatch_event(payload)
        event_name = payload.get('t')
        if event_name == 'RESUMED' or (
            event_name == 'READY'
            and self._session_id is not None
            and self._resume_gateway_url is not None
            and self._bot_user_id is not None
        ):
            established.set()
        if event is not None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                await websocket.close(code=4000)
                return False
        if sequence is not None:
            self._sequence = sequence
        return True

    def _hello_interval(self, payload: dict[str, object], *, repeated: bool) -> float:
        if repeated:
            raise DiscordChannelError('Discord Gateway sent more than one Hello')
        data = self._object(payload.get('d'))
        interval_ms = self._integer(data.get('heartbeat_interval'))
        if interval_ms is None or interval_ms <= 0:
            raise DiscordChannelError('Discord Gateway Hello omitted a valid heartbeat interval')
        try:
            return interval_ms / 1000
        except OverflowError:
            raise DiscordChannelError('Discord Gateway Hello omitted a valid heartbeat interval') from None

    async def _authenticate(self, websocket: ClientConnection) -> None:
        if self._session_id is not None and self._sequence is not None:
            await websocket.send(
                json.dumps(
                    {
                        'op': 6,
                        'd': {'token': self._token, 'session_id': self._session_id, 'seq': self._sequence},
                    }
                )
            )
            return
        now = anyio.current_time()
        if self._identify_attempts >= _MAX_CONSECUTIVE_IDENTIFY_ATTEMPTS:
            raise DiscordChannelError('Discord Gateway stopped after 100 consecutive Identify attempts without READY')
        if self._last_identify_at is not None:
            await anyio.sleep(max(0, 5 - (now - self._last_identify_at)))
        self._last_identify_at = anyio.current_time()
        self._identify_attempts += 1
        await websocket.send(
            json.dumps(
                {
                    'op': 2,
                    'd': {
                        'token': self._token,
                        'intents': self._intents,
                        'properties': {
                            'os': 'pydantic-ai-harness',
                            'browser': 'pydantic-ai-harness',
                            'device': 'pydantic-ai-harness',
                        },
                    },
                }
            )
        )

    async def _heartbeat(self, websocket: ClientConnection, interval: float) -> None:
        await anyio.sleep(interval * random.random())
        while self._heartbeat_acknowledged:
            await self._send_heartbeat(websocket)
            await anyio.sleep(interval)
        await websocket.close(code=4000)

    async def _send_heartbeat(self, websocket: ClientConnection, *, expect_ack: bool = True) -> None:
        if expect_ack:
            self._heartbeat_acknowledged = False
        await websocket.send(json.dumps({'op': 1, 'd': self._sequence}))

    def _dispatch_event(self, payload: dict[str, object]) -> ChannelEvent | None:
        event_name = payload.get('t')
        data = self._object(payload.get('d'))
        if event_name == 'READY':
            self._session_id = self._string(data.get('session_id'))
            self._resume_gateway_url = self._string(data.get('resume_gateway_url'))
            user = self._object(data.get('user'))
            self._bot_user_id = self._string(user.get('id'))
            if not self._session_id or not self._resume_gateway_url or not self._bot_user_id:
                self._clear_session()
                raise DiscordChannelError('Discord Gateway READY omitted required session fields')
            self._identify_attempts = 0
            return None
        if event_name != 'MESSAGE_CREATE':
            return None
        return self._message_event(data)

    def _message_event(self, data: dict[str, object]) -> ChannelEvent | None:
        if self._integer(data.get('type')) not in _USER_MESSAGE_TYPES:
            return None
        message_id = self._string(data.get('id'))
        channel_id = self._string(data.get('channel_id'))
        content = self._string(data.get('content'))
        author = self._object(data.get('author'))
        author_id = self._string(author.get('id'))
        if not message_id or not channel_id or not content or not author_id:
            return None
        bot = author.get('bot')
        if ('bot' in author and not isinstance(bot, bool)) or bot is True:
            return None
        if 'webhook_id' in data or author_id == self._bot_user_id:
            return None
        if self._allowed_user_ids is not None and author_id not in self._allowed_user_ids:
            return None

        guild_id = self._string(data.get('guild_id'))
        if 'guild_id' in data and not guild_id:
            return None
        if guild_id is not None:
            if self._allowed_guild_ids is not None and guild_id not in self._allowed_guild_ids:
                return None
            if self._require_mention:
                bot_user_id = self._bot_user_id
                mentions = data.get('mentions')
                if bot_user_id is None or not self._mentions_user(mentions, bot_user_id):
                    return None
                content = content.replace(f'<@{bot_user_id}>', '').replace(f'<@!{bot_user_id}>', '').strip()
                if not content:
                    return None
        return ChannelEvent(
            event_id=message_id,
            conversation_id=channel_id,
            sender_id=author_id,
            text=content,
            reply_to_id=message_id,
        )

    @classmethod
    def _mentions_user(cls, value: object, user_id: str) -> bool:
        try:
            mentions = _JSON_OBJECTS.validate_python(value)
        except ValidationError:
            return False
        return any(cls._string(item.get('id')) == user_id for item in mentions)

    async def _send_chunk(self, event: ChannelEvent, text: str, index: int, reply_digest: bytes) -> None:
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient()
            self._http_client = client
        nonce_input = event.event_id.encode() + b'\0' + str(index).encode() + b'\0' + reply_digest
        nonce = hashlib.blake2s(nonce_input, digest_size=12).hexdigest()
        body: dict[str, object] = {
            'content': text,
            'allowed_mentions': {'parse': [], 'replied_user': False},
            'nonce': nonce,
            'enforce_nonce': True,
        }
        if event.reply_to_id is not None:
            body['message_reference'] = {'message_id': event.reply_to_id, 'fail_if_not_exists': False}

        url = f'{self._api_base_url}/channels/{event.conversation_id}/messages'
        for attempt in range(3):  # pragma: no branch
            blocked_for = self._rest_blocked_until - anyio.current_time()
            if blocked_for > 0:
                await anyio.sleep(blocked_for)
                self._rest_blocked_until = 0
            response = await client.post(
                url,
                json=body,
                headers={
                    'Authorization': f'Bot {self._token}',
                    'User-Agent': 'DiscordBot (https://github.com/pydantic/pydantic-ai-harness, 1)',
                },
            )
            if response.status_code != 429:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise DiscordChannelError(
                        f'Discord Create Message failed with HTTP {response.status_code}'
                    ) from exc
                return
            retry_after = self._retry_after(response)
            self._rest_blocked_until = anyio.current_time() + retry_after
            if attempt == 2:
                break
            await anyio.sleep(retry_after)
            self._rest_blocked_until = 0
        raise DiscordChannelError('Discord Create Message remained rate limited after 3 attempts')

    @classmethod
    def _retry_after(cls, response: httpx.Response) -> float:
        value = response.headers.get('Retry-After')
        if value is None:
            try:
                payload = _JSON_OBJECT.validate_json(response.content)
                retry_after = payload.get('retry_after')
                if not isinstance(retry_after, str | int | float) or isinstance(retry_after, bool):
                    raise DiscordChannelError('Discord rate-limit response omitted a valid retry delay')
                value = str(retry_after)
            except ValidationError:
                raise DiscordChannelError('Discord rate-limit response omitted a valid retry delay') from None
        try:
            delay = float(value)
        except ValueError:
            raise DiscordChannelError('Discord rate-limit response omitted a valid retry delay') from None
        if not math.isfinite(delay) or delay < 0 or delay > _MAX_RETRY_AFTER:
            raise DiscordChannelError('Discord rate-limit retry delay must be between 0 and 60 seconds')
        return delay

    def _clear_session(self) -> None:
        self._sequence = None
        self._session_id = None
        self._resume_gateway_url = None

    @staticmethod
    def _parse_payload(raw_message: str | bytes) -> dict[str, object] | None:
        if not isinstance(raw_message, str):
            return None
        try:
            return _JSON_OBJECT.validate_json(raw_message)
        except ValidationError:
            return None

    @staticmethod
    def _object(value: object) -> dict[str, object]:
        try:
            return _JSON_OBJECT.validate_python(value)
        except ValidationError:
            return {}

    @staticmethod
    def _string(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _integer(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ('DiscordChannel', 'DiscordChannelError')
