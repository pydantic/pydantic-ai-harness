"""Telegram Bot API channel adapter.

External assumptions verified 2026-08-31:

* The Bot API base is `https://api.telegram.org/bot<token>/METHOD`.
* `getUpdates` is the long-polling transport, confirms updates through the next
  offset, and cannot run while a webhook is configured.
* After at least a week without updates, Telegram may choose a random next
  `update_id`, so a stale local offset must be discarded.
* `sendMessage` text is limited to 4096 Unicode characters. UTF-16 units apply
  to message entity offsets, not the text limit.

Re-check these assumptions against https://core.telegram.org/bots/api before
changing authentication, polling, or text delivery.
"""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncGenerator, Mapping
from time import monotonic
from types import TracebackType

import anyio
import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self

from pydantic_ai_harness.channels._types import ChannelError, InboundMessage

_API_BASE = 'https://api.telegram.org'
_MAX_TEXT_CHARS = 4096
_MAX_INLINE_RETRY_DELAY = 60.0
_REQUEST_TIMEOUT = 10.0
_UPDATE_ID_RESET_SECONDS = 7 * 24 * 60 * 60
_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_LIST_ADAPTER = TypeAdapter(list[object])
_JSON_ADAPTER = TypeAdapter(object)
_FATAL_ERROR_CODES = frozenset({401, 404, 409})

logger = logging.getLogger(__name__)


class _TelegramTokenFilter(logging.Filter):
    """Redact Bot API credentials from HTTPX's completed-request log."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _redact_telegram_url(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


_TOKEN_FILTER = _TelegramTokenFilter()


class TelegramChannel:
    """Connect a Telegram bot through the official Bot API's long-polling transport."""

    def __init__(
        self,
        token: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        poll_timeout: int = 30,
        retry_delay: float = 1.0,
    ) -> None:
        """Configure the Telegram adapter without opening it.

        Args:
            token: Bot token issued by BotFather.
            http_client: Optional client whose lifecycle remains owned by the caller.
            poll_timeout: Long-poll timeout sent to `getUpdates`, in seconds.
            retry_delay: Delay after a transient polling error, in seconds.

        Raises:
            ValueError: If the token is empty, `poll_timeout` is not positive,
                or `retry_delay` is not positive or non-finite.
        """
        if not token:
            raise ValueError('token must not be empty')
        if type(poll_timeout) is not int or poll_timeout <= 0:
            raise ValueError('poll_timeout must be a positive integer')
        if retry_delay <= 0 or not math.isfinite(retry_delay):
            raise ValueError('retry_delay must be finite and positive')
        self._token = token
        self._provided_client = http_client
        self._client: httpx.AsyncClient | None = None
        self._poll_timeout = poll_timeout
        self._retry_delay = retry_delay
        self._offset: int | None = None
        self._last_update_at: float | None = None

    async def __aenter__(self) -> Self:
        """Open the HTTP client and validate the bot token with `getMe`."""
        if self._client is not None:
            raise RuntimeError('TelegramChannel is already open')
        httpx_logger = logging.getLogger('httpx')
        # Keep the singleton installed because HTTPX's logger is process-wide and adapters may overlap.
        if _TOKEN_FILTER not in httpx_logger.filters:
            httpx_logger.addFilter(_TOKEN_FILTER)
        self._client = self._provided_client or httpx.AsyncClient()
        try:
            await self._call('getMe')
        except BaseException:
            await self._close_owned_client()
            self._client = None
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the internally created HTTP client, if any."""
        await self._close_owned_client()
        self._client = None

    async def _close_owned_client(self) -> None:
        if self._provided_client is None and self._client is not None:
            with anyio.CancelScope(shield=True):
                await self._client.aclose()

    async def send_text(self, conversation_id: str, text: str) -> None:
        """Send text in ordered chunks no longer than Telegram's limit."""
        if not text:
            raise ValueError('text must not be empty')
        for start in range(0, len(text), _MAX_TEXT_CHARS):
            params = {'chat_id': conversation_id, 'text': text[start : start + _MAX_TEXT_CHARS]}
            try:
                await self._call('sendMessage', params)
            except _RateLimited as exc:
                if exc.retry_after > _MAX_INLINE_RETRY_DELAY:
                    raise
                await anyio.sleep(exc.retry_after)
                await self._call('sendMessage', params)

    async def messages(self) -> AsyncGenerator[InboundMessage, None]:
        """Yield private text messages from `getUpdates` until cancelled."""
        while True:
            if (
                self._offset is not None
                and self._last_update_at is not None
                and monotonic() - self._last_update_at >= _UPDATE_ID_RESET_SECONDS
            ):
                self._offset = None
                self._last_update_at = None
            params: dict[str, object] = {
                'timeout': self._poll_timeout,
                'allowed_updates': ['message'],
            }
            if self._offset is not None:
                params['offset'] = self._offset
            try:
                payload = await self._call('getUpdates', params, timeout=self._poll_timeout + 5)
                updates = _LIST_ADAPTER.validate_python(payload)
            except _FatalTelegramError:
                raise
            except _RateLimited as exc:
                logger.warning('Telegram polling rate limited; retrying in %s seconds', exc.retry_after)
                await anyio.sleep(exc.retry_after)
                continue
            except ChannelError as exc:
                logger.warning('Telegram polling failed: %s; retrying in %s seconds', exc, self._retry_delay)
                await anyio.sleep(self._retry_delay)
                continue
            except ValidationError:
                logger.warning(
                    'Telegram getUpdates returned an invalid result; retrying in %s seconds', self._retry_delay
                )
                await anyio.sleep(self._retry_delay)
                continue

            advanced_offset = False
            for raw_update in updates:
                update = _mapping(raw_update)
                if update is None:
                    continue
                update_id = update.get('update_id')
                if isinstance(update_id, bool) or not isinstance(update_id, int):
                    continue
                self._offset = update_id + 1
                self._last_update_at = monotonic()
                advanced_offset = True
                message = _parse_message(update.get('message'))
                if message is not None:
                    yield message
            if updates and not advanced_offset:
                logger.warning(
                    'Telegram returned updates without usable ids; retrying in %s seconds', self._retry_delay
                )
                await anyio.sleep(self._retry_delay)

    async def _call(
        self,
        method: str,
        params: Mapping[str, object] | None = None,
        *,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> object:
        client = self._client
        if client is None:
            raise RuntimeError('TelegramChannel must be opened before use')
        try:
            response = await client.post(
                f'{_API_BASE}/bot{self._token}/{method}',
                json=dict(params or {}),
                timeout=timeout,
            )
        except httpx.RequestError:
            # httpx exceptions include the request URL, whose path contains the
            # bot token. Suppress the exception chain as well as replacing the
            # message so logging this error cannot expose credentials.
            raise ChannelError(f'Telegram Bot API request failed: {method}') from None
        try:
            payload = _JSON_ADAPTER.validate_json(response.content)
        except ValidationError:
            if response.status_code in _FATAL_ERROR_CODES:
                raise _FatalTelegramError(
                    f'Telegram Bot API rejected {method} with HTTP {response.status_code}'
                ) from None
            raise ChannelError(f'Telegram Bot API returned an invalid response: {method}') from None

        envelope = _mapping(payload)
        if response.status_code in _FATAL_ERROR_CODES:
            description = envelope.get('description') if envelope is not None else None
            message = (
                _redact_telegram_url(description)
                if isinstance(description, str)
                else f'Telegram Bot API rejected {method} with HTTP {response.status_code}'
            )
            raise _FatalTelegramError(message)
        if envelope is None:
            raise ChannelError(f'Telegram Bot API returned an invalid response: {method}')
        if envelope.get('ok') is True:
            return envelope.get('result')

        description = envelope.get('description')
        message = (
            _redact_telegram_url(description) if isinstance(description, str) else 'unknown Telegram Bot API error'
        )
        error_code = envelope.get('error_code')
        if isinstance(error_code, int) and error_code in _FATAL_ERROR_CODES:
            raise _FatalTelegramError(message)
        retry_after = _retry_after(envelope)
        if retry_after is not None:
            raise _RateLimited(message, retry_after=retry_after)
        raise ChannelError(message)


class _RateLimited(ChannelError):
    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _FatalTelegramError(ChannelError):
    """A Telegram configuration or credential error that retrying cannot fix."""


def _redact_telegram_url(message: str) -> str:
    prefix = f'{_API_BASE}/bot'
    before, separator, token_and_suffix = message.partition(prefix)
    if not separator:
        return message
    _, separator, suffix = token_and_suffix.partition('/')
    if not separator:  # pragma: no cover - HTTPX logs the method path used by `_call`
        return f'{before}{prefix}<redacted>'
    return f'{before}{prefix}<redacted>/{suffix}'


def _mapping(value: object) -> dict[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _retry_after(envelope: Mapping[str, object]) -> float | None:
    parameters = _mapping(envelope.get('parameters'))
    if parameters is None:
        return None
    retry_after = parameters.get('retry_after')
    if (
        isinstance(retry_after, int | float)
        and not isinstance(retry_after, bool)
        and retry_after >= 0
        and math.isfinite(retry_after)
    ):
        return float(retry_after)
    return None


def _parse_message(value: object) -> InboundMessage | None:
    message = _mapping(value)
    if message is None:
        return None
    chat = _mapping(message.get('chat'))
    sender = _mapping(message.get('from'))
    text = message.get('text')
    message_id = message.get('message_id')
    if chat is None or sender is None or not isinstance(text, str) or not isinstance(message_id, int):
        return None
    if chat.get('type') != 'private':
        return None
    conversation_id = chat.get('id')
    sender_id = sender.get('id')
    if not isinstance(conversation_id, int | str) or not isinstance(sender_id, int | str):
        return None
    return InboundMessage(
        conversation_id=str(conversation_id),
        sender_id=str(sender_id),
        message_id=str(message_id),
        text=text,
    )
