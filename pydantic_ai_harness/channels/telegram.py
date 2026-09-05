"""Telegram Bot API adapter for the provider-neutral Channels host.

External assumptions verified 2026-09-04 against https://core.telegram.org/bots/api:

* webhook requests carry the configured secret in `X-Telegram-Bot-Api-Secret-Token`;
* `update_id` identifies deliveries, while messages identify their chat, optional topic, and source message;
* Bot API requests use `https://api.telegram.org/bot<token>/METHOD_NAME`;
* `sendMessage` accepts 1 to 4096 characters and supports topics and reply parameters;
* flood-control responses may include an integer `parameters.retry_after` delay.

Re-check the linked primary documentation before changing authentication, identity, or retry policy.
"""

from __future__ import annotations

import hmac
import logging
import re
from collections.abc import Collection, Mapping
from threading import Lock
from typing import TypeGuard
from weakref import finalize

import anyio
import httpx
from pydantic import TypeAdapter, ValidationError

from ._host import ChannelEvent

__all__ = (
    'TelegramChannel',
    'TelegramError',
    'TelegramPartialDeliveryError',
    'TelegramRateLimitError',
    'TelegramWebhookError',
)

_DEFAULT_API_URL = 'https://api.telegram.org'
_MAX_TEXT_CHARS = 4096
_MAX_RETRY_AFTER_SECONDS = 60
_SECRET_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,256}')
_CONVERSATION_PATTERN = re.compile(
    r'telegram:bot:([1-9][0-9]*):chat:(-?[1-9][0-9]*)(?::(topic|direct-topic):([1-9][0-9]*))?'
)
_DELIVERY_PATTERN = re.compile(r'-?[1-9][0-9]*')
_MESSAGE_PATTERN = re.compile(r'telegram:message:([1-9][0-9]*)')
_JSON_ADAPTER: TypeAdapter[object] = TypeAdapter(object)
_MAPPING_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


class TelegramError(RuntimeError):
    """A Telegram webhook payload or Bot API request is invalid."""


class TelegramWebhookError(TelegramError):
    """A Telegram webhook request fails secret-token verification."""


class TelegramPartialDeliveryError(TelegramError):
    """A multi-message reply fails after at least one chunk is confirmed."""

    def __init__(self, message: str, *, sent_chunks: int, retry_after: int | None = None) -> None:
        """Record confirmed chunks and any Telegram-provided retry delay."""
        super().__init__(message)
        self.sent_chunks = sent_chunks
        self.retry_after = retry_after


class TelegramRateLimitError(TelegramError):
    """Telegram rate-limits a reply with a caller-visible retry delay."""

    def __init__(self, message: str, *, retry_after: int) -> None:
        """Record Telegram's requested delay without forcing an unbounded wait."""
        super().__init__(message)
        self.retry_after = retry_after


class _TelegramTokenFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._lock = Lock()
        self._prefixes: dict[str, tuple[str, int]] = {}

    def register(self, token_url_prefix: str, redacted_url_prefix: str) -> None:
        with self._lock:
            current = self._prefixes.get(token_url_prefix)
            count = current[1] + 1 if current is not None else 1
            self._prefixes[token_url_prefix] = redacted_url_prefix, count

    def unregister(self, token_url_prefix: str) -> None:
        with self._lock:
            redacted_url_prefix, count = self._prefixes[token_url_prefix]
            if count == 1:
                del self._prefixes[token_url_prefix]
            else:
                self._prefixes[token_url_prefix] = redacted_url_prefix, count - 1

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == 'httpx' and record.msg == 'HTTP Request: %s %s "%s %d %s"':
            args = record.args
            if isinstance(args, tuple):  # pragma: no branch - HTTPX supplies positional logging arguments
                with self._lock:
                    replacements = tuple(
                        (token_url_prefix, redacted_url_prefix)
                        for token_url_prefix, (redacted_url_prefix, _count) in sorted(
                            self._prefixes.items(), key=lambda item: len(item[0]), reverse=True
                        )
                    )
                record.args = tuple(_redact_log_argument(value, replacements) for value in args)
        return True


_TOKEN_FILTER = _TelegramTokenFilter()


class TelegramChannel:
    """Translate verified Telegram updates and send replies through the Bot API."""

    def __init__(
        self,
        *,
        bot_id: int,
        bot_token: str,
        webhook_secret: str,
        allowed_senders: Collection[int],
        client: httpx.AsyncClient | None = None,
        api_url: str = _DEFAULT_API_URL,
    ) -> None:
        """Configure Telegram authentication, admission, and HTTP transport.

        Args:
            bot_id: Public Telegram user ID of the bot, used to scope stored identities.
            bot_token: Token issued by BotFather for Bot API requests.
            webhook_secret: Secret configured through Telegram's `setWebhook` method.
            allowed_senders: Telegram user or sender-chat IDs allowed to start runs.
            client: Optional pooled HTTP client. The caller retains ownership.
            api_url: Bot API server base URL.
        """
        if not _is_integer_id(bot_id) or bot_id <= 0:
            raise ValueError('bot_id must be a positive integer Telegram ID')
        if not bot_token:
            raise ValueError('bot_token must not be empty')
        if _SECRET_PATTERN.fullmatch(webhook_secret) is None:
            raise ValueError('webhook_secret must contain 1-256 ASCII letters, digits, underscores, or hyphens')
        if isinstance(allowed_senders, str):
            raise TypeError('allowed_senders must be a collection of integer Telegram IDs')
        sender_values: Collection[object] = allowed_senders
        if any(not _is_integer_id(sender_id) for sender_id in sender_values):
            raise TypeError('allowed_senders must contain only integer Telegram IDs')
        if any(sender_id == 0 for sender_id in allowed_senders):
            raise ValueError('allowed_senders must contain only nonzero Telegram IDs')
        sender_ids = frozenset(allowed_senders)
        if not sender_ids:
            raise ValueError('allowed_senders must contain at least one Telegram ID')
        if not api_url:
            raise ValueError('api_url must not be empty')

        self._bot_id = bot_id
        self._bot_token = bot_token
        self._webhook_secret = webhook_secret
        self._allowed_senders = sender_ids
        self._client = client
        self._api_url = api_url.rstrip('/')

        httpx_logger = logging.getLogger('httpx')
        token_url_prefix = str(httpx.URL(f'{self._api_url}/bot{self._bot_token}/'))
        _TOKEN_FILTER.register(token_url_prefix, f'{token_url_prefix.rpartition("/bot")[0]}/bot<redacted>/')
        if _TOKEN_FILTER not in httpx_logger.filters:
            httpx_logger.addFilter(_TOKEN_FILTER)
        finalize(self, _TOKEN_FILTER.unregister, token_url_prefix)

    def parse_request(self, raw_body: bytes, headers: Mapping[str, str]) -> ChannelEvent | None:
        """Verify and normalize one caller-owned Telegram webhook request.

        Unsupported update types, non-text messages, bot messages, and disallowed senders return
        `None`. The caller should atomically claim the returned `event_id` before passing the event
        to [`ChannelHost.handle`][pydantic_ai_harness.channels.ChannelHost.handle].
        """
        update_id, message = _verified_update(raw_body, headers, self._webhook_secret)
        if message is None:
            return None

        text = message.get('text')
        if not isinstance(text, str) or not text:
            return None

        identity = _message_identity(message)
        if identity is None:
            return None
        chat_id, topic_kind, topic_id, message_id = identity

        sender = self._sender(message)
        if sender is None:
            return None
        sender_kind, sender_id = sender
        if sender_id not in self._allowed_senders:
            return None

        conversation_id = f'telegram:bot:{self._bot_id}:chat:{chat_id}'
        if topic_kind is not None:
            conversation_id += f':{topic_kind}:{topic_id}'
        return ChannelEvent(
            event_id=f'telegram:bot:{self._bot_id}:update:{update_id}',
            conversation_id=conversation_id,
            sender_id=f'telegram:{sender_kind}:{sender_id}',
            text=text,
            reply_to_id=f'telegram:message:{message_id}',
            delivery_id=str(chat_id),
        )

    def _sender(self, message: Mapping[str, object]) -> tuple[str, int] | None:
        raw_sender_chat = message.get('sender_chat')
        if raw_sender_chat is not None:
            sender_chat = _mapping(raw_sender_chat)
            if sender_chat is None:
                raise TelegramError('Telegram webhook payload has invalid sender-chat identity')
            sender_chat_id = sender_chat.get('id')
            if not _is_integer_id(sender_chat_id) or sender_chat_id == 0:
                raise TelegramError('Telegram webhook payload has invalid sender-chat identity')
            return 'chat', sender_chat_id

        raw_sender = message.get('from')
        sender = _mapping(raw_sender)
        if sender is None:
            raise TelegramError('Telegram webhook payload has no sender identity')
        sender_id = sender.get('id')
        is_bot = sender.get('is_bot')
        if not _is_integer_id(sender_id) or sender_id == 0 or not isinstance(is_bot, bool):
            raise TelegramError('Telegram webhook payload has invalid sender identity')
        if is_bot:
            return None
        return 'user', sender_id

    async def reply(self, event: ChannelEvent, text: str) -> None:
        """Reply in Telegram, splitting text at the provider limit.

        A flood-control delay of at most 60 seconds is retried once. Larger and repeated delays are
        surfaced without retry. Transport failures are also surfaced because delivery may already
        have happened. If a later chunk fails, `TelegramPartialDeliveryError.sent_chunks` reports
        how many preceding chunks Telegram confirmed and `retry_after` preserves a rate-limit delay.
        """
        if not text:
            raise ValueError('text must not be empty')
        conversation_chat_id, topic_kind, topic_id = _decode_conversation(event.conversation_id, bot_id=self._bot_id)
        chat_id = _decode_delivery(event.delivery_id)
        if chat_id != conversation_chat_id:
            raise TelegramError('event delivery_id does not match its Telegram conversation_id')
        message_id = _decode_message(event.reply_to_id)

        chunks = [text[start : start + _MAX_TEXT_CHARS] for start in range(0, len(text), _MAX_TEXT_CHARS)]
        for index, chunk in enumerate(chunks):
            payload: dict[str, object] = {
                'chat_id': chat_id,
                'text': chunk,
            }
            if topic_kind == 'topic':
                payload['message_thread_id'] = topic_id
            elif topic_kind == 'direct-topic':
                payload['direct_messages_topic_id'] = topic_id
            if message_id is not None:
                payload['reply_parameters'] = {'message_id': message_id}
            try:
                await self._send_with_one_flood_retry(payload)
            except TelegramError as exc:
                if index == 0:
                    raise
                raise TelegramPartialDeliveryError(
                    f'Telegram reply failed after {index} of {len(chunks)} chunks were confirmed: {exc}',
                    sent_chunks=index,
                    retry_after=exc.retry_after if isinstance(exc, TelegramRateLimitError) else None,
                ) from None

    async def _send_with_one_flood_retry(self, payload: Mapping[str, object]) -> None:
        try:
            await self._send(payload)
        except TelegramRateLimitError as exc:
            if exc.retry_after > _MAX_RETRY_AFTER_SECONDS:
                raise
            await anyio.sleep(exc.retry_after)
            try:
                await self._send(payload)
            except TelegramRateLimitError:
                raise

    async def _send(self, payload: Mapping[str, object]) -> None:
        if self._client is not None:
            await self._post(self._client, payload)
            return
        async with httpx.AsyncClient() as client:
            await self._post(client, payload)

    async def _post(self, client: httpx.AsyncClient, payload: Mapping[str, object]) -> None:
        method = 'sendMessage'
        try:
            response = await client.post(
                f'{self._api_url}/bot{self._bot_token}/{method}',
                json=dict(payload),
            )
        except httpx.RequestError:
            raise TelegramError(f'Telegram Bot API request failed: {method}') from None

        try:
            envelope = _mapping(_JSON_ADAPTER.validate_json(response.content))
        except ValidationError:
            envelope = None
        if envelope is None:
            raise TelegramError(f'Telegram Bot API returned an invalid response: {method}')
        ok = envelope.get('ok')
        if not isinstance(ok, bool):
            raise TelegramError(f'Telegram Bot API returned an invalid response: {method}')
        if response.is_success and ok:
            result = _mapping(envelope.get('result'))
            sent_message_id = result.get('message_id') if result is not None else None
            if _is_integer_id(sent_message_id) and sent_message_id >= 0:
                return
            raise TelegramError(f'Telegram Bot API returned an invalid response: {method}')
        if ok:
            raise TelegramError(f'Telegram Bot API returned an invalid response: {method}')

        description = envelope.get('description')
        message = description if isinstance(description, str) else f'Telegram Bot API rejected {method}'
        message = message.replace(self._bot_token, '<redacted>')
        retry_after = _retry_after(envelope) if response.status_code == 429 else None
        if retry_after is not None:
            raise TelegramRateLimitError(message, retry_after=retry_after)
        raise TelegramError(message)


def _mapping(value: object) -> dict[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _verified_update(
    raw_body: bytes, headers: Mapping[str, str], webhook_secret: str
) -> tuple[int, dict[str, object] | None]:
    secret_values = [value for name, value in headers.items() if name.casefold() == 'x-telegram-bot-api-secret-token']
    if len(secret_values) > 1:
        raise TelegramWebhookError('Telegram webhook has ambiguous secret token headers')
    if (
        len(secret_values) != 1
        or _SECRET_PATTERN.fullmatch(secret_values[0]) is None
        or not hmac.compare_digest(secret_values[0], webhook_secret)
    ):
        raise TelegramWebhookError('Telegram webhook secret token is missing or invalid')

    try:
        payload = _mapping(_JSON_ADAPTER.validate_json(raw_body))
    except ValidationError:
        payload = None
    if payload is None:
        raise TelegramError('Telegram webhook payload must be a JSON object')

    update_id = payload.get('update_id')
    if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id <= 0:
        raise TelegramError('Telegram webhook payload has an invalid update_id')

    raw_message = payload.get('message')
    if raw_message is None:
        return update_id, None
    message = _mapping(raw_message)
    if message is None:
        raise TelegramError('Telegram webhook payload has an invalid message')
    return update_id, message


def _message_identity(message: Mapping[str, object]) -> tuple[int, str | None, int | None, int] | None:
    message_id = message.get('message_id')
    if not _is_integer_id(message_id):
        raise TelegramError('Telegram webhook payload has invalid message identity')
    if message_id == 0 or message.get('ephemeral_message_id') is not None:
        return None
    chat = _mapping(message.get('chat'))
    if chat is None or message_id < 0:
        raise TelegramError('Telegram webhook payload has invalid message identity')
    chat_id = chat.get('id')
    if not _is_integer_id(chat_id) or chat_id == 0:
        raise TelegramError('Telegram webhook payload has invalid chat identity')

    topic_id = message.get('message_thread_id')
    if topic_id is not None and (not _is_integer_id(topic_id) or topic_id <= 0):
        raise TelegramError('Telegram webhook payload has invalid topic identity')
    is_topic_message = message.get('is_topic_message')
    if is_topic_message is not None and is_topic_message is not True:
        raise TelegramError('Telegram webhook payload has invalid topic identity')
    if is_topic_message is True and topic_id is None:
        raise TelegramError('Telegram webhook payload has invalid topic identity')
    forum_topic_id = topic_id if is_topic_message is True else None
    direct_topic = message.get('direct_messages_topic')
    direct_topic_id: int | None = None
    if direct_topic is not None:
        direct_topic_mapping = _mapping(direct_topic)
        if direct_topic_mapping is None:
            raise TelegramError('Telegram webhook payload has invalid direct-topic identity')
        raw_direct_topic_id = direct_topic_mapping.get('topic_id')
        if not _is_integer_id(raw_direct_topic_id) or raw_direct_topic_id <= 0:
            raise TelegramError('Telegram webhook payload has invalid direct-topic identity')
        direct_topic_id = raw_direct_topic_id
    if forum_topic_id is not None and direct_topic_id is not None:
        raise TelegramError('Telegram webhook payload has ambiguous topic identity')
    if forum_topic_id is not None:
        return chat_id, 'topic', forum_topic_id, message_id
    if direct_topic_id is not None:
        return chat_id, 'direct-topic', direct_topic_id, message_id
    return chat_id, None, None, message_id


def _is_integer_id(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _redact_log_argument(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    if isinstance(value, (str, httpx.URL)):
        original_text = str(value)
        text = original_text
        for token_url_prefix, redacted_url_prefix in replacements:
            text = text.replace(token_url_prefix, redacted_url_prefix)
        return text if text != original_text else value
    return value


def _decode_conversation(conversation_id: str, *, bot_id: int) -> tuple[int, str | None, int | None]:
    match = _CONVERSATION_PATTERN.fullmatch(conversation_id)
    if match is None or int(match.group(1)) != bot_id:
        raise TelegramError('event conversation_id is not a Telegram chat identity')
    topic = match.group(4)
    return int(match.group(2)), match.group(3), int(topic) if topic is not None else None


def _decode_message(reply_to_id: str | None) -> int | None:
    if reply_to_id is None:
        return None
    match = _MESSAGE_PATTERN.fullmatch(reply_to_id)
    if match is None:
        raise TelegramError('event reply_to_id is not a Telegram message identity')
    return int(match.group(1))


def _decode_delivery(delivery_id: str | None) -> int:
    if delivery_id is None or _DELIVERY_PATTERN.fullmatch(delivery_id) is None:
        raise TelegramError('event delivery_id is not a Telegram chat identity')
    return int(delivery_id)


def _retry_after(envelope: Mapping[str, object]) -> int | None:
    parameters = _mapping(envelope.get('parameters'))
    if parameters is None:
        return None
    retry_after = parameters.get('retry_after')
    if isinstance(retry_after, bool) or not isinstance(retry_after, int) or retry_after < 0:
        return None
    return retry_after
