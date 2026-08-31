"""WhatsApp Cloud API channel adapter.

External assumptions verified 2026-08-31:

* Webhook verification uses the configured GET challenge and POST requests use
  `X-Hub-Signature-256` over the exact request body.
* Webhooks may contain batched messages and Meta retries failed deliveries for
  up to seven days.
* Free-form replies are limited to 4096 characters and to the 24-hour customer
  service window.
* Graph API v26.0 is the current default. Callers can select another supported
  version with `api_version`.

Re-check these assumptions against
https://developers.facebook.com/documentation/business-messaging/whatsapp/
before changing verification, deduplication, parsing, or delivery.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import AsyncGenerator, Iterator, Mapping
from types import TracebackType
from typing import NoReturn

import anyio
import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import Self

from pydantic_ai_harness.channels._types import (
    ChannelError,
    InboundMessage,
    WebhookRequest,
    WebhookResponse,
)
from pydantic_ai_harness.channels._webhook import RecentMessageIds, WebhookInbox

_API_BASE = 'https://graph.facebook.com'
_DEFAULT_API_VERSION = 'v26.0'
_MAX_TEXT_CHARS = 4096
_MAX_SEEN_MESSAGES = 10_000
_REQUEST_TIMEOUT = 10.0
_THROTTLING_ERROR_CODES = frozenset({4, 80007, 130429, 131056})
_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_LIST_ADAPTER = TypeAdapter(list[object])


class WhatsAppChannel:
    """Connect one business phone number through the official Cloud API."""

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        app_secret: str,
        verify_token: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        api_version: str = _DEFAULT_API_VERSION,
        max_queued_messages: int = 100,
        retry_delay: float = 1.0,
    ) -> None:
        """Configure the WhatsApp adapter without opening it.

        Args:
            access_token: System User token with WhatsApp messaging access.
            phone_number_id: Business phone number id used for replies.
            app_secret: Meta app secret used to authenticate POST webhooks.
            verify_token: Private value checked during the GET webhook handshake.
            http_client: Optional client whose lifecycle remains owned by the caller.
            api_version: Graph API version, including the leading `v`.
            max_queued_messages: Maximum verified messages waiting for the host.
            retry_delay: Delay before one retry for an explicit throttling error.

        Raises:
            ValueError: If configuration is empty or a limit is invalid.
        """
        credentials = {
            'access_token': access_token,
            'phone_number_id': phone_number_id,
            'app_secret': app_secret,
            'verify_token': verify_token,
        }
        for name, value in credentials.items():
            if not value:
                raise ValueError(f'{name} must not be empty')
        if not _valid_api_version(api_version):
            raise ValueError('api_version must look like v26.0')
        if type(max_queued_messages) is not int or max_queued_messages <= 0:
            raise ValueError('max_queued_messages must be a positive integer')
        if retry_delay < 0 or not math.isfinite(retry_delay):
            raise ValueError('retry_delay must be finite and not negative')

        self._access_token = access_token
        self._phone_number_id = phone_number_id
        self._app_secret = app_secret.encode()
        self._verify_token = verify_token
        self._provided_client = http_client
        self._client: httpx.AsyncClient | None = None
        self._api_version = api_version
        self._inbox = WebhookInbox(max_queued_messages)
        self._seen_messages = RecentMessageIds(_MAX_SEEN_MESSAGES)
        self._retry_delay = retry_delay

    async def __aenter__(self) -> Self:
        """Open the HTTP client and begin accepting message webhooks."""
        if self._client is not None:
            raise RuntimeError('WhatsAppChannel is already open')
        self._inbox.open()
        self._client = self._provided_client or httpx.AsyncClient()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the internally created HTTP client, if any."""
        self._inbox.close()
        # A failed batch can cache IDs for messages left in the old inbox.
        self._seen_messages.clear()
        await self._close_owned_client()
        self._client = None

    async def _close_owned_client(self) -> None:
        if self._provided_client is None and self._client is not None:
            with anyio.CancelScope(shield=True):
                await self._client.aclose()

    async def handle_webhook(self, request: WebhookRequest) -> WebhookResponse:
        """Verify and enqueue messages on the async thread that runs `messages()`."""
        method = request.method.upper()
        if method == 'GET':
            return self._verify_challenge(request.query)
        if method != 'POST':
            return WebhookResponse(405)
        if not self._valid_signature(request):
            return WebhookResponse(401)

        try:
            payload = _MAPPING_ADAPTER.validate_json(request.body)
        except ValidationError:
            return WebhookResponse(400)
        if payload.get('object') != 'whatsapp_business_account':
            return WebhookResponse(200)

        for message in self._parse_messages(payload):
            if message.message_id in self._seen_messages:
                continue
            if not self._inbox.put(message):
                return WebhookResponse(503)
            self._seen_messages.add(message.message_id)
        return WebhookResponse(200)

    def _verify_challenge(self, query: Mapping[str, str]) -> WebhookResponse:
        mode = query.get('hub.mode')
        token = query.get('hub.verify_token')
        challenge = query.get('hub.challenge')
        if (
            mode == 'subscribe'
            and token is not None
            and hmac.compare_digest(
                self._verify_token.encode(),
                token.encode('utf-8', 'replace'),
            )
            and challenge is not None
        ):
            return WebhookResponse(200, challenge)
        return WebhookResponse(403)

    def _valid_signature(self, request: WebhookRequest) -> bool:
        headers = {name.lower(): value for name, value in request.headers.items()}
        signature = headers.get('x-hub-signature-256')
        if signature is None:
            return False
        expected = 'sha256=' + hmac.new(self._app_secret, request.body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected.encode(), signature.encode('utf-8', 'replace'))

    def _parse_messages(self, payload: Mapping[str, object]) -> Iterator[InboundMessage]:
        entries = _list(payload.get('entry'))
        if entries is None:
            return
        for raw_entry in entries:
            entry = _mapping(raw_entry)
            if entry is None:
                continue
            changes = _list(entry.get('changes'))
            if changes is None:
                continue
            for raw_change in changes:
                change = _mapping(raw_change)
                if change is None or change.get('field') != 'messages':
                    continue
                value = _mapping(change.get('value'))
                if value is None or not self._matches_phone_number(value):
                    continue
                messages = _list(value.get('messages'))
                if messages is None:
                    continue
                for raw_message in messages:
                    message = _parse_text_message(raw_message)
                    if message is not None:
                        yield message

    def _matches_phone_number(self, value: Mapping[str, object]) -> bool:
        metadata = _mapping(value.get('metadata'))
        return metadata is not None and metadata.get('phone_number_id') == self._phone_number_id

    def messages(self) -> AsyncGenerator[InboundMessage, None]:
        """Yield verified text messages until the channel closes or the task is cancelled."""
        return self._inbox.messages()

    async def send_text(self, conversation_id: str, text: str) -> None:
        """Send free-form text in ordered chunks within the Cloud API limit."""
        if not text:
            raise ValueError('text must not be empty')
        for start in range(0, len(text), _MAX_TEXT_CHARS):
            payload: dict[str, object] = {
                'messaging_product': 'whatsapp',
                'recipient_type': 'individual',
                'to': conversation_id,
                'type': 'text',
                'text': {'body': text[start : start + _MAX_TEXT_CHARS]},
            }
            try:
                await self._post_message(payload)
            except _Throttled:
                await anyio.sleep(self._retry_delay)
                await self._post_message(payload)

    async def _post_message(self, payload: Mapping[str, object]) -> None:
        client = self._client
        if client is None:
            raise RuntimeError('WhatsAppChannel must be opened before use')
        try:
            response = await client.post(
                f'{_API_BASE}/{self._api_version}/{self._phone_number_id}/messages',
                headers={'Authorization': f'Bearer {self._access_token}'},
                json=dict(payload),
                timeout=_REQUEST_TIMEOUT,
            )
        except httpx.RequestError:
            raise ChannelError('WhatsApp Cloud API request failed: send message') from None
        try:
            envelope = _MAPPING_ADAPTER.validate_json(response.content)
        except ValidationError:
            raise ChannelError('WhatsApp Cloud API returned an invalid response: send message') from None
        messages = _list(envelope.get('messages'))
        if messages is not None:
            return
        _raise_api_error(envelope)


class _Throttled(ChannelError):
    """A Cloud API throttling error that is safe to retry once."""


def _valid_api_version(value: str) -> bool:
    if not value.startswith('v'):
        return False
    major, separator, minor = value[1:].partition('.')
    return bool(separator and major.isdigit() and minor.isdigit())


def _mapping(value: object) -> dict[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _list(value: object) -> list[object] | None:
    try:
        return _LIST_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _parse_text_message(value: object) -> InboundMessage | None:
    message = _mapping(value)
    if message is None or message.get('type') != 'text':
        return None
    sender_id = message.get('from')
    message_id = message.get('id')
    text = _mapping(message.get('text'))
    body = text.get('body') if text is not None else None
    if not isinstance(sender_id, str) or not isinstance(message_id, str) or not isinstance(body, str) or not body:
        return None
    return InboundMessage(
        conversation_id=sender_id,
        sender_id=sender_id,
        message_id=message_id,
        text=body,
    )


def _raise_api_error(envelope: Mapping[str, object]) -> NoReturn:
    error = _mapping(envelope.get('error'))
    if error is None:
        raise ChannelError('unknown WhatsApp Cloud API error')
    code = error.get('code')
    message = error.get('message')
    normalized_code = code if isinstance(code, int) else None
    normalized_message = message if isinstance(message, str) else 'unknown WhatsApp Cloud API error'
    detail = (
        f'WhatsApp Cloud API error {normalized_code}: {normalized_message}'
        if normalized_code is not None
        else normalized_message
    )
    if normalized_code in _THROTTLING_ERROR_CODES:
        raise _Throttled(detail)
    raise ChannelError(detail)
