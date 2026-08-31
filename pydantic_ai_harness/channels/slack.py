"""Slack Events API channel adapter.

External assumptions verified 2026-08-31:

* Events API requests are signed as HMAC-SHA256 over the raw request body and
  must be acknowledged within three seconds.
* `event_id` is globally unique and Slack retries failed deliveries.
* `chat.postMessage` recommends text no longer than 4000 characters and
  returns HTTP 429 with `Retry-After` when rate limited.

Re-check these assumptions against https://docs.slack.dev/apis/events-api/
before changing authentication, acknowledgement, deduplication, or delivery.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from collections.abc import AsyncGenerator, Collection, Mapping
from types import TracebackType

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

_API_BASE = 'https://slack.com/api'
_MAX_TEXT_CHARS = 4000
_MAX_SEEN_EVENTS = 10_000
_MAX_INLINE_RETRY_DELAY = 60.0
_REQUEST_TIMEOUT = 10.0
_SIGNATURE_TOLERANCE_SECONDS = 300
_MAPPING_ADAPTER = TypeAdapter(dict[str, object])


class SlackChannel:
    """Connect one Slack app installation through the HTTP Events API."""

    def __init__(
        self,
        bot_token: str,
        signing_secret: str,
        *,
        allowed_channel_ids: Collection[str] = (),
        http_client: httpx.AsyncClient | None = None,
        api_base_url: str = _API_BASE,
        max_queued_messages: int = 100,
    ) -> None:
        """Configure the Slack adapter without opening it.

        Args:
            bot_token: Bot token with `chat:write` and subscribed-event scopes.
            signing_secret: Secret used to authenticate Events API requests.
            allowed_channel_ids: Channel ids where app mentions may invoke the agent.
                Direct messages remain enabled when this is empty.
            http_client: Optional client whose lifecycle remains owned by the caller.
            api_base_url: Slack Web API root. Use `https://slack-gov.com/api` for GovSlack.
            max_queued_messages: Maximum verified messages waiting for the host.

        Raises:
            TypeError: If `allowed_channel_ids` is a string instead of a collection of ids.
            ValueError: If a credential is empty or the queue limit is not positive.
        """
        if not bot_token:
            raise ValueError('bot_token must not be empty')
        if not signing_secret:
            raise ValueError('signing_secret must not be empty')
        if not api_base_url:
            raise ValueError('api_base_url must not be empty')
        if isinstance(allowed_channel_ids, str):
            raise TypeError('allowed_channel_ids must be a collection of channel ids, not a string')
        if type(max_queued_messages) is not int or max_queued_messages <= 0:
            raise ValueError('max_queued_messages must be a positive integer')

        self._bot_token = bot_token
        self._signing_secret = signing_secret.encode()
        self._allowed_channel_ids = frozenset(allowed_channel_ids)
        self._provided_client = http_client
        self._api_base_url = api_base_url.rstrip('/')
        self._client: httpx.AsyncClient | None = None
        self._inbox = WebhookInbox(max_queued_messages)
        self._seen_events = RecentMessageIds(_MAX_SEEN_EVENTS)
        self._team_id: str | None = None
        self._bot_user_id: str | None = None

    async def __aenter__(self) -> Self:
        """Open the HTTP client and validate the bot token with `auth.test`."""
        if self._client is not None:
            raise RuntimeError('SlackChannel is already open')
        self._inbox.open()
        self._client = self._provided_client or httpx.AsyncClient()
        try:
            identity = await self._call('auth.test')
            team_id = identity.get('team_id')
            user_id = identity.get('user_id')
            if not isinstance(team_id, str) or not isinstance(user_id, str):
                raise ChannelError('Slack auth.test returned an invalid identity')
            self._team_id = team_id
            self._bot_user_id = user_id
        except BaseException:
            self._inbox.close()
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
        self._inbox.close()
        # Events still queued at shutdown were acknowledged but not handled.
        self._seen_events.clear()
        await self._close_owned_client()
        self._client = None
        self._team_id = None
        self._bot_user_id = None

    async def _close_owned_client(self) -> None:
        if self._provided_client is None and self._client is not None:
            with anyio.CancelScope(shield=True):
                await self._client.aclose()

    async def handle_webhook(self, request: WebhookRequest) -> WebhookResponse:
        """Authenticate and enqueue one Slack Events API request.

        This method performs no agent work and acknowledges Slack promptly. Call
        it on the same async event-loop thread that runs `messages()`.
        """
        if request.method.upper() != 'POST':
            return WebhookResponse(405)
        if not self._valid_signature(request):
            return WebhookResponse(401)

        try:
            payload = _MAPPING_ADAPTER.validate_json(request.body)
        except ValidationError:
            return WebhookResponse(400)

        payload_type = payload.get('type')
        if payload_type == 'url_verification':
            challenge = payload.get('challenge')
            return WebhookResponse(200, challenge) if isinstance(challenge, str) else WebhookResponse(400)
        if payload_type != 'event_callback':
            return WebhookResponse(200)

        team_id = self._team_id
        bot_user_id = self._bot_user_id
        if team_id is None or bot_user_id is None:
            return WebhookResponse(503)
        if payload.get('team_id') != team_id:
            return WebhookResponse(200)

        event_id = payload.get('event_id')
        if not isinstance(event_id, str):
            return WebhookResponse(400)
        if event_id in self._seen_events:
            return WebhookResponse(200)

        message = self._parse_event(payload.get('event'), event_id, bot_user_id)
        if message is None:
            return WebhookResponse(200)
        if not self._inbox.put(message):
            return WebhookResponse(503)

        self._seen_events.add(event_id)
        return WebhookResponse(200)

    def _valid_signature(self, request: WebhookRequest) -> bool:
        headers = {name.lower(): value for name, value in request.headers.items()}
        timestamp = headers.get('x-slack-request-timestamp')
        signature = headers.get('x-slack-signature')
        if timestamp is None or signature is None:
            return False
        try:
            request_time = int(timestamp)
        except ValueError:
            return False
        now = int(time.time())
        if request_time < now - _SIGNATURE_TOLERANCE_SECONDS or request_time > now + _SIGNATURE_TOLERANCE_SECONDS:
            return False
        base = b'v0:' + timestamp.encode() + b':' + request.body
        expected = 'v0=' + hmac.new(self._signing_secret, base, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected.encode(), signature.encode('utf-8', 'replace'))

    def _parse_event(self, value: object, event_id: str, bot_user_id: str) -> InboundMessage | None:
        event = _mapping(value)
        if event is None or event.get('subtype') is not None or event.get('bot_id') is not None:
            return None

        event_type = event.get('type')
        user_id = event.get('user')
        channel_id = event.get('channel')
        text = event.get('text')
        timestamp = event.get('ts')
        if (
            not isinstance(user_id, str)
            or user_id == bot_user_id
            or not isinstance(channel_id, str)
            or not isinstance(text, str)
            or not isinstance(timestamp, str)
        ):
            return None

        if event_type == 'message' and event.get('channel_type') == 'im':
            thread_timestamp = event.get('thread_ts')
            conversation_id = (
                f'thread:{channel_id}:{thread_timestamp}'
                if isinstance(thread_timestamp, str) and thread_timestamp
                else f'dm:{channel_id}'
            )
        elif event_type == 'app_mention' and channel_id in self._allowed_channel_ids:
            thread_timestamp = event.get('thread_ts')
            if not isinstance(thread_timestamp, str) or not thread_timestamp:
                thread_timestamp = timestamp
            conversation_id = f'thread:{channel_id}:{thread_timestamp}'
            text = text.replace(f'<@{bot_user_id}>', '').strip()
        else:
            return None

        if not text:
            return None
        return InboundMessage(
            conversation_id=conversation_id,
            sender_id=user_id,
            message_id=event_id,
            text=text,
        )

    def messages(self) -> AsyncGenerator[InboundMessage, None]:
        """Yield verified direct messages and app mentions until cancelled."""
        return self._inbox.messages()

    async def send_text(self, conversation_id: str, text: str) -> None:
        """Post text in ordered chunks no longer than Slack's recommended limit."""
        if not text:
            raise ValueError('text must not be empty')
        channel_id, thread_timestamp = _decode_conversation_id(conversation_id)
        for start in range(0, len(text), _MAX_TEXT_CHARS):
            params: dict[str, object] = {
                'channel': channel_id,
                'text': text[start : start + _MAX_TEXT_CHARS],
                'mrkdwn': False,
                'unfurl_links': False,
                'unfurl_media': False,
            }
            if thread_timestamp is not None:
                params['thread_ts'] = thread_timestamp
            try:
                await self._call('chat.postMessage', params)
            except _RateLimited as exc:
                if exc.retry_after > _MAX_INLINE_RETRY_DELAY:
                    raise
                await anyio.sleep(exc.retry_after)
                await self._call('chat.postMessage', params)

    async def _call(self, method: str, params: Mapping[str, object] | None = None) -> dict[str, object]:
        client = self._client
        if client is None:
            raise RuntimeError('SlackChannel must be opened before use')
        try:
            response = await client.post(
                f'{self._api_base_url}/{method}',
                headers={'Authorization': f'Bearer {self._bot_token}'},
                json=dict(params or {}),
                timeout=_REQUEST_TIMEOUT,
            )
        except httpx.RequestError:
            raise ChannelError(f'Slack Web API request failed: {method}') from None

        if response.status_code == 429:
            retry_after = _retry_after(response.headers)
            if retry_after is not None:
                raise _RateLimited('Slack Web API rate limited the request', retry_after=retry_after)

        try:
            envelope = _MAPPING_ADAPTER.validate_json(response.content)
        except ValidationError:
            raise ChannelError(f'Slack Web API returned an invalid response: {method}') from None
        if envelope.get('ok') is True:
            return envelope

        error = envelope.get('error')
        message = error if isinstance(error, str) else 'unknown Slack Web API error'
        raise ChannelError(message)


class _RateLimited(ChannelError):
    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _mapping(value: object) -> dict[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get('Retry-After')
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _decode_conversation_id(conversation_id: str) -> tuple[str, str | None]:
    kind, separator, route = conversation_id.partition(':')
    if not separator:
        raise ChannelError('invalid Slack conversation id')
    if kind == 'dm' and route:
        return route, None
    if kind == 'thread':
        channel_id, thread_separator, thread_timestamp = route.partition(':')
        if channel_id and thread_separator and thread_timestamp:
            return channel_id, thread_timestamp
    raise ChannelError('invalid Slack conversation id')
