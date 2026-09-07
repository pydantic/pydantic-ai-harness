"""Slack Events API adapter for the provider-neutral channel host.

External assumptions verified 2026-09:

* Slack signs `v0:{timestamp}:{raw_body}` with HMAC-SHA256 and recommends rejecting timestamps
  more than five minutes from local time: https://docs.slack.dev/authentication/verifying-requests-from-slack/
* `chat.postMessage` replies to a thread when passed its root `thread_ts`:
  https://docs.slack.dev/reference/methods/chat.postmessage
* `chat.postMessage` reports truncation in `response_metadata.warnings`:
  https://docs.slack.dev/changelog/2018-truncating-really-long-messages/
* Slack returns HTTP 429 with a `Retry-After` delay:
  https://docs.slack.dev/apis/web-api/rate-limits/

Re-check these pages before changing request verification or reply mapping.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping
from dataclasses import dataclass

import anyio
import httpx
from pydantic import TypeAdapter, ValidationError

from ._host import ChannelEvent

__all__ = ('SlackAPIError', 'SlackChannel', 'SlackError', 'SlackSignatureError', 'SlackUrlVerification')

_DEFAULT_API_URL = 'https://slack.com/api/chat.postMessage'
_MAX_REQUEST_AGE_SECONDS = 60 * 5
_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, object])
_STRING_LIST_ADAPTER = TypeAdapter(list[str])


class SlackError(Exception):
    """Base class for Slack adapter failures."""


class SlackSignatureError(SlackError):
    """The Slack request signature or timestamp is invalid."""


class SlackAPIError(SlackError):
    """An outbound Slack reply failed or was only partially delivered."""


@dataclass(frozen=True, slots=True)
class SlackUrlVerification:
    """A challenge that the caller's HTTP framework should return as plain text."""

    challenge: str


class SlackChannel:
    """Verify and normalize Slack Events API requests and post threaded replies.

    One instance represents one Slack app installation. The caller owns HTTP routing, prompt
    acknowledgment, queueing, and the lifetime of any supplied `httpx.AsyncClient`.
    """

    def __init__(
        self,
        *,
        signing_secret: str,
        bot_token: str,
        team_id: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure one Slack app installation without taking ownership of `client`."""
        if not signing_secret:
            raise ValueError('signing_secret must not be empty')
        if not bot_token:
            raise ValueError('bot_token must not be empty')
        if not team_id:
            raise ValueError('team_id must not be empty')
        self._signing_secret = signing_secret.encode()
        self._bot_token = bot_token
        self._team_id = team_id
        self._client = client
        self._post_lock = anyio.Lock()

    def parse_request(self, raw_body: bytes, headers: Mapping[str, str]) -> ChannelEvent | SlackUrlVerification | None:
        """Verify a raw Slack request and normalize supported message events.

        URL verification returns a challenge. Unsupported events and bot-authored messages return
        `None`, allowing the caller to acknowledge them without starting an agent run.
        """
        self._verify(raw_body, headers)
        payload = _json_object(raw_body)
        request_type = _string(payload, 'type')
        if request_type == 'url_verification':
            return SlackUrlVerification(_required_nonempty_string(payload, 'challenge'))
        if request_type != 'event_callback':
            return None
        if _required_nonempty_string(payload, 'team_id') != self._team_id:
            raise SlackError('Slack event does not belong to this workspace installation')

        event = _mapping(payload.get('event'))
        if event is None:
            raise SlackError("Slack payload field 'event' must be an object")
        if _string(event, 'type') != 'app_mention':
            return None
        if 'bot_id' in event or 'subtype' in event:
            return None

        channel_id = _required_nonempty_string(event, 'channel')
        timestamp = _required_nonempty_string(event, 'ts')
        if 'thread_ts' in event:
            thread_timestamp = _required_nonempty_string(event, 'thread_ts')
        else:
            thread_timestamp = timestamp
        return ChannelEvent(
            event_id=_required_nonempty_string(payload, 'event_id'),
            conversation_id=f'slack:{self._team_id}:{channel_id}:{thread_timestamp}',
            sender_id=_required_nonempty_string(event, 'user'),
            text=_required_string(event, 'text'),
            reply_to_id=thread_timestamp,
            delivery_id=channel_id,
        )

    async def reply(self, event: ChannelEvent, text: str) -> None:
        """Post `text` to the Slack conversation and original thread for `event`."""
        payload: dict[str, str] = {'channel': event.delivery_id or event.conversation_id, 'text': text}
        if event.reply_to_id is not None:
            payload['thread_ts'] = event.reply_to_id

        async with self._post_lock:
            if self._client is None:
                async with httpx.AsyncClient() as client:
                    await self._post(client, payload)
            else:
                await self._post(self._client, payload)

    def _verify(self, raw_body: bytes, headers: Mapping[str, str]) -> None:
        normalized_headers = {name.lower(): value for name, value in headers.items()}
        timestamp_text = normalized_headers.get('x-slack-request-timestamp')
        signature = normalized_headers.get('x-slack-signature')
        if timestamp_text is None or signature is None:
            raise SlackSignatureError('Missing Slack signature headers')
        try:
            timestamp = int(timestamp_text)
        except ValueError as exc:
            raise SlackSignatureError('Invalid Slack request timestamp') from exc
        now = int(time.time())
        if timestamp < now - _MAX_REQUEST_AGE_SECONDS or timestamp > now + _MAX_REQUEST_AGE_SECONDS:
            raise SlackSignatureError('Slack request timestamp is outside the five-minute window')

        signed = b'v0:' + timestamp_text.encode() + b':' + raw_body
        expected = 'v0=' + hmac.new(self._signing_secret, signed, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise SlackSignatureError('Invalid Slack request signature')

    async def _post(self, client: httpx.AsyncClient, payload: Mapping[str, str]) -> None:
        response = await client.post(
            _DEFAULT_API_URL,
            headers={'Authorization': f'Bearer {self._bot_token}'},
            json=payload,
        )
        if response.status_code == 429:
            await anyio.sleep(_retry_after_seconds(response))
            response = await client.post(
                _DEFAULT_API_URL,
                headers={'Authorization': f'Bearer {self._bot_token}'},
                json=payload,
            )
            if response.status_code == 429:
                await anyio.sleep(_retry_after_seconds(response))
        response.raise_for_status()
        try:
            body = _json_object(response.content)
        except SlackError as exc:
            raise SlackAPIError('Slack chat.postMessage returned an invalid response') from exc
        if body.get('ok') is not True:
            error = _string(body, 'error') or 'unknown_error'
            raise SlackAPIError(f'Slack chat.postMessage failed: {error}')
        if 'response_metadata' not in body:
            return
        response_metadata = _mapping(body['response_metadata'])
        if response_metadata is None:
            raise SlackAPIError('Slack chat.postMessage returned an invalid response')
        if 'warnings' in response_metadata:
            try:
                warnings = _STRING_LIST_ADAPTER.validate_python(response_metadata['warnings'])
            except ValidationError as exc:
                raise SlackAPIError('Slack chat.postMessage returned an invalid response') from exc
            if 'message_truncated' in warnings:
                raise SlackAPIError('Slack chat.postMessage truncated the reply')


def _retry_after_seconds(response: httpx.Response) -> int:
    value = response.headers.get('Retry-After')
    if value is None:
        raise SlackAPIError('Slack rate limit response has no Retry-After delay')
    try:
        seconds = int(value)
    except ValueError as exc:
        raise SlackAPIError('Slack rate limit response has an invalid Retry-After delay') from exc
    if seconds < 0:
        raise SlackAPIError('Slack rate limit response has an invalid Retry-After delay')
    return seconds


def _json_object(raw: bytes) -> dict[str, object]:
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        raise SlackError('Slack request body is not valid JSON') from exc


def _mapping(value: object) -> dict[str, object] | None:
    try:
        return _JSON_OBJECT_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = _string(mapping, key)
    if value is None:
        raise SlackError(f'Slack payload field {key!r} must be a string')
    return value


def _required_nonempty_string(mapping: Mapping[str, object], key: str) -> str:
    value = _required_string(mapping, key)
    if not value:
        raise SlackError(f'Slack payload field {key!r} must not be empty')
    return value
