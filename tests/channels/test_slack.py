from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing

import anyio
import httpx
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent

from pydantic_ai_harness.channels import ChannelError, ChannelHost, InboundMessage, WebhookRequest
from pydantic_ai_harness.channels.slack import SlackChannel

_BODY_ADAPTER = TypeAdapter(dict[str, object])
_SIGNING_SECRET = 'signing-secret'


def response(payload: object, *, status_code: int = 200, headers: Mapping[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status_code, json=payload, headers=headers)


def signed_request(
    payload: object,
    *,
    secret: str = _SIGNING_SECRET,
    timestamp: int | None = None,
    method: str = 'POST',
) -> WebhookRequest:
    body = json.dumps(payload, separators=(',', ':')).encode()
    return signed_body(body, secret=secret, timestamp=timestamp, method=method)


def signed_body(
    body: bytes,
    *,
    secret: str = _SIGNING_SECRET,
    timestamp: int | None = None,
    method: str = 'POST',
) -> WebhookRequest:
    timestamp_text = str(int(time.time()) if timestamp is None else timestamp)
    signature = (
        'v0='
        + hmac.new(
            secret.encode(),
            b'v0:' + timestamp_text.encode() + b':' + body,
            hashlib.sha256,
        ).hexdigest()
    )
    return WebhookRequest(
        method=method,
        headers={
            'X-Slack-Request-Timestamp': timestamp_text,
            'X-Slack-Signature': signature,
        },
        query={},
        body=body,
    )


def event_request(event_id: str, event: object, *, team_id: str = 'T1') -> WebhookRequest:
    return signed_request(
        {
            'type': 'event_callback',
            'team_id': team_id,
            'event_id': event_id,
            'event': event,
        }
    )


def direct_message(*, text: str = 'hello', user: str = 'U1') -> dict[str, object]:
    return {
        'type': 'message',
        'channel_type': 'im',
        'channel': 'D1',
        'user': user,
        'text': text,
        'ts': '123.45',
    }


def auth_response() -> httpx.Response:
    return response({'ok': True, 'team_id': 'T1', 'user_id': 'UBOT'})


@pytest.mark.anyio
class TestSlackChannel:
    async def test_normalizes_direct_messages_and_suppresses_duplicate_events(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel('xoxb-token', _SIGNING_SECRET, http_client=client)

        async with channel:
            async with aclosing(channel.messages()) as messages:
                request = event_request('Ev1', direct_message())
                assert (await channel.handle_webhook(request)).status_code == 200
                assert (await channel.handle_webhook(request)).status_code == 200
                message = await anext(messages)

        assert (message.conversation_id, message.sender_id, message.message_id, message.text) == (
            'dm:D1',
            'U1',
            'Ev1',
            'hello',
        )
        assert client.is_closed is False
        await client.aclose()

    async def test_handles_challenge_and_rejects_invalid_requests(self) -> None:
        channel = SlackChannel('xoxb-token', _SIGNING_SECRET)
        challenge = signed_request({'type': 'url_verification', 'challenge': 'verify-me'})
        assert (await channel.handle_webhook(challenge)).body == 'verify-me'
        assert (await channel.handle_webhook(signed_request({'type': 'url_verification'}))).status_code == 400
        assert (await channel.handle_webhook(signed_request({'type': 'other'}))).status_code == 200
        assert (await channel.handle_webhook(signed_request({}, method='GET'))).status_code == 405

        bad_signature = WebhookRequest(
            method='POST',
            headers={
                'x-slack-request-timestamp': str(int(time.time())),
                'x-slack-signature': 'v0=wrong',
            },
            query={},
            body=b'{}',
        )
        assert (await channel.handle_webhook(bad_signature)).status_code == 401
        non_ascii_signature = WebhookRequest(
            method='POST',
            headers={
                'x-slack-request-timestamp': str(int(time.time())),
                'x-slack-signature': 'v0=é',
            },
            query={},
            body=b'{}',
        )
        assert (await channel.handle_webhook(non_ascii_signature)).status_code == 401
        assert (await channel.handle_webhook(WebhookRequest('POST', {}, {}, b'{}'))).status_code == 401
        assert (
            await channel.handle_webhook(
                WebhookRequest(
                    'POST',
                    {
                        'x-slack-request-timestamp': 'invalid',
                        'x-slack-signature': 'v0=wrong',
                    },
                    {},
                    b'{}',
                )
            )
        ).status_code == 401
        assert (await channel.handle_webhook(signed_request({}, timestamp=int(time.time()) - 301))).status_code == 401
        assert (await channel.handle_webhook(signed_request({}, timestamp=int(time.time()) + 301))).status_code == 401
        oversized_timestamp = WebhookRequest(
            'POST',
            {
                'x-slack-request-timestamp': '9' * 1000,
                'x-slack-signature': 'v0=wrong',
            },
            {},
            b'{}',
        )
        assert (await channel.handle_webhook(oversized_timestamp)).status_code == 401

        assert (await channel.handle_webhook(signed_body(b'not json'))).status_code == 400

    async def test_requires_open_channel_for_events(self) -> None:
        channel = SlackChannel('xoxb-token', _SIGNING_SECRET)
        assert (await channel.handle_webhook(event_request('Ev1', direct_message()))).status_code == 503

    async def test_filters_unaddressed_and_bot_authored_events(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel('xoxb-token', _SIGNING_SECRET, http_client=client)

        async with channel:
            assert (
                await channel.handle_webhook(event_request('Ev0', direct_message(), team_id='other'))
            ).status_code == 200
            assert (
                await channel.handle_webhook(
                    signed_request(
                        {
                            'type': 'event_callback',
                            'team_id': 'T1',
                            'event': direct_message(),
                        }
                    )
                )
            ).status_code == 400
            ignored = [
                {**direct_message(user='UBOT')},
                {**direct_message(), 'bot_id': 'B1'},
                {**direct_message(), 'subtype': 'message_changed'},
                {**direct_message(), 'channel_type': 'channel'},
                {
                    'type': 'app_mention',
                    'channel': 'C2',
                    'user': 'U1',
                    'text': '<@UBOT> hello',
                    'ts': '123.45',
                },
                {**direct_message(), 'text': ''},
                {'type': 'reaction_added'},
                123,
            ]
            for index, event in enumerate(ignored):
                assert (await channel.handle_webhook(event_request(f'Ev{index + 1}', event))).status_code == 200

        await client.aclose()

    async def test_queue_full_is_retryable_and_not_recorded_as_seen(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel(
            'xoxb-token',
            _SIGNING_SECRET,
            http_client=client,
            max_queued_messages=1,
        )

        async with channel:
            async with aclosing(channel.messages()) as messages:
                first = event_request('Ev1', direct_message(text='first'))
                second = event_request('Ev2', direct_message(text='second'))
                assert (await channel.handle_webhook(first)).status_code == 200
                assert (await channel.handle_webhook(second)).status_code == 503
                assert (await anext(messages)).text == 'first'
                assert (await channel.handle_webhook(second)).status_code == 200
                assert (await anext(messages)).text == 'second'

        await client.aclose()

    async def test_bounds_recent_event_deduplication(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel(
            'xoxb-token',
            _SIGNING_SECRET,
            http_client=client,
            max_queued_messages=1,
        )

        async with channel:
            async with aclosing(channel.messages()) as messages:
                for index in range(10_001):
                    request = event_request(f'Ev{index}', direct_message(text=str(index)))
                    assert (await channel.handle_webhook(request)).status_code == 200
                    await anext(messages)
                first = event_request('Ev0', direct_message(text='first again'))
                assert (await channel.handle_webhook(first)).status_code == 200
                assert (await anext(messages)).text == 'first again'

        await client.aclose()

    async def test_mentions_start_or_continue_threads_and_strip_bot_mention(self) -> None:
        posts: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith('/auth.test'):
                return auth_response()
            posts.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return response({'ok': True, 'ts': '124.00'})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = SlackChannel(
            'xoxb-token',
            _SIGNING_SECRET,
            allowed_channel_ids={'C1'},
            http_client=client,
        )
        mention = {
            'type': 'app_mention',
            'channel': 'C1',
            'user': 'U1',
            'text': '<@UBOT> explain this',
            'ts': '123.45',
        }

        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert (await channel.handle_webhook(event_request('Ev1', mention))).status_code == 200
                message = await anext(messages)
                await channel.send_text(message.conversation_id, 'reply')
                continued = {**mention, 'thread_ts': '100.00'}
                assert (await channel.handle_webhook(event_request('Ev2', continued))).status_code == 200
                continued_message = await anext(messages)
                empty_thread = {**mention, 'thread_ts': ''}
                assert (await channel.handle_webhook(event_request('Ev3', empty_thread))).status_code == 200
                empty_thread_message = await anext(messages)
                threaded_dm = {**direct_message(), 'thread_ts': '110.00'}
                assert (await channel.handle_webhook(event_request('Ev4', threaded_dm))).status_code == 200
                dm_message = await anext(messages)
                await channel.send_text(dm_message.conversation_id, 'dm reply')

        assert message.conversation_id == 'thread:C1:123.45'
        assert message.text == 'explain this'
        assert continued_message.conversation_id == 'thread:C1:100.00'
        assert empty_thread_message.conversation_id == 'thread:C1:123.45'
        assert dm_message.conversation_id == 'thread:D1:110.00'
        assert posts == [
            {
                'channel': 'C1',
                'text': 'reply',
                'mrkdwn': False,
                'unfurl_links': False,
                'unfurl_media': False,
                'thread_ts': '123.45',
            },
            {
                'channel': 'D1',
                'text': 'dm reply',
                'mrkdwn': False,
                'unfurl_links': False,
                'unfurl_media': False,
                'thread_ts': '110.00',
            },
        ]
        await client.aclose()

    async def test_webhook_runs_agent_and_posts_reply(self) -> None:
        opened = anyio.Event()
        posted = anyio.Event()
        posts: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith('/auth.test'):
                opened.set()
                return auth_response()
            posts.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            posted.set()
            return response({'ok': True, 'ts': '124.00'})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = SlackChannel('xoxb-token', _SIGNING_SECRET, http_client=client)
        host = ChannelHost(Agent('test'), channel, allowed_senders={'U1'})
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(host.serve)
            with anyio.fail_after(1):
                await opened.wait()
            assert (await channel.handle_webhook(event_request('Ev1', direct_message()))).status_code == 200
            with anyio.fail_after(1):
                await posted.wait()
            task_group.cancel_scope.cancel()
        await client.aclose()

        assert posts == [
            {
                'channel': 'D1',
                'text': 'success (no tool calls)',
                'mrkdwn': False,
                'unfurl_links': False,
                'unfurl_media': False,
            }
        ]

    async def test_chunks_messages_and_retries_one_explicit_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        posts: list[dict[str, object]] = []
        delays: list[float] = []

        async def sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(anyio, 'sleep', sleep)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith('/auth.test'):
                return auth_response()
            posts.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            if len(posts) == 1:
                return response(
                    {'ok': False, 'error': 'ratelimited'},
                    status_code=429,
                    headers={'Retry-After': '60'},
                )
            return response({'ok': True, 'ts': str(len(posts))})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = SlackChannel('xoxb-token', _SIGNING_SECRET, http_client=client)

        async with channel:
            await channel.send_text('dm:D1', 'a' * 4001)

        assert posts == [
            {'channel': 'D1', 'text': 'a' * 4000, 'mrkdwn': False, 'unfurl_links': False, 'unfurl_media': False},
            {'channel': 'D1', 'text': 'a' * 4000, 'mrkdwn': False, 'unfurl_links': False, 'unfurl_media': False},
            {'channel': 'D1', 'text': 'a', 'mrkdwn': False, 'unfurl_links': False, 'unfurl_media': False},
        ]
        assert delays == [60]
        await client.aclose()

    async def test_does_not_hold_conversation_lane_for_long_rate_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith('/auth.test'):
                return auth_response()
            return response(
                {'ok': False, 'error': 'ratelimited'},
                status_code=429,
                headers={'Retry-After': '61'},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = SlackChannel('xoxb-token', _SIGNING_SECRET, http_client=client)
        async with channel:
            with pytest.raises(ChannelError, match='rate limited'):
                await channel.send_text('dm:D1', 'hello')
        await client.aclose()

    async def test_uses_configured_api_root(self) -> None:
        urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            urls.append(str(request.url))
            return auth_response()

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with SlackChannel(
            'token',
            _SIGNING_SECRET,
            http_client=client,
            api_base_url='https://slack-gov.com/api/',
        ):
            pass

        assert urls == ['https://slack-gov.com/api/auth.test']
        await client.aclose()

    async def test_surfaces_api_and_transport_errors_without_credentials(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return auth_response()
            if calls == 2:
                return httpx.Response(500, content=b'not json')
            if calls == 3:
                return response({'ok': False, 'error': 'channel_not_found'})
            if calls == 4:
                return response(
                    {'ok': False, 'error': 'ratelimited'},
                    status_code=429,
                    headers={'Retry-After': 'invalid'},
                )
            if calls == 5:
                return response({'ok': False, 'error': 'ratelimited'}, status_code=429)
            if calls == 6:
                return response(
                    {'ok': False, 'error': 'ratelimited'},
                    status_code=429,
                    headers={'Retry-After': '-1'},
                )
            if calls == 7:
                return response(
                    {'ok': False, 'error': 'ratelimited'},
                    status_code=429,
                    headers={'Retry-After': 'inf'},
                )
            if calls == 8:
                return response({'ok': False})
            raise httpx.ConnectError('offline', request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = SlackChannel('secret-token', _SIGNING_SECRET, http_client=client)

        async with channel:
            with pytest.raises(ChannelError, match='invalid response'):
                await channel.send_text('dm:D1', 'first')
            with pytest.raises(ChannelError, match='channel_not_found'):
                await channel.send_text('dm:D1', 'second')
            with pytest.raises(ChannelError, match='ratelimited'):
                await channel.send_text('dm:D1', 'third')
            with pytest.raises(ChannelError, match='ratelimited'):
                await channel.send_text('dm:D1', 'fourth')
            with pytest.raises(ChannelError, match='ratelimited'):
                await channel.send_text('dm:D1', 'fifth')
            with pytest.raises(ChannelError, match='ratelimited'):
                await channel.send_text('dm:D1', 'sixth')
            with pytest.raises(ChannelError, match='unknown Slack Web API error'):
                await channel.send_text('dm:D1', 'seventh')
            with pytest.raises(ChannelError, match='request failed') as exc_info:
                await channel.send_text('dm:D1', 'eighth')

        assert exc_info.value.__suppress_context__ is True
        assert 'secret-token' not in str(exc_info.value)
        await client.aclose()

    async def test_owned_client_closes_on_success_and_invalid_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client_class = httpx.AsyncClient
        client = client_class(transport=httpx.MockTransport(lambda request: auth_response()))
        monkeypatch.setattr(httpx, 'AsyncClient', lambda: client)

        async with SlackChannel('token', _SIGNING_SECRET):
            pass
        assert client.is_closed

        rejected = client_class(transport=httpx.MockTransport(lambda request: response({'ok': True, 'team_id': 'T1'})))
        monkeypatch.setattr(httpx, 'AsyncClient', lambda: rejected)
        with pytest.raises(ChannelError, match='invalid identity'):
            async with SlackChannel('token', _SIGNING_SECRET):
                pass  # pragma: no cover
        assert rejected.is_closed

    async def test_requires_open_channel_and_valid_output(self) -> None:
        channel = SlackChannel('token', _SIGNING_SECRET)
        with pytest.raises(RuntimeError, match='opened'):
            await channel.send_text('dm:D1', 'hello')

        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel('token', _SIGNING_SECRET, http_client=client)
        async with channel:
            with pytest.raises(ValueError, match='text'):
                await channel.send_text('dm:D1', '')
            with pytest.raises(ChannelError, match='invalid Slack conversation id'):
                await channel.send_text('invalid', 'hello')
            with pytest.raises(ChannelError, match='invalid Slack conversation id'):
                await channel.send_text('thread:C1', 'hello')
            with pytest.raises(ChannelError, match='invalid Slack conversation id'):
                await channel.send_text('other:route', 'hello')
        await client.aclose()

    async def test_closing_channel_ends_pending_message_iterator(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel('token', _SIGNING_SECRET, http_client=client)
        iterator_stopped = anyio.Event()

        async def wait_for_message(messages: AsyncGenerator[InboundMessage, None]) -> None:
            with pytest.raises(StopAsyncIteration):
                await anext(messages)
            iterator_stopped.set()

        async with anyio.create_task_group() as task_group:
            async with channel:
                messages = channel.messages()
                task_group.start_soon(wait_for_message, messages)
                await anyio.sleep(0)
            with anyio.fail_after(1):
                await iterator_stopped.wait()

        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert (await channel.handle_webhook(event_request('Ev1', direct_message()))).status_code == 200
                assert (await anext(messages)).message_id == 'Ev1'
        await client.aclose()

    async def test_closed_message_iterator_rejects_new_webhooks(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel('token', _SIGNING_SECRET, http_client=client)
        async with channel:
            messages = channel.messages()
            assert (await channel.handle_webhook(event_request('Ev1', direct_message()))).status_code == 200
            assert (await anext(messages)).message_id == 'Ev1'
            await messages.aclose()

            assert (await channel.handle_webhook(event_request('Ev2', direct_message()))).status_code == 503
        await client.aclose()

    async def test_old_message_iterator_ends_after_channel_reopens(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel('token', _SIGNING_SECRET, http_client=client)
        messages = channel.messages()
        async with channel:
            assert (await channel.handle_webhook(event_request('Ev1', direct_message()))).status_code == 200
            assert (await anext(messages)).message_id == 'Ev1'

        async with channel:
            with pytest.raises(StopAsyncIteration):
                await anext(messages)
            async with aclosing(channel.messages()) as reopened_messages:
                assert (await channel.handle_webhook(event_request('Ev1', direct_message()))).status_code == 200
                assert (await anext(reopened_messages)).message_id == 'Ev1'
        await client.aclose()

    async def test_open_failure_ends_pending_message_iterator(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response({'ok': True, 'team_id': 'T1'}))
        )
        channel = SlackChannel('token', _SIGNING_SECRET, http_client=client)
        with pytest.raises(ChannelError, match='invalid identity'):
            await channel.__aenter__()
        with pytest.raises(StopAsyncIteration):
            await anext(channel.messages())
        await client.aclose()

    async def test_rejects_invalid_configuration_and_double_open(self) -> None:
        with pytest.raises(ValueError, match='bot_token'):
            SlackChannel('', _SIGNING_SECRET)
        with pytest.raises(ValueError, match='signing_secret'):
            SlackChannel('token', '')
        with pytest.raises(ValueError, match='api_base_url'):
            SlackChannel('token', _SIGNING_SECRET, api_base_url='')
        with pytest.raises(TypeError, match='allowed_channel_ids'):
            SlackChannel('token', _SIGNING_SECRET, allowed_channel_ids='C1')
        for value in (0, True):
            with pytest.raises(ValueError, match='max_queued_messages'):
                SlackChannel('token', _SIGNING_SECRET, max_queued_messages=value)

        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: auth_response()))
        channel = SlackChannel('token', _SIGNING_SECRET, http_client=client)
        async with channel:
            with pytest.raises(RuntimeError, match='already open'):
                await channel.__aenter__()
        await client.aclose()
