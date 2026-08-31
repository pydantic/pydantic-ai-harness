from __future__ import annotations

import hashlib
import hmac
import json
from contextlib import aclosing

import anyio
import httpx
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent

from pydantic_ai_harness.channels import ChannelError, ChannelHost, WebhookRequest
from pydantic_ai_harness.channels.whatsapp import WhatsAppChannel

_BODY_ADAPTER = TypeAdapter(dict[str, object])
_APP_SECRET = 'app-secret'
_PHONE_ID = '123456789'


def response(payload: object, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def signed_request(payload: object, *, secret: str = _APP_SECRET, method: str = 'POST') -> WebhookRequest:
    body = json.dumps(payload, separators=(',', ':')).encode()
    signature = 'sha256=' + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return WebhookRequest(
        method=method,
        headers={'X-Hub-Signature-256': signature},
        query={},
        body=body,
    )


def challenge_request(*, token: str = 'verify-token', challenge: str = 'challenge') -> WebhookRequest:
    return WebhookRequest(
        method='GET',
        headers={},
        query={
            'hub.mode': 'subscribe',
            'hub.verify_token': token,
            'hub.challenge': challenge,
        },
        body=b'',
    )


def webhook_payload(messages: object, *, phone_number_id: str = _PHONE_ID) -> dict[str, object]:
    return {
        'object': 'whatsapp_business_account',
        'entry': [
            {
                'id': 'WABA1',
                'changes': [
                    {
                        'field': 'messages',
                        'value': {
                            'metadata': {'phone_number_id': phone_number_id},
                            'messages': messages,
                        },
                    }
                ],
            }
        ],
    }


def text_message(message_id: str = 'wamid.1', *, body: str = 'hello', sender: str = '15551234567') -> dict[str, object]:
    return {
        'from': sender,
        'id': message_id,
        'type': 'text',
        'text': {'body': body},
    }


async def webhook_status(channel: WhatsAppChannel, request: WebhookRequest) -> int:
    return (await channel.handle_webhook(request)).status_code


@pytest.mark.anyio
class TestWhatsAppChannel:
    async def test_verifies_challenge(self) -> None:
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token')
        accepted = await channel.handle_webhook(challenge_request())
        assert (accepted.status_code, accepted.body) == (200, 'challenge')

        assert await webhook_status(channel, challenge_request(token='wrong')) == 403
        missing_challenge = WebhookRequest(
            method='GET',
            headers={},
            query={'hub.mode': 'subscribe', 'hub.verify_token': 'verify-token'},
            body=b'',
        )
        assert await webhook_status(channel, missing_challenge) == 403
        wrong_mode = WebhookRequest(
            method='GET',
            headers={},
            query={
                'hub.mode': 'unsubscribe',
                'hub.verify_token': 'verify-token',
                'hub.challenge': 'challenge',
            },
            body=b'',
        )
        assert await webhook_status(channel, wrong_mode) == 403
        assert await webhook_status(channel, signed_request({}, method='PUT')) == 405

    async def test_normalizes_batched_text_and_suppresses_duplicates(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response({})))
        channel = WhatsAppChannel(
            'token',
            _PHONE_ID,
            _APP_SECRET,
            'verify-token',
            http_client=client,
            max_queued_messages=2,
        )
        request = signed_request(
            webhook_payload(
                [
                    text_message('wamid.1', body='first'),
                    text_message('wamid.2', body='second'),
                ]
            )
        )

        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert await webhook_status(channel, request) == 200
                assert await webhook_status(channel, request) == 200
                first = await anext(messages)
                second = await anext(messages)

        assert (first.conversation_id, first.sender_id, first.message_id, first.text) == (
            '15551234567',
            '15551234567',
            'wamid.1',
            'first',
        )
        assert second.message_id == 'wamid.2'
        assert client.is_closed is False
        await client.aclose()

    async def test_rejects_invalid_signatures_and_payloads(self) -> None:
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token')
        assert await webhook_status(channel, WebhookRequest('POST', {}, {}, b'{}')) == 401

        non_ascii = WebhookRequest(
            'POST',
            {'x-hub-signature-256': 'sha256=é'},
            {},
            b'{}',
        )
        assert await webhook_status(channel, non_ascii) == 401
        assert await webhook_status(channel, signed_request({}, secret='wrong')) == 401

        malformed_body = b'not json'
        malformed_signature = 'sha256=' + hmac.new(_APP_SECRET.encode(), malformed_body, hashlib.sha256).hexdigest()
        malformed = WebhookRequest(
            'POST',
            {'x-hub-signature-256': malformed_signature},
            {},
            malformed_body,
        )
        assert await webhook_status(channel, malformed) == 400
        assert await webhook_status(channel, signed_request({'object': 'other'})) == 200

    async def test_filters_statuses_other_numbers_and_malformed_messages(self) -> None:
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token')
        payload: dict[str, object] = {
            'object': 'whatsapp_business_account',
            'entry': [
                123,
                {'changes': 'invalid'},
                {
                    'changes': [
                        123,
                        {'field': 'statuses', 'value': {}},
                        {'field': 'messages', 'value': 123},
                        {
                            'field': 'messages',
                            'value': {
                                'metadata': {'phone_number_id': 'other'},
                                'messages': [text_message('ignored')],
                            },
                        },
                        {
                            'field': 'messages',
                            'value': {
                                'metadata': {'phone_number_id': _PHONE_ID},
                                'messages': [
                                    123,
                                    {'type': 'image'},
                                    {**text_message('empty'), 'text': {'body': ''}},
                                    {**text_message('bad-sender'), 'from': 123},
                                    text_message('accepted'),
                                ],
                            },
                        },
                    ]
                },
            ],
        }

        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert await webhook_status(channel, signed_request(payload)) == 200
                assert (await anext(messages)).message_id == 'accepted'

                invalid_entries = webhook_payload('invalid')
                assert await webhook_status(channel, signed_request(invalid_entries)) == 200
                request = signed_request({'object': 'whatsapp_business_account', 'entry': 'invalid'})
                assert await webhook_status(channel, request) == 200

    async def test_queue_full_leaves_message_retryable(self) -> None:
        channel = WhatsAppChannel(
            'token',
            _PHONE_ID,
            _APP_SECRET,
            'verify-token',
            max_queued_messages=1,
        )
        first = signed_request(webhook_payload([text_message('wamid.1', body='first')]))
        second = signed_request(webhook_payload([text_message('wamid.2', body='second')]))

        assert await webhook_status(channel, first) == 503
        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert await webhook_status(channel, first) == 200
                assert await webhook_status(channel, second) == 503
                assert (await anext(messages)).text == 'first'
                assert await webhook_status(channel, second) == 200
                assert (await anext(messages)).text == 'second'
        assert await webhook_status(channel, second) == 503

    async def test_queue_full_mid_batch_resumes_on_redelivery(self) -> None:
        channel = WhatsAppChannel(
            'token',
            _PHONE_ID,
            _APP_SECRET,
            'verify-token',
            max_queued_messages=1,
        )
        request = signed_request(
            webhook_payload(
                [
                    text_message('wamid.1', body='first'),
                    text_message('wamid.2', body='second'),
                ]
            )
        )

        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert await webhook_status(channel, request) == 503
                assert (await anext(messages)).text == 'first'
                assert await webhook_status(channel, request) == 200
                assert (await anext(messages)).text == 'second'

    async def test_webhook_runs_agent_and_posts_reply(self) -> None:
        posted = anyio.Event()
        posts: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            posted.set()
            return response({'messages': [{'id': 'wamid.reply'}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token', http_client=client)
        host = ChannelHost(Agent('test'), channel, allowed_senders={'15551234567'})
        request = signed_request(webhook_payload([text_message()]))

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(host.serve)
            with anyio.fail_after(1):
                while await webhook_status(channel, request) == 503:
                    await anyio.sleep(0)
                await posted.wait()
            task_group.cancel_scope.cancel()
        await client.aclose()

        assert posts == [
            {
                'messaging_product': 'whatsapp',
                'recipient_type': 'individual',
                'to': '15551234567',
                'type': 'text',
                'text': {'body': 'success (no tool calls)'},
            }
        ]

    async def test_chunks_messages_and_retries_one_throttling_error(self) -> None:
        posts: list[dict[str, object]] = []
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            posts.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            if len(posts) == 1:
                return response(
                    {
                        'error': {
                            'code': 130429,
                            'message': 'Rate limit hit',
                        }
                    },
                    status_code=429,
                )
            return response({'messages': [{'id': f'wamid.{len(posts)}'}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = WhatsAppChannel(
            'token',
            _PHONE_ID,
            _APP_SECRET,
            'verify-token',
            http_client=client,
            api_version='v24.0',
            retry_delay=0,
        )

        async with channel:
            await channel.send_text('15551234567', 'a' * 4097)

        assert [post['text'] for post in posts] == [
            {'body': 'a' * 4096},
            {'body': 'a' * 4096},
            {'body': 'a'},
        ]
        assert all(post['to'] == '15551234567' for post in posts)
        assert paths == [
            f'/v24.0/{_PHONE_ID}/messages',
            f'/v24.0/{_PHONE_ID}/messages',
            f'/v24.0/{_PHONE_ID}/messages',
        ]
        await client.aclose()

    async def test_surfaces_service_window_and_unknown_api_errors(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(
                    {
                        'error': {
                            'code': 131047,
                            'message': 'Re-engagement message',
                        }
                    },
                    status_code=400,
                )
            return response({})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token', http_client=client)

        async with channel:
            with pytest.raises(ChannelError, match='Re-engagement message'):
                await channel.send_text('15551234567', 'outside window')
            with pytest.raises(ChannelError, match='unknown WhatsApp Cloud API error'):
                await channel.send_text('15551234567', 'unknown')

        assert calls == 2
        await client.aclose()

    async def test_surfaces_invalid_and_transport_responses_without_token(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(500, content=b'not json')
            raise httpx.ConnectError('offline', request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = WhatsAppChannel(
            'secret-token',
            _PHONE_ID,
            _APP_SECRET,
            'verify-token',
            http_client=client,
        )

        async with channel:
            with pytest.raises(ChannelError, match='invalid response'):
                await channel.send_text('15551234567', 'invalid')
            with pytest.raises(ChannelError, match='request failed') as exc_info:
                await channel.send_text('15551234567', 'offline')

        assert exc_info.value.__suppress_context__ is True
        assert 'secret-token' not in str(exc_info.value)
        await client.aclose()

    async def test_owned_client_closes_and_webhooks_require_open_channel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client_class = httpx.AsyncClient
        client = client_class(transport=httpx.MockTransport(lambda request: response({})))
        monkeypatch.setattr(httpx, 'AsyncClient', lambda: client)
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token')
        request = signed_request(webhook_payload([text_message()]))

        assert await webhook_status(channel, request) == 503
        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert await webhook_status(channel, request) == 200
                assert (await anext(messages)).message_id == 'wamid.1'
        assert client.is_closed
        assert await webhook_status(channel, request) == 503

    async def test_reopening_resets_recent_message_ids(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response({})))
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token', http_client=client)
        request = signed_request(webhook_payload([text_message()]))
        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert await webhook_status(channel, request) == 200
                assert (await anext(messages)).message_id == 'wamid.1'

        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert await webhook_status(channel, request) == 200
                assert (await anext(messages)).message_id == 'wamid.1'
        await client.aclose()

    async def test_requires_open_channel_and_nonempty_text(self) -> None:
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token')
        with pytest.raises(RuntimeError, match='opened'):
            await channel.send_text('15551234567', 'hello')

        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response({})))
        channel = WhatsAppChannel('token', _PHONE_ID, _APP_SECRET, 'verify-token', http_client=client)
        async with channel:
            with pytest.raises(ValueError, match='text'):
                await channel.send_text('15551234567', '')
            with pytest.raises(RuntimeError, match='already open'):
                await channel.__aenter__()
        await client.aclose()

    def test_validates_configuration(self) -> None:
        base: tuple[str, str, str, str] = ('token', _PHONE_ID, _APP_SECRET, 'verify-token')
        for index, name in enumerate(('access_token', 'phone_number_id', 'app_secret', 'verify_token')):
            values: list[str] = list(base)
            values[index] = ''
            with pytest.raises(ValueError, match=name):
                WhatsAppChannel(*values)

        with pytest.raises(ValueError, match='api_version'):
            WhatsAppChannel(*base, api_version='latest')
        for value in (0, True):
            with pytest.raises(ValueError, match='max_queued_messages'):
                WhatsAppChannel(*base, max_queued_messages=value)
        with pytest.raises(ValueError, match='retry_delay'):
            WhatsAppChannel(*base, retry_delay=-1)
        with pytest.raises(ValueError, match='retry_delay'):
            WhatsAppChannel(*base, retry_delay=float('nan'))
