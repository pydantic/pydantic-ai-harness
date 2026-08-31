from __future__ import annotations

import json
import logging
from contextlib import aclosing

import httpx
import pytest
from pydantic import TypeAdapter

from pydantic_ai_harness.channels import ChannelAdapter, ChannelError
from pydantic_ai_harness.channels.telegram import TelegramChannel

_BODY_ADAPTER = TypeAdapter(dict[str, object])


def response(payload: object, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


@pytest.mark.anyio
class TestTelegramChannel:
    async def test_polls_private_text_and_advances_offset(self) -> None:
        poll_bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            method = request.url.path.rsplit('/', 1)[-1]
            if method == 'getMe':
                return response({'ok': True, 'result': {'id': 42}})
            body = _BODY_ADAPTER.validate_python(json.loads(request.content))
            poll_bodies.append(body)
            if len(poll_bodies) == 1:
                return response(
                    {
                        'ok': True,
                        'result': [
                            {
                                'update_id': 10,
                                'message': {
                                    'message_id': 20,
                                    'from': {'id': 30},
                                    'chat': {'id': 40, 'type': 'group'},
                                    'text': 'group',
                                },
                            },
                            {'update_id': 11, 'message': {'message_id': 21}},
                            {
                                'update_id': 12,
                                'message': {
                                    'message_id': 22,
                                    'from': {'id': 30},
                                    'chat': {'id': 40, 'type': 'private'},
                                    'text': 'hello',
                                },
                            },
                        ],
                    }
                )
            return response(
                {
                    'ok': True,
                    'result': [
                        {
                            'update_id': 13,
                            'message': {
                                'message_id': 23,
                                'from': {'id': '30'},
                                'chat': {'id': '40', 'type': 'private'},
                                'text': 'again',
                            },
                        }
                    ],
                }
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel: ChannelAdapter = TelegramChannel('token', http_client=client, poll_timeout=7)
        async with channel:
            async with aclosing(channel.messages()) as messages:
                first = await anext(messages)
                second = await anext(messages)

        assert (first.conversation_id, first.sender_id, first.message_id, first.text) == ('40', '30', '22', 'hello')
        assert second.text == 'again'
        assert poll_bodies == [
            {'timeout': 7, 'allowed_updates': ['message']},
            {'timeout': 7, 'allowed_updates': ['message'], 'offset': 13},
        ]
        assert client.is_closed is False
        await client.aclose()

    async def test_chunks_text_by_unicode_characters_and_retries_one_rate_limit(self) -> None:
        texts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            method = request.url.path.rsplit('/', 1)[-1]
            if method == 'getMe':
                return response({'ok': True, 'result': {}})
            body = _BODY_ADAPTER.validate_python(json.loads(request.content))
            text = body.get('text')
            assert isinstance(text, str)
            texts.append(text)
            if len(texts) == 1:
                return response(
                    {
                        'ok': False,
                        'description': 'Too Many Requests',
                        'parameters': {'retry_after': 0},
                    },
                    status_code=429,
                )
            return response({'ok': True, 'result': {'message_id': len(texts)}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('token', http_client=client)
        async with channel:
            await channel.send_text('chat', 'a' * 4097)
            await channel.send_text('chat', '😀' * 2049)
            with pytest.raises(ValueError, match='text must not be empty'):
                await channel.send_text('chat', '')

        assert texts == ['a' * 4096, 'a' * 4096, 'a', '😀' * 2049]
        await client.aclose()

    async def test_does_not_hold_conversation_lane_for_long_rate_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith('/getMe'):
                return response({'ok': True, 'result': {}})
            return response(
                {
                    'ok': False,
                    'description': 'Too Many Requests',
                    'parameters': {'retry_after': 61},
                },
                status_code=429,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('token', http_client=client)
        async with channel:
            with pytest.raises(ChannelError, match='Too Many Requests'):
                await channel.send_text('chat', 'hello')
        await client.aclose()

    async def test_provider_error_does_not_expose_token(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return response({'ok': True, 'result': {}})
            return response(
                {
                    'ok': False,
                    'description': 'chat not found: https://api.telegram.org/botsecret-token/sendMessage',
                }
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('secret-token', http_client=client)
        async with channel:
            with pytest.raises(ChannelError, match='chat not found') as exc_info:
                await channel.send_text('chat', 'hello')

        assert 'secret-token' not in str(exc_info.value)
        assert '/bot<redacted>/' in str(exc_info.value)
        await client.aclose()

    async def test_transport_error_suppresses_token_bearing_exception(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return response({'ok': True, 'result': {}})
            raise httpx.ConnectError('offline', request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('secret-token', http_client=client)
        async with channel:
            with pytest.raises(ChannelError, match='request failed: sendMessage') as exc_info:
                await channel.send_text('chat', 'hello')

        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True
        assert 'secret-token' not in str(exc_info.value)
        await client.aclose()

    async def test_httpx_request_log_redacts_token(self, caplog: pytest.LogCaptureFixture) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response({'ok': True, 'result': {}})))
        channel = TelegramChannel('secret-token', http_client=client)

        with caplog.at_level('INFO', logger='httpx'):
            async with channel:
                await channel.send_text('chat', 'hello')
            logging.getLogger('httpx').info('unrelated request')

        assert 'secret-token' not in caplog.text
        assert '/bot<redacted>/' in caplog.text
        assert 'unrelated request' in caplog.text
        await client.aclose()

    async def test_poll_recovers_from_transport_error(self) -> None:
        poll_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count
            method = request.url.path.rsplit('/', 1)[-1]
            if method == 'getMe':
                return response({'ok': True, 'result': {}})
            poll_count += 1
            if poll_count == 1:
                raise httpx.ConnectError('offline', request=request)
            return response(
                {
                    'ok': True,
                    'result': [
                        {
                            'update_id': 1,
                            'message': {
                                'message_id': 2,
                                'from': {'id': 3},
                                'chat': {'id': 4, 'type': 'private'},
                                'text': 'recovered',
                            },
                        }
                    ],
                }
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('token', http_client=client, retry_delay=0.001)
        async with channel:
            async with aclosing(channel.messages()) as messages:
                message = await anext(messages)

        assert message.text == 'recovered'
        assert poll_count == 2
        await client.aclose()

    async def test_resets_offset_after_one_week_without_updates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        poll_bodies: list[dict[str, object]] = []
        update_ids = iter((10, 5))
        clock = iter((0.0, 604_801.0, 604_801.0))

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith('/getMe'):
                return response({'ok': True, 'result': {}})
            poll_bodies.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            update_id = next(update_ids)
            return response(
                {
                    'ok': True,
                    'result': [
                        {
                            'update_id': update_id,
                            'message': {
                                'message_id': update_id,
                                'from': {'id': 3},
                                'chat': {'id': 4, 'type': 'private'},
                                'text': str(update_id),
                            },
                        }
                    ],
                }
            )

        monkeypatch.setattr('pydantic_ai_harness.channels.telegram.monotonic', lambda: next(clock))
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('token', http_client=client)
        async with channel:
            async with aclosing(channel.messages()) as messages:
                assert (await anext(messages)).text == '10'
                assert (await anext(messages)).text == '5'

        assert poll_bodies == [
            {'timeout': 30, 'allowed_updates': ['message']},
            {'timeout': 30, 'allowed_updates': ['message']},
        ]
        await client.aclose()

    async def test_poll_honors_rate_limit_and_rejects_permanent_error(self, caplog: pytest.LogCaptureFixture) -> None:
        poll_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count
            method = request.url.path.rsplit('/', 1)[-1]
            if method == 'getMe':
                return response({'ok': True, 'result': {}})
            poll_count += 1
            if poll_count == 1:
                return response(
                    {
                        'ok': False,
                        'error_code': 429,
                        'description': 'Too Many Requests',
                        'parameters': {'retry_after': 0},
                    },
                    status_code=429,
                )
            return response({'ok': False, 'error_code': 409, 'description': 'webhook is active'})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('token', http_client=client)
        async with channel:
            with pytest.raises(ChannelError, match='webhook is active'):
                await anext(channel.messages())

        assert poll_count == 2
        assert 'polling rate limited' in caplog.text
        await client.aclose()

    async def test_poll_recovers_from_invalid_results_and_skips_malformed_updates(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        poll_count = 0
        invalid_sender: list[object] = []
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_count
            method = request.url.path.rsplit('/', 1)[-1]
            if method == 'getMe':
                return response({'ok': True, 'result': {}})
            poll_count += 1
            if poll_count == 1:
                return response({'ok': True, 'result': 'not a list'})
            if poll_count == 2:
                return response({'ok': True, 'result': [123, {'update_id': 'unknown'}, {'update_id': True}]})
            return response(
                {
                    'ok': True,
                    'result': [
                        123,
                        {'update_id': 'unknown', 'message': 123},
                        {'update_id': 1, 'message': 123},
                        {
                            'update_id': 2,
                            'message': {
                                'message_id': 2,
                                'from': {'id': invalid_sender},
                                'chat': {'id': 4, 'type': 'private'},
                                'text': 'invalid sender',
                            },
                        },
                        {
                            'update_id': 'unknown',
                            'message': {
                                'message_id': 3,
                                'from': {'id': 5},
                                'chat': {'id': 4, 'type': 'private'},
                                'text': 'valid',
                            },
                        },
                        {
                            'update_id': 4,
                            'message': {
                                'message_id': 4,
                                'from': {'id': 5},
                                'chat': {'id': 4, 'type': 'private'},
                                'text': 'valid update',
                            },
                        },
                    ],
                }
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('token', http_client=client, retry_delay=0.25)
        monkeypatch.setattr('pydantic_ai_harness.channels.telegram.anyio.sleep', sleep)
        async with channel:
            async with aclosing(channel.messages()) as messages:
                message = await anext(messages)

        assert message.text == 'valid update'
        assert 'invalid result' in caplog.text
        assert 'without usable ids' in caplog.text
        assert sleeps == [0.25, 0.25]
        await client.aclose()

    async def test_invalid_response_is_safe_and_not_rate_limited(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return response({'ok': True, 'result': {}})
            if calls == 2:
                return response(['invalid envelope'])
            if calls == 3:
                return httpx.Response(500, content=b'upstream unavailable')
            if calls == 4:
                return response(
                    {
                        'ok': False,
                        'description': 'ordinary error',
                        'parameters': {'retry_after': -1},
                    }
                )
            if calls == 5:
                return httpx.Response(
                    200,
                    content=b'{"ok":false,"description":"non-finite error","parameters":{"retry_after":Infinity}}',
                )
            if calls == 6:
                return response(
                    {
                        'ok': False,
                        'description': 'boolean delay',
                        'parameters': {'retry_after': False},
                    }
                )
            return response({'ok': False})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel('token', http_client=client)
        async with channel:
            with pytest.raises(ChannelError, match='invalid response'):
                await channel.send_text('chat', 'first')
            with pytest.raises(ChannelError, match='invalid response'):
                await channel.send_text('chat', 'second')
            with pytest.raises(ChannelError, match='ordinary error'):
                await channel.send_text('chat', 'third')
            with pytest.raises(ChannelError, match='non-finite error'):
                await channel.send_text('chat', 'fourth')
            with pytest.raises(ChannelError, match='boolean delay'):
                await channel.send_text('chat', 'fifth')
            with pytest.raises(ChannelError, match='unknown Telegram Bot API error'):
                await channel.send_text('chat', 'sixth')
        await client.aclose()

    async def test_owned_client_closes_on_success_and_open_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client_class = httpx.AsyncClient
        client = client_class(transport=httpx.MockTransport(lambda request: response({'ok': True, 'result': {}})))
        monkeypatch.setattr(httpx, 'AsyncClient', lambda: client)
        channel = TelegramChannel('token')

        async with channel:
            with pytest.raises(RuntimeError, match='already open'):
                await channel.__aenter__()

        assert client.is_closed

        rejected = client_class(
            transport=httpx.MockTransport(
                lambda request: response(
                    {'ok': False, 'error_code': 401, 'description': 'unauthorized'}, status_code=401
                )
            )
        )
        monkeypatch.setattr(httpx, 'AsyncClient', lambda: rejected)
        with pytest.raises(ChannelError, match='unauthorized'):
            async with TelegramChannel('token'):
                pass  # pragma: no cover
        assert rejected.is_closed

    @pytest.mark.parametrize('body', [b'Unauthorized', b'{}'])
    async def test_fatal_http_response_does_not_retry(self, body: bytes) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, content=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with pytest.raises(ChannelError, match='HTTP 401'):
            async with TelegramChannel('token', http_client=client):
                pass  # pragma: no cover
        assert calls == 1
        await client.aclose()

    async def test_requires_open_channel(self) -> None:
        channel = TelegramChannel('token')
        with pytest.raises(RuntimeError, match='opened'):
            await channel.send_text('chat', 'hello')

    @pytest.mark.parametrize(
        ('token', 'poll_timeout', 'retry_delay', 'message'),
        [
            ('', 30, 1.0, 'token'),
            ('x', 0, 1.0, 'poll_timeout'),
            ('x', True, 1.0, 'poll_timeout'),
            ('x', 30, 0, 'retry_delay'),
            ('x', 30, -1.0, 'retry_delay'),
            ('x', 30, float('nan'), 'retry_delay'),
        ],
    )
    def test_validates_configuration(
        self,
        token: str,
        poll_timeout: int,
        retry_delay: float,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            TelegramChannel(token, poll_timeout=poll_timeout, retry_delay=retry_delay)
