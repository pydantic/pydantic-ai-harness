from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import TypeGuard

import anyio
import httpx
import pytest
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import TypeAdapter
from pydantic_ai import Agent
from starlette.applications import Starlette

from pydantic_ai_harness.channels import ChannelEvent, ChannelHost, InMemoryConversationStore
from pydantic_ai_harness.channels.telegram import (
    TelegramChannel,
    TelegramError,
    TelegramPartialDeliveryError,
    TelegramRateLimitError,
    TelegramWebhookError,
)

pytestmark = pytest.mark.anyio

_BODY_ADAPTER = TypeAdapter(dict[str, object])


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _response(payload: object, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _headers(secret: str = 'webhook-secret') -> Mapping[str, str]:
    return {'x-telegram-bot-api-secret-token': secret}


def _update(
    *,
    update_id: int = 11,
    chat_id: int = -100123,
    sender_id: int = 22,
    message_id: int = 33,
    topic_id: int | None = 44,
    text: str = 'hello',
) -> bytes:
    message: dict[str, object] = {
        'message_id': message_id,
        'from': {'id': sender_id, 'is_bot': False},
        'chat': {'id': chat_id, 'type': 'supergroup'},
        'text': text,
    }
    if topic_id is not None:
        message['message_thread_id'] = topic_id
        message['is_topic_message'] = True
    return json.dumps({'update_id': update_id, 'message': message}).encode()


def _channel(client: httpx.AsyncClient | None = None) -> TelegramChannel:
    return TelegramChannel(
        bot_id=9001,
        bot_token='bot-secret',
        webhook_secret='webhook-secret',
        allowed_senders={22},
        client=client,
    )


def _telegram_docs_example() -> str:
    readme = Path(__file__).parents[2] / 'pydantic_ai_harness' / 'channels' / 'README.md'
    telegram_section = readme.read_text(encoding='utf-8').split('Save this as `telegram_bot.py`:', 1)[1]
    return telegram_section.split('```python {names="defined"}', 1)[1].split('```', 1)[0]


def _is_async_no_arg(value: object) -> TypeGuard[Callable[[], Awaitable[None]]]:
    return callable(value)


class TestTelegramChannel:
    async def test_maps_verified_topic_reply_update(self) -> None:
        channel = _channel()

        event = channel.parse_request(
            _update(),
            {'X-TELEGRAM-BOT-API-SECRET-TOKEN': 'webhook-secret'},
        )

        assert event == ChannelEvent(
            event_id='telegram:bot:9001:update:11',
            conversation_id='telegram:bot:9001:chat:-100123:topic:44',
            sender_id='telegram:user:22',
            text='hello',
            reply_to_id='telegram:message:33',
            delivery_id='-100123',
        )

    async def test_scopes_claims_and_history_to_bot(self) -> None:
        first = _channel().parse_request(_update(), _headers())
        second = TelegramChannel(
            bot_id=9002,
            bot_token='other-secret',
            webhook_secret='webhook-secret',
            allowed_senders={22},
        ).parse_request(_update(), _headers())

        assert first is not None
        assert second is not None
        assert first.event_id != second.event_id
        assert first.conversation_id != second.conversation_id

    async def test_rejects_unverified_or_malformed_webhook(self) -> None:
        channel = _channel()

        for headers in (
            {},
            _headers('wrong'),
            {'X-Telegram-Bot-Api-Secret-Token': 'wrong'},
            {'X-Telegram-Bot-Api-Secret-Token': 'é'},
        ):
            with pytest.raises(TelegramWebhookError, match='secret token'):
                channel.parse_request(_update(), headers)
            with pytest.raises(TelegramWebhookError, match='secret token'):
                channel.parse_request(b'not json', headers)

        with pytest.raises(TelegramWebhookError, match='ambiguous'):
            channel.parse_request(
                _update(),
                {
                    'X-Telegram-Bot-Api-Secret-Token': 'webhook-secret',
                    'x-telegram-bot-api-secret-token': 'webhook-secret',
                },
            )

        for body in (
            b'not json',
            b'[]',
            b'{"update_id":true}',
            b'{"update_id":0}',
            b'{"update_id":-1}',
            b'{"message":{}}',
            b'{"update_id":1,"message":"invalid"}',
        ):
            with pytest.raises(TelegramError, match='payload'):
                channel.parse_request(body, _headers())

    async def test_ignores_unsupported_or_disallowed_updates(self) -> None:
        channel = _channel()

        assert channel.parse_request(b'{"update_id":1,"callback_query":{}}', _headers()) is None
        assert (
            channel.parse_request(
                b'{"update_id":2,"message":{"message_id":3,"from":{"id":22},"chat":{"id":4},"photo":[]}}',
                _headers(),
            )
            is None
        )
        assert channel.parse_request(_update(update_id=3, sender_id=99), _headers()) is None

        bot_message = _BODY_ADAPTER.validate_python(json.loads(_update(update_id=4)))
        message = _BODY_ADAPTER.validate_python(bot_message['message'])
        sender = _BODY_ADAPTER.validate_python(message['from'])
        sender['is_bot'] = True
        message['from'] = sender
        bot_message['message'] = message
        assert channel.parse_request(json.dumps(bot_message).encode(), _headers()) is None

    async def test_maps_anonymous_chat_sender_without_trusting_fake_user(self) -> None:
        body = _BODY_ADAPTER.validate_python(json.loads(_update()))
        message = _BODY_ADAPTER.validate_python(body['message'])
        message['sender_chat'] = {'id': -100999}
        body['message'] = message
        channel = TelegramChannel(
            bot_id=9001,
            bot_token='bot-secret',
            webhook_secret='webhook-secret',
            allowed_senders={-100999},
        )

        event = channel.parse_request(json.dumps(body).encode(), _headers())

        assert event is not None
        assert event.sender_id == 'telegram:chat:-100999'

    async def test_rejects_malformed_message_identities(self) -> None:
        channel = _channel()

        malformed_messages = (
            {'message_id': True, 'from': {'id': 22, 'is_bot': False}, 'chat': {'id': 1}, 'text': 'hello'},
            {'message_id': -1, 'from': {'id': 22, 'is_bot': False}, 'chat': {'id': 1}, 'text': 'hello'},
            {'message_id': 1, 'from': {'id': 22, 'is_bot': False}, 'chat': {'id': True}, 'text': 'hello'},
            {'message_id': 1, 'from': {'id': 22, 'is_bot': False}, 'chat': {'id': 0}, 'text': 'hello'},
            {
                'message_id': 1,
                'message_thread_id': 0,
                'is_topic_message': True,
                'from': {'id': 22, 'is_bot': False},
                'chat': {'id': 1},
                'text': 'hello',
            },
            {'message_id': 1, 'from': {'id': True, 'is_bot': False}, 'chat': {'id': 1}, 'text': 'hello'},
            {'message_id': 1, 'from': {'id': 0, 'is_bot': False}, 'chat': {'id': 1}, 'text': 'hello'},
            {'message_id': 1, 'from': {'id': 22}, 'chat': {'id': 1}, 'text': 'hello'},
            {
                'message_id': 1,
                'sender_chat': 'invalid',
                'from': {'id': 22, 'is_bot': False},
                'chat': {'id': 1},
                'text': 'hello',
            },
            {
                'message_id': 1,
                'sender_chat': {'id': True},
                'from': {'id': 22, 'is_bot': False},
                'chat': {'id': 1},
                'text': 'hello',
            },
            {
                'message_id': 1,
                'sender_chat': {'id': 0},
                'from': {'id': 22, 'is_bot': False},
                'chat': {'id': 1},
                'text': 'hello',
            },
            {'message_id': 1, 'chat': {'id': 1}, 'text': 'hello'},
            {
                'message_id': 1,
                'direct_messages_topic': 'invalid',
                'from': {'id': 22, 'is_bot': False},
                'chat': {'id': 1},
                'text': 'hello',
            },
            {
                'message_id': 1,
                'direct_messages_topic': {'topic_id': 0},
                'from': {'id': 22, 'is_bot': False},
                'chat': {'id': 1},
                'text': 'hello',
            },
        )

        for message in malformed_messages:
            body = json.dumps({'update_id': 1, 'message': message}).encode()
            with pytest.raises(TelegramError, match='identity'):
                channel.parse_request(body, _headers())

        ephemeral = {
            'message_id': 0,
            'ephemeral_message_id': 9,
            'from': {'id': 22, 'is_bot': False},
            'chat': {'id': 1},
            'text': 'hello',
        }
        assert channel.parse_request(json.dumps({'update_id': 1, 'message': ephemeral}).encode(), _headers()) is None

        scheduled = {**ephemeral, 'ephemeral_message_id': None}
        assert channel.parse_request(json.dumps({'update_id': 2, 'message': scheduled}).encode(), _headers()) is None

        plain_event = channel.parse_request(_update(topic_id=None), _headers())
        assert plain_event is not None
        assert plain_event.conversation_id == 'telegram:bot:9001:chat:-100123'

    async def test_keeps_generic_message_thread_in_chat_conversation(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return _response({'ok': True, 'result': {'message_id': 1}})

        body = _BODY_ADAPTER.validate_python(json.loads(_update(topic_id=None)))
        message = _BODY_ADAPTER.validate_python(body['message'])
        message['message_thread_id'] = 2
        body['message'] = message
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)

        event = channel.parse_request(json.dumps(body).encode(), _headers())

        assert event is not None
        assert event.conversation_id == 'telegram:bot:9001:chat:-100123'
        await channel.reply(event, 'answer')
        assert requests == [
            {
                'chat_id': -100123,
                'reply_parameters': {'message_id': 33},
                'text': 'answer',
            }
        ]
        await client.aclose()

    async def test_rejects_forum_topic_marker_without_thread_id(self) -> None:
        for topic_fields in ({'is_topic_message': True}, {'is_topic_message': False, 'message_thread_id': 2}):
            body = _BODY_ADAPTER.validate_python(json.loads(_update(topic_id=None)))
            message = _BODY_ADAPTER.validate_python(body['message'])
            message.update(topic_fields)
            body['message'] = message

            with pytest.raises(TelegramError, match='invalid topic identity'):
                _channel().parse_request(json.dumps(body).encode(), _headers())

    async def test_preserves_direct_message_topic_identity(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return _response({'ok': True, 'result': {'message_id': 1}})

        body = _BODY_ADAPTER.validate_python(json.loads(_update(topic_id=None)))
        message = _BODY_ADAPTER.validate_python(body['message'])
        message['direct_messages_topic'] = {'topic_id': 55, 'user': {'id': 22, 'is_bot': False}}
        body['message'] = message
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)

        event = channel.parse_request(json.dumps(body).encode(), _headers())

        assert event is not None
        assert event.conversation_id == 'telegram:bot:9001:chat:-100123:direct-topic:55'
        await channel.reply(event, 'answer')
        assert requests == [
            {
                'chat_id': -100123,
                'direct_messages_topic_id': 55,
                'reply_parameters': {'message_id': 33},
                'text': 'answer',
            }
        ]

        message['message_thread_id'] = 44
        message['is_topic_message'] = True
        with pytest.raises(TelegramError, match='ambiguous topic'):
            channel.parse_request(json.dumps(body).encode(), _headers())

        message.pop('message_thread_id')
        message.pop('is_topic_message')
        direct_topic = _BODY_ADAPTER.validate_python(message['direct_messages_topic'])
        direct_topic['topic_id'] = 56
        message['direct_messages_topic'] = direct_topic
        other_event = channel.parse_request(json.dumps(body).encode(), _headers())
        assert other_event is not None
        assert other_event.conversation_id != event.conversation_id
        await client.aclose()

    async def test_duplicate_update_is_idempotent_at_admission_boundary(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return _response({'ok': True, 'result': {'message_id': 1}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        host = ChannelHost(Agent('test'), channel)
        claimed: set[str] = set()

        for _ in range(2):
            event = channel.parse_request(_update(), _headers())
            assert event is not None
            if event.event_id in claimed:
                continue
            claimed.add(event.event_id)
            await host.handle(event)

        assert claimed == {'telegram:bot:9001:update:11'}
        assert len(requests) == 1
        await client.aclose()

    async def test_documented_server_bounds_ingress_and_claims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('TELEGRAM_BOT_ID', '9001')
        monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'bot-secret')
        monkeypatch.setenv('TELEGRAM_ALLOWED_SENDER_ID', '22')
        monkeypatch.setenv('TELEGRAM_WEBHOOK_SECRET', 'webhook-secret')
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        namespace: dict[str, object] = {}
        exec(compile(_telegram_docs_example(), 'telegram_bot.py', 'exec'), namespace)
        app = namespace.get('app')
        assert isinstance(app, Starlette)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='https://example.test') as client:
            oversized = await client.post('/telegram', content=b'x' * 1_000_001, headers=_headers())
            unauthorized = await client.post('/telegram', content=b'{}')
            malformed = await client.post('/telegram', content=b'not json', headers=_headers())
            ignored = await client.post('/telegram', content=b'{"update_id":1}', headers=_headers())

            assert [oversized.status_code, unauthorized.status_code, malformed.status_code, ignored.status_code] == [
                413,
                401,
                400,
                204,
            ]

            move_on_after = anyio.move_on_after

            def expire_body_read(delay: float | None) -> anyio.CancelScope:
                return move_on_after(0 if delay == 5 else delay)

            monkeypatch.setattr(anyio, 'move_on_after', expire_body_read)
            timed_out = await client.post('/telegram', content=b'{}', headers=_headers())
            assert timed_out.status_code == 408
            monkeypatch.setattr(anyio, 'move_on_after', move_on_after)

            for update_id in range(100):
                queued = await client.post('/telegram', content=_update(update_id=update_id + 1), headers=_headers())
                assert queued.status_code == 204

            def expire_enqueue(delay: float | None) -> anyio.CancelScope:
                return move_on_after(0 if delay == 2 else delay)

            monkeypatch.setattr(anyio, 'move_on_after', expire_enqueue)
            full = await client.post('/telegram', content=_update(update_id=101), headers=_headers())
            assert full.status_code == 503

        send_events = namespace.get('send_events')
        receive_events = namespace.get('receive_events')
        assert isinstance(send_events, MemoryObjectSendStream)
        assert isinstance(receive_events, MemoryObjectReceiveStream)
        await send_events.aclose()
        await receive_events.aclose()

        events = [
            ChannelEvent(
                event_id=f'telegram:bot:9001:update:{update_id}',
                conversation_id='telegram:bot:9001:chat:1',
                sender_id='telegram:user:22',
                text='hello',
            )
            for update_id in range(10_001)
        ]
        events.append(events[0])

        class FakeHost:
            def __init__(self) -> None:
                self.handled: list[str] = []

            async def handle(self, event: ChannelEvent) -> None:
                self.handled.append(event.event_id)

        class FakeReceive:
            def __init__(self, values: list[ChannelEvent]) -> None:
                self._values: Iterator[ChannelEvent] = iter(values)

            async def __aenter__(self) -> FakeReceive:
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                return None

            def __aiter__(self) -> FakeReceive:
                return self

            async def __anext__(self) -> ChannelEvent:
                try:
                    return next(self._values)
                except StopIteration:
                    raise StopAsyncIteration from None

        host = FakeHost()
        namespace['host'] = host
        namespace['receive_events'] = FakeReceive(events)
        consume_events = namespace.get('consume_events')
        assert _is_async_no_arg(consume_events)

        await consume_events()

        assert len(host.handled) == 10_002
        assert host.handled.count(events[0].event_id) == 2

    async def test_sends_to_original_topic_and_message(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return _response({'ok': True, 'result': {'message_id': 1}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        await channel.reply(event, 'answer')

        assert requests == [
            {
                'chat_id': -100123,
                'message_thread_id': 44,
                'reply_parameters': {'message_id': 33},
                'text': 'answer',
            }
        ]
        assert not client.is_closed
        await client.aclose()

    async def test_topics_keep_separate_history_and_share_raw_chat_delivery(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return _response({'ok': True, 'result': {'message_id': len(requests)}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        store = InMemoryConversationStore()
        host = ChannelHost(Agent('test'), channel, store=store)
        first = channel.parse_request(_update(topic_id=44, text='first topic'), _headers())
        second = channel.parse_request(
            _update(update_id=12, message_id=34, topic_id=45, text='second topic'), _headers()
        )
        assert first is not None
        assert second is not None

        first_result = await host.handle(first)
        second_result = await host.handle(second)

        assert first.conversation_id != second.conversation_id
        assert first.delivery_id == second.delivery_id == '-100123'
        assert await store.load(first.conversation_id) == first_result.all_messages()
        assert await store.load(second.conversation_id) == second_result.all_messages()
        assert len(first_result.all_messages()) == len(second_result.all_messages())
        assert requests == [
            {
                'chat_id': -100123,
                'message_thread_id': 44,
                'reply_parameters': {'message_id': 33},
                'text': first_result.output,
            },
            {
                'chat_id': -100123,
                'message_thread_id': 45,
                'reply_parameters': {'message_id': 34},
                'text': second_result.output,
            },
        ]
        await client.aclose()

    async def test_chunks_text_at_telegram_limit(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return _response({'ok': True, 'result': {'message_id': len(requests)}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = ChannelEvent(
            event_id='telegram:bot:9001:update:1',
            conversation_id='telegram:bot:9001:chat:7',
            sender_id='telegram:user:22',
            text='prompt',
            delivery_id='7',
        )

        await channel.reply(event, 'a' * 4097)
        await channel.reply(event, '😀' * 2049)

        assert [request['text'] for request in requests] == ['a' * 4096, 'a', '😀' * 2049]
        assert all('message_thread_id' not in request and 'reply_parameters' not in request for request in requests)
        with pytest.raises(ValueError, match='text must not be empty'):
            await channel.reply(event, '')
        await client.aclose()

    async def test_reports_confirmed_chunks_after_partial_delivery(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise httpx.ReadError('connection lost', request=request)
            return _response({'ok': True, 'result': {'message_id': calls}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        with pytest.raises(TelegramPartialDeliveryError, match='1 of 2') as exc_info:
            await channel.reply(event, 'a' * 4097)

        assert exc_info.value.sent_chunks == 1
        assert exc_info.value.retry_after is None
        assert calls == 2
        await client.aclose()

    async def test_preserves_rate_limit_after_partial_delivery(self) -> None:
        calls = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 2:
                return _response(
                    {
                        'ok': False,
                        'error_code': 429,
                        'description': 'Too Many Requests',
                        'parameters': {'retry_after': 2_282},
                    },
                    status_code=429,
                )
            return _response({'ok': True, 'result': {'message_id': calls}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        with pytest.raises(TelegramPartialDeliveryError, match='1 of 2') as exc_info:
            await channel.reply(event, 'a' * 4097)

        assert exc_info.value.sent_chunks == 1
        assert exc_info.value.retry_after == 2_282
        assert calls == 2
        await client.aclose()

    async def test_retry_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _response(
                    {
                        'ok': False,
                        'error_code': 429,
                        'description': 'Too Many Requests',
                        'parameters': {'retry_after': 2},
                    },
                    status_code=429,
                )
            return _response({'ok': True, 'result': {'message_id': 1}})

        monkeypatch.setattr('pydantic_ai_harness.channels.telegram.anyio.sleep', sleep)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        await channel.reply(event, 'answer')

        assert calls == 2
        assert sleeps == [2.0]
        await client.aclose()

    async def test_does_not_retry_ambiguous_transport_failure(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadError('connection lost', request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        with pytest.raises(TelegramError, match='request failed') as exc_info:
            await channel.reply(event, 'answer')

        assert calls == 1
        assert exc_info.value.__cause__ is None
        assert 'bot-secret' not in str(exc_info.value)
        await client.aclose()

    async def test_surfaces_unretried_rate_limits_and_rejects_malformed_retry_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retry_after: object = 0
        calls = 0
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            payload: dict[str, object] = {
                'ok': False,
                'error_code': 429,
                'description': 'Too Many Requests',
            }
            if retry_after is not None:
                payload['parameters'] = {'retry_after': retry_after}
            return _response(payload, status_code=429)

        monkeypatch.setattr('pydantic_ai_harness.channels.telegram.anyio.sleep', sleep)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        with pytest.raises(TelegramRateLimitError, match='Too Many Requests') as repeated:
            await channel.reply(event, 'first')
        assert calls == 2
        assert repeated.value.retry_after == 0
        assert sleeps == [0]

        retry_after = False
        with pytest.raises(TelegramError, match='Too Many Requests'):
            await channel.reply(event, 'second')
        assert calls == 3

        retry_after = -1
        with pytest.raises(TelegramError, match='Too Many Requests'):
            await channel.reply(event, 'third')
        assert calls == 4

        retry_after = 1.5
        with pytest.raises(TelegramError, match='Too Many Requests'):
            await channel.reply(event, 'fourth')
        assert calls == 5

        retry_after = None
        with pytest.raises(TelegramError, match='Too Many Requests'):
            await channel.reply(event, 'fifth')
        assert calls == 6

        retry_after = 2_282
        with pytest.raises(TelegramRateLimitError, match='Too Many Requests') as long_delay:
            await channel.reply(event, 'sixth')
        assert calls == 7
        assert long_delay.value.retry_after == 2_282
        assert sleeps == [0]
        await client.aclose()

    async def test_rejects_invalid_provider_responses_and_event_identity(self) -> None:
        responses = iter(
            (
                httpx.Response(200, content=b'not json'),
                _response([], status_code=500),
                _response({}),
                _response({'ok': False}, status_code=401),
                _response({'ok': True}, status_code=500),
                _response({'ok': True}),
            )
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return next(responses)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = ChannelEvent(
            event_id='telegram:bot:9001:update:1',
            conversation_id='telegram:bot:9001:chat:7',
            sender_id='telegram:user:22',
            text='prompt',
            delivery_id='7',
        )

        for _ in range(6):
            with pytest.raises(TelegramError):
                await channel.reply(event, 'answer')

        with pytest.raises(TelegramError, match='conversation_id'):
            await channel.reply(
                ChannelEvent(
                    event_id='telegram:bot:9001:update:1',
                    conversation_id='slack:channel:7',
                    sender_id='telegram:user:22',
                    text='prompt',
                ),
                'answer',
            )
        with pytest.raises(TelegramError, match='conversation_id'):
            await channel.reply(
                ChannelEvent(
                    event_id='telegram:bot:9002:update:1',
                    conversation_id='telegram:bot:9002:chat:7',
                    sender_id='telegram:user:22',
                    text='prompt',
                ),
                'answer',
            )
        with pytest.raises(TelegramError, match='conversation_id'):
            await channel.reply(
                ChannelEvent(
                    event_id='telegram:bot:9001:update:1',
                    conversation_id='telegram:bot:9001:chat:7:topic:0',
                    sender_id='telegram:user:22',
                    text='prompt',
                ),
                'answer',
            )
        with pytest.raises(TelegramError, match='reply_to_id'):
            await channel.reply(
                ChannelEvent(
                    event_id='telegram:bot:9001:update:1',
                    conversation_id='telegram:bot:9001:chat:7',
                    sender_id='telegram:user:22',
                    text='prompt',
                    reply_to_id='slack:message:2',
                    delivery_id='7',
                ),
                'answer',
            )
        with pytest.raises(TelegramError, match='reply_to_id'):
            await channel.reply(
                ChannelEvent(
                    event_id='telegram:bot:9001:update:1',
                    conversation_id='telegram:bot:9001:chat:7',
                    sender_id='telegram:user:22',
                    text='prompt',
                    reply_to_id='telegram:message:0',
                    delivery_id='7',
                ),
                'answer',
            )
        for delivery_id in (None, '0', 'not-a-chat', '8'):
            with pytest.raises(TelegramError, match='delivery_id'):
                await channel.reply(
                    ChannelEvent(
                        event_id='telegram:bot:9001:update:1',
                        conversation_id='telegram:bot:9001:chat:7',
                        sender_id='telegram:user:22',
                        text='prompt',
                        delivery_id=delivery_id,
                    ),
                    'answer',
                )
        await client.aclose()

    async def test_does_not_retry_authentication_failure_with_retry_after(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        async def sleep(_delay: float) -> None:  # pragma: no cover - this test asserts no retry
            pytest.fail('authentication failure must not sleep')

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(
                {
                    'ok': False,
                    'error_code': 401,
                    'description': 'Unauthorized',
                    'parameters': {'retry_after': 1},
                },
                status_code=401,
            )

        monkeypatch.setattr('pydantic_ai_harness.channels.telegram.anyio.sleep', sleep)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        with pytest.raises(TelegramError, match='Unauthorized'):
            await channel.reply(event, 'answer')

        assert calls == 1
        await client.aclose()

    async def test_does_not_retry_malformed_rate_limit_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        async def sleep(_delay: float) -> None:  # pragma: no cover - this test asserts no retry
            pytest.fail('malformed response must not sleep')

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _response(
                {'ok': True, 'parameters': {'retry_after': 1}},
                status_code=429,
            )

        monkeypatch.setattr('pydantic_ai_harness.channels.telegram.anyio.sleep', sleep)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        with pytest.raises(TelegramError, match='invalid response'):
            await channel.reply(event, 'answer')

        assert calls == 1
        await client.aclose()

    async def test_uses_custom_api_url(self, caplog: pytest.LogCaptureFixture) -> None:
        requests: list[httpx.Request] = []
        unrelated_log_arguments: list[object] = []

        class ObserveUnrelatedUrl(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                args = record.args
                if isinstance(args, tuple) and len(args) > 1:  # pragma: no branch - the test emits positional args
                    unrelated_log_arguments.append(args[1])
                return True

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _response({'ok': True, 'result': {'message_id': 1}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = TelegramChannel(
            bot_id=9001,
            bot_token='bot-secret',
            webhook_secret='webhook-secret',
            allowed_senders={22},
            client=client,
            api_url='https://telegram.test/bot-decoy/root/',
        )
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        await channel.reply(event, 'answer')

        observer = ObserveUnrelatedUrl()
        logging.getLogger('httpx').addFilter(observer)
        with caplog.at_level(logging.INFO, logger='httpx'):
            await channel.reply(event, 'second')
            unrelated_url = httpx.URL('https://example.com/botinventory/items')
            logging.getLogger('httpx').info(
                'HTTP Request: %s %s "%s %d %s"',
                'GET',
                unrelated_url,
                'HTTP/1.1',
                200,
                'OK',
            )
        logging.getLogger('httpx').removeFilter(observer)

        assert str(requests[0].url) == 'https://telegram.test/bot-decoy/root/botbot-secret/sendMessage'
        assert 'bot-secret' not in caplog.text
        assert '<redacted>' in caplog.text
        assert 'https://example.com/botinventory/items' in caplog.text
        assert unrelated_log_arguments[-1] is unrelated_url
        await client.aclose()

    async def test_redacts_overlapping_active_bot_tokens(self, caplog: pytest.LogCaptureFixture) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: _response({'ok': True, 'result': {'message_id': 1}}))
        )
        _first = TelegramChannel(
            bot_id=9001,
            bot_token='first-secret',
            webhook_secret='webhook-secret',
            allowed_senders={22},
            client=client,
            api_url='https://proxy.test',
        )
        second = TelegramChannel(
            bot_id=9002,
            bot_token='second-secret',
            webhook_secret='webhook-secret',
            allowed_senders={22},
            client=client,
            api_url='https://proxy.test/botfirst-secret/tenant',
        )
        event = second.parse_request(_update(), _headers())
        assert event is not None

        with caplog.at_level(logging.INFO, logger='httpx'):
            await second.reply(event, 'answer')

        assert 'first-secret' not in caplog.text
        assert 'second-secret' not in caplog.text
        assert caplog.text.count('<redacted>') == 2
        await client.aclose()

    async def test_closes_short_lived_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clients: list[TrackingClient] = []

        class TrackingClient(httpx.AsyncClient):
            def __init__(self) -> None:
                super().__init__(
                    transport=httpx.MockTransport(lambda _request: _response({'ok': True, 'result': {'message_id': 1}}))
                )
                clients.append(self)

        monkeypatch.setattr('pydantic_ai_harness.channels.telegram.httpx.AsyncClient', TrackingClient)
        channel = _channel()
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        await channel.reply(event, 'answer')

        assert len(clients) == 1
        assert clients[0].is_closed

    async def test_bot_token_is_redacted(self, caplog: pytest.LogCaptureFixture) -> None:
        calls = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _response({'ok': True, 'result': {'message_id': 1}})
            return _response(
                {
                    'ok': False,
                    'error_code': 401,
                    'description': 'Unauthorized https://api.telegram.org/botbot-secret/sendMessage',
                },
                status_code=401,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(), _headers())
        assert event is not None

        with caplog.at_level(logging.INFO, logger='httpx'):
            logging.getLogger('httpx').info('unrelated %s', 'request')
            await channel.reply(event, 'first')
            with pytest.raises(TelegramError, match='Unauthorized') as exc_info:
                await channel.reply(event, 'second')

        assert 'bot-secret' not in str(exc_info.value)
        assert 'bot-secret' not in caplog.text
        assert '<redacted>' in caplog.text
        await client.aclose()

    async def test_agent_integration(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(_BODY_ADAPTER.validate_python(json.loads(request.content)))
            return _response({'ok': True, 'result': {'message_id': 1}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        channel = _channel(client)
        event = channel.parse_request(_update(text='run the agent'), _headers())
        assert event is not None

        result = await ChannelHost(Agent('test'), channel).handle(event)

        assert result.output == 'success (no tool calls)'
        assert requests[0]['text'] == result.output
        assert all(message.conversation_id == event.conversation_id for message in result.all_messages())
        await client.aclose()

    async def test_validates_configuration(self) -> None:
        with pytest.raises(ValueError, match='bot_id'):
            TelegramChannel(bot_id=0, bot_token='token', webhook_secret='secret', allowed_senders={22})
        with pytest.raises(ValueError, match='bot_token'):
            TelegramChannel(bot_id=9001, bot_token='', webhook_secret='secret', allowed_senders={22})
        with pytest.raises(ValueError, match='webhook_secret'):
            TelegramChannel(bot_id=9001, bot_token='token', webhook_secret='', allowed_senders={22})
        with pytest.raises(ValueError, match='webhook_secret'):
            TelegramChannel(bot_id=9001, bot_token='token', webhook_secret='contains spaces', allowed_senders={22})
        with pytest.raises(ValueError, match='allowed_senders'):
            TelegramChannel(bot_id=9001, bot_token='token', webhook_secret='secret', allowed_senders=set())
        with pytest.raises(TypeError, match='allowed_senders'):
            TelegramChannel(
                bot_id=9001,
                bot_token='token',
                webhook_secret='secret',
                allowed_senders='22',  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match='allowed_senders'):
            TelegramChannel(bot_id=9001, bot_token='token', webhook_secret='secret', allowed_senders={True})
        with pytest.raises(ValueError, match='allowed_senders'):
            TelegramChannel(bot_id=9001, bot_token='token', webhook_secret='secret', allowed_senders={0})
        with pytest.raises(ValueError, match='api_url'):
            TelegramChannel(bot_id=9001, bot_token='token', webhook_secret='secret', allowed_senders={22}, api_url='')
