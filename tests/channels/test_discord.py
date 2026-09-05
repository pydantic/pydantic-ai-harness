from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest

pytest.importorskip('websockets')

from pydantic import TypeAdapter
from pydantic_ai import Agent
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.http11 import Response

from pydantic_ai_harness.channels import ChannelEvent, ChannelHost
from pydantic_ai_harness.channels.discord import DiscordChannel, DiscordChannelError

pytestmark = pytest.mark.anyio

_JSON_OBJECT = TypeAdapter(dict[str, object])
_SOCKET_ADDRESS = TypeAdapter(tuple[str, int])


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@asynccontextmanager
async def _gateway(handler: Callable[[ServerConnection], Awaitable[None]]) -> AsyncGenerator[str, None]:
    failures: list[Exception] = []

    async def checked_handler(connection: ServerConnection) -> None:
        if failures:
            await connection.close(code=4014)
            return
        try:
            await handler(connection)
        except Exception as exc:
            failures.append(exc)
            raise

    server: Server = await serve(checked_handler, '127.0.0.1', 0)
    socket = next(iter(server.sockets))
    host, port = _SOCKET_ADDRESS.validate_python(socket.getsockname()[:2])
    try:
        yield f'ws://{host}:{port}'
    finally:
        server.close()
        await server.wait_closed()
        if failures:
            raise failures[0]


def _message(*, content: str = 'hello', guild_id: str | None = None, message_type: int = 0) -> dict[str, object]:
    data: dict[str, object] = {
        'id': 'message-1',
        'channel_id': 'thread-1',
        'content': content,
        'type': message_type,
        'author': {'id': 'user-1', 'bot': False},
        'mentions': [{'id': 'bot-1'}],
    }
    if guild_id is not None:
        data['guild_id'] = guild_id
    return {'op': 0, 't': 'MESSAGE_CREATE', 's': 2, 'd': data}


class TestDiscordChannel:
    async def test_gateway_fixture_propagates_handler_failure(self) -> None:
        async def handler(_connection: ServerConnection) -> None:
            raise RuntimeError('sentinel gateway failure')

        with pytest.raises(RuntimeError, match='sentinel gateway failure'):
            async with _gateway(handler) as gateway_url:
                channel = DiscordChannel('token', allowed_user_ids=None, gateway_url=gateway_url)
                with pytest.raises(DiscordChannelError, match='4014'):  # pragma: no branch
                    await anext(channel.events())

    async def test_identifies_and_maps_guild_thread_message(self) -> None:
        identified: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            identified.append(_JSON_OBJECT.validate_json(await connection.recv()))
            await connection.send(
                json.dumps(
                    {
                        'op': 0,
                        't': 'READY',
                        's': 1,
                        'd': {
                            'session_id': 'session-1',
                            'resume_gateway_url': 'ws://unused',
                            'user': {'id': 'bot-1'},
                        },
                    }
                )
            )
            await connection.send(json.dumps(_message(content='<@bot-1> explain this', guild_id='guild-1')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel(
                'secret-token',
                allowed_user_ids={'user-1'},
                allowed_guild_ids={'guild-1'},
                gateway_url=gateway_url,
            )
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        identify_data = _JSON_OBJECT.validate_python(identified[0]['d'])
        assert identified[0]['op'] == 2
        assert identify_data['token'] == 'secret-token'
        assert identify_data['intents'] == 4_608
        assert event == ChannelEvent(
            event_id='message-1',
            conversation_id='thread-1',
            sender_id='user-1',
            text='explain this',
            reply_to_id='message-1',
        )
        assert event.delivery_id is None

    async def test_unmentioned_guild_messages_request_message_content_intent(self) -> None:
        identified: dict[str, object] = {}

        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            identified.update(_JSON_OBJECT.validate_json(await connection.recv()))
            await connection.send(json.dumps(_message(content='unmentioned', guild_id='guild-1')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel(
                'token',
                allowed_user_ids={'user-1'},
                allowed_guild_ids={'guild-1'},
                require_mention=False,
                gateway_url=gateway_url,
            )
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        identify_data = _JSON_OBJECT.validate_python(identified['d'])
        assert identify_data['intents'] == 37_376
        assert event.text == 'unmentioned'

    async def test_explicit_intents_override_defaults(self) -> None:
        identified: dict[str, object] = {}

        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            identified.update(_JSON_OBJECT.validate_json(await connection.recv()))
            await connection.send(json.dumps(_message(content='event')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, intents=1, gateway_url=gateway_url)
            events = channel.events()
            await anext(events)
            await events.aclose()

        identify_data = _JSON_OBJECT.validate_python(identified['d'])
        assert identify_data['intents'] == 1

    async def test_forces_json_gateway_encoding_without_compression(self) -> None:
        request_path = ''

        async def handler(connection: ServerConnection) -> None:
            nonlocal request_path
            assert connection.request is not None
            request_path = connection.request.path
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(json.dumps(_message(content='json event')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel(
                'token',
                allowed_user_ids={'user-1'},
                gateway_url=f'{gateway_url}?v=9&encoding=etf&compress=zlib-stream&trace=kept',
            )
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'json event'
        assert request_path == '/?trace=kept&v=10&encoding=json'

    async def test_handshake_ignores_malformed_payloads_before_hello(self) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(b'{}')
            await connection.send('{not json')
            await connection.send(json.dumps({'op': 'unknown'}))
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(json.dumps(_message(content='after malformed handshake payloads')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'after malformed handshake payloads'

    @pytest.mark.parametrize('reconnect_payload', [{'op': 7, 'd': None}, {'op': 9, 'd': True}, {'op': 9, 'd': False}])
    async def test_reconnect_request_before_hello_starts_new_connection(
        self, reconnect_payload: dict[str, object]
    ) -> None:
        connections = 0

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            if connections == 1:
                await connection.send(json.dumps(reconnect_payload))
                await connection.wait_closed()
                return
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(json.dumps(_message(content='after reconnect request')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert connections == 2
        assert event.text == 'after reconnect request'

    async def test_gateway_http_failures_retry_server_error_then_surface_client_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statuses = iter((503, 401))

        class RejectedConnection:
            async def __aenter__(self) -> None:
                status = next(statuses)
                raise InvalidStatus(Response(status, 'rejected', Headers()))

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: object,
            ) -> None:
                return None  # pragma: no cover

        def reject_connection(*_args: object, **_kwargs: object) -> RejectedConnection:
            return RejectedConnection()

        monkeypatch.setattr('pydantic_ai_harness.channels.discord.connect', reject_connection)
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.sleep', AsyncMock())
        channel = DiscordChannel('token', allowed_user_ids=None)

        with pytest.raises(DiscordChannelError, match='HTTP 401'):
            await anext(channel.events())

    @pytest.mark.parametrize('failure', ['network', 'http'])
    async def test_failed_resume_endpoint_falls_back_to_fresh_identify(
        self, failure: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        authentications: list[dict[str, object]] = []
        connections = 0

        async def reject_handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.readuntil(b'\r\n\r\n')
            writer.write(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        rejection_server = await asyncio.start_server(reject_handshake, '127.0.0.1', 0)
        rejection_socket = rejection_server.sockets[0]
        rejection_host, rejection_port = _SOCKET_ADDRESS.validate_python(rejection_socket.getsockname()[:2])
        resume_gateway_url = 'ws://127.0.0.1:1' if failure == 'network' else f'ws://{rejection_host}:{rejection_port}'

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
            if connections == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 1,
                            'd': {
                                'session_id': 'session',
                                'resume_gateway_url': resume_gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                await connection.close(code=4000)
            else:
                await connection.send(json.dumps(_message(content='fresh connection')))
                await connection.wait_closed()

        never_release = asyncio.Event()

        async def skip_transport_delays(delay: float) -> None:
            if delay >= 60:
                await never_release.wait()

        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.sleep', skip_transport_delays)
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.random.random', lambda: 1.0)
        try:
            async with _gateway(handler) as gateway_url:
                channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
                events = channel.events()
                event = await anext(events)
                await events.aclose()
        finally:
            rejection_server.close()
            await rejection_server.wait_closed()

        assert event.text == 'fresh connection'
        assert [authentication['op'] for authentication in authentications] == [2, 2]

    @pytest.mark.parametrize(
        'ready_data',
        [
            {'resume_gateway_url': 'ws://resume', 'user': {'id': 'bot'}},
            {'session_id': '', 'resume_gateway_url': 'ws://resume', 'user': {'id': 'bot'}},
            {'session_id': 'session', 'user': {'id': 'bot'}},
            {'session_id': 'session', 'resume_gateway_url': '', 'user': {'id': 'bot'}},
            {'session_id': 'session', 'resume_gateway_url': 'ws://resume'},
            {'session_id': 'session', 'resume_gateway_url': 'ws://resume', 'user': {'id': ''}},
        ],
    )
    async def test_rejects_ready_without_required_session_fields(self, ready_data: dict[str, object]) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(json.dumps({'op': 0, 't': 'READY', 's': 1, 'd': ready_data}))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids=None, gateway_url=gateway_url)
            with pytest.raises(DiscordChannelError, match='READY omitted required'):
                await anext(channel.events())

    async def test_ignores_large_gateway_event_before_message(self) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(
                json.dumps(
                    {
                        'op': 0,
                        't': 'READY',
                        's': 1,
                        'd': {
                            'session_id': 'session-1',
                            'resume_gateway_url': 'ws://unused',
                            'user': {'id': 'bot-1'},
                        },
                    }
                )
            )
            await connection.send(json.dumps({'op': 0, 't': 'GUILD_CREATE', 's': 2, 'd': {'x': 'x' * 1_100_000}}))
            await connection.send(json.dumps(_message(content='after large event')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'after large event'

    async def test_buffer_overload_resumes_without_losing_events(self) -> None:
        total_messages = 110
        connections = 0
        overflowed = asyncio.Event()

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            connection_number = connections
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            authentication = _JSON_OBJECT.validate_json(await connection.recv())
            if connection_number == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 0,
                            'd': {
                                'session_id': 'session',
                                'resume_gateway_url': gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                first_sequence = 1
            else:
                resume = _JSON_OBJECT.validate_python(authentication['d'])
                first_sequence = int(str(resume['seq'])) + 1
            try:
                for sequence in range(first_sequence, total_messages + 1):
                    payload = _message(content=f'message {sequence}')
                    data = _JSON_OBJECT.validate_python(payload['d'])
                    data['id'] = f'message-{sequence}'
                    payload['d'] = data
                    payload['s'] = sequence
                    await connection.send(json.dumps(payload))
                await connection.wait_closed()
            except ConnectionClosed:  # pragma: no cover -- close timing varies by event loop
                pass
            finally:
                if connection_number == 1:
                    overflowed.set()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            received = [(await anext(events)).event_id]
            await asyncio.wait_for(overflowed.wait(), timeout=3)
            while len(received) < total_messages:
                received.append((await anext(events)).event_id)
            await events.aclose()

        assert received == [f'message-{sequence}' for sequence in range(1, total_messages + 1)]
        assert connections >= 2

    async def test_rejects_a_second_active_event_iterator(self) -> None:
        identified = asyncio.Event()

        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            identified.set()
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids=None, gateway_url=gateway_url)
            first_events = channel.events()
            first_consumer = asyncio.create_task(anext(first_events))
            await asyncio.wait_for(identified.wait(), timeout=1)
            with pytest.raises(DiscordChannelError, match='one active event iterator'):
                await anext(channel.events())
            first_consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_consumer
            await first_events.aclose()

    @pytest.mark.parametrize('reconnect_payload', [{'op': 7, 'd': None}, {'op': 9, 'd': True}])
    async def test_resumes_with_latest_sequence_and_preserves_event_identity(
        self, reconnect_payload: dict[str, object]
    ) -> None:
        connections = 0
        gateway_url = ''
        authentications: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
            if connections == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 8,
                            'd': {
                                'session_id': 'session-1',
                                'resume_gateway_url': gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                await connection.send(json.dumps(_message(content='first')))
                await connection.send(json.dumps(reconnect_payload))
                await connection.wait_closed()
            else:
                await connection.send(json.dumps(_message(content='replayed')))
                await connection.wait_closed()

        async with _gateway(handler) as running_gateway_url:
            gateway_url = running_gateway_url
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            first = await anext(events)
            second = await anext(events)
            await events.aclose()

        resume_data = _JSON_OBJECT.validate_python(authentications[1]['d'])
        assert first.event_id == second.event_id == 'message-1'
        assert authentications[1]['op'] == 6
        assert resume_data == {'token': 'token', 'session_id': 'session-1', 'seq': 2}

    async def test_invalid_session_starts_a_new_session(self) -> None:
        connections = 0
        authentications: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
            if connections == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 1,
                            'd': {
                                'session_id': 'stale',
                                'resume_gateway_url': gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                await connection.send(json.dumps({'op': 9, 'd': False}))
                await connection.wait_closed()
            else:
                await connection.send(json.dumps(_message(content='new session')))
                await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'new session'
        assert [authentication['op'] for authentication in authentications] == [2, 2]

    @pytest.mark.parametrize('close_code', [1000, 1001])
    async def test_clean_close_reidentifies(self, close_code: int, monkeypatch: pytest.MonkeyPatch) -> None:
        times = iter((0.0, 0.0, 10.0, 10.0))
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.current_time', lambda: next(times))
        connections = 0
        authentications: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
            if connections == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 1,
                            'd': {
                                'session_id': 'stale',
                                'resume_gateway_url': gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                await connection.close(code=close_code)
            else:
                await connection.send(json.dumps(_message(content='new session')))
                await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            await anext(events)
            await events.aclose()

        assert [authentication['op'] for authentication in authentications] == [2, 2]

    @pytest.mark.parametrize('close_code', [4007, 4009])
    async def test_stale_session_close_codes_reidentify(self, close_code: int, monkeypatch: pytest.MonkeyPatch) -> None:
        times = iter((0.0, 0.0, 10.0, 10.0))
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.current_time', lambda: next(times))
        connections = 0
        authentications: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
            if connections == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 1,
                            'd': {
                                'session_id': 'stale',
                                'resume_gateway_url': gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                await connection.close(code=close_code)
            else:
                await connection.send(json.dumps(_message(content='new session')))
                await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            await anext(events)
            await events.aclose()

        assert [authentication['op'] for authentication in authentications] == [2, 2]

    async def test_answers_server_heartbeat_with_latest_sequence(self) -> None:
        heartbeat: dict[str, object] = {}

        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(
                json.dumps(
                    {
                        'op': 0,
                        't': 'READY',
                        's': 7,
                        'd': {'session_id': 's', 'resume_gateway_url': 'ws://unused', 'user': {'id': 'bot-1'}},
                    }
                )
            )
            await connection.send(json.dumps({'op': 1, 'd': None}))
            heartbeat.update(_JSON_OBJECT.validate_json(await connection.recv()))
            await connection.send(json.dumps({'op': 11, 'd': None}))
            await connection.send(json.dumps(_message(content='after heartbeat')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            await anext(events)
            await events.aclose()

        assert heartbeat == {'op': 1, 'd': 7}

    async def test_server_requested_heartbeat_does_not_consume_periodic_ack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.random.random', lambda: 1)

        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 20}}))
            await connection.recv()
            await connection.send(json.dumps({'op': 1, 'd': None}))
            requested = _JSON_OBJECT.validate_json(await connection.recv())
            periodic = _JSON_OBJECT.validate_json(await connection.recv())
            await connection.send(json.dumps({'op': 11, 'd': None}))
            await connection.send(json.dumps(_message(content='connection stayed healthy')))
            await connection.wait_closed()
            assert requested['op'] == periodic['op'] == 1

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'connection stayed healthy'

    async def test_reads_heartbeat_ack_while_event_consumer_is_paused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.random.random', lambda: 0)
        heartbeat_count = 0
        received_multiple_heartbeats = asyncio.Event()

        async def handler(connection: ServerConnection) -> None:
            nonlocal heartbeat_count
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 20}}))
            await connection.recv()
            await connection.send(
                json.dumps(
                    {
                        'op': 0,
                        't': 'READY',
                        's': 1,
                        'd': {'session_id': 's', 'resume_gateway_url': gateway_url, 'user': {'id': 'bot-1'}},
                    }
                )
            )
            await connection.send(json.dumps(_message(content='pause consumer')))
            while heartbeat_count < 2:
                payload = _JSON_OBJECT.validate_json(await connection.recv())
                assert payload.get('op') == 1
                heartbeat_count += 1
                await connection.send(json.dumps({'op': 11, 'd': None}))
            received_multiple_heartbeats.set()
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            await anext(events)
            await asyncio.wait_for(received_multiple_heartbeats.wait(), timeout=1)
            await events.aclose()

        assert heartbeat_count >= 2

    async def test_first_heartbeat_uses_gateway_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.random.random', lambda: 0.25)
        requested_delays: list[float] = []
        original_sleep = anyio.sleep

        async def recording_sleep(delay: float) -> None:
            requested_delays.append(delay)
            await original_sleep(delay)

        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.sleep', recording_sleep)

        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 100}}))
            await connection.recv()
            heartbeat = _JSON_OBJECT.validate_json(await connection.recv())
            assert heartbeat['op'] == 1
            await connection.send(json.dumps({'op': 11, 'd': None}))
            await connection.send(json.dumps(_message(content='after jitter')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            await anext(events)
            await events.aclose()

        assert abs(requested_delays[0] - 0.025) < 1e-9

    @pytest.mark.parametrize('heartbeat_interval', ['false', '1e309', '1' + '0' * 1000])
    async def test_rejects_invalid_gateway_hello(self, heartbeat_interval: str) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(f'{{"op":10,"d":{{"heartbeat_interval":{heartbeat_interval}}}}}')
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids=None, gateway_url=gateway_url)
            with pytest.raises(DiscordChannelError, match='valid heartbeat interval'):
                await anext(channel.events())

    async def test_rejects_repeated_gateway_hello(self) -> None:
        async def handler(connection: ServerConnection) -> None:
            hello = json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}})
            await connection.send(hello)
            await connection.recv()
            await connection.send(hello)
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids=None, gateway_url=gateway_url)
            with pytest.raises(DiscordChannelError, match='more than one Hello'):
                await anext(channel.events())

    async def test_reconnects_when_gateway_omits_hello(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.channels.discord._HELLO_TIMEOUT', 0.01)
        connections = 0

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            if connections == 1:
                await connection.wait_closed()
                return
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(json.dumps(_message(content='after timeout')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'after timeout'
        assert connections == 2

    async def test_cancelling_waiting_consumer_closes_gateway_tasks(self) -> None:
        identified = asyncio.Event()
        connections = 0
        authentications: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
            if connections == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 1,
                            'd': {
                                'session_id': 'session',
                                'resume_gateway_url': gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                identified.set()
            else:
                await connection.send(json.dumps(_message(content='fresh connection')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids=None, gateway_url=gateway_url)
            events = channel.events()
            consumer = asyncio.create_task(anext(events))
            await asyncio.wait_for(identified.wait(), timeout=1)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            await events.aclose()
            next_events = channel.events()
            event = await anext(next_events)
            await next_events.aclose()

        assert consumer.done()
        assert event.text == 'fresh connection'
        assert [authentication['op'] for authentication in authentications] == [2, 2]

    async def test_anyio_cancellation_closes_gateway_and_allows_reuse(self) -> None:
        first_connected = asyncio.Event()
        first_disconnected = asyncio.Event()
        connections = 0

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            connection_number = connections
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            if connection_number == 1:
                first_connected.set()
                await connection.wait_closed()
                first_disconnected.set()
                return
            await connection.send(json.dumps(_message(content='after cancellation')))
            await connection.wait_closed()

        async def consume(channel: DiscordChannel) -> None:
            async for _event in channel.events():
                pass  # pragma: no cover

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            async with anyio.create_task_group() as group:
                group.start_soon(consume, channel)
                await first_connected.wait()
                group.cancel_scope.cancel()
            await asyncio.wait_for(first_disconnected.wait(), timeout=1)

            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert connections == 2
        assert event.text == 'after cancellation'

    async def test_context_exit_closes_a_paused_event_connection(self) -> None:
        disconnected = asyncio.Event()
        connections = 0

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            content = 'pause iterator' if connections == 1 else 'replacement iterator'
            await connection.send(json.dumps(_message(content=content)))
            await connection.wait_closed()
            if connections == 1:
                disconnected.set()

        async with _gateway(handler) as gateway_url:
            async with DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url) as channel:
                events = channel.events()
                event = await anext(events)
            await asyncio.wait_for(disconnected.wait(), timeout=1)
            replacement_events = channel.events()
            replacement_event = await anext(replacement_events)
            await replacement_events.aclose()
            await events.aclose()

        assert event.text == 'pause iterator'
        assert replacement_event.text == 'replacement iterator'

    async def test_aclose_blocks_replacement_until_gateway_cleanup_finishes(self) -> None:
        gateway_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        class SlowCleanupChannel(DiscordChannel):
            async def _gateway_loop(self, queue: asyncio.Queue[ChannelEvent]) -> None:
                gateway_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await asyncio.shield(release_cleanup.wait())

        channel = SlowCleanupChannel('token', allowed_user_ids=None)
        events = channel.events()
        pending_event = asyncio.create_task(anext(events))
        await gateway_started.wait()

        close_task = asyncio.create_task(channel.aclose())
        await cleanup_started.wait()
        concurrent_close_task = asyncio.create_task(channel.aclose())
        await asyncio.sleep(0)
        replacement_events = channel.events()
        with pytest.raises(DiscordChannelError, match='one active event iterator'):
            await anext(replacement_events)

        release_cleanup.set()
        await asyncio.gather(close_task, concurrent_close_task)
        await asyncio.gather(pending_event, return_exceptions=True)
        await events.aclose()

    async def test_missing_heartbeat_ack_reconnects_and_resumes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.random.random', lambda: 0)
        connections = 0
        authentications: list[dict[str, object]] = []

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 10}}))
            authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
            if connections == 1:
                await connection.send(
                    json.dumps(
                        {
                            'op': 0,
                            't': 'READY',
                            's': 4,
                            'd': {
                                'session_id': 'session',
                                'resume_gateway_url': gateway_url,
                                'user': {'id': 'bot-1'},
                            },
                        }
                    )
                )
                await connection.recv()
                await connection.wait_closed()
            else:
                await connection.send(json.dumps(_message(content='after reconnect')))
                await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'after reconnect'
        assert [authentication['op'] for authentication in authentications] == [2, 6]

    async def test_pre_ready_disconnects_back_off_exponentially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_sleep = anyio.sleep
        reconnect_sleeps: list[float] = []
        connections = 0

        async def controlled_sleep(delay: float) -> None:
            if 0 < delay < 60:
                reconnect_sleeps.append(delay)
                await original_sleep(0)
            elif delay >= 60:
                await asyncio.Event().wait()

        times = iter(float(value) for value in range(0, 1000, 10))
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.sleep', controlled_sleep)
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.current_time', lambda: next(times))
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.random.random', lambda: 1)

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            if connections < 3:
                await connection.close(code=4000)
                return
            await connection.send(json.dumps(_message(content='eventually ready')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'eventually ready'
        assert reconnect_sleeps[:2] == [1, 2]

    async def test_stops_after_consecutive_identifies_without_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original_sleep = anyio.sleep
        connections = 0
        authentications: list[dict[str, object]] = []

        async def controlled_sleep(delay: float) -> None:
            if delay < 60:
                await original_sleep(0)
            else:
                await asyncio.Event().wait()

        times = iter(float(value) for value in range(0, 1000, 10))
        monkeypatch.setattr('pydantic_ai_harness.channels.discord._MAX_CONSECUTIVE_IDENTIFY_ATTEMPTS', 2)
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.sleep', controlled_sleep)
        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.current_time', lambda: next(times))

        async def handler(connection: ServerConnection) -> None:
            nonlocal connections
            connections += 1
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            try:
                authentications.append(_JSON_OBJECT.validate_json(await connection.recv()))
                await connection.close(code=4000)
            except ConnectionClosed:
                pass

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('token', allowed_user_ids=None, gateway_url=gateway_url)
            with pytest.raises(DiscordChannelError, match='consecutive Identify'):
                await anext(channel.events())

        assert connections == 3
        assert [authentication['op'] for authentication in authentications] == [2, 2]

    async def test_surfaces_fatal_gateway_close_without_logging_token(self, caplog: pytest.LogCaptureFixture) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.close(code=4014, reason='privileged intent denied')

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel('do-not-leak', allowed_user_ids=None, gateway_url=gateway_url)
            with caplog.at_level(logging.DEBUG, logger='websockets.client'):
                with pytest.raises(DiscordChannelError, match='4014') as exc_info:  # pragma: no branch
                    await anext(channel.events())

        assert 'do-not-leak' not in str(exc_info.value)
        assert 'do-not-leak' not in caplog.text

    async def test_ignores_unadmitted_and_looping_messages(self) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(
                json.dumps(
                    {
                        'op': 0,
                        't': 'READY',
                        's': 1,
                        'd': {'session_id': 's', 'resume_gateway_url': 'ws://unused', 'user': {'id': 'bot-1'}},
                    }
                )
            )
            await connection.send(b'{}')
            await connection.send('{not json')
            await connection.send(json.dumps({'op': 'unknown'}))
            await connection.send(json.dumps({'op': 0, 't': 'MESSAGE_CREATE', 's': None, 'd': {}}))
            await connection.send(json.dumps({'op': 0, 't': 'MESSAGE_CREATE', 's': None, 'd': {'type': 0}}))
            bot_message = _message()
            bot_data = _JSON_OBJECT.validate_python(bot_message['d'])
            bot_data['author'] = {'id': 'other-bot', 'bot': True}
            bot_message['d'] = bot_data
            await connection.send(json.dumps(bot_message))
            malformed_bot_message = _message()
            malformed_bot_data = _JSON_OBJECT.validate_python(malformed_bot_message['d'])
            malformed_bot_data['author'] = {'id': 'user-1', 'bot': 1}
            malformed_bot_message['d'] = malformed_bot_data
            await connection.send(json.dumps(malformed_bot_message))
            self_message = _message()
            self_data = _JSON_OBJECT.validate_python(self_message['d'])
            self_data['author'] = {'id': 'bot-1', 'bot': False}
            self_message['d'] = self_data
            await connection.send(json.dumps(self_message))
            webhook_message = _message()
            webhook_data = _JSON_OBJECT.validate_python(webhook_message['d'])
            webhook_data['webhook_id'] = 'webhook-1'
            webhook_message['d'] = webhook_data
            await connection.send(json.dumps(webhook_message))
            null_webhook_message = _message()
            null_webhook_data = _JSON_OBJECT.validate_python(null_webhook_message['d'])
            null_webhook_data['webhook_id'] = None
            null_webhook_message['d'] = null_webhook_data
            await connection.send(json.dumps(null_webhook_message))
            disallowed_user = _message()
            disallowed_user_data = _JSON_OBJECT.validate_python(disallowed_user['d'])
            disallowed_user_data['author'] = {'id': 'user-2', 'bot': False}
            disallowed_user['d'] = disallowed_user_data
            await connection.send(json.dumps(disallowed_user))
            await connection.send(json.dumps(_message(content='<@bot-1> wrong guild', guild_id='guild-2')))
            malformed_guild = _message(content='invalid guild identity')
            malformed_guild_data = _JSON_OBJECT.validate_python(malformed_guild['d'])
            malformed_guild_data['guild_id'] = 1
            malformed_guild['d'] = malformed_guild_data
            await connection.send(json.dumps(malformed_guild))
            await connection.send(json.dumps(_message(content='system message', message_type=21)))
            malformed_type = _message(content='malformed message type')
            malformed_type_data = _JSON_OBJECT.validate_python(malformed_type['d'])
            malformed_type_data['type'] = '0'
            malformed_type['d'] = malformed_type_data
            await connection.send(json.dumps(malformed_type))
            unmentioned = _message(content='no mention', guild_id='guild-1')
            unmentioned_data = _JSON_OBJECT.validate_python(unmentioned['d'])
            unmentioned_data['mentions'] = []
            unmentioned['d'] = unmentioned_data
            await connection.send(json.dumps(unmentioned))
            malformed_mentions = _message(content='bad mentions', guild_id='guild-1')
            malformed_mentions_data = _JSON_OBJECT.validate_python(malformed_mentions['d'])
            malformed_mentions_data['mentions'] = 'not a list'
            malformed_mentions['d'] = malformed_mentions_data
            await connection.send(json.dumps(malformed_mentions))
            await connection.send(json.dumps(_message(content='<@bot-1>', guild_id='guild-1')))
            await connection.send(json.dumps(_message(content='accepted DM', message_type=19)))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel(
                'token', allowed_user_ids={'user-1'}, allowed_guild_ids={'guild-1'}, gateway_url=gateway_url
            )
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'accepted DM'

    async def test_can_explicitly_accept_unmentioned_guild_messages(self) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(
                json.dumps(
                    {
                        'op': 0,
                        't': 'READY',
                        's': 1,
                        'd': {'session_id': 's', 'resume_gateway_url': 'ws://unused', 'user': {'id': 'bot-1'}},
                    }
                )
            )
            message = _message(content='no mention required', guild_id='guild-1')
            data = _JSON_OBJECT.validate_python(message['d'])
            data['mentions'] = []
            message['d'] = data
            await connection.send(json.dumps(message))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            channel = DiscordChannel(
                'token',
                allowed_user_ids={'user-1'},
                allowed_guild_ids={'guild-1'},
                require_mention=False,
                gateway_url=gateway_url,
            )
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == 'no mention required'

    @pytest.mark.parametrize(('admit_all_guilds', 'expected'), [(False, 'accepted DM'), (True, 'guild message')])
    async def test_guild_allowlist_default_and_explicit_wildcard(self, admit_all_guilds: bool, expected: str) -> None:
        async def handler(connection: ServerConnection) -> None:
            await connection.send(json.dumps({'op': 10, 'd': {'heartbeat_interval': 60_000}}))
            await connection.recv()
            await connection.send(
                json.dumps(
                    {
                        'op': 0,
                        't': 'READY',
                        's': 1,
                        'd': {'session_id': 's', 'resume_gateway_url': 'ws://unused', 'user': {'id': 'bot-1'}},
                    }
                )
            )
            await connection.send(json.dumps(_message(content='<@bot-1> guild message', guild_id='guild-1')))
            await connection.send(json.dumps(_message(content='accepted DM')))
            await connection.wait_closed()

        async with _gateway(handler) as gateway_url:
            if admit_all_guilds:
                channel = DiscordChannel(
                    'token', allowed_user_ids={'user-1'}, allowed_guild_ids=None, gateway_url=gateway_url
                )
            else:
                channel = DiscordChannel('token', allowed_user_ids={'user-1'}, gateway_url=gateway_url)
            events = channel.events()
            event = await anext(events)
            await events.aclose()

        assert event.text == expected

    async def test_public_host_path_posts_reply_with_safe_retry(self) -> None:
        requests: list[httpx.Request] = []
        bodies: list[dict[str, object]] = []

        async def send(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            bodies.append(_JSON_OBJECT.validate_json(request.content))
            if len(requests) == 1:
                return httpx.Response(429, headers={'Retry-After': '0'}, request=request)
            return httpx.Response(200, json={'id': 'reply'}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel(
            'secret-token',
            allowed_user_ids={'user-1'},
            api_base_url='https://discord.test/api/v10',
            http_client=client,
        )
        host = ChannelHost(Agent('test'), channel)
        event = ChannelEvent(
            event_id='message-1',
            conversation_id='channel-1',
            sender_id='user-1',
            text='question',
            reply_to_id='message-1',
        )

        result = await host.handle(event)
        await client.aclose()

        assert result.output == 'success (no tool calls)'
        assert len(requests) == 2
        assert requests[0].url == 'https://discord.test/api/v10/channels/channel-1/messages'
        assert requests[0].headers['authorization'] == 'Bot secret-token'
        assert requests[0].headers['user-agent'].startswith('DiscordBot (')
        assert bodies[0] == bodies[1]
        assert bodies[0]['allowed_mentions'] == {'parse': [], 'replied_user': False}
        assert bodies[0]['message_reference'] == {'message_id': 'message-1', 'fail_if_not_exists': False}
        assert bodies[0]['enforce_nonce'] is True
        assert len(str(bodies[0]['nonce'])) <= 25

    async def test_adapter_owned_http_client_is_created_and_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def send(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))

        def client_factory() -> httpx.AsyncClient:
            return client

        monkeypatch.setattr('pydantic_ai_harness.channels.discord.httpx.AsyncClient', client_factory)
        event = ChannelEvent(event_id='event', conversation_id='channel', sender_id='user', text='in')

        channel = DiscordChannel('token', allowed_user_ids=None)
        await channel.reply(event, 'reply')
        with anyio.CancelScope() as cancel_scope:
            cancel_scope.cancel()
            await channel.aclose()

        assert client.is_closed

    async def test_splits_long_replies_without_retrying_http_errors(self) -> None:
        contents: list[str] = []

        def send(request: httpx.Request) -> httpx.Response:
            body = _JSON_OBJECT.validate_json(request.content)
            contents.append(str(body['content']))
            status = 404 if len(contents) == 2 else 200
            return httpx.Response(status, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel('token', allowed_user_ids=None, http_client=client)
        event = ChannelEvent(event_id='event', conversation_id='channel', sender_id='user', text='in')

        with pytest.raises(DiscordChannelError, match='HTTP 404'):
            await channel.reply(event, 'a' * 2001 + 'b' * 2000)
        await client.aclose()

        assert contents == ['a' * 2000, 'a' + 'b' * 1999]

    async def test_sends_all_long_reply_chunks_in_order_with_distinct_nonces(self) -> None:
        bodies: list[dict[str, object]] = []

        def send(request: httpx.Request) -> httpx.Response:
            bodies.append(_JSON_OBJECT.validate_json(request.content))
            return httpx.Response(200, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel('token', allowed_user_ids=None, http_client=client)
        event = ChannelEvent(event_id='event', conversation_id='channel', sender_id='user', text='in')

        await channel.reply(event, 'a' * 2000 + 'b' * 2000 + 'c')
        await client.aclose()

        assert [body['content'] for body in bodies] == ['a' * 2000, 'b' * 2000, 'c']
        nonces = [str(body['nonce']) for body in bodies]
        assert len(set(nonces)) == 3
        assert all(len(nonce) <= 25 for nonce in nonces)

    async def test_manual_retry_reuses_nonce_without_retrying_unknown_outcome(self) -> None:
        bodies: list[dict[str, object]] = []

        def send(request: httpx.Request) -> httpx.Response:
            bodies.append(_JSON_OBJECT.validate_json(request.content))
            if len(bodies) == 1:
                raise httpx.ReadTimeout('unknown send outcome', request=request)
            return httpx.Response(200, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel('token', allowed_user_ids=None, http_client=client)
        event = ChannelEvent(event_id='event', conversation_id='channel', sender_id='user', text='in')

        with pytest.raises(httpx.ReadTimeout, match='unknown send outcome'):
            await channel.reply(event, 'reply')
        assert len(bodies) == 1

        await channel.reply(event, 'reply')
        await channel.reply(event, 'different reply')
        await client.aclose()

        assert bodies[0]['nonce'] == bodies[1]['nonce']
        assert bodies[1]['nonce'] != bodies[2]['nonce']

    async def test_rate_limit_wait_blocks_concurrent_replies(self) -> None:
        contents: list[str] = []
        first_request_started = asyncio.Event()

        async def send(request: httpx.Request) -> httpx.Response:
            body = _JSON_OBJECT.validate_json(request.content)
            contents.append(str(body['content']))
            if len(contents) == 1:
                first_request_started.set()
                return httpx.Response(429, json={'retry_after': 0, 'global': True}, request=request)
            return httpx.Response(200, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel('token', allowed_user_ids=None, http_client=client)
        first = ChannelEvent(event_id='first', conversation_id='one', sender_id='user', text='in')
        second = ChannelEvent(event_id='second', conversation_id='two', sender_id='user', text='in')

        async def send_second() -> None:
            await first_request_started.wait()
            await channel.reply(second, 'second reply')

        async with anyio.create_task_group() as group:
            group.start_soon(channel.reply, first, 'first reply')
            group.start_soon(send_second)
        await client.aclose()

        assert contents == ['first reply', 'first reply', 'second reply']

    async def test_exhausted_rate_limit_does_not_sleep_without_another_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []

        async def record_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.sleep', record_sleep)
        requests = 0

        def send(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            return httpx.Response(429, headers={'Retry-After': '2'}, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel('token', allowed_user_ids=None, http_client=client)
        event = ChannelEvent(event_id='event', conversation_id='channel', sender_id='user', text='in')

        with pytest.raises(DiscordChannelError, match='remained rate limited'):
            await channel.reply(event, 'reply')
        await client.aclose()

        assert requests == 3
        assert sleeps == [2, 2]

    async def test_cancelled_rate_limit_wait_blocks_the_next_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        first_sleep_started = asyncio.Event()
        never_release = asyncio.Event()

        async def controlled_sleep(delay: float) -> None:
            sleeps.append(delay)
            if len(sleeps) == 1:
                first_sleep_started.set()
                await never_release.wait()

        monkeypatch.setattr('pydantic_ai_harness.channels.discord.anyio.sleep', controlled_sleep)
        contents: list[str] = []

        def send(request: httpx.Request) -> httpx.Response:
            body = _JSON_OBJECT.validate_json(request.content)
            contents.append(str(body['content']))
            if len(contents) == 1:
                return httpx.Response(429, headers={'Retry-After': '2'}, request=request)
            return httpx.Response(200, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel('token', allowed_user_ids=None, http_client=client)
        first = ChannelEvent(event_id='first', conversation_id='one', sender_id='user', text='in')
        second = ChannelEvent(event_id='second', conversation_id='two', sender_id='user', text='in')

        first_reply = asyncio.create_task(channel.reply(first, 'first reply'))
        await first_sleep_started.wait()
        first_reply.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_reply
        await channel.reply(second, 'second reply')
        await client.aclose()

        assert contents == ['first reply', 'second reply']
        assert len(sleeps) == 2
        assert sleeps[1] > 0

    @pytest.mark.parametrize(
        ('headers', 'content'),
        [
            ({'Retry-After': 'nan'}, b''),
            ({'Retry-After': '-1'}, b''),
            ({'Retry-After': '61'}, b''),
            ({'Retry-After': 'not-a-number'}, b''),
            ({}, b'{not json'),
            ({}, b'{"retry_after": false}'),
        ],
    )
    async def test_rejects_invalid_rate_limit_delay_and_redacts_token(
        self, headers: dict[str, str], content: bytes
    ) -> None:
        def send(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers=headers, content=content, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        channel = DiscordChannel('do-not-leak', allowed_user_ids=None, http_client=client)
        event = ChannelEvent(event_id='event', conversation_id='channel', sender_id='user', text='in')

        with pytest.raises(DiscordChannelError, match='retry delay') as exc_info:
            await channel.reply(event, 'reply')
        await client.aclose()

        assert 'do-not-leak' not in repr(channel)
        assert 'do-not-leak' not in str(exc_info.value)

    async def test_requires_explicit_nonempty_credentials_and_reply(self) -> None:
        with pytest.raises(ValueError, match='must not be empty'):
            DiscordChannel('', allowed_user_ids=None)

        channel = DiscordChannel('token', allowed_user_ids=None)
        event = ChannelEvent(event_id='event', conversation_id='channel', sender_id='user', text='in')
        with pytest.raises(DiscordChannelError, match='must not be empty'):
            await channel.reply(event, '')
        with pytest.raises(DiscordChannelError, match='must not exceed 20,000'):
            await channel.reply(event, 'x' * 20_001)
        await channel.aclose()
