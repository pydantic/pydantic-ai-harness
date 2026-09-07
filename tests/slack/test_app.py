from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence

import anyio
import httpx
import pytest
from aiohttp import FormData
from pydantic import TypeAdapter
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from slack_bolt.context.async_context import AsyncBoltContext
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.errors import SlackApiError
from slack_sdk.oauth.installation_store.async_installation_store import AsyncInstallationStore
from slack_sdk.oauth.installation_store.models.installation import Installation
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse
from typing_extensions import TypeVar

from pydantic_ai_harness.slack import InMemorySlackHistory, SlackApp, SlackContext

pytestmark = pytest.mark.anyio

_SIGNING_SECRET = 'test-signing-secret'


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(autouse=True)
def slack_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-test')


class _FakeSlack:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.error_methods: set[str] = set()
        self.stream_index = 0

    def response(self, client: AsyncWebClient, method: str, data: dict[str, object]) -> AsyncSlackResponse:
        return AsyncSlackResponse(
            client=client,
            http_verb='POST',
            api_url=f'https://slack.test/api/{method}',
            req_args={},
            data=data,
            headers={'x-oauth-scopes': 'chat:write'},
            status_code=200,
        )


def _fake_client(monkeypatch: pytest.MonkeyPatch, fake: _FakeSlack | None = None) -> tuple[AsyncWebClient, _FakeSlack]:
    recorder = fake if fake is not None else _FakeSlack()

    class FakeAsyncWebClient(AsyncWebClient):
        async def api_call(
            self,
            api_method: str,
            *,
            http_verb: str = 'POST',
            files: dict[object, object] | None = None,
            data: dict[object, object] | FormData | None = None,
            params: dict[object, object] | None = None,
            json: dict[object, object] | None = None,
            headers: dict[object, object] | None = None,
            auth: dict[object, object] | None = None,
        ) -> AsyncSlackResponse:
            del http_verb, files, headers, auth
            raw_payload = json or params or (data if isinstance(data, dict) else None) or {}
            payload = {str(key): value for key, value in raw_payload.items()}
            recorder.calls.append((api_method, payload))
            if api_method in recorder.error_methods:
                response = recorder.response(self, api_method, {'ok': False, 'error': 'fake_error'})
                raise SlackApiError('Fake Slack API error', response)
            if api_method == 'auth.test':
                token = payload.get('token')
                if token == 'xoxp-user':
                    response_data: dict[str, object] = {
                        'ok': True,
                        'user_id': 'U1',
                        'team_id': 'T1',
                        'enterprise_id': 'E1',
                    }
                else:
                    response_data = {
                        'ok': True,
                        'user_id': 'UBOT',
                        'team_id': 'T1',
                        'enterprise_id': 'E1',
                        'bot_id': 'B1',
                    }
            elif api_method == 'chat.startStream':
                recorder.stream_index += 1
                response_data = {'ok': True, 'ts': f'stream-{recorder.stream_index}'}
            else:
                response_data = {'ok': True}
            return recorder.response(self, api_method, response_data)

    monkeypatch.setattr('slack_bolt.app.async_app.AsyncWebClient', FakeAsyncWebClient)
    return FakeAsyncWebClient(token='xoxb-test'), recorder


def _event(
    text: str,
    *,
    event_type: str = 'message',
    channel: str = 'D1',
    channel_type: str = 'im',
    ts: str = '1.1',
    thread_ts: str | None = None,
    user: str = 'U1',
    **extra: object,
) -> dict[str, object]:
    event: dict[str, object] = {
        'type': event_type,
        'user': user,
        'channel': channel,
        'channel_type': channel_type,
        'text': text,
        'ts': ts,
        **extra,
    }
    if thread_ts is not None:
        event['thread_ts'] = thread_ts
    return event


_PostDepsT = TypeVar('_PostDepsT')


async def _post(
    app: SlackApp[_PostDepsT],
    event: dict[str, object],
    *,
    event_id: str,
    valid_signature: bool = True,
    retry_num: str | None = None,
) -> httpx.Response:
    payload = {
        'type': 'event_callback',
        'team_id': 'T1',
        'enterprise_id': 'E1',
        'event_id': event_id,
        'event_time': int(time.time()),
        'event': event,
    }
    body = json.dumps(payload, separators=(',', ':'))
    timestamp = str(int(time.time()))
    digest = hmac.new(_SIGNING_SECRET.encode(), f'v0:{timestamp}:{body}'.encode(), hashlib.sha256).hexdigest()
    headers = {
        'content-type': 'application/json',
        'x-slack-request-timestamp': timestamp,
        'x-slack-signature': f'v0={digest}' if valid_signature else 'v0=garbage',
    }
    if retry_num is not None:
        headers['x-slack-retry-num'] = retry_num
    transport = httpx.ASGITransport(app=app)  # pyright: ignore[reportArgumentType]
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        return await client.post('/slack/events', content=body, headers=headers)


async def _wait_for_calls(fake: _FakeSlack, method: str, count: int) -> None:
    with anyio.fail_after(5):
        while sum(call_method == method for call_method, _ in fake.calls) < count:
            await anyio.sleep(0)


def _text_agent(output: str, prompts: list[str] | None = None) -> Agent[None, str]:
    async def stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        if prompts is not None:
            prompts.append(_latest_prompt(messages))
        yield output

    return Agent(FunctionModel(stream_function=stream))


def _latest_prompt(messages: Sequence[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in reversed(message.parts):
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    raise AssertionError('No string user prompt found')


def _stream_text(fake: _FakeSlack) -> str:
    text = ''
    for method, payload in fake.calls:
        if method not in {'chat.startStream', 'chat.appendStream', 'chat.stopStream'}:
            continue
        chunks = payload.get('chunks')
        if isinstance(chunks, list):
            for chunk in TypeAdapter(list[dict[str, object]]).validate_python(chunks):
                chunk_text = chunk.get('text')
                if isinstance(chunk_text, str):
                    text += chunk_text
    return text


class _RecordingHistory:
    def __init__(self) -> None:
        self.values: dict[str, list[ModelMessage]] = {}
        self.loads: list[str] = []
        self.saves: list[tuple[str, list[ModelMessage]]] = []

    async def load(self, thread_key: str) -> Sequence[ModelMessage]:
        self.loads.append(thread_key)
        return list(self.values.get(thread_key, ()))

    async def save(self, thread_key: str, messages: Sequence[ModelMessage]) -> None:
        saved = list(messages)
        self.values[thread_key] = saved
        self.saves.append((thread_key, saved))

    async def delete(self, thread_key: str) -> None:
        self.values.pop(thread_key, None)


class _InstallationStore(AsyncInstallationStore):
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    async def async_find_installation(
        self,
        *,
        enterprise_id: str | None,
        team_id: str | None,
        user_id: str | None = None,
        is_enterprise_install: bool | None = False,
    ) -> Installation | None:
        del enterprise_id, team_id, user_id, is_enterprise_install
        return Installation(
            enterprise_id='E1',
            team_id='T1',
            bot_token='xoxb-test',
            bot_id='B1',
            bot_user_id='UBOT',
            user_id='U1',
            user_token='xoxp-user',
        )


class TestSlackApp:
    async def test_rejects_invalid_signature_before_running_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _fake_client(monkeypatch)
        prompts: list[str] = []
        app = SlackApp(
            _text_agent('unused', prompts), bot_token='xoxb-test', signing_secret=_SIGNING_SECRET, client=client
        )

        response = await _post(app, _event('hello'), event_id='Ev-invalid', valid_signature=False)

        assert response.status_code == 401
        assert prompts == []
        assert fake.calls == []

    async def test_dm_replies_in_thread_when_status_scope_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _fake_client(monkeypatch)
        fake.error_methods.add('assistant.threads.setStatus')
        app = SlackApp(_text_agent('hello from agent'), signing_secret=_SIGNING_SECRET, client=client)

        response = await _post(app, _event('hello', ts='1.2'), event_id='Ev-dm')
        await _wait_for_calls(fake, 'chat.stopStream', 1)

        assert response.status_code == 200
        assert _stream_text(fake) == 'hello from agent'
        start = next(payload for method, payload in fake.calls if method == 'chat.startStream')
        assert start['thread_ts'] == '1.2'
        assert sum(method == 'assistant.threads.setStatus' for method, _ in fake.calls) == 1

    async def test_channel_thread_routing_and_custom_history_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _fake_client(monkeypatch)
        prompts: list[str] = []
        history = _RecordingHistory()
        app = SlackApp(
            _text_agent('channel reply', prompts),
            signing_secret=_SIGNING_SECRET,
            client=client,
            history=history,
        )

        ignored = _event('not engaged', channel='C1', channel_type='channel', ts='9.2', thread_ts='9.1')
        assert (await _post(app, ignored, event_id='Ev-unengaged')).status_code == 200
        mention = _event(
            '<@UBOT> first question',
            event_type='app_mention',
            channel='C1',
            channel_type='channel',
            ts='2.1',
        )
        await _post(app, mention, event_id='Ev-mention')
        await _wait_for_calls(fake, 'chat.stopStream', 1)
        reply = _event('follow up', channel='C1', channel_type='channel', ts='2.2', thread_ts='2.1')
        await _post(app, reply, event_id='Ev-thread')
        await _wait_for_calls(fake, 'chat.stopStream', 2)

        assert prompts == ['first question', 'follow up']
        assert [key for key, _ in history.saves] == ['T1:C1:2.1', 'T1:C1:2.1']
        assert history.loads == ['T1:C1:9.1', 'T1:C1:2.1', 'T1:C1:2.1']
        assert len(history.saves[1][1]) > len(history.saves[0][1])

    async def test_bot_subtype_missing_user_and_missing_timestamp_events_are_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, fake = _fake_client(monkeypatch)
        prompts: list[str] = []
        app = SlackApp(_text_agent('ok', prompts), signing_secret=_SIGNING_SECRET, client=client)

        await _post(app, _event('bot', bot_id='B2'), event_id='Ev-bot')
        await _post(app, _event('edited', subtype='message_changed'), event_id='Ev-subtype')
        no_user = _event('anonymous')
        del no_user['user']
        await _post(app, no_user, event_id='Ev-no-user')
        no_ts = _event('untimed')
        del no_ts['ts']
        await _post(app, no_ts, event_id='Ev-no-ts')
        await _post(app, _event(''), event_id='Ev-empty')
        await _post(app, _event('valid', files='invalid'), event_id='Ev-valid')
        await _wait_for_calls(fake, 'chat.stopStream', 1)

        assert prompts == ['valid']

    async def test_retry_event_id_runs_once_and_oldest_event_is_evicted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.slack._app._MAX_RECENT_EVENTS', 2)
        client, fake = _fake_client(monkeypatch)
        prompts: list[str] = []
        app = SlackApp(_text_agent('ok', prompts), signing_secret=_SIGNING_SECRET, client=client)

        await _post(app, _event('one', ts='1.1'), event_id='Ev-1')
        await _wait_for_calls(fake, 'chat.stopStream', 1)
        await _post(app, _event('duplicate', ts='1.2'), event_id='Ev-1', retry_num='1')
        await _post(app, _event('duplicate again', ts='1.3'), event_id='Ev-1')
        await _post(app, _event('two', ts='2.1'), event_id='Ev-2')
        await _post(app, _event('three', ts='3.1'), event_id='Ev-3')
        await _wait_for_calls(fake, 'chat.stopStream', 3)
        await _post(app, _event('one again', ts='4.1'), event_id='Ev-1', retry_num='2')
        await _wait_for_calls(fake, 'chat.stopStream', 4)

        assert prompts == ['one', 'two', 'three', 'one again']

    async def test_threads_are_ordered_while_different_threads_run_concurrently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, fake = _fake_client(monkeypatch)
        first_started = anyio.Event()
        other_started = anyio.Event()
        second_same_started = anyio.Event()
        release_first = anyio.Event()
        starts: list[str] = []

        async def stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            prompt = _latest_prompt(messages)
            starts.append(prompt)
            if prompt == 'first':
                first_started.set()
                await release_first.wait()
            elif prompt == 'other':
                other_started.set()
            else:
                second_same_started.set()
            yield prompt

        app = SlackApp(Agent(FunctionModel(stream_function=stream)), signing_secret=_SIGNING_SECRET, client=client)

        await _post(app, _event('first', channel='D1', ts='1.1'), event_id='Ev-order-1')
        with anyio.fail_after(5):
            await first_started.wait()
        await _post(app, _event('second', channel='D1', ts='1.2', thread_ts='1.1'), event_id='Ev-order-2')
        await _post(app, _event('other', channel='D2', ts='2.1'), event_id='Ev-order-3')
        with anyio.fail_after(5):
            await other_started.wait()
        assert not second_same_started.is_set()
        release_first.set()
        with anyio.fail_after(5):
            await second_same_started.wait()
        await _wait_for_calls(fake, 'chat.stopStream', 3)

        assert starts.index('other') < starts.index('second')

    async def test_deps_factory_receives_context_files_and_run_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _fake_client(monkeypatch)
        contexts: list[SlackContext] = []
        metadata: list[dict[str, object] | None] = []

        async def deps_factory(context: SlackContext) -> SlackContext:
            contexts.append(context)
            return context

        async def instructions(ctx: RunContext[SlackContext]) -> str:
            metadata.append(ctx.metadata)
            return ''

        async def stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            assert 'report.txt' in _latest_prompt(messages)
            yield 'received'

        agent = Agent(FunctionModel(stream_function=stream), deps_type=SlackContext, instructions=instructions)
        app = SlackApp(agent, signing_secret=_SIGNING_SECRET, client=client, deps_factory=deps_factory)
        event = _event(
            '',
            event_type='app_mention',
            channel='C1',
            channel_type='channel',
            ts='4.1',
            files=[
                {'id': 'F1', 'name': 'report.txt', 'mimetype': 'text/plain'},
                {'name': 'missing-id.txt'},
            ],
        )

        await _post(app, event, event_id='Ev-deps')
        await _wait_for_calls(fake, 'chat.stopStream', 1)

        assert contexts == [
            SlackContext(
                team_id='T1',
                channel_id='C1',
                thread_ts='4.1',
                message_ts='4.1',
                user_id='U1',
                bot_token='xoxb-test',
                user_token=None,
            )
        ]
        assert 'xoxb-test' not in repr(contexts[0])
        assert metadata == [
            {
                'team_id': 'T1',
                'channel_id': 'C1',
                'thread_ts': '4.1',
                'message_ts': '4.1',
                'user_id': 'U1',
            }
        ]

    async def test_context_carries_user_token_under_oauth_and_bot_token_otherwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contexts: list[SlackContext] = []

        def deps_factory(context: SlackContext) -> None:
            contexts.append(context)

        oauth_client, oauth_fake = _fake_client(monkeypatch)
        oauth_settings = AsyncOAuthSettings(
            client_id='client-id',
            client_secret='client-secret',
            scopes=['chat:write'],
            user_scopes=['channels:history'],
            installation_store=_InstallationStore(),
            state_validation_enabled=False,
        )
        oauth_app = SlackApp(
            _text_agent('oauth'),
            signing_secret=_SIGNING_SECRET,
            oauth_settings=oauth_settings,
            client=oauth_client,
            deps_factory=deps_factory,
        )
        await _post(oauth_app, _event('oauth'), event_id='Ev-oauth')
        await _wait_for_calls(oauth_fake, 'chat.stopStream', 1)

        bot_client, bot_fake = _fake_client(monkeypatch)
        bot_app = SlackApp(
            _text_agent('bot'), signing_secret=_SIGNING_SECRET, client=bot_client, deps_factory=deps_factory
        )
        await _post(bot_app, _event('bot'), event_id='Ev-bot-token')
        await _wait_for_calls(bot_fake, 'chat.stopStream', 1)

        assert [(context.user_token, context.bot_token) for context in contexts] == [
            ('xoxp-user', contexts[0].bot_token),
            (None, 'xoxb-test'),
        ]
        assert contexts[0].bot_token is not None

    async def test_agent_failure_posts_error_once_and_does_not_save_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, fake = _fake_client(monkeypatch)
        history = _RecordingHistory()

        async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            yield 'partial'
            raise RuntimeError('model failed')

        app = SlackApp(
            Agent(FunctionModel(stream_function=fail)),
            signing_secret=_SIGNING_SECRET,
            client=client,
            history=history,
            error_reply='try later',
        )

        await _post(app, _event('fail'), event_id='Ev-fail')
        await _wait_for_calls(fake, 'chat.postMessage', 1)

        error_posts = [payload for method, payload in fake.calls if method == 'chat.postMessage']
        assert [post['text'] for post in error_posts] == ['try later']
        assert error_posts[0]['thread_ts'] == '1.1'
        assert history.saves == []

    async def test_run_cancellation_propagates_without_error_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _fake_client(monkeypatch)
        history = _RecordingHistory()
        propagated: list[Exception] = []
        handled = anyio.Event()

        async def cancel(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            yield 'partial'
            raise RunCancelled('cancelled by test')

        app = SlackApp(
            Agent(FunctionModel(stream_function=cancel)),
            signing_secret=_SIGNING_SECRET,
            client=client,
            history=history,
        )

        async def capture_error(error: Exception) -> None:
            propagated.append(error)
            handled.set()

        app.bolt.error(capture_error)
        await _post(app, _event('cancel'), event_id='Ev-cancel')
        with anyio.fail_after(5):
            await handled.wait()

        assert len(propagated) == 1
        assert isinstance(propagated[0], RunCancelled)
        assert all(method != 'chat.postMessage' for method, _ in fake.calls)
        assert history.saves == []

    def test_missing_credentials_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        monkeypatch.delenv('SLACK_SIGNING_SECRET', raising=False)
        with pytest.raises(ValueError, match='SLACK_SIGNING_SECRET'):
            SlackApp(Agent(TestModel()), bot_token='xoxb-test')
        with pytest.raises(ValueError, match='SLACK_BOT_TOKEN'):
            SlackApp(Agent(TestModel()), signing_secret=_SIGNING_SECRET)

    def test_oauth_allows_missing_bot_token_and_non_text_output_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        client, fake = _fake_client(monkeypatch)
        oauth_settings = AsyncOAuthSettings(
            client_id='client-id',
            client_secret='client-secret',
            installation_store=_InstallationStore(),
            state_validation_enabled=False,
        )
        SlackApp(
            Agent(TestModel()),
            signing_secret=_SIGNING_SECRET,
            oauth_settings=oauth_settings,
            client=client,
        )
        assert fake.calls == []
        with pytest.raises(TypeError, match='text replies'):
            SlackApp(
                Agent(TestModel(), output_type=int),  # pyright: ignore[reportArgumentType]
                bot_token='xoxb-test',
                signing_secret=_SIGNING_SECRET,
                client=client,
            )

    async def test_falls_back_to_say_when_streaming_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(AsyncBoltContext, 'say_stream', property(lambda _self: None))
        monkeypatch.setattr(AsyncBoltContext, 'set_status', property(lambda _self: None))
        client, fake = _fake_client(monkeypatch)
        app = SlackApp(_text_agent('fallback text'), signing_secret=_SIGNING_SECRET, client=client)

        await _post(app, _event('fallback'), event_id='Ev-fallback')
        await _wait_for_calls(fake, 'chat.postMessage', 1)

        post = next(payload for method, payload in fake.calls if method == 'chat.postMessage')
        assert post['text'] == 'fallback text'
        assert post['thread_ts'] == '1.1'

    async def test_streaming_uses_start_append_and_stop_without_losing_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, fake = _fake_client(monkeypatch)
        expected = 'a' * 250 + 'b' * 250 + 'c' * 50

        async def stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            yield 'a' * 250
            yield 'b' * 250
            yield 'c' * 50

        app = SlackApp(Agent(FunctionModel(stream_function=stream)), signing_secret=_SIGNING_SECRET, client=client)

        await _post(app, _event('stream'), event_id='Ev-stream')
        await _wait_for_calls(fake, 'chat.stopStream', 1)

        methods = [method for method, _ in fake.calls if method.startswith('chat.')]
        assert methods[0] == 'chat.startStream'
        assert methods[-1] == 'chat.stopStream'
        assert set(methods) <= {'chat.startStream', 'chat.appendStream', 'chat.stopStream'}
        assert _stream_text(fake) == expected

    async def test_streaming_flushes_partial_text_after_half_a_second(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _fake_client(monkeypatch)
        release = anyio.Event()

        async def stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            yield 'initial'
            await release.wait()
            yield ' final'

        app = SlackApp(Agent(FunctionModel(stream_function=stream)), signing_secret=_SIGNING_SECRET, client=client)

        await _post(app, _event('stream slowly'), event_id='Ev-stream-slow')
        await _wait_for_calls(fake, 'chat.startStream', 1)
        assert _stream_text(fake) == 'initial'
        release.set()
        await _wait_for_calls(fake, 'chat.stopStream', 1)
        assert _stream_text(fake) == 'initial final'

    async def test_serve_requires_token_and_closes_handler_when_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _fake_client(monkeypatch)
        app = SlackApp(_text_agent('unused'), signing_secret=_SIGNING_SECRET, client=client)
        monkeypatch.delenv('SLACK_APP_TOKEN', raising=False)
        with pytest.raises(ValueError, match='SLACK_APP_TOKEN'):
            await app.serve()

        started: list[tuple[object, str]] = []
        closed = False

        class FakeSocketModeHandler:
            def __init__(self, bolt: object, app_token: str) -> None:
                started.append((bolt, app_token))

            async def start_async(self) -> None:
                raise asyncio.CancelledError

            async def close_async(self) -> None:
                nonlocal closed
                closed = True

        monkeypatch.setattr('pydantic_ai_harness.slack._app.AsyncSocketModeHandler', FakeSocketModeHandler)
        monkeypatch.setenv('SLACK_APP_TOKEN', 'xapp-test')

        with pytest.raises(asyncio.CancelledError):
            await app.serve()

        assert started == [(app.bolt, 'xapp-test')]
        assert closed is True


class TestInMemorySlackHistory:
    async def test_expiry_eviction_copy_and_delete(self) -> None:
        first = [ModelRequest(parts=[UserPromptPart('first')])]
        second = [ModelRequest(parts=[UserPromptPart('second')])]
        history = InMemorySlackHistory(max_threads=1)

        await history.save('first', first)
        loaded = await history.load('first')
        assert loaded == first
        assert loaded is not first
        await history.save('second', second)
        assert await history.load('first') == ()
        assert await history.load('second') == second
        await history.delete('second')
        assert await history.load('second') == ()

        expiring = InMemorySlackHistory(ttl_seconds=0)
        await expiring.save('expired', first)
        assert await expiring.load('expired') == ()
