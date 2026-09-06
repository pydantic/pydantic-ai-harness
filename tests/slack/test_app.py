"""Integration tests for the caller-owned Slack host contract."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.authorization.authorize_result import AuthorizeResult
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_sdk.web.async_client import AsyncWebClient

import pydantic_ai_harness.slack._app as app_module
from pydantic_ai_harness.slack import ConversationStore, Slack, SlackApp, SlackContext, current_slack_context
from tests._recording_durability import RecordingDurability  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio

DepsT = TypeVar('DepsT')
Handler = Callable[..., Awaitable[None]]
INVALID_AUDIENCES: tuple[dict[str, set[str]], ...] = ({'': {'U1'}}, {'T1': set[str]()}, {'T1': {''}})


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class SlackCall:
    method: str
    kwargs: dict[str, object]


@dataclass
class FakeSlackClient:
    calls: list[SlackCall] = field(default_factory=list[SlackCall])
    status_error: Exception | None = None
    post_error: Exception | None = None
    post_error_after: int | None = None

    async def agents_sessions_setStatus(
        self, *, channel_id: str, thread_ts: str | None = None, status: str
    ) -> Mapping[str, object]:
        self.calls.append(
            SlackCall(
                'agents_sessions_setStatus',
                {'channel_id': channel_id, 'thread_ts': thread_ts, 'status': status},
            )
        )
        if self.status_error is not None:
            raise self.status_error
        return {'ok': True}

    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str | None = None,
        markdown_text: str | None = None,
        thread_ts: str | None = None,
        mrkdwn: bool | None = None,
    ) -> Mapping[str, object]:
        self.calls.append(
            SlackCall(
                'chat_postMessage',
                {
                    'channel': channel,
                    'text': text,
                    'markdown_text': markdown_text,
                    'thread_ts': thread_ts,
                    'mrkdwn': mrkdwn,
                },
            )
        )
        post_count = sum(call.method == 'chat_postMessage' for call in self.calls)
        if self.post_error is not None and (self.post_error_after is None or post_count > self.post_error_after):
            raise self.post_error
        return {'ok': True}

    def posts(self) -> list[SlackCall]:
        return [call for call in self.calls if call.method == 'chat_postMessage']

    def statuses(self) -> list[str | None]:
        return [call.kwargs['status'] for call in self.calls if call.method == 'agents_sessions_setStatus']  # type: ignore[return-value]


@dataclass
class FakeBoltApp:
    events: dict[str, Handler] = field(default_factory=dict[str, Handler])

    def event(self, name: str) -> Callable[[Handler], Handler]:
        def register(handler: Handler) -> Handler:
            self.events[name] = handler
            return handler

        return register


@dataclass
class FakeStore:
    histories: dict[str, list[ModelMessage]] = field(default_factory=dict[str, list[ModelMessage]])
    loads: list[str] = field(default_factory=list[str])
    saves: list[tuple[str, Sequence[ModelMessage]]] = field(default_factory=list[tuple[str, Sequence[ModelMessage]]])
    load_error: Exception | None = None
    save_error: Exception | None = None

    async def load(self, key: str) -> Sequence[ModelMessage]:
        self.loads.append(key)
        if self.load_error is not None:
            raise self.load_error
        return list(self.histories.get(key, ()))

    async def save(self, key: str, messages: Sequence[ModelMessage]) -> None:
        self.saves.append((key, messages))
        if self.save_error is not None:
            raise self.save_error
        self.histories[key] = list(messages)


class BlockingStore(FakeStore):
    def __init__(
        self,
        *,
        blocked_key: str,
        load_started: anyio.Event,
        release_load: anyio.Event,
        load_cancelled: anyio.Event | None = None,
        history: Sequence[ModelMessage] = (),
    ) -> None:
        super().__init__(histories={blocked_key: list(history)})
        self._blocked_key = blocked_key
        self._load_started = load_started
        self._release_load = release_load
        self._load_cancelled = load_cancelled
        self._blocked = False

    async def load(self, key: str) -> Sequence[ModelMessage]:
        self.loads.append(key)
        if key == self._blocked_key and not self._blocked:
            self._blocked = True
            self._load_started.set()
            try:
                await self._release_load.wait()
            except BaseException:
                if self._load_cancelled is not None:  # pragma: no branch - cancellation tests supply the event
                    self._load_cancelled.set()
                raise
        return list(self.histories.get(key, ()))


class CustomStringSet(Set[str]):
    def __init__(self, *values: str) -> None:
        self._values = frozenset(values)

    def __contains__(self, value: object) -> bool:  # pragma: no cover - required by Set ABC, unused by tests
        return value in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:  # pragma: no cover - required by Set ABC, unused by tests
        return len(self._values)


def auth_context(
    *,
    team_id: str = 'T1',
    user_id: str = 'U1',
    bot_user_id: str = 'UBOT',
    user_token: str | None = 'xoxp-user',
    auth_team_id: str | None = None,
    auth_user_id: str | None = None,
) -> dict[str, object]:
    return {
        'team_id': team_id,
        'user_id': user_id,
        'enterprise_id': 'E1',
        'bot_user_id': bot_user_id,
        'authorize_result': AuthorizeResult(
            enterprise_id='E1',
            team_id=auth_team_id or team_id,
            user_id=auth_user_id or user_id,
            user_token=user_token,
            bot_user_id=bot_user_id,
        ),
    }


def event(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        'user': 'U1',
        'team': 'T1',
        'channel': 'C1',
        'channel_type': 'channel',
        'text': '<@UBOT> hello',
        'ts': '1.1',
    }
    result.update(overrides)
    return result


def recording_agent(output: str = 'done', prompts: list[str] | None = None) -> Agent[None, str]:
    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if prompts is not None:
            for message in messages:
                if isinstance(message, ModelRequest):
                    for part in message.parts:
                        if isinstance(part, UserPromptPart) and isinstance(part.content, str):  # pragma: no branch
                            prompts.append(part.content)
        return ModelResponse(parts=[TextPart(output)])

    return Agent(FunctionModel(respond))


def build(
    agent: Agent[DepsT, str],
    *,
    allowed_users: Mapping[str, Set[str]] | str = {'T1': {'U1'}},
    store: ConversationStore | FakeStore | None = None,
    deps: DepsT | None = None,
    deps_factory: Callable[[SlackContext], DepsT] | None = None,
    install_url: str | None = None,
) -> tuple[SlackApp[DepsT], FakeBoltApp]:
    app = FakeBoltApp()
    bot = SlackApp(  # pyright: ignore[reportArgumentType, reportCallIssue]
        agent,
        app=app,  # pyright: ignore[reportArgumentType]
        allowed_users=allowed_users,  # pyright: ignore[reportArgumentType]
        store=store,
        deps=deps,  # pyright: ignore[reportArgumentType]
        deps_factory=deps_factory,  # pyright: ignore[reportArgumentType]
        install_url=install_url,
    )
    return bot, app


async def dispatch(
    app: FakeBoltApp,
    name: str,
    client: FakeSlackClient,
    payload: Mapping[str, object],
    context: Mapping[str, object],
    *,
    body: Mapping[str, object] | None = None,
) -> None:
    await app.events[name](  # type: ignore[call-arg]
        event=payload,
        client=client,
        context=context,
        body=body or {},
    )


class TestConstructor:
    def test_registers_only_the_public_event_listeners(self) -> None:
        _, app = build(recording_agent())
        assert set(app.events) == {'agent_session_stopped', 'app_mention', 'message'}

    def test_copies_and_validates_workspace_audience(self) -> None:
        audience = {'T1': {'U1'}}
        _, _ = build(recording_agent(), allowed_users=audience)
        audience['T1'].add('U2')
        with pytest.raises((TypeError, ValueError)):
            build(recording_agent(), allowed_users={'T1': ['U1']})  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            build(recording_agent(), allowed_users={})
        build(recording_agent(), allowed_users='all')
        with pytest.raises(TypeError):
            build(recording_agent(), allowed_users='ALL')

    async def test_mapping_variants_are_copied_without_widening_authorization(self) -> None:
        mutable_users = {'U1'}
        audience = UserDict({'T1': mutable_users})
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts), allowed_users=audience)  # type: ignore[arg-type]
        mutable_users.add('U2')
        await dispatch(app, 'app_mention', FakeSlackClient(), event(user='U2'), auth_context(user_id='U2'))
        assert prompts == []

        mapping = MappingProxyType({'T1': frozenset({'U1'})})
        prompts.clear()
        _, app = build(recording_agent(prompts=prompts), allowed_users=mapping)  # type: ignore[arg-type]
        await dispatch(app, 'app_mention', FakeSlackClient(), event(), auth_context())
        assert prompts

    def test_abstract_sets_are_accepted_and_copied(self) -> None:
        users = CustomStringSet('U1')
        build(recording_agent(), allowed_users={'T1': users})
        build(recording_agent(), allowed_users={'T1': {'U1': 1}.keys()})  # type: ignore[arg-type]

    def test_valid_install_url_is_accepted(self) -> None:
        build(recording_agent(), install_url='https://slack.example/install')

    @pytest.mark.parametrize('audience', INVALID_AUDIENCES)
    def test_audience_ids_must_be_non_empty(self, audience: object) -> None:
        with pytest.raises(ValueError):
            build(recording_agent(), allowed_users=audience)  # type: ignore[arg-type]

    def test_audience_rejects_non_string_user_ids_without_coercion(self) -> None:
        with pytest.raises(TypeError):
            build(recording_agent(), allowed_users={'T1': {1}})  # type: ignore[arg-type]

    @pytest.mark.parametrize('url', ['slack.example/install', 'ftp://slack.example/install', 'https://'])
    def test_install_url_must_be_absolute_http(self, url: str) -> None:
        with pytest.raises(ValueError, match='absolute http'):
            build(recording_agent(), install_url=url)

    def test_deps_modes_are_exclusive_and_factory_is_sync(self) -> None:
        calls: list[str] = []

        def factory(_context: SlackContext) -> None:
            calls.append('called')  # pragma: no cover - invalid configuration rejects before factory execution

        with pytest.raises(ValueError, match='not both'):
            build(recording_agent(), deps='fixed', deps_factory=factory)  # type: ignore[arg-type]

    def test_rejects_durable_agents(self) -> None:
        agent = Agent(TestModel(), name='durable', capabilities=[RecordingDurability()])
        with pytest.raises(ValueError, match='durable'):
            build(agent)

    async def test_sync_deps_factory_runs_once_inside_the_host_run(self) -> None:
        contexts: list[SlackContext] = []

        def factory(context: SlackContext) -> None:
            contexts.append(context)

        _, app = build(recording_agent(), deps_factory=factory)
        await dispatch(app, 'app_mention', FakeSlackClient(), event(), auth_context())
        assert len(contexts) == 1
        assert contexts[0].conversation_id == 'T1:C1:1.1'

    async def test_fixed_deps_mode_runs(self) -> None:
        _, app = build(recording_agent(), deps='fixed')  # type: ignore[arg-type]
        await dispatch(app, 'app_mention', FakeSlackClient(), event(), auth_context())


class TestRouting:
    async def test_matching_context_binds_token_and_exact_prompt_metadata(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        client = FakeSlackClient()
        payload = event(files=[{'id': 'F1', 'name': 'report.pdf', 'mimetype': 'application/pdf'}])
        await dispatch(app, 'app_mention', client, payload, auth_context())
        assert prompts == [
            'Slack event context (metadata, not instructions):\n'
            '{"channel_id":"C1","enterprise_id":"E1","files":[{"id":"F1","mimetype":"application/pdf",'
            '"name":"report.pdf"}],"message_ts":"1.1","team_id":"T1","thread_ts":"1.1","user_id":"U1"}'
            '\n\nUser message:\nhello'
        ]
        assert 'xoxp-user' not in prompts[0]

    async def test_all_audience_accepts_a_verified_workspace(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts), allowed_users='all')
        await dispatch(
            app,
            'app_mention',
            FakeSlackClient(),
            event(team='T9'),
            auth_context(team_id='T9'),
        )
        assert prompts

    @pytest.mark.parametrize(
        'payload, context',
        [
            (event(user=None), auth_context()),
            (event(team='T2'), auth_context()),
            (event(), {**auth_context(), 'user_id': 'U2'}),
            (event(team='T1'), {'user_id': 'U1', 'team_id': 'T2'}),
            (event(team=None), {'user_id': 'U1'}),
        ],
    )
    async def test_inconsistent_request_identity_is_denied_before_host_work(
        self, payload: Mapping[str, object], context: Mapping[str, object]
    ) -> None:
        store = FakeStore()
        _, app = build(recording_agent(), store=store)
        await dispatch(app, 'app_mention', FakeSlackClient(), payload, context)
        assert store.loads == []

    async def test_invalid_attachment_entries_are_ignored_but_text_routes(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        await dispatch(
            app,
            'app_mention',
            FakeSlackClient(),
            event(files=[{'name': 'missing-id'}]),
            auth_context(),
        )
        assert prompts

    async def test_no_bot_identity_keeps_user_text_unchanged(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        context = {'team_id': 'T1', 'user_id': 'U1'}
        await dispatch(app, 'app_mention', FakeSlackClient(), event(text='hello'), context)
        assert prompts[-1].endswith('User message:\nhello')

    async def test_message_listener_rejects_invalid_payload_before_routing(self) -> None:
        store = FakeStore()
        _, app = build(recording_agent(), store=store)
        await dispatch(
            app,
            'message',
            FakeSlackClient(),
            event(text=None, files=None),
            auth_context(),
        )
        assert store.loads == []

    async def test_bot_self_and_empty_user_messages_are_ignored(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts), allowed_users='all')
        await dispatch(
            app,
            'app_mention',
            FakeSlackClient(),
            event(user='UBOT'),
            auth_context(user_id='UBOT'),
        )
        await dispatch(app, 'app_mention', FakeSlackClient(), event(text='   ', ts='2.1'), auth_context())
        assert prompts == []

    async def test_dm_and_initial_channel_mention_route_but_unengaged_chatter_does_not(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        client = FakeSlackClient()
        await dispatch(app, 'message', client, event(channel_type='im', text='hello'), auth_context())
        await dispatch(app, 'message', client, event(text='chatter'), auth_context())
        await dispatch(app, 'app_mention', client, event(channel='C2', ts='2.1', text='<@UBOT> start'), auth_context())
        assert len(prompts) == 2

    async def test_unengaged_message_does_not_claim_a_later_mention(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts), store=FakeStore())
        client = FakeSlackClient()
        payload = event(text='chatter')
        await dispatch(app, 'message', client, payload, auth_context())
        await dispatch(app, 'app_mention', client, {**payload, 'text': '<@UBOT> start'}, auth_context())
        assert len(prompts) == 1

    async def test_file_only_event_uses_exact_marker(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        await dispatch(
            app,
            'message',
            FakeSlackClient(),
            event(text=None, files=[{'id': 'F1', 'name': 'a.txt'}], channel_type='im'),
            auth_context(),
        )
        assert prompts and prompts[0].endswith('User message:\nThe user shared files without a text message')
        assert 'F1' in prompts[0].split('\n\nUser message:\n', 1)[0]

    @pytest.mark.parametrize('field', ['bot_id', 'subtype'])
    async def test_bot_and_unsupported_events_are_ignored(self, field: str) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        payload = event(**{field: 'B1' if field == 'bot_id' else 'message_changed'})
        await dispatch(app, 'app_mention', FakeSlackClient(), payload, auth_context())
        assert prompts == []

    async def test_cross_workspace_same_user_is_denied_before_store(self) -> None:
        store = FakeStore()
        _, app = build(recording_agent(), store=store, allowed_users={'T1': {'U1'}})
        await dispatch(app, 'app_mention', FakeSlackClient(), event(team='T2'), auth_context(team_id='T2'))
        assert store.loads == []

    async def test_duplicate_app_mention_and_message_runs_once(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        client = FakeSlackClient()
        payload = event(channel_type='im')
        await dispatch(app, 'app_mention', client, payload, auth_context(), body={'event_id': 'one'})
        await dispatch(app, 'message', client, payload, auth_context(), body={'event_id': 'two'})
        assert len(prompts) == 1

    async def test_message_dedupe_expires_and_stays_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = [0.0]
        monkeypatch.setattr(app_module, 'monotonic', lambda: clock[0])
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts))
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(), auth_context())
        await dispatch(app, 'app_mention', client, event(), auth_context())
        assert len(prompts) == 1
        clock[0] = 601.0
        await dispatch(app, 'app_mention', client, event(), auth_context())
        assert len(prompts) == 3

        monkeypatch.setattr(app_module, '_MAX_EVENTS_PER_TEAM', 1)
        await dispatch(app, 'app_mention', client, event(ts='2.1'), auth_context())
        await dispatch(app, 'app_mention', client, event(ts='2.2'), auth_context())
        await dispatch(app, 'app_mention', client, event(ts='2.1'), auth_context())
        assert len(prompts) > 3


class TestBoltIntegration:
    @pytest.mark.parametrize('auth_team_id, auth_user_id', [('T1', 'U2'), ('T2', 'U1')])
    async def test_real_bolt_authorization_cannot_retag_sender(
        self, monkeypatch: pytest.MonkeyPatch, auth_team_id: str, auth_user_id: str
    ) -> None:
        seen_users: list[str] = []
        posts: list[Mapping[str, object]] = []

        async def authorize(team_id: str | None, user_id: str | None) -> AuthorizeResult:
            assert team_id == 'T1'
            assert user_id == 'U1'
            return AuthorizeResult(
                enterprise_id='E1',
                team_id=auth_team_id,
                user_id=auth_user_id,
                user_token='xoxp-other-user',
                bot_user_id='UBOT',
            )

        async def post_message(self: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
            _ = self
            posts.append(kwargs)
            return {'ok': True}

        async def set_status(self: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
            _ = self, kwargs
            return {'ok': True}

        monkeypatch.setattr(AsyncWebClient, 'chat_postMessage', post_message)
        monkeypatch.setattr(AsyncWebClient, 'agents_sessions_setStatus', set_status)

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            context = current_slack_context()
            assert context is not None
            seen_users.append(context.user_id)
            return ModelResponse(parts=[TextPart('done')])

        bolt_app = AsyncApp(
            authorize=authorize,
            client=AsyncWebClient(token='xoxb-test'),
            process_before_response=True,
            request_verification_enabled=False,
            ignoring_self_events_enabled=False,
        )
        SlackApp(Agent(FunctionModel(model)), app=bolt_app, allowed_users={'T1': {'U1'}})
        response = await bolt_app.async_dispatch(
            AsyncBoltRequest(
                body={
                    'type': 'event_callback',
                    'team_id': 'T1',
                    'event': {
                        'type': 'app_mention',
                        'team': 'T1',
                        'user': 'U1',
                        'channel': 'C1',
                        'text': '<@UBOT> hello',
                        'ts': '1.1',
                    },
                },
                mode='socket_mode',
            )
        )
        assert response.status == 200
        assert seen_users == ['U1']
        assert posts[-1]['markdown_text'] == 'done'

    async def test_real_bolt_identity_mismatch_leaves_slack_capability_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        posts: list[Mapping[str, object]] = []

        async def authorize(team_id: str | None, user_id: str | None) -> AuthorizeResult:
            assert team_id == 'T1'
            assert user_id == 'U1'
            return AuthorizeResult(
                enterprise_id='E1',
                team_id='T1',
                user_id='U2',
                user_token='xoxp-other-user',
                bot_user_id='UBOT',
            )

        async def post_message(self: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
            _ = self
            posts.append(kwargs)
            return {'ok': True}

        async def set_status(self: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
            _ = self, kwargs
            return {'ok': True}

        monkeypatch.setattr(AsyncWebClient, 'chat_postMessage', post_message)
        monkeypatch.setattr(AsyncWebClient, 'agents_sessions_setStatus', set_status)
        bolt_app = AsyncApp(
            authorize=authorize,
            client=AsyncWebClient(token='xoxb-test'),
            process_before_response=True,
            request_verification_enabled=False,
            ignoring_self_events_enabled=False,
        )
        SlackApp(Agent(TestModel(), capabilities=[Slack()]), app=bolt_app, allowed_users={'T1': {'U1'}})
        await bolt_app.async_dispatch(
            AsyncBoltRequest(
                body={
                    'type': 'event_callback',
                    'team_id': 'T1',
                    'event': {
                        'type': 'app_mention',
                        'team': 'T1',
                        'user': 'U1',
                        'channel': 'C1',
                        'text': '<@UBOT> hello',
                        'ts': '2.1',
                    },
                },
                mode='socket_mode',
            )
        )
        assert posts[-1]['markdown_text'] == 'Connect your Slack account before using this agent.'


class TestPersistenceAndDelivery:
    async def test_concurrent_followups_load_history_after_the_previous_save(self) -> None:
        load_started = anyio.Event()
        release_load = anyio.Event()
        initial = [ModelRequest(parts=[UserPromptPart('old')])]
        store = BlockingStore(
            blocked_key='T1:C1:1.1',
            load_started=load_started,
            release_load=release_load,
            history=initial,
        )
        history_sizes: list[int] = []

        async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            history_sizes.append(len(messages))
            return ModelResponse(parts=[TextPart('done')])

        _, app = build(Agent(FunctionModel(model)), store=store)
        client = FakeSlackClient()
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                dispatch, app, 'message', client, event(thread_ts='1.1', ts='1.2', text='first'), auth_context()
            )
            await load_started.wait()
            tg.start_soon(
                dispatch, app, 'message', client, event(thread_ts='1.1', ts='1.3', text='second'), auth_context()
            )
            await anyio.sleep(0)
            release_load.set()
        assert len(history_sizes) == 2
        assert history_sizes[1] > history_sizes[0]

    async def test_stop_cancels_a_run_blocked_loading_history(self) -> None:
        load_started = anyio.Event()
        release_load = anyio.Event()
        load_cancelled = anyio.Event()
        store = BlockingStore(
            blocked_key='T1:C1:1.1',
            load_started=load_started,
            release_load=release_load,
            load_cancelled=load_cancelled,
        )
        model_called = False

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:  # pragma: no cover
            nonlocal model_called
            model_called = True
            return ModelResponse(parts=[TextPart('done')])

        _, app = build(Agent(FunctionModel(model)), store=store)
        client = FakeSlackClient()
        async with anyio.create_task_group() as tg:
            tg.start_soon(dispatch, app, 'app_mention', client, event(), auth_context())
            await load_started.wait()
            stop = event(thread_ts='1.1', ts='stop-ts', event_ts='stop-load', text='')
            await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
            await load_cancelled.wait()
        assert not model_called
        assert [post.kwargs['markdown_text'] for post in client.posts()] == ['Stopped.']

    async def test_thread_followup_uses_live_or_stored_engagement(self) -> None:
        prompts: list[str] = []
        store = FakeStore()
        _, app = build(recording_agent(prompts=prompts), store=store)
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(), auth_context())
        prompts.clear()
        await dispatch(app, 'message', client, event(text='followup', thread_ts='1.1', ts='1.2'), auth_context())
        assert len(prompts) == 2

        prompts.clear()
        store.histories['T1:C2:2.2'] = [ModelRequest(parts=[UserPromptPart('old')])]
        await dispatch(
            app,
            'message',
            client,
            event(channel='C2', ts='2.3', thread_ts='2.2', text='stored followup'),
            auth_context(),
        )
        assert len(prompts) == 2

    async def test_stored_empty_history_keeps_unengaged_thread_ignored(self) -> None:
        prompts: list[str] = []
        _, app = build(recording_agent(prompts=prompts), store=FakeStore())
        await dispatch(
            app,
            'message',
            FakeSlackClient(),
            event(channel='C2', thread_ts='2.2', ts='2.3', text='not engaged'),
            auth_context(),
        )
        assert prompts == []

    async def test_thread_history_load_failure_posts_generic_reply(self) -> None:
        store = FakeStore(load_error=RuntimeError('history'))
        _, app = build(recording_agent(), store=store)
        client = FakeSlackClient()
        await dispatch(
            app,
            'message',
            client,
            event(channel='C2', thread_ts='2.2', ts='2.3', text='followup'),
            auth_context(),
        )
        assert client.posts()[-1].kwargs['markdown_text'] == "I couldn't complete that request. Please try again."

    async def test_load_or_agent_failure_is_generic_and_not_saved(self) -> None:
        store = FakeStore(load_error=RuntimeError('load'))
        _, app = build(recording_agent(), store=store)
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(), auth_context())
        assert client.posts()[-1].kwargs['markdown_text'] == "I couldn't complete that request. Please try again."
        assert store.saves == []

        async def fails(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('agent')

        _, app = build(Agent(FunctionModel(fails)), store=FakeStore())
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(ts='2.1'), auth_context())
        assert client.posts()[-1].kwargs['markdown_text'] == "I couldn't complete that request. Please try again."

    @pytest.mark.parametrize('install_url', [None, 'https://slack.example/install'])
    async def test_missing_slack_identity_uses_exact_reply(self, install_url: str | None) -> None:
        _, app = build(Agent(TestModel(), capabilities=[Slack()]), install_url=install_url)
        client = FakeSlackClient()
        context = {'team_id': 'T1', 'user_id': 'U1', 'bot_user_id': 'UBOT'}
        await dispatch(app, 'app_mention', client, event(), context)
        expected = 'Connect your Slack account before using this agent.'
        if install_url is not None:
            expected += f' {install_url}'
        assert client.posts()[-1].kwargs['markdown_text'] == expected

    async def test_delivery_failure_does_not_save_and_save_failure_posts_warning(self) -> None:
        store = FakeStore()
        _, app = build(recording_agent(), store=store)
        client = FakeSlackClient(post_error=RuntimeError('delivery'))
        await dispatch(app, 'app_mention', client, event(), auth_context())
        assert store.saves == []

        store = FakeStore(save_error=RuntimeError('save'))
        _, app = build(recording_agent(), store=store)
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(ts='3.1'), auth_context())
        assert (
            client.posts()[-1].kwargs['markdown_text']
            == "I sent the reply, but couldn't save this conversation's history."
        )

        store = FakeStore(save_error=RuntimeError('save'))
        _, app = build(recording_agent(), store=store)
        client = FakeSlackClient(post_error=RuntimeError('warning'), post_error_after=1)
        await dispatch(app, 'app_mention', client, event(ts='3.2'), auth_context())
        assert len(client.posts()) == 2

    async def test_empty_output_is_generic_and_not_saved(self) -> None:
        store = FakeStore()
        _, app = build(recording_agent(output=''), store=store)
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(), auth_context())
        assert client.posts()[-1].kwargs['markdown_text'] == "I couldn't complete that request. Please try again."
        assert store.saves == []

    async def test_whitespace_output_is_generic_and_not_saved(self) -> None:
        store = FakeStore()
        _, app = build(recording_agent(output='   '), store=store)
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(ts='empty-whitespace'), auth_context())
        assert client.posts()[-1].kwargs['markdown_text'] == "I couldn't complete that request. Please try again."
        assert store.saves == []

    async def test_long_reply_uses_bounded_plain_chunks(self) -> None:
        _, app = build(recording_agent(output='x' * 12_001))
        client = FakeSlackClient()
        await dispatch(app, 'app_mention', client, event(), auth_context())
        posts = client.posts()
        assert len(posts) == 4
        assert all(isinstance(post.kwargs['text'], str) and len(post.kwargs['text']) <= 3_500 for post in posts)
        assert all(post.kwargs['mrkdwn'] is False for post in posts)

    async def test_generic_error_delivery_failure_is_only_logged(self) -> None:
        async def fails(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('agent')

        _, app = build(Agent(FunctionModel(fails)))
        await dispatch(
            app,
            'app_mention',
            FakeSlackClient(post_error=RuntimeError('delivery')),
            event(),
            auth_context(),
        )

    async def test_default_stores_are_isolated(self) -> None:
        first, first_app = build(recording_agent())
        second, second_app = build(recording_agent())
        assert first is not second
        first_client = FakeSlackClient()
        second_client = FakeSlackClient()
        await dispatch(first_app, 'app_mention', first_client, event(), auth_context())
        await dispatch(second_app, 'app_mention', second_client, event(), auth_context())
        assert first_client.posts() and second_client.posts()


class TestConcurrencyAndStops:
    async def test_same_thread_runs_are_serialized(self) -> None:
        first_started = anyio.Event()
        release_first = anyio.Event()
        calls = 0

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
            return ModelResponse(parts=[TextPart('done')])

        _, app = build(Agent(FunctionModel(model)))
        client = FakeSlackClient()
        async with anyio.create_task_group() as tg:
            tg.start_soon(dispatch, app, 'app_mention', client, event(), auth_context())
            await first_started.wait()
            tg.start_soon(
                dispatch,
                app,
                'app_mention',
                client,
                event(thread_ts='1.1', ts='1.2', text='<@UBOT> second'),
                auth_context(),
            )
            await anyio.sleep(0)
            assert calls == 1
            release_first.set()
        assert calls == 2

    async def test_same_thread_serializes_and_different_threads_overlap(self) -> None:
        active = 0
        maximum = 0
        release = anyio.Event()
        both_started = anyio.Event()

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                both_started.set()
            await release.wait()
            active -= 1
            return ModelResponse(parts=[TextPart('done')])

        _, app = build(Agent(FunctionModel(model)))
        client = FakeSlackClient()
        async with anyio.create_task_group() as tg:
            tg.start_soon(dispatch, app, 'app_mention', client, event(), auth_context())
            await anyio.sleep(0)
            tg.start_soon(
                dispatch,
                app,
                'app_mention',
                client,
                event(text='other', channel='C2', ts='2.1'),
                auth_context(),
            )
            await both_started.wait()
            assert maximum == 2
            release.set()

    async def test_status_active_survives_agent_error_and_status_errors_do_not_hide_answer(self) -> None:
        _, app = build(recording_agent())
        client = FakeSlackClient(status_error=RuntimeError('status'))
        await dispatch(app, 'app_mention', client, event(), auth_context())
        assert client.posts()
        assert client.statuses() == ['processing', 'active']

    async def test_stop_cancels_active_and_waiting_runs_and_dedupes_retry(self) -> None:
        started = anyio.Event()
        release = anyio.Event()

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart('done')])  # pragma: no cover - stop cancels before return

        _, app = build(Agent(FunctionModel(model)))
        client = FakeSlackClient()
        async with anyio.create_task_group() as tg:
            tg.start_soon(dispatch, app, 'app_mention', client, event(), auth_context())
            await started.wait()
            tg.start_soon(
                dispatch,
                app,
                'message',
                client,
                event(text='queued', thread_ts='1.1', ts='1.2'),
                auth_context(),
            )
            await anyio.sleep(0)
            stop = event(user='U1', text='', thread_ts='1.1', ts='1.3', event_ts='stop-1')
            await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
            await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
        assert [post.kwargs['markdown_text'] for post in client.posts()] == ['Stopped.']
        assert client.statuses().count('active') == 1

    async def test_stop_dedupes_a_canceled_followup_but_allows_a_new_timestamp(self) -> None:
        load_started = anyio.Event()
        release_load = anyio.Event()
        load_cancelled = anyio.Event()
        store = BlockingStore(
            blocked_key='T1:C1:1.1',
            load_started=load_started,
            release_load=release_load,
            load_cancelled=load_cancelled,
            history=[ModelRequest(parts=[UserPromptPart('persisted')])],
        )
        calls = 0

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[TextPart('new timestamp')])

        _, app = build(Agent(FunctionModel(model)), store=store)
        client = FakeSlackClient()
        followup = event(channel='C1', thread_ts='1.1', ts='1.2', text='followup')
        stop = event(channel='C1', thread_ts='1.1', ts='stop-ts', event_ts='stop-followup', text='')
        async with anyio.create_task_group() as tg:
            tg.start_soon(dispatch, app, 'message', client, followup, auth_context())
            await load_started.wait()
            await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
            await load_cancelled.wait()
        await dispatch(app, 'message', client, followup, auth_context())
        await dispatch(app, 'message', client, {**followup, 'ts': '1.3'}, auth_context())
        assert calls == 1
        assert [post.kwargs['markdown_text'] for post in client.posts()] == ['Stopped.', 'new timestamp']

    async def test_retried_stop_does_not_cancel_a_new_turn(self) -> None:
        first_started = anyio.Event()
        second_started = anyio.Event()
        release_second = anyio.Event()
        calls = 0

        async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await anyio.sleep_forever()
            second_started.set()
            await release_second.wait()
            return ModelResponse(parts=[TextPart('new turn')])

        _, app = build(Agent(FunctionModel(model)))
        client = FakeSlackClient()
        stop = event(thread_ts='1.1', ts='stop-ts', event_ts='stop-retry', text='')
        async with anyio.create_task_group() as tg:
            tg.start_soon(dispatch, app, 'app_mention', client, event(), auth_context())
            await first_started.wait()
            await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
            tg.start_soon(
                dispatch,
                app,
                'app_mention',
                client,
                event(thread_ts='1.1', ts='1.2', text='<@UBOT> new turn'),
                auth_context(),
            )
            await second_started.wait()
            await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
            await anyio.sleep(0)
            release_second.set()
        assert calls == 2
        assert [post.kwargs['markdown_text'] for post in client.posts()] == ['Stopped.', 'new turn']

    async def test_accepted_stop_without_a_live_run_still_confirms_once(self) -> None:
        _, app = build(recording_agent())
        client = FakeSlackClient()
        stop = event(channel='C9', thread_ts='9.1', ts='9.2', event_ts='stop-9', text='')
        await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
        await dispatch(app, 'agent_session_stopped', client, stop, auth_context())
        assert [post.kwargs['markdown_text'] for post in client.posts()] == ['Stopped.']

    @pytest.mark.parametrize(
        'payload, context',
        [
            (event(thread_ts=None), auth_context()),
            (event(user='U2'), auth_context(user_id='U2')),
        ],
    )
    async def test_malformed_or_denied_stop_does_nothing(
        self, payload: Mapping[str, object], context: Mapping[str, object]
    ) -> None:
        _, app = build(recording_agent())
        client = FakeSlackClient()
        await dispatch(app, 'agent_session_stopped', client, payload, context)
        assert client.posts() == []

    async def test_stop_delivery_failure_is_logged(self) -> None:
        _, app = build(recording_agent())
        stop = event(channel='C9', thread_ts='9.1', ts='9.2', event_ts='stop-9', text='')
        await dispatch(
            app,
            'agent_session_stopped',
            FakeSlackClient(post_error=RuntimeError('stop delivery')),
            stop,
            auth_context(),
        )
