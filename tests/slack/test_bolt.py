"""Public Bolt dispatch acceptance tests for Slack registration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping

import anyio
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.authorization.authorize_result import AuthorizeResult
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_bolt.response.response import BoltResponse
from slack_sdk.web.async_client import AsyncWebClient

from pydantic_ai_harness.slack import SlackContext, current_slack_context, register_slack
from tests._recording_durability import RecordingDurability  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def event(event_type: str = 'message', **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        'type': event_type,
        'team': 'T1',
        'user': 'U1',
        'channel': 'C1',
        'channel_type': 'im',
        'text': 'hello',
        'ts': '1.1',
    }
    value.update(overrides)
    return value


def auth(
    *,
    user: str | None = 'U1',
    token: str | None = 'xoxp-user',
    bot_token: str | None = 'xoxb-bot',
    bot_user_id: str | None = 'UBOT',
    team_id: str = 'T1',
) -> AuthorizeResult:
    return AuthorizeResult(
        enterprise_id='E1',
        team_id=team_id,
        user_id=user,
        user_token=token,
        bot_user_id=bot_user_id,
        bot_token=bot_token,
    )


class Harness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agent: Agent[object, str],
        *,
        authorization: AuthorizeResult | None = None,
        deps_factory: Callable[[SlackContext], object] | None = None,
    ) -> None:
        self.posts: list[dict[str, object]] = []
        self.tokens: list[str | None] = []
        self.replies_calls: list[dict[str, object]] = []
        self.reply_tokens: list[str | None] = []
        self.errors: list[Exception] = []
        self.authorize_calls: list[tuple[str | None, str | None]] = []
        self.authorization = authorization or auth()

        async def authorize(team_id: str | None, user_id: str | None) -> AuthorizeResult:
            self.authorize_calls.append((team_id, user_id))
            del team_id, user_id
            return self.authorization

        async def post(client: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
            self.tokens.append(client.token)
            self.posts.append(kwargs)
            return {'ok': True}

        async def replies(client: AsyncWebClient, **kwargs: object) -> Mapping[str, object]:
            self.reply_tokens.append(client.token)
            self.replies_calls.append(kwargs)
            return {'ok': True, 'messages': []}

        async def unexpected_api_call(_: AsyncWebClient, *args: object, **kwargs: object) -> Mapping[str, object]:
            raise AssertionError(f'unexpected Slack Web API call: {args!r} {kwargs!r}')  # pragma: no cover

        monkeypatch.setattr(AsyncWebClient, 'chat_postMessage', post)
        monkeypatch.setattr(AsyncWebClient, 'conversations_replies', replies)
        monkeypatch.setattr(AsyncWebClient, 'api_call', unexpected_api_call)
        self.app = AsyncApp(
            authorize=authorize,
            request_verification_enabled=False,
            ignoring_self_events_enabled=False,
            process_before_response=True,
        )

        async def error_handler(error: Exception, **_: object) -> BoltResponse:
            self.errors.append(error)
            return BoltResponse(status=500, body='error')

        self.app.error(error_handler)
        if deps_factory is None:
            register_slack(self.app, agent)
        else:
            register_slack(self.app, agent, deps_factory=deps_factory)

    async def dispatch(
        self,
        value: Mapping[str, object],
        *,
        context: Mapping[str, object] | None = None,
        envelope: Mapping[str, object] | None = None,
        expected_status: int = 200,
    ) -> BoltResponse:
        body = {'type': 'event_callback', 'team_id': 'T1', 'event': value, **dict(envelope or {})}
        response = await self.app.async_dispatch(
            AsyncBoltRequest(body=body, mode='socket_mode', context=dict(context or {}))
        )
        assert response.status == expected_status
        if response.status == 200:
            assert not self.errors
        return response


def kwargs(text: str, thread_ts: str) -> dict[str, object]:
    return {
        'text': text,
        'thread_ts': thread_ts,
        'mrkdwn': False,
        'parse': 'none',
        'unfurl_links': False,
        'unfurl_media': False,
    }


def assert_post(post: Mapping[str, object], text: str, thread_ts: str) -> None:
    expected = kwargs(text, thread_ts)
    assert {key: post[key] for key in expected} == expected


async def test_envelope_only_workspace_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    await h.dispatch(event('app_mention', team=None, text='<@UBOT> hi', channel_type='channel'), expected_status=200)
    await h.dispatch(event(team=None, text='dm'), expected_status=200)
    assert calls == ['run', 'run']
    assert len(h.posts) == 2
    for post in h.posts:
        assert_post(post, 'ok', '1.1')


async def test_unrelated_message_reaches_later_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(monkeypatch, Agent(TestModel()))
    seen: list[str] = []

    async def later(**_: object) -> None:
        seen.append('called')

    h.app.message()(later)  # pyright: ignore[reportUnknownMemberType]
    await h.dispatch(event(text='unrelated', channel_type='channel'))
    assert seen == ['called']


async def test_missing_bot_id_channel_message_reaches_later_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    await h.dispatch(event('app_mention', text='<@UBOT> start', channel_type='channel', thread_ts='1.1'))
    h.authorization = auth(bot_user_id=None)
    seen: list[str] = []

    async def later(**_: object) -> None:
        seen.append('called')

    h.app.message()(later)  # pyright: ignore[reportUnknownMemberType]
    await h.dispatch(event(text='reply', channel_type='channel', thread_ts='1.1'))
    assert seen == ['called']
    assert calls == ['run']
    assert len(h.posts) == 1


@pytest.mark.parametrize('thread_ts', [None, '0.9'])
async def test_dm_mention_still_routes(monkeypatch: pytest.MonkeyPatch, thread_ts: str | None) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    await h.dispatch(event(text='<@UBOT> hi', thread_ts=thread_ts), expected_status=200)
    assert calls == ['run']
    assert len(h.posts) == 1
    assert_post(h.posts[0], 'ok', thread_ts or '1.1')
    if thread_ts is not None:
        assert h.replies_calls == [
            {
                'channel': 'C1',
                'ts': '0.9',
                'oldest': '0.9',
                'limit': 4,
                'include_all_metadata': True,
            }
        ]
        assert h.reply_tokens == ['xoxb-bot']


@pytest.mark.parametrize('ordinary_first', [False, True])
async def test_channel_mention_dual_delivery_runs_once(monkeypatch: pytest.MonkeyPatch, ordinary_first: bool) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    mention = event('app_mention', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1')
    ordinary = event(text='<@UBOT> hi', channel_type='channel', thread_ts='1.1')
    for value in (ordinary, mention) if ordinary_first else (mention, ordinary):
        await h.dispatch(value, expected_status=200 if value is mention else 404)
    assert calls == ['run']
    assert len(h.posts) == 1
    assert not h.errors


async def test_other_user_mention_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    async def model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        prompts.extend(
            p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    await h.dispatch(event('app_mention', text='<@UBOT> hello <@U2>', channel_type='channel', thread_ts='1.1'))
    assert 'hello <@U2>' in prompts[0]
    assert len(h.posts) == 1
    assert_post(h.posts[0], 'ok', '1.1')


async def test_shared_engagement_is_registration_local(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    agent = Agent(FunctionModel(model))
    h = Harness(monkeypatch, agent)
    await h.dispatch(
        event('app_mention', text='<@UBOT> start', channel_type='channel', thread_ts='1.1'), expected_status=200
    )
    h.authorization = auth(user='U2')
    await h.dispatch(event(user='U2', text='reply', channel_type='channel', thread_ts='1.1'), expected_status=200)
    fresh = Harness(monkeypatch, agent)
    fresh.authorization = auth(user='U2')
    await fresh.dispatch(event(user='U2', text='reply', channel_type='channel', thread_ts='1.1'), expected_status=404)
    assert calls == ['run', 'run']
    assert len(h.posts) == 2


async def test_native_message_subtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(monkeypatch, Agent(TestModel()))
    for subtype in ('message_changed', 'message_deleted'):
        await h.dispatch(event(subtype=subtype), expected_status=404)
    assert not h.posts
    await h.dispatch(
        event(subtype='file_share', files=[{'id': 'F1', 'name': 'a.txt', 'mimetype': 'text/plain'}]),
        expected_status=200,
    )
    await h.dispatch(
        event('app_mention', text='<@UBOT> start', channel_type='channel', thread_ts='1.1'), expected_status=200
    )
    await h.dispatch(event(subtype='thread_broadcast', channel_type='channel', thread_ts='1.1'), expected_status=200)
    await h.dispatch(event(bot_id='B1'), expected_status=404)
    assert len(h.posts) == 3


async def test_file_only_marker_and_invalid_files(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    async def model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        prompts.extend(
            p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    value = event(subtype='file_share', text=' ', files=[{'id': 'F1', 'name': 'a.txt', 'mimetype': 'text/plain'}])
    await h.dispatch(value, expected_status=200)
    assert prompts and 'The user shared files without a text message' in prompts[0]
    assert len(h.posts) == 1
    await h.dispatch(event(subtype='file_share', text='', files=[{'name': 'missing-id'}]), expected_status=404)
    assert len(h.posts) == 1


@pytest.mark.parametrize('overrides', [{'text': ''}, {'text': None}, {'ts': None}, {'user': None}])
async def test_malformed_required_coordinates_are_not_delivered(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    value = event('app_mention', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1')
    value.update(overrides)
    await h.dispatch(value, expected_status=200)
    assert not calls and not h.posts


@pytest.mark.parametrize('identity', [{'user': None}, {'token': None}, {'token': ''}])
async def test_missing_user_oauth_gets_connection_reply(
    monkeypatch: pytest.MonkeyPatch, identity: dict[str, str | None]
) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)), authorization=auth(**identity))  # type: ignore[arg-type]
    await h.dispatch(
        event('app_mention', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'), expected_status=200
    )
    assert len(h.posts) == 1
    assert_post(h.posts[0], 'Connect your Slack account before using this agent.', '1.1')
    h.authorization = auth()
    seen: list[str] = []

    async def later(**_: object) -> None:
        seen.append('called')

    h.app.message()(later)  # pyright: ignore[reportUnknownMemberType]
    await h.dispatch(event(user='U1', text='reply', channel_type='channel', thread_ts='1.1'), expected_status=200)
    assert calls == [] and seen == ['called']


@pytest.mark.parametrize('sender', [None, '', {}])
async def test_missing_message_sender_has_no_oauth_reply_or_engagement(
    monkeypatch: pytest.MonkeyPatch, sender: object
) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    await h.dispatch(event(user=sender), expected_status=404)
    await h.dispatch(event(user='U1', text='reply', channel_type='channel', thread_ts='1.1'), expected_status=404)
    assert not h.posts and not calls


@pytest.mark.parametrize('missing', [{'channel': None}, {'team': None}])
async def test_missing_message_coordinates_fall_through_to_later_listener(
    monkeypatch: pytest.MonkeyPatch, missing: dict[str, None]
) -> None:
    calls: list[str] = []
    seen: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    async def later(**_: object) -> None:
        seen.append('called')

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    h.app.message()(later)  # pyright: ignore[reportUnknownMemberType]
    value = event(text='reply', channel_type='channel', thread_ts='1.1')
    value.update(missing)
    await h.dispatch(value, expected_status=200)
    assert seen == ['called']
    assert not calls and not h.posts


@pytest.mark.parametrize('missing', [{'channel': None}, {'team': None}])
async def test_missing_channel_or_workspace_mention_is_not_delivered(
    monkeypatch: pytest.MonkeyPatch, missing: dict[str, None]
) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    value = event('app_mention', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1')
    value.update(missing)
    envelope = {'team_id': None} if missing == {'team': None} else None
    await h.dispatch(value, envelope=envelope, expected_status=200)
    assert not calls and not h.posts


async def test_mpim_message_falls_through_to_later_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    seen: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    async def later(**_: object) -> None:
        seen.append('called')

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    h.app.message()(later)  # pyright: ignore[reportUnknownMemberType]
    await h.dispatch(event(channel_type='mpim', text='group message'), expected_status=200)
    assert seen == ['called']
    assert not calls and not h.posts


async def test_mention_only_does_not_engage_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    seen: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    async def later(**_: object) -> None:
        seen.append('called')

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    h.app.message()(later)  # pyright: ignore[reportUnknownMemberType]
    await h.dispatch(
        event('app_mention', text='<@UBOT>   ', channel_type='channel', thread_ts='1.1'), expected_status=200
    )
    await h.dispatch(event(text='reply', channel_type='channel', thread_ts='1.1'), expected_status=200)
    assert seen == ['called']
    assert not calls and not h.posts


async def test_installer_cannot_retag_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(monkeypatch, Agent(TestModel()), authorization=auth(user='U2'))
    await h.dispatch(event(user='U1'))
    assert not h.posts


async def test_known_actor_workspace_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(monkeypatch, Agent(TestModel()))
    await h.dispatch(
        event('app_mention', team=None, user_team='T2', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'),
        envelope={'is_ext_shared_channel': True},
    )
    assert not h.posts
    assert h.authorize_calls == [('T1', 'U1')]


@pytest.mark.parametrize(
    ('value', 'context', 'envelope', 'authorization'),
    [
        pytest.param(
            event('app_mention', team='T1', team_id='T2', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'),
            {},
            {},
            auth(),
            id='inner-team-conflict',
        ),
        pytest.param(
            event('app_mention', team='T2', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'),
            {},
            {'team_id': 'T1'},
            auth(),
            id='event-vs-envelope',
        ),
        pytest.param(
            event('app_mention', team=None, user_team='T2', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'),
            {},
            {'is_ext_shared_channel': True},
            auth(),
            id='actor-team',
        ),
        pytest.param(
            event('app_mention', team='T1', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'),
            {},
            {},
            auth(team_id='T2'),
            id='auth-team',
        ),
        pytest.param(
            event('app_mention', team='T1', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'),
            {},
            {},
            auth(user='U2'),
            id='auth-user',
        ),
    ],
)
async def test_identity_mismatches_reject_app_mentions(
    monkeypatch: pytest.MonkeyPatch,
    value: Mapping[str, object],
    context: Mapping[str, object],
    envelope: Mapping[str, object],
    authorization: AuthorizeResult,
) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)), authorization=authorization)
    await h.dispatch(value, context=context, envelope=envelope, expected_status=200)
    assert h.authorize_calls
    assert not calls and not h.posts


async def test_bot_token_is_required_for_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.append('run')
        return ModelResponse(parts=[TextPart('ok')])

    for token, bot_user_id in ((None, 'UBOT'), ('', 'UBOT'), ('xoxb-bot', None)):
        h = Harness(
            monkeypatch, Agent(FunctionModel(model)), authorization=auth(bot_token=token, bot_user_id=bot_user_id)
        )
        await h.dispatch(
            event('app_mention', text='<@UBOT> hi', channel_type='channel', thread_ts='1.1'),
            context={'bot_token': 'stale-token'},
        )
        assert not h.posts and not calls


async def test_valid_delivery_uses_authorized_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(monkeypatch, Agent(TestModel()))
    await h.dispatch(event(), expected_status=200)
    assert h.tokens == ['xoxb-bot']


async def test_model_receives_exact_context_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    async def model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        prompts.extend(
            p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        return ModelResponse(parts=[TextPart('ok')])

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    await h.dispatch(
        event(text='<@UBOT> hello', thread_ts='0.9', files=[{'id': 'F1', 'name': 'x', 'mimetype': 'text/plain'}]),
        envelope={'enterprise_id': 'E1'},
    )
    header, text = prompts[0].split('\n\nUser message:\n', 1)
    assert text == 'hello'
    assert json.loads(header.split('\n', 1)[1]) == {
        'team_id': 'T1',
        'channel_id': 'C1',
        'thread_ts': '0.9',
        'message_ts': '1.1',
        'user_id': 'U1',
        'enterprise_id': 'E1',
        'files': [{'file_id': 'F1', 'name': 'x', 'mimetype': 'text/plain'}],
    }
    assert 'xox' not in prompts[0]
    assert len(h.posts) == 1
    assert_post(h.posts[0], 'ok', '0.9')


async def test_dependency_factory_and_conversation_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[object, str | None]] = []
    count = 0

    def factory(ctx: SlackContext) -> object:
        nonlocal count
        count += 1
        assert current_slack_context() == ctx
        return ctx

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('ok')])

    agent = Agent(FunctionModel(model))

    @agent.instructions
    def inspect(ctx: RunContext[object]) -> str:
        seen.append((ctx.deps, ctx.conversation_id))
        return ''

    h = Harness(monkeypatch, agent, deps_factory=factory)
    await h.dispatch(event())
    assert count == 1
    assert seen[0][0] == SlackContext(
        team_id='T1', channel_id='C1', thread_ts='1.1', message_ts='1.1', user_id='U1', enterprise_id=None
    )
    assert seen[0][1] == 'slack:T1:U1:C1:1.1'
    assert len(h.posts) == 1
    assert_post(h.posts[0], 'ok', '1.1')


@pytest.mark.parametrize('output', ['', '   '])
async def test_blank_output_uses_generic_reply(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    h = Harness(monkeypatch, Agent(TestModel(custom_output_text=output)))
    await h.dispatch(event())
    assert h.posts[0]['text'] == "I couldn't complete that request. Please try again."
    assert len(h.posts) == 1


async def test_run_failure_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        raise RuntimeError('boom')

    h = Harness(monkeypatch, Agent(FunctionModel(model)))
    await h.dispatch(event())
    assert h.posts[0]['text'] == "I couldn't complete that request. Please try again."
    assert 'boom' not in str(h.posts[0]['text'])
    assert len(h.posts) == 1


async def test_delivery_failure_reaches_bolt_error_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(monkeypatch, Agent(TestModel()))
    attempts = 0

    async def fail(_: AsyncWebClient, **__: object) -> Mapping[str, object]:
        nonlocal attempts
        attempts += 1
        raise RuntimeError('network')

    monkeypatch.setattr(AsyncWebClient, 'chat_postMessage', fail)
    response = await h.dispatch(event(), expected_status=500)
    assert response.status == 500 and attempts == 1 and len(h.errors) == 1
    assert isinstance(h.errors[0], RuntimeError)


async def test_first_party_cancellation_is_not_a_generic_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = Agent(TestModel())

    @agent.tool
    async def cancel(ctx: RunContext[object]) -> str:
        ctx.cancel()
        await anyio.sleep(0)
        return 'unreachable'

    h = Harness(monkeypatch, agent)
    response = await h.dispatch(event(), expected_status=500)
    assert response.status == 500 and not h.posts
    assert len(h.errors) == 1 and isinstance(h.errors[0], RunCancelled)


async def test_external_cancellation_restores_context(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    observed: list[SlackContext | None] = []

    async def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        entered.set()
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart('late')])  # pragma: no cover

    h = Harness(monkeypatch, Agent(FunctionModel(model)))

    async def run_dispatch() -> BoltResponse:
        try:
            return await h.dispatch(event())
        finally:
            observed.append(current_slack_context())

    task = asyncio.create_task(run_dispatch())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observed == [None]
    assert not h.posts


async def test_durable_agent_rejected_before_registration() -> None:
    with pytest.raises(ValueError, match='durable'):
        register_slack(
            AsyncApp(token='xoxb-test', request_verification_enabled=False),
            Agent(TestModel(), name='durable', capabilities=[RecordingDurability()]),
        )
