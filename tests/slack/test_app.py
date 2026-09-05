from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anyio
import httpx
import pytest
import slack_bolt.adapter.socket_mode.async_handler as slack_socket_mode
import slack_bolt.app.async_app as slack_async_app
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from slack_bolt.app.async_app import AsyncApp as RealAsyncApp
from starlette.applications import Starlette
from starlette.routing import Mount

import pydantic_ai_harness.slack as slack_package
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.slack import (
    InMemoryConversationStore,
    Slack,
    SlackAccess,
    SlackApp,
    SlackContext,
    SlackContextEntity,
    SlackFile,
    SlackMessageContext,
    SlackThread,
    SlackTools,
    current_slack_context,
)

from .conftest import FakeSlackClient, fake_mcp, prompt_block_id

pytestmark = pytest.mark.anyio

Handler = Callable[..., Any]

DepsT = TypeVar('DepsT')


@dataclass
class Warehouse:
    """Deps an agent already had before anyone thought about Slack."""

    dsn: str


def warehouse_for(thread: SlackThread) -> Warehouse:
    return Warehouse(dsn=f'postgres://{thread.channel_id}')


@dataclass
class FakeBoltApp:
    """Records the listeners `SlackApp` registers, without touching Slack."""

    token: str
    signing_secret: str | None = None
    events: dict[str, Handler] = field(default_factory=dict[str, Handler])
    actions: list[Handler] = field(default_factory=list[Handler])

    def event(self, name: str) -> Callable[[Handler], Handler]:
        def register(handler: Handler) -> Handler:
            self.events[name] = handler
            return handler

        return register

    def action(self, _constraint: object) -> Callable[[Handler], Handler]:
        def register(handler: Handler) -> Handler:
            self.actions.append(handler)
            return handler

        return register


@pytest.fixture(autouse=True)
def bolt(monkeypatch: pytest.MonkeyPatch) -> Callable[[], FakeBoltApp]:
    """Replace Bolt's app with a recorder. Autouse, so no test reaches real Slack;
    request it by name only to inspect what was registered."""
    built: list[FakeBoltApp] = []

    def factory(*, token: str, signing_secret: str | None = None) -> FakeBoltApp:
        app = FakeBoltApp(token=token, signing_secret=signing_secret)
        built.append(app)
        return app

    monkeypatch.setattr(slack_async_app, 'AsyncApp', factory)
    return lambda: built[-1]


@pytest.fixture
def slack_agent_with_store(agent: Agent[None, str]) -> tuple[SlackApp[None], InMemoryConversationStore]:
    store = InMemoryConversationStore()
    return build(agent, store=store), store


@pytest.fixture
def agent() -> Agent[None, str]:
    return Agent(TestModel(custom_output_text='done'))


def message(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        'user': 'U0ASKER',
        'channel': 'C123',
        'text': '<@U0BOT> ship it',
        'ts': '1700000000.000001',
        'team': 'T1',
    }
    event.update(overrides)
    return event


def build(agent: Agent[DepsT, str], **kwargs: object) -> SlackApp[DepsT]:
    defaults: dict[str, object] = {'bot_token': 'xoxb-t', 'app_token': 'xapp-t', 'access': SlackAccess.users('U0ASKER')}
    defaults.update(kwargs)
    return SlackApp(agent, **defaults)  # pyright: ignore[reportCallIssue, reportArgumentType]


async def dispatch_message(
    bot: SlackApp[DepsT],
    event: Mapping[str, object],
    client: FakeSlackClient,
    *,
    bot_user_id: str | None = None,
    team_id: str | None = None,
    enterprise_id: str | None = None,
    user_token: str | None = None,
) -> None:
    """Deliver a message through the listener registered on the public Bolt app."""
    bolt_app = bot.app
    assert isinstance(bolt_app, FakeBoltApp)
    context = {
        key: value
        for key, value in {
            'bot_user_id': bot_user_id,
            'team_id': team_id,
            'enterprise_id': enterprise_id,
            'user_token': user_token,
        }.items()
        if value is not None
    }
    await bolt_app.events['app_mention'](event=event, client=client, context=context, body={})


class TestLazyExport:
    def test_the_bolt_app_is_reachable_from_the_package(self) -> None:
        # Named on the package, imported only when asked for, so the rest of the
        # package stays usable without `slack-bolt`.
        assert slack_package.SlackApp is SlackApp

    def test_an_unknown_name_still_raises(self) -> None:
        with pytest.raises(AttributeError, match='has no attribute'):
            _ = slack_package.SlackWidget  # pyright: ignore[reportAttributeAccessIssue]


class TestConfiguration:
    def test_accepts_a_caller_configured_bolt_app_for_oauth(
        self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        configured = FakeBoltApp(token='resolved-by-oauth')
        slack_app = SlackApp(  # pyright: ignore[reportCallIssue] - fake stands in for the configured Bolt app
            agent,
            app=configured,  # pyright: ignore[reportArgumentType] - fake records Bolt listener registration
            access=SlackAccess.users('U0ASKER'),
        )
        assert slack_app.app is configured

    def test_caller_configured_app_rejects_credentials_for_a_new_app(self, agent: Agent[None, str]) -> None:
        configured = FakeBoltApp(token='resolved-by-oauth')
        with pytest.raises(ValueError, match='cannot be passed with app'):
            SlackApp(  # pyright: ignore[reportCallIssue] - deliberately invalid construction mode
                agent,
                app=configured,  # pyright: ignore[reportArgumentType] - deliberate invalid combination
                access=SlackAccess.users('U0ASKER'),
                bot_token='xoxb-t',  # pyright: ignore[reportArgumentType] - deliberate invalid combination
            )
        with pytest.raises(ValueError, match='cannot be passed with app'):
            SlackApp(  # pyright: ignore[reportCallIssue] - deliberately invalid construction mode
                agent,
                app=configured,  # pyright: ignore[reportArgumentType] - deliberate invalid combination
                access=SlackAccess.users('U0ASKER'),
                signing_secret='secret',  # pyright: ignore[reportArgumentType] - deliberate invalid combination
            )

    def test_reads_tokens_from_the_environment(
        self, agent: Agent[None, str], bolt: Callable[[], FakeBoltApp], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-env')
        monkeypatch.setenv('SLACK_APP_TOKEN', 'xapp-env')
        monkeypatch.setenv('SLACK_ALLOWED_USER_IDS', 'U1, U2 ,')
        slack_agent = SlackApp(agent)
        assert bolt().token == 'xoxb-env'
        assert slack_agent.access.allows('U1')
        assert slack_agent.access.allows('U2')
        assert not slack_agent.access.allows('U3')

    def test_a_missing_bot_token_is_refused(self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        with pytest.raises(ValueError, match='SLACK_BOT_TOKEN'):
            SlackApp(agent, app_token='xapp-t')

    async def test_socket_mode_asks_for_its_token_only_when_started(
        self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An Events API deployment has no app token and must still construct.
        monkeypatch.delenv('SLACK_APP_TOKEN', raising=False)
        bot = SlackApp(agent, bot_token='xoxb-t', signing_secret='s', access=SlackAccess.users('U0ASKER'))
        with pytest.raises(ValueError, match='SLACK_APP_TOKEN'):
            await bot.start()

    def test_the_events_api_asks_for_its_secret_only_when_mounted(
        self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # And a Socket Mode deployment has no signing secret.
        monkeypatch.delenv('SLACK_SIGNING_SECRET', raising=False)
        bot = SlackApp(agent, bot_token='xoxb-t', app_token='xapp-t', access=SlackAccess.users('U0ASKER'))
        with pytest.raises(ValueError, match='SLACK_SIGNING_SECRET'):
            bot.http_app()

    def test_the_signing_secret_reaches_bolt(self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str]) -> None:
        build(agent, signing_secret='shhh')
        assert bolt().signing_secret == 'shhh'

    def test_slack_access_rejects_an_empty_user_list(self) -> None:
        with pytest.raises(ValueError, match='non-empty Slack user ID'):
            SlackAccess.users()

    def test_slack_access_rejects_an_empty_user_id(self) -> None:
        with pytest.raises(ValueError, match='non-empty Slack user ID'):
            SlackAccess.users('')

    def test_slack_access_normalizes_surrounding_whitespace(self) -> None:
        access = SlackAccess.users(' U0ASKER ')
        assert access.allows('U0ASKER')
        assert not access.allows(' U0ASKER ')

    def test_workspace_access_is_an_explicit_allow_all_policy(self) -> None:
        access = SlackAccess.workspace()
        assert access.allows('U0ANYONE')
        assert access.allowed_user_ids is None

    def test_selected_user_access_exposes_an_immutable_policy(self) -> None:
        assert SlackAccess.users('U1', 'U2').allowed_user_ids == frozenset({'U1', 'U2'})

    @pytest.mark.parametrize('access', [SlackAccess.workspace(), SlackAccess.users('U1', 'U2')])
    def test_environment_mcp_token_requires_exactly_one_allowed_user(
        self, access: SlackAccess, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SLACK_MCP_TOKEN', 'xoxp-one-person')
        with pytest.raises(ValueError, match='requires exactly one allowed Slack user'):
            SlackApp(
                Agent(TestModel(), capabilities=[Slack()]),
                bot_token='xoxb-t',
                access=access,
            )

    def test_environment_mcp_token_accepts_its_single_owner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_MCP_TOKEN', 'xoxp-one-person')
        SlackApp(  # pyright: ignore[reportCallIssue] - fake stands in for the configured Bolt app
            Agent(TestModel(), capabilities=[Slack()]),
            bot_token='xoxb-t',
            access=SlackAccess.users('U1'),
        )

    def test_explicit_mcp_token_is_an_opt_in_to_a_shared_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_MCP_TOKEN', 'xoxp-ignored')
        SlackApp(
            Agent(TestModel(), capabilities=[Slack(mcp_token='xoxp-explicit')]),
            bot_token='xoxb-t',
            access=SlackAccess.workspace(),
        )

    def test_environment_mcp_token_does_not_constrain_an_agent_without_slack(
        self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SLACK_MCP_TOKEN', 'xoxp-unused')
        SlackApp(agent, bot_token='xoxb-t', access=SlackAccess.workspace())

    def test_slack_access_rejects_a_non_string_user_id(self) -> None:
        with pytest.raises(ValueError, match='non-empty Slack user ID'):
            SlackAccess.users(None)  # pyright: ignore[reportArgumentType]

    def test_missing_access_policy_is_refused(
        self,
        agent: Agent[None, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv('SLACK_ALLOWED_USER_IDS', raising=False)
        with pytest.raises(ValueError, match='access is not configured'):
            SlackApp(agent, bot_token='xoxb-t', app_token='xapp-t')

    def test_registers_the_listeners_a_slack_agent_needs(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str]
    ) -> None:
        build(agent)
        assert set(bolt().events) == {
            'agent_session_stopped',
            'app_context_changed',
            'app_mention',
            'message',
        }
        assert len(bolt().actions) == 1


class TestHandleMessage:
    async def test_runs_the_agent_and_replies_in_the_thread(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        await dispatch_message(build(agent), message(), slack_client, bot_user_id='U0BOT')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert call.kwargs['markdown_text'] == 'done'
        assert call.kwargs['thread_ts'] == '1700000000.000001'
        assert call.kwargs['text'] is None

    async def test_shows_and_clears_a_working_status(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        await dispatch_message(build(agent), message(), slack_client, bot_user_id='U0BOT')
        statuses = slack_client.method_calls('agents_sessions_setStatus')
        assert [call.kwargs['status'] for call in statuses] == ['processing', 'active']

    async def test_status_failure_does_not_fail_the_run(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.status_error = RuntimeError('not an agent view')
        await dispatch_message(build(agent), message(), slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['markdown_text'] == 'done'

    async def test_a_reply_continues_the_thread_it_arrived_in(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        event = message(ts='1700000000.000009', thread_ts='1700000000.000001')
        await dispatch_message(build(agent), event, slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['thread_ts'] == '1700000000.000001'

    async def test_history_accumulates_across_turns_in_one_thread(
        self, slack_agent_with_store: tuple[SlackApp[None], InMemoryConversationStore], slack_client: FakeSlackClient
    ) -> None:
        slack_agent, store = slack_agent_with_store
        await dispatch_message(slack_agent, message(), slack_client, bot_user_id='U0BOT')
        reply_in_same_thread = message(ts='1700000000.000002', thread_ts='1700000000.000001')
        await dispatch_message(slack_agent, reply_in_same_thread, slack_client, bot_user_id='U0BOT')
        assert len(await store.load('T1:C123:1700000000.000001')) == 4

    async def test_a_separate_thread_keeps_its_own_history(
        self, slack_agent_with_store: tuple[SlackApp[None], InMemoryConversationStore], slack_client: FakeSlackClient
    ) -> None:
        slack_agent, store = slack_agent_with_store
        await dispatch_message(slack_agent, message(), slack_client, bot_user_id='U0BOT')
        await dispatch_message(slack_agent, message(ts='1700000000.000002'), slack_client, bot_user_id='U0BOT')
        assert len(await store.load('T1:C123:1700000000.000001')) == 2
        assert len(await store.load('T1:C123:1700000000.000002')) == 2

    async def test_only_the_bots_own_mention_is_stripped(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        captured: list[str] = []

        def remember(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            part = messages[-1].parts[-1]
            captured.append(part.content if isinstance(part, UserPromptPart) and isinstance(part.content, str) else '')
            return ModelResponse(parts=[TextPart('done')])

        recording: Agent[None, str] = Agent(FunctionModel(remember))

        event = message(text='<@U0BOT> ask <@U0ALICE> about the deploy')
        await dispatch_message(build(recording), event, slack_client, bot_user_id='U0BOT')
        assert captured == ['ask <@U0ALICE> about the deploy']

    @pytest.mark.parametrize(
        'event',
        [
            message(bot_id='B1'),
            message(subtype='channel_join'),
            message(user='U0BOT'),
            message(user='U0STRANGER'),
            message(text='<@U0BOT>'),
            message(user=None),
            message(channel=None),
            message(text=None),
            message(ts=None),
        ],
        ids=[
            'from a bot',
            'a subtype such as a join',
            'from the bot itself',
            'from someone off the allowlist',
            'a bare mention with nothing to do',
            'no user',
            'no channel',
            'no text',
            'no timestamp',
        ],
    )
    async def test_ignores(
        self,
        agent: Agent[None, str],
        slack_client: FakeSlackClient,
        event: dict[str, object],
    ) -> None:
        await dispatch_message(build(agent), event, slack_client, bot_user_id='U0BOT')
        assert slack_client.calls == []

    async def test_an_empty_answer_posts_nothing(self, slack_client: FakeSlackClient) -> None:
        quiet = Agent(TestModel(custom_output_text='   '))
        await dispatch_message(build(quiet), message(), slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage') == []

    async def test_a_long_answer_is_split_rather_than_dropped(self, slack_client: FakeSlackClient) -> None:
        chatty = Agent(TestModel(custom_output_text='x' * 14_001))
        await dispatch_message(build(chatty), message(), slack_client, bot_user_id='U0BOT')
        posts = slack_client.method_calls('chat_postMessage')
        assert [len(str(post.kwargs['text'])) for post in posts] == [3500, 3500, 3500, 3500, 1]
        assert all(post.kwargs['mrkdwn'] is False for post in posts)

    async def test_output_keeps_markdown_but_neutralizes_slack_mentions(self, slack_client: FakeSlackClient) -> None:
        agent = Agent(TestModel(custom_output_text='**Ready** <@U123> <!channel> & done'))
        await dispatch_message(build(agent), message(), slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['markdown_text'] == (
            '**Ready** &lt;@U123&gt; &lt;!channel&gt; &amp; done'
        )

    async def test_a_second_message_waits_for_the_running_turn(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        # Two turns in one thread share a history. Running them at once means the
        # second loads the history the first has not finished writing.
        order: list[str] = []
        release = anyio.Event()
        slow = Agent(TestModel(custom_output_text='done'))

        @slow.instructions
        async def gate() -> str:
            order.append('enter')
            await release.wait()
            order.append('exit')
            return ''

        slack_agent = build(slow)
        first = message()
        second = message(ts='1700000000.000002', thread_ts='1700000000.000001')

        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: dispatch_message(slack_agent, first, slack_client, bot_user_id='U0BOT'))
            while order != ['enter']:
                await anyio.sleep(0)
            tg.start_soon(lambda: dispatch_message(slack_agent, second, slack_client, bot_user_id='U0BOT'))
            # Real time, not a yield count: the second turn has to get far enough
            # to reach the agent run before "it did not start" means anything.
            await anyio.sleep(0.2)
            assert order == ['enter'], 'the second turn started while the first was still running'
            release.set()

        assert order == ['enter', 'exit', 'enter', 'exit']

    async def test_an_undelivered_reply_is_not_written_into_the_history(
        self,
        slack_agent_with_store: tuple[SlackApp[None], InMemoryConversationStore],
        slack_client: FakeSlackClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Saving first would leave the next turn answering a message nobody saw.
        slack_agent, store = slack_agent_with_store
        slack_client.recorder.post_error = RuntimeError('slack is down')
        await dispatch_message(slack_agent, message(), slack_client, bot_user_id='U0BOT')
        assert list(await store.load('T1:C123:1700000000.000001')) == []
        assert 'Could not post the error reply' in caplog.text

    async def test_a_failing_run_says_so_in_the_thread(
        self, slack_client: FakeSlackClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        broken = Agent(TestModel(custom_output_text='ok'))

        @broken.instructions
        def explode() -> str:
            raise RuntimeError('no instructions today')

        await dispatch_message(build(broken, error_reply='It broke.'), message(), slack_client, bot_user_id='U0BOT')

        assert slack_client.method_calls('chat_postMessage')[0].kwargs['markdown_text'] == 'It broke.'
        assert 'Slack agent run failed' in caplog.text

    async def test_each_message_uses_the_invoking_users_mcp_token(
        self, monkeypatch: pytest.MonkeyPatch, slack_client: FakeSlackClient
    ) -> None:
        authorizations = fake_mcp(monkeypatch)
        agent = Agent(TestModel(custom_output_text='done'), capabilities=[Slack()])
        bot = build(agent, access=SlackAccess.workspace())
        await dispatch_message(
            bot,
            message(user='U1', ts='1.1'),
            slack_client,
            bot_user_id='U0BOT',
            user_token='xoxp-first',
        )
        await dispatch_message(
            bot,
            message(user='U2', ts='1.2'),
            slack_client,
            bot_user_id='U0BOT',
            user_token='xoxp-second',
        )
        assert authorizations == [
            {'Authorization': 'Bearer xoxp-first'},
            {'Authorization': 'Bearer xoxp-second'},
        ]


class TestHttpApp:
    def test_mounts_the_bolt_app_at_the_given_path(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str]
    ) -> None:
        bot = build(agent, signing_secret='shhh')
        mounted = bot.http_app(path='/hooks/slack')
        assert mounted.app is bot.app
        assert mounted.path == '/hooks/slack'

    def test_defaults_to_the_path_slack_suggests(self, agent: Agent[None, str]) -> None:
        assert build(agent, signing_secret='shhh').http_app().path == '/slack/events'

    async def test_root_mount_preserves_the_full_bolt_event_path(self, agent: Agent[None, str]) -> None:
        bolt = RealAsyncApp(token='xoxb-test', signing_secret='shhh')
        slack_app = SlackApp(agent, app=bolt, access=SlackAccess.users('U0ASKER'))
        combined_app = Starlette(
            routes=[Mount('/', app=slack_app.http_app())]  # pyright: ignore[reportArgumentType] - Bolt's ASGI types differ
        )

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=combined_app), base_url='http://test') as client:
            response = await client.post('/slack/events', content='{}')

        # Invalid Slack authentication reaches Bolt. A mismatched mount returns 404.
        assert response.status_code == 401


class TestRetries:
    async def test_a_redelivered_event_does_not_run_twice(
        self,
        slack_agent_with_store: tuple[SlackApp[None], InMemoryConversationStore],
        slack_client: FakeSlackClient,
    ) -> None:
        # Slack retries what it thinks was not delivered. For an agent with write
        # access, running it again means doing the work again.
        slack_bot, store = slack_agent_with_store
        for _ in range(2):
            await dispatch_message(slack_bot, message(), slack_client, bot_user_id='U0BOT')
        assert len(slack_client.method_calls('chat_postMessage')) == 1
        assert len(await store.load('T1:C123:1700000000.000001')) == 2

    async def test_a_different_event_still_runs(self, agent: Agent[None, str], slack_client: FakeSlackClient) -> None:
        bot = build(agent)
        await dispatch_message(bot, message(), slack_client, bot_user_id='U0BOT')
        await dispatch_message(bot, message(ts='1700000000.000002'), slack_client, bot_user_id='U0BOT')
        assert len(slack_client.method_calls('chat_postMessage')) == 2


class TestListeners:
    async def test_a_direct_message_runs_without_a_mention(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['message'](
            event=message(text='ship it', channel_type='im'),
            client=slack_client,
            context={'bot_user_id': 'U0BOT'},
            body={},
        )
        assert slack_client.method_calls('chat_postMessage')

    async def test_a_group_dm_requires_a_mention_to_start(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['message'](
            event=message(text='ship it', channel_type='mpim'),
            client=slack_client,
            context={'bot_user_id': 'U0BOT'},
            body={},
        )
        assert slack_client.calls == []

    async def test_a_group_dm_mention_starts_a_thread(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['message'](
            event=message(channel_type='mpim'),
            client=slack_client,
            context={'bot_user_id': 'U0BOT'},
            body={},
        )
        assert slack_client.method_calls('chat_postMessage')

    async def test_slacks_stop_button_cancels_the_active_run(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        entered = anyio.Event()
        slow = Agent(TestModel(custom_output_text='should not be posted'))

        @slow.instructions
        async def wait_forever() -> str:
            entered.set()
            await anyio.sleep_forever()
            return ''

        build(slow)
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: bolt().events['app_mention'](
                    event=message(),
                    client=slack_client,
                    context={'bot_user_id': 'U0BOT', 'team_id': 'T1'},
                    body={},
                )
            )
            await entered.wait()
            await bolt().events['agent_session_stopped'](
                event={'channel': 'C123', 'thread_ts': '1700000000.000001', 'user': 'U0ASKER'},
                client=slack_client,
                context={'team_id': 'T1'},
            )

        assert [call.kwargs['markdown_text'] for call in slack_client.method_calls('chat_postMessage')] == ['Stopped.']
        assert [call.kwargs['status'] for call in slack_client.method_calls('agents_sessions_setStatus')] == [
            'processing',
            'active',
        ]

    async def test_a_user_outside_the_access_policy_cannot_stop_the_active_run(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        entered = anyio.Event()
        release = anyio.Event()
        slow = Agent(TestModel(custom_output_text='done'))

        @slow.instructions
        async def wait_for_release() -> str:
            entered.set()
            await release.wait()
            return ''

        build(slow)
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: bolt().events['app_mention'](
                    event=message(),
                    client=slack_client,
                    context={'bot_user_id': 'U0BOT', 'team_id': 'T1'},
                    body={},
                )
            )
            await entered.wait()
            await bolt().events['agent_session_stopped'](
                event={'channel': 'C123', 'thread_ts': '1700000000.000001', 'user': 'U0OTHER'},
                client=slack_client,
                context={'team_id': 'T1'},
            )
            release.set()

        assert [call.kwargs['markdown_text'] for call in slack_client.method_calls('chat_postMessage')] == ['done']

    async def test_another_allowed_user_can_stop_the_active_run(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        entered = anyio.Event()
        release = anyio.Event()
        slow = Agent(TestModel(custom_output_text='done'))

        @slow.instructions
        async def wait_for_release() -> str:
            entered.set()
            await release.wait()
            return ''

        build(slow, access=SlackAccess.users('U0ASKER', 'U0OTHER'))
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: bolt().events['app_mention'](
                    event=message(),
                    client=slack_client,
                    context={'bot_user_id': 'U0BOT', 'team_id': 'T1'},
                    body={},
                )
            )
            await entered.wait()
            await bolt().events['agent_session_stopped'](
                event={'channel': 'C123', 'thread_ts': '1700000000.000001', 'user': 'U0OTHER'},
                client=slack_client,
                context={'team_id': 'T1'},
            )

        assert [call.kwargs['markdown_text'] for call in slack_client.method_calls('chat_postMessage')] == ['Stopped.']

    async def test_malformed_or_idle_stop_events_do_nothing(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['app_context_changed'](event={})
        await bolt().events['agent_session_stopped'](event={}, client=slack_client, context={})
        await bolt().events['agent_session_stopped'](
            event={'channel': 'C123', 'thread_ts': '1.1', 'user': 'U0ASKER'},
            client=slack_client,
            context={'team_id': 'T1'},
        )
        assert slack_client.calls == []

    async def test_a_channel_message_without_a_mention_is_left_alone(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['message'](
            event=message(text='ship it', channel_type='channel'), client=slack_client, context={}, body={}
        )
        assert slack_client.calls == []

    async def test_a_thread_reply_runs_without_another_mention_after_the_agent_joined(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT', 'team_id': 'T1'}, body={}
        )
        await bolt().events['message'](
            event=message(text='what about today?', ts='1700000000.000002', thread_ts='1700000000.000001'),
            client=slack_client,
            context={'bot_user_id': 'U0BOT', 'team_id': 'T1'},
            body={},
        )
        assert len(slack_client.method_calls('chat_postMessage')) == 2

    async def test_a_mention_delivered_to_both_listeners_runs_once(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        event = message(thread_ts='1700000000.000000')
        context = {'bot_user_id': 'U0BOT', 'team_id': 'T1'}
        await bolt().events['app_mention'](event=event, client=slack_client, context=context, body={'event_id': 'Ev1'})
        await bolt().events['message'](event=event, client=slack_client, context=context, body={'event_id': 'Ev2'})
        assert len(slack_client.method_calls('chat_postMessage')) == 1

    @pytest.mark.parametrize('subtype', ['file_share', 'thread_broadcast', 'me_message'])
    async def test_user_authored_message_subtypes_continue_an_engaged_thread(
        self,
        subtype: str,
        bolt: Callable[[], FakeBoltApp],
        agent: Agent[None, str],
        slack_client: FakeSlackClient,
    ) -> None:
        build(agent)
        await bolt().events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT', 'team_id': 'T1'}, body={}
        )
        await bolt().events['message'](
            event=message(
                text='see this',
                ts='1700000000.000002',
                thread_ts='1700000000.000001',
                subtype=subtype,
            ),
            client=slack_client,
            context={'bot_user_id': 'U0BOT', 'team_id': 'T1'},
            body={},
        )
        assert len(slack_client.method_calls('chat_postMessage')) == 2

    async def test_a_file_shared_without_text_still_continues_an_engaged_thread(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT', 'team_id': 'T1'}, body={}
        )
        await bolt().events['message'](
            event=message(
                text='',
                ts='1700000000.000002',
                thread_ts='1700000000.000001',
                files=[{'id': 'F123', 'name': 'report.pdf', 'mimetype': 'application/pdf'}],
            ),
            client=slack_client,
            context={'bot_user_id': 'U0BOT', 'team_id': 'T1'},
            body={},
        )
        assert len(slack_client.method_calls('chat_postMessage')) == 2

    async def test_a_reply_in_an_unrelated_channel_thread_is_left_alone(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['message'](
            event=message(text='team discussion', thread_ts='1700000000.000000'),
            client=slack_client,
            context={'team_id': 'T1'},
            body={},
        )
        assert slack_client.calls == []

    async def test_a_mention_runs(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
        )
        assert slack_client.method_calls('chat_postMessage')

    async def test_missing_mcp_authorization_gets_an_actionable_reply(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('SLACK_MCP_TOKEN', raising=False)
        monkeypatch.delenv('SLACK_INSTALL_URL', raising=False)
        build(Agent(TestModel(), capabilities=[Slack()]))
        await bolt().events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
        )
        reply = slack_client.method_calls('chat_postMessage')[0].kwargs['markdown_text']
        assert isinstance(reply, str)
        assert 'Slack workspace access is not connected' in reply

    async def test_missing_mcp_authorization_links_the_users_oauth_flow(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('SLACK_MCP_TOKEN', raising=False)
        build(
            Agent(TestModel(), capabilities=[Slack()]),
            install_url='https://agent.example/slack/install',
        )
        await bolt().events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
        )
        reply = slack_client.method_calls('chat_postMessage')[0].kwargs['markdown_text']
        assert isinstance(reply, str)
        assert 'https://agent.example/slack/install' in reply

    async def test_oauth_app_does_not_fall_back_to_a_process_wide_user_token(
        self, slack_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SLACK_MCP_TOKEN', 'xoxp-someone-else')
        configured = FakeBoltApp(token='resolved-by-oauth')
        SlackApp(  # pyright: ignore[reportCallIssue] - fake stands in for the configured Bolt app
            Agent(TestModel(), capabilities=[Slack()]),
            app=configured,  # pyright: ignore[reportArgumentType] - fake records Bolt listener registration
            access=SlackAccess.users('U0ASKER'),
        )
        await configured.events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
        )
        reply = slack_client.method_calls('chat_postMessage')[0].kwargs['markdown_text']
        assert isinstance(reply, str)
        assert 'Slack workspace access is not connected' in reply

    async def test_workspace_app_does_not_adopt_an_environment_token_added_later(
        self, slack_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('SLACK_MCP_TOKEN', raising=False)
        bot = build(Agent(TestModel(), capabilities=[Slack()]), access=SlackAccess.workspace())
        monkeypatch.setenv('SLACK_MCP_TOKEN', 'xoxp-someone-else')
        await dispatch_message(bot, message(), slack_client, bot_user_id='U0BOT')
        reply = slack_client.method_calls('chat_postMessage')[0].kwargs['markdown_text']
        assert isinstance(reply, str)
        assert 'Slack workspace access is not connected' in reply

    async def test_the_workspace_comes_from_the_listener_context(
        self,
        bolt: Callable[[], FakeBoltApp],
        slack_agent_with_store: tuple[SlackApp[None], InMemoryConversationStore],
        slack_client: FakeSlackClient,
    ) -> None:
        # Bolt puts the workspace on the context; the event body may not carry it,
        # and without it two workspaces share one thread's history.
        _, store = slack_agent_with_store
        event = message()
        del event['team']
        await bolt().events['app_mention'](event=event, client=slack_client, context={'team_id': 'T9'}, body={})
        assert len(await store.load('T9:C123:1700000000.000001')) == 2


class TestPromptClicks:
    def _body(self, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            'actions': [{'block_id': 'T1:C123:1.1#1', 'value': 'Yes'}],
            'user': {'id': 'U0ASKER'},
        }
        body.update(overrides)
        return body

    async def test_a_click_reaches_the_waiting_prompt(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        ran: list[str] = []
        agent = Agent(TestModel(call_tools=['change_record']), capabilities=[Slack(tools=SlackTools.none())])

        @agent.tool_plain(requires_approval=True)
        def change_record() -> str:
            ran.append('changed')
            return 'changed'

        build(agent)
        acked: list[bool] = []

        async def ack() -> None:
            acked.append(True)

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: bolt().events['app_mention'](
                    event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
                )
            )
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            prompt = slack_client.method_calls('chat_postMessage')[0]
            assert prompt.kwargs['channel'] == 'D-U0ASKER'
            assert prompt.kwargs['thread_ts'] is None
            assert 'Workspace: T1\nChannel: C123\nThread: 1700000000.000001\nRequested by: U0ASKER' in str(
                prompt.kwargs['text']
            )
            body = self._body()
            body['actions'] = [{'block_id': prompt_block_id(slack_client), 'value': 'Approve'}]
            await bolt().actions[0](ack=ack, body=body)
        assert acked == [True]
        assert ran == ['changed']
        assert (
            slack_client.method_calls('chat_postMessage')[-1].kwargs['markdown_text'] == '{"change_record":"changed"}'
        )
        assert [call.kwargs['status'] for call in slack_client.method_calls('agents_sessions_setStatus')] == [
            'processing',
            'suspended',
            'processing',
            'active',
        ]

    async def test_denying_keeps_the_tool_from_running(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.status_error = RuntimeError('Status UI unavailable')
        ran: list[int] = []
        agent = Agent(TestModel(call_tools=['change_record']), capabilities=[Slack(tools=SlackTools.none())])

        @agent.tool_plain(requires_approval=True)
        def change_record(number: int) -> str:  # pragma: no cover - denial prevents execution
            ran.append(number)
            return 'changed'

        build(agent)

        async def ack() -> None:
            pass

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: bolt().events['app_mention'](
                    event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
                )
            )
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            body = self._body()
            body['actions'] = [{'block_id': prompt_block_id(slack_client), 'value': 'Deny'}]
            await bolt().actions[0](ack=ack, body=body)

        assert ran == []
        assert 'chose Deny' in str(slack_client.method_calls('chat_update')[0].kwargs['text'])

    async def test_an_unanswered_approval_expires_without_running_the_tool(self, slack_client: FakeSlackClient) -> None:
        ran: list[int] = []
        agent = Agent(
            TestModel(call_tools=['change_record']),
            capabilities=[Slack(tools=SlackTools.none(), approval_timeout_seconds=0.01)],
        )

        @agent.tool_plain(requires_approval=True)
        def change_record(number: int) -> str:  # pragma: no cover - timeout prevents execution
            ran.append(number)
            return 'changed'

        await dispatch_message(build(agent), message(), slack_client, bot_user_id='U0BOT')
        assert ran == []
        assert 'expired' in str(slack_client.method_calls('chat_update')[0].kwargs['text'])

    async def test_an_approval_too_long_to_review_is_denied_without_buttons(
        self, slack_client: FakeSlackClient
    ) -> None:
        ran: list[str] = []
        requested = False

        def request_long_call(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal requested
            if requested:
                return ModelResponse(parts=[TextPart('done')])
            requested = True
            return ModelResponse(
                parts=[ToolCallPart('write_record', {'body': 'x' * 4000}, tool_call_id='write-record')]
            )

        agent = Agent(
            FunctionModel(request_long_call),
            capabilities=[Slack(tools=SlackTools.none())],
        )

        @agent.tool_plain(requires_approval=True)
        def write_record(body: str) -> str:  # pragma: no cover - oversized approval prevents execution
            ran.append(body)
            return 'written'

        await dispatch_message(build(agent), message(), slack_client, bot_user_id='U0BOT')
        assert ran == []
        assert all(call.kwargs['blocks'] is None for call in slack_client.method_calls('chat_postMessage'))

    async def test_missing_prompt_timestamp_fails_without_running_the_tool(self, slack_client: FakeSlackClient) -> None:
        ran: list[str] = []
        agent = Agent(TestModel(call_tools=['change_record']), capabilities=[Slack(tools=SlackTools.none())])

        @agent.tool_plain(requires_approval=True)
        def change_record() -> str:  # pragma: no cover - malformed Slack response prevents execution
            ran.append('changed')
            return 'changed'

        slack_client.recorder.post_response = {'ok': True}
        await dispatch_message(
            build(agent, error_reply='Approval failed.'), message(), slack_client, bot_user_id='U0BOT'
        )
        assert ran == []
        assert slack_client.method_calls('chat_postMessage')[-1].kwargs['markdown_text'] == 'Approval failed.'

    async def test_only_an_explicit_approver_can_answer(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        ran: list[str] = []
        agent = Agent(
            TestModel(call_tools=['change_record']),
            capabilities=[Slack(tools=SlackTools.none(), approver_ids=['U0REVIEWER', 'U0BACKUP'])],
        )

        @agent.tool_plain(requires_approval=True)
        def change_record() -> str:
            ran.append('changed')
            return 'changed'

        build(agent)

        async def ack() -> None:
            pass

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: bolt().events['app_mention'](
                    event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
                )
            )
            while len(slack_client.method_calls('chat_postMessage')) < 2:
                await anyio.sleep(0)
            assert {call.kwargs['channel'] for call in slack_client.method_calls('chat_postMessage')} == {
                'D-U0BACKUP',
                'D-U0REVIEWER',
            }
            block_id = prompt_block_id(slack_client)
            requester = self._body(user={'id': 'U0ASKER'})
            requester['actions'] = [{'block_id': block_id, 'value': 'Approve'}]
            await bolt().actions[0](ack=ack, body=requester)
            assert ran == []
            reviewer = self._body(user={'id': 'U0REVIEWER'})
            reviewer['actions'] = [{'block_id': block_id, 'value': 'Approve'}]
            await bolt().actions[0](ack=ack, body=reviewer)

        assert ran == ['changed']
        assert len(slack_client.method_calls('chat_update')) == 2

    async def test_partial_approver_delivery_failure_abandons_existing_prompts(
        self, slack_client: FakeSlackClient
    ) -> None:
        ran: list[str] = []
        agent = Agent(
            TestModel(call_tools=['change_record']),
            capabilities=[Slack(tools=SlackTools.none(), approver_ids=['U0FIRST', 'U0SECOND'])],
        )

        @agent.tool_plain(requires_approval=True)
        def change_record() -> str:  # pragma: no cover - prompt setup failure prevents execution
            ran.append('changed')
            return 'changed'

        slack_client.recorder.open_error_user_id = 'U0SECOND'
        await dispatch_message(
            build(agent, error_reply='Approval failed.'), message(), slack_client, bot_user_id='U0BOT'
        )

        assert ran == []
        private_prompt = slack_client.method_calls('chat_postMessage')[0]
        assert private_prompt.kwargs['channel'] == 'D-U0FIRST'
        assert private_prompt.kwargs['blocks'] is not None
        settled = slack_client.method_calls('chat_update')
        assert len(settled) == 1
        assert settled[0].kwargs['channel'] == 'D-U0FIRST'
        assert settled[0].kwargs['blocks'] == []
        assert 'abandoned before a decision' in str(settled[0].kwargs['text'])
        assert slack_client.method_calls('chat_postMessage')[-1].kwargs['markdown_text'] == 'Approval failed.'

    async def test_approval_fails_closed_when_slack_cannot_open_the_private_conversation(
        self, slack_client: FakeSlackClient
    ) -> None:
        ran: list[str] = []
        agent = Agent(TestModel(call_tools=['change_record']), capabilities=[Slack(tools=SlackTools.none())])

        @agent.tool_plain(requires_approval=True)
        def change_record() -> str:  # pragma: no cover - malformed Slack response prevents execution
            ran.append('changed')
            return 'changed'

        slack_client.recorder.open_response = {'ok': True, 'channel': {}}
        await dispatch_message(
            build(agent, error_reply='Approval failed.'), message(), slack_client, bot_user_id='U0BOT'
        )
        assert ran == []
        assert slack_client.method_calls('chat_postMessage')[-1].kwargs['markdown_text'] == 'Approval failed.'

    async def test_approval_fails_closed_when_slack_cannot_remove_the_buttons(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        ran: list[int] = []
        agent = Agent(TestModel(call_tools=['change_record']), capabilities=[Slack(tools=SlackTools.none())])

        @agent.tool_plain(requires_approval=True)
        def change_record(number: int) -> str:  # pragma: no cover - failed settlement prevents execution
            ran.append(number)
            return 'changed'

        build(agent, error_reply='Approval could not be settled.')
        slack_client.recorder.update_error = RuntimeError('Slack update failed')

        async def ack() -> None:
            pass

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: bolt().events['app_mention'](
                    event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
                )
            )
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            body = self._body()
            body['actions'] = [{'block_id': prompt_block_id(slack_client), 'value': 'Approve'}]
            await bolt().actions[0](ack=ack, body=body)

        assert ran == []
        assert (
            slack_client.method_calls('chat_postMessage')[-1].kwargs['markdown_text']
            == 'Approval could not be settled.'
        )
        assert slack_client.recorder.update_attempts == 3

    async def test_a_valid_click_is_acknowledged_by_an_agent_without_slack(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str]
    ) -> None:
        build(agent)
        acked: list[bool] = []

        async def ack() -> None:
            acked.append(True)

        await bolt().actions[0](ack=ack, body=self._body())
        assert acked == [True]

    async def test_an_unknown_valid_prompt_click_is_acknowledged_and_ignored(
        self, bolt: Callable[[], FakeBoltApp]
    ) -> None:
        build(Agent(TestModel(), capabilities=[Slack(tools=SlackTools.none())]))
        acked: list[bool] = []

        async def ack() -> None:
            acked.append(True)

        await bolt().actions[0](ack=ack, body=self._body())
        assert acked == [True]

    @pytest.mark.parametrize(
        'body',
        [
            {'actions': [], 'user': {'id': 'U0ASKER'}},
            {'actions': 'not a list', 'user': {'id': 'U0ASKER'}},
            {'actions': [{'value': 'Yes'}], 'user': {'id': 'U0ASKER'}},
            {'actions': [{'block_id': 'b', 'value': 'Yes'}], 'user': 'not a mapping'},
            {'actions': [{'block_id': 'b', 'value': 'Yes'}]},
        ],
        ids=['empty', 'not a list', 'missing block id', 'user is not a mapping', 'no user'],
    )
    async def test_a_malformed_click_payload_is_acknowledged_and_dropped(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient, body: Mapping[str, object]
    ) -> None:
        build(Agent(TestModel(), capabilities=[Slack(tools=SlackTools.none())]))
        acked: list[bool] = []

        async def ack() -> None:
            acked.append(True)

        await bolt().actions[0](ack=ack, body=body)
        assert acked == [True]


class TestStarting:
    @pytest.fixture
    def socket_handler(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, str]]:
        started: list[tuple[object, str]] = []

        class FakeSocketModeHandler:
            def __init__(self, app: object, app_token: str) -> None:
                self._app = app
                self._app_token = app_token

            async def start_async(self) -> None:
                started.append((self._app, self._app_token))

        monkeypatch.setattr(slack_socket_mode, 'AsyncSocketModeHandler', FakeSocketModeHandler)
        return started

    async def test_start_connects_socket_mode_with_the_app_token(
        self,
        agent: Agent[None, str],
        socket_handler: list[tuple[object, str]],
    ) -> None:
        slack_agent = build(agent)
        await slack_agent.start()
        assert socket_handler == [(slack_agent.app, 'xapp-t')]

    def test_run_is_the_blocking_entry_point(
        self,
        agent: Agent[None, str],
        socket_handler: list[tuple[object, str]],
    ) -> None:
        build(agent).run()
        assert len(socket_handler) == 1


class TestDeps:
    async def test_an_agent_keeps_the_deps_it_already_had(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        seen: list[Warehouse] = []

        agent: Agent[Warehouse, str] = Agent(TestModel(call_tools=['record']), deps_type=Warehouse)

        @agent.tool
        def record(ctx: RunContext[Warehouse], note: str) -> str:
            seen.append(ctx.deps)
            return 'noted'

        build(agent, deps=Warehouse(dsn='postgres://one'))
        await bolt().events['app_mention'](event=message(), client=slack_client, context={}, body={})
        assert seen == [Warehouse(dsn='postgres://one')]

    async def test_deps_can_be_built_per_thread(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        seen: list[Warehouse] = []

        agent: Agent[Warehouse, str] = Agent(TestModel(call_tools=['record']), deps_type=Warehouse)

        @agent.tool
        def record(ctx: RunContext[Warehouse], note: str) -> str:
            seen.append(ctx.deps)
            return 'noted'

        build(agent, deps_factory=warehouse_for)
        await bolt().events['app_mention'](event=message(), client=slack_client, context={}, body={})
        assert seen == [Warehouse(dsn='postgres://C123')]

    def test_a_value_and_a_factory_together_is_refused(self, agent: Agent[None, str]) -> None:
        # Both set means one is being ignored, and which one is not obvious.
        with pytest.raises(ValueError, match='not both'):
            build(agent, deps=Warehouse(dsn='x'), deps_factory=warehouse_for)

    async def test_public_slack_context_excludes_the_invoking_user_token(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        seen: list[SlackContext | None] = []

        def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(current_slack_context())
            return ModelResponse(parts=[TextPart('done')])

        build(Agent(FunctionModel(record)))
        await bolt().events['app_mention'](
            event=message(),
            client=slack_client,
            context={'team_id': 'T1', 'enterprise_id': 'E1', 'user_token': 'xoxp-user'},
            body={},
        )
        context = seen[0]
        assert context is not None
        assert not hasattr(context, 'user_token')
        assert context.enterprise_id == 'E1'
        assert current_slack_context() is None

    async def test_agent_view_context_is_typed_and_keeps_relevance_order(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        seen: list[SlackContext | None] = []

        def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(current_slack_context())
            return ModelResponse(parts=[TextPart('done')])

        build(Agent(FunctionModel(record)))
        event = message(
            channel='D123',
            channel_type='im',
            text='summarize this',
            app_context={
                'entities': [
                    {'type': 'slack#/types/channel_id', 'value': 'C456', 'team_id': 'T1'},
                    {'type': 'slack#/types/thread_ts', 'value': '1700.5', 'team_id': 'T1'},
                    {
                        'type': 'slack#/types/message_context',
                        'value': {'channel_id': 'C999', 'message_ts': '1700.9'},
                        'team_id': 'T1',
                    },
                    {'type': 7, 'value': 'ignored'},
                ]
            },
        )
        await bolt().events['message'](event=event, client=slack_client, context={}, body={})
        context = seen[0]
        assert context is not None
        assert context.active_entities == (
            SlackContextEntity(entity_type='slack#/types/channel_id', value='C456', team_id='T1'),
            SlackContextEntity(entity_type='slack#/types/thread_ts', value='1700.5', team_id='T1'),
            SlackContextEntity(
                entity_type='slack#/types/message_context',
                value=SlackMessageContext(channel_id='C999', message_ts='1700.9'),
                team_id='T1',
            ),
        )

    async def test_attached_files_are_typed_and_named_in_the_prompt(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        seen: list[tuple[str, SlackContext | None]] = []

        def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            part = messages[-1].parts[-1]
            prompt = part.content if isinstance(part, UserPromptPart) and isinstance(part.content, str) else ''
            seen.append((prompt, current_slack_context()))
            return ModelResponse(parts=[TextPart('done')])

        build(Agent(FunctionModel(record)))
        await bolt().events['app_mention'](
            event=message(
                text='<@U0BOT> summarize this',
                subtype='file_share',
                files=[{'id': 'F123', 'name': 'report.pdf', 'mimetype': 'application/pdf'}],
            ),
            client=slack_client,
            context={'bot_user_id': 'U0BOT'},
            body={},
        )
        assert seen == [
            (
                'summarize this',
                SlackContext(
                    channel_id='C123',
                    thread_ts='1700000000.000001',
                    message_ts='1700000000.000001',
                    user_id='U0ASKER',
                    team_id='T1',
                    files=(SlackFile(file_id='F123', name='report.pdf', mimetype='application/pdf'),),
                ),
            )
        ]


class TestFindingTheCapability:
    def test_a_slack_capability_nested_in_a_combined_capability_is_found(self) -> None:
        chat = Slack()
        agent: Agent[None, str] = Agent(TestModel(custom_output_text='done'), capabilities=[Planning(), chat])
        build(agent)

    def test_two_slack_capabilities_are_refused(self) -> None:
        agent: Agent[None, str] = Agent(
            TestModel(custom_output_text='done'),
            capabilities=[Slack(), Slack()],
        )
        with pytest.raises(ValueError, match='exactly one Slack capability'):
            build(agent)
