from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anyio
import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from slack_bolt.app.async_app import AsyncApp as RealAsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from starlette.applications import Starlette
from starlette.routing import Mount

import pydantic_ai_harness.slack as slack_package
import pydantic_ai_harness.slack._app as app_module
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.slack import (
    InMemoryConversationStore,
    Slack,
    SlackAccess,
    SlackApp,
    SlackContext,
    SlackContextEntity,
    SlackInteractions,
    SlackThread,
    current_slack_context,
)

from .conftest import FakeSlackClient

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

    monkeypatch.setattr(app_module, 'AsyncApp', factory)
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


def asking(interactions: SlackInteractions, client: FakeSlackClient) -> Agent[None, str]:
    """An agent whose `Slack` is where the bot should find the registry."""
    return Agent(
        TestModel(custom_output_text='done'),
        capabilities=[Slack(delivery_client=client, interactions=interactions)],
    )


def build(agent: Agent[DepsT, str], **kwargs: object) -> SlackApp[DepsT]:
    defaults: dict[str, object] = {'bot_token': 'xoxb-t', 'app_token': 'xapp-t', 'access': SlackAccess.users('U0ASKER')}
    defaults.update(kwargs)
    return SlackApp(agent, **defaults)  # pyright: ignore[reportArgumentType]


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
        slack_app = SlackApp(
            agent,
            app=configured,  # pyright: ignore[reportArgumentType] - fake records Bolt listener registration
            access=SlackAccess.users('U0ASKER'),
        )
        assert slack_app.app is configured

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
        SlackApp(
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
        assert set(bolt().events) == {'app_mention', 'message'}
        assert len(bolt().actions) == 1


class TestHandleMessage:
    async def test_runs_the_agent_and_replies_in_the_thread(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        await build(agent).handle_message(message(), slack_client, bot_user_id='U0BOT')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert call.kwargs['text'] == 'done'
        assert call.kwargs['thread_ts'] == '1700000000.000001'

    async def test_a_reply_continues_the_thread_it_arrived_in(
        self, agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        event = message(ts='1700000000.000009', thread_ts='1700000000.000001')
        await build(agent).handle_message(event, slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['thread_ts'] == '1700000000.000001'

    async def test_history_accumulates_across_turns_in_one_thread(
        self, slack_agent_with_store: tuple[SlackApp[None], InMemoryConversationStore], slack_client: FakeSlackClient
    ) -> None:
        slack_agent, store = slack_agent_with_store
        await slack_agent.handle_message(message(), slack_client, bot_user_id='U0BOT')
        reply_in_same_thread = message(ts='1700000000.000002', thread_ts='1700000000.000001')
        await slack_agent.handle_message(reply_in_same_thread, slack_client, bot_user_id='U0BOT')
        assert len(await store.load('T1:C123:1700000000.000001')) == 4

    async def test_a_separate_thread_keeps_its_own_history(
        self, slack_agent_with_store: tuple[SlackApp[None], InMemoryConversationStore], slack_client: FakeSlackClient
    ) -> None:
        slack_agent, store = slack_agent_with_store
        await slack_agent.handle_message(message(), slack_client, bot_user_id='U0BOT')
        await slack_agent.handle_message(message(ts='1700000000.000002'), slack_client, bot_user_id='U0BOT')
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
        await build(recording).handle_message(event, slack_client, bot_user_id='U0BOT')
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
        await build(agent).handle_message(event, slack_client, bot_user_id='U0BOT')
        assert slack_client.calls == []

    async def test_an_empty_answer_posts_nothing(self, slack_client: FakeSlackClient) -> None:
        quiet = Agent(TestModel(custom_output_text='   '))
        await build(quiet).handle_message(message(), slack_client, bot_user_id='U0BOT')
        assert slack_client.calls == []

    async def test_a_long_answer_is_split_rather_than_dropped(self, slack_client: FakeSlackClient) -> None:
        chatty = Agent(TestModel(custom_output_text='x' * 7001))
        await build(chatty).handle_message(message(), slack_client, bot_user_id='U0BOT')
        posts = slack_client.method_calls('chat_postMessage')
        assert [len(str(post.kwargs['text'])) for post in posts] == [3500, 3500, 1]

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
            tg.start_soon(lambda: slack_agent.handle_message(first, slack_client, bot_user_id='U0BOT'))
            while order != ['enter']:
                await anyio.sleep(0)
            tg.start_soon(lambda: slack_agent.handle_message(second, slack_client, bot_user_id='U0BOT'))
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
        await slack_agent.handle_message(message(), slack_client, bot_user_id='U0BOT')
        assert list(await store.load('T1:C123:1700000000.000001')) == []
        assert 'Could not post the error reply' in caplog.text

    async def test_a_failing_run_says_so_in_the_thread(
        self, slack_client: FakeSlackClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        broken = Agent(TestModel(custom_output_text='ok'))

        @broken.instructions
        def explode() -> str:
            raise RuntimeError('no instructions today')

        await build(broken, error_reply='It broke.').handle_message(message(), slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['text'] == 'It broke.'
        assert 'Slack agent run failed' in caplog.text


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
            await slack_bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev1')
        assert len(slack_client.method_calls('chat_postMessage')) == 1
        assert len(await store.load('T1:C123:1700000000.000001')) == 2

    async def test_a_different_event_still_runs(self, agent: Agent[None, str], slack_client: FakeSlackClient) -> None:
        bot = build(agent)
        await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev1')
        await bot.handle_message(message(ts='1700000000.000002'), slack_client, bot_user_id='U0BOT', event_id='Ev2')
        assert len(slack_client.method_calls('chat_postMessage')) == 2

    async def test_event_ids_are_retained_for_slacks_retry_window(
        self, agent: Agent[None, str], slack_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        times = iter([100.0, 100.0, 699.0, 701.0])
        monkeypatch.setattr(app_module, 'monotonic', lambda: next(times))
        bot = build(agent)
        await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev1')
        await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev1')
        assert len(slack_client.method_calls('chat_postMessage')) == 1
        await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev2')
        await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev1')
        assert len(slack_client.method_calls('chat_postMessage')) == 3

    async def test_engaged_thread_memory_is_bounded(
        self,
        bolt: Callable[[], FakeBoltApp],
        agent: Agent[None, str],
        slack_client: FakeSlackClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(app_module, '_REMEMBERED_THREADS', 1)
        store = InMemoryConversationStore()
        bot = build(agent, store=store)
        first = message(ts='1.1')
        second = message(ts='2.2')
        await bot.handle_message(first, slack_client, bot_user_id='U0BOT')
        await store.delete('T1:C123:1.1')
        await bot.handle_message(second, slack_client, bot_user_id='U0BOT')

        await bolt().events['message'](
            event=message(ts='1.2', thread_ts='1.1'),
            client=slack_client,
            context={'team_id': 'T1'},
            body={},
        )
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
        reply = slack_client.method_calls('chat_postMessage')[0].kwargs['text']
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
        reply = slack_client.method_calls('chat_postMessage')[0].kwargs['text']
        assert isinstance(reply, str)
        assert 'https://agent.example/slack/install' in reply

    async def test_oauth_app_does_not_fall_back_to_a_process_wide_user_token(
        self, slack_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SLACK_MCP_TOKEN', 'xoxp-someone-else')
        configured = FakeBoltApp(token='resolved-by-oauth')
        SlackApp(
            Agent(TestModel(), capabilities=[Slack()]),
            app=configured,  # pyright: ignore[reportArgumentType]
            access=SlackAccess.users('U0ASKER'),
        )
        await configured.events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
        )
        reply = slack_client.method_calls('chat_postMessage')[0].kwargs['text']
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
        resolved: list[tuple[str, str, str]] = []

        class RecordingInteractions(SlackInteractions):
            def resolve(self, *, block_id: str, value: str, user_id: str) -> bool:
                resolved.append((block_id, value, user_id))
                return True

        build(asking(RecordingInteractions(), slack_client))
        acked: list[bool] = []

        async def ack() -> None:
            acked.append(True)

        await bolt().actions[0](ack=ack, body=self._body())
        assert acked == [True]
        assert resolved == [('T1:C123:1.1#1', 'Yes', 'U0ASKER')]

    def test_clicks_are_dropped_by_an_agent_with_no_slack_chat(self, agent: Agent[None, str]) -> None:
        assert build(agent).resolve_prompt(block_id='b', value='Yes', user_id='U0ASKER') is False

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
        build(asking(SlackInteractions(), slack_client))
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

        monkeypatch.setattr(app_module, 'AsyncSocketModeHandler', FakeSocketModeHandler)
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

    async def test_the_typed_slack_context_includes_the_invoking_user_token(
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
        assert context.user_token == 'xoxp-user'
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
                    {'type': 7, 'value': 'ignored'},
                ]
            },
        )
        await bolt().events['message'](event=event, client=slack_client, context={}, body={})
        context = seen[0]
        assert context is not None
        assert context.active_entities == (
            SlackContextEntity('slack#/types/channel_id', 'C456', 'T1'),
            SlackContextEntity('slack#/types/thread_ts', '1700.5', 'T1'),
        )


class TestFindingTheCapability:
    def test_a_slack_capability_nested_in_a_combined_capability_is_found(self, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        chat = Slack(delivery_client=slack_client, interactions=interactions)
        agent: Agent[None, str] = Agent(TestModel(custom_output_text='done'), capabilities=[Planning(), chat])
        assert build(agent).resolve_prompt(block_id='nothing', value='Yes', user_id='U0ASKER') is False
        assert chat.resolve_interactions() is interactions

    def test_the_bots_token_reaches_a_capability_that_has_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Configuring the bot is enough; the capability does not need
        # SLACK_BOT_TOKEN set as well.
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        chat = Slack()
        build(Agent(TestModel(custom_output_text='done'), capabilities=[chat]))
        client = chat.resolve_client()
        assert isinstance(client, AsyncWebClient)
        assert client.token == 'xoxb-t'

    def test_the_capability_you_passed_in_is_not_written_to(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The bot's token is remembered beside the field, not put in it: a token
        # nobody configured should not turn up in an object the caller still holds.
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        chat = Slack()
        build(Agent(TestModel(custom_output_text='done'), capabilities=[chat]))
        assert chat.delivery_token is None

    def test_a_token_you_configured_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        chat = Slack(delivery_token='xoxb-mine')
        build(Agent(TestModel(custom_output_text='done'), capabilities=[chat]))
        client = chat.resolve_client()
        assert isinstance(client, AsyncWebClient)
        assert client.token == 'xoxb-mine'
        assert chat.resolve_client() is client

    def test_resolving_a_delivery_client_without_a_token_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        with pytest.raises(ValueError, match='Slack bot token is required'):
            Slack().resolve_client()

    def test_two_slack_capabilities_are_refused(self, slack_client: FakeSlackClient) -> None:
        agent: Agent[None, str] = Agent(
            TestModel(custom_output_text='done'),
            capabilities=[Slack(delivery_client=slack_client), Slack(delivery_client=slack_client)],
        )
        with pytest.raises(ValueError, match='exactly one Slack capability'):
            build(agent)
