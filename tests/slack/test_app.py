from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from slack_sdk.web.async_client import AsyncWebClient

import pydantic_ai_harness.slack as slack_package
import pydantic_ai_harness.slack._app as app_module
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.slack import (
    InMemoryConversationStore,
    SlackBot,
    SlackChat,
    SlackInteractions,
    SlackThread,
    current_thread,
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
    """Records the listeners `SlackBot` registers, without touching Slack."""

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
def slack_agent_with_store(agent: Agent[None, str]) -> tuple[SlackBot[None], InMemoryConversationStore]:
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
    """An agent whose `SlackChat` is where the bot should find the registry."""
    return Agent(
        TestModel(custom_output_text='done'),
        capabilities=[SlackChat(client=client, ask_user=True, interactions=interactions)],
    )


def build(agent: Agent[DepsT, str], **kwargs: object) -> SlackBot[DepsT]:
    defaults: dict[str, object] = {'bot_token': 'xoxb-t', 'app_token': 'xapp-t', 'allowed_user_ids': ['U0ASKER']}
    defaults.update(kwargs)
    return SlackBot(agent, **defaults)  # pyright: ignore[reportArgumentType]


class TestLazyExport:
    def test_the_bolt_app_is_reachable_from_the_package(self) -> None:
        # Named on the package, imported only when asked for, so the rest of the
        # package stays usable without `slack-bolt`.
        assert slack_package.SlackBot is SlackBot

    def test_an_unknown_name_still_raises(self) -> None:
        with pytest.raises(AttributeError, match='has no attribute'):
            _ = slack_package.SlackWidget  # pyright: ignore[reportAttributeAccessIssue]


class TestConfiguration:
    def test_reads_tokens_from_the_environment(
        self, agent: Agent[None, str], bolt: Callable[[], FakeBoltApp], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-env')
        monkeypatch.setenv('SLACK_APP_TOKEN', 'xapp-env')
        monkeypatch.setenv('SLACK_ALLOWED_USER_IDS', 'U1, U2 ,')
        slack_agent = SlackBot(agent)
        assert bolt().token == 'xoxb-env'
        assert slack_agent._allowed_user_ids == frozenset({'U1', 'U2'})  # pyright: ignore[reportPrivateUsage]

    def test_a_missing_bot_token_is_refused(self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        with pytest.raises(ValueError, match='SLACK_BOT_TOKEN'):
            SlackBot(agent, app_token='xapp-t')

    async def test_socket_mode_asks_for_its_token_only_when_started(
        self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An Events API deployment has no app token and must still construct.
        monkeypatch.delenv('SLACK_APP_TOKEN', raising=False)
        bot = SlackBot(agent, bot_token='xoxb-t', signing_secret='s')
        with pytest.raises(ValueError, match='SLACK_APP_TOKEN'):
            await bot.start()

    def test_the_events_api_asks_for_its_secret_only_when_mounted(
        self, agent: Agent[None, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # And a Socket Mode deployment has no signing secret.
        monkeypatch.delenv('SLACK_SIGNING_SECRET', raising=False)
        bot = SlackBot(agent, bot_token='xoxb-t', app_token='xapp-t')
        with pytest.raises(ValueError, match='SLACK_SIGNING_SECRET'):
            bot.http_app()

    def test_the_signing_secret_reaches_bolt(self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str]) -> None:
        build(agent, signing_secret='shhh')
        assert bolt().signing_secret == 'shhh'

    def test_a_string_allowlist_is_refused(self, agent: Agent[None, str]) -> None:
        with pytest.raises(ValueError, match='one entry per character'):
            build(agent, allowed_user_ids='U0ASKER')

    def test_an_empty_allowlist_warns(
        self,
        agent: Agent[None, str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv('SLACK_ALLOWED_USER_IDS', raising=False)
        build(agent, allowed_user_ids=None)
        assert 'anyone who can reach this bot' in caplog.text

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
        self, slack_agent_with_store: tuple[SlackBot[None], InMemoryConversationStore], slack_client: FakeSlackClient
    ) -> None:
        slack_agent, store = slack_agent_with_store
        await slack_agent.handle_message(message(), slack_client, bot_user_id='U0BOT')
        reply_in_same_thread = message(ts='1700000000.000002', thread_ts='1700000000.000001')
        await slack_agent.handle_message(reply_in_same_thread, slack_client, bot_user_id='U0BOT')
        assert len(await store.load('T1:C123:1700000000.000001')) == 4

    async def test_a_separate_thread_keeps_its_own_history(
        self, slack_agent_with_store: tuple[SlackBot[None], InMemoryConversationStore], slack_client: FakeSlackClient
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
        slack_agent_with_store: tuple[SlackBot[None], InMemoryConversationStore],
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


class TestRetries:
    async def test_a_redelivered_event_does_not_run_twice(
        self,
        slack_agent_with_store: tuple[SlackBot[None], InMemoryConversationStore],
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

    async def test_remembering_events_stays_bounded(
        self, agent: Agent[None, str], slack_client: FakeSlackClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The oldest id is dropped rather than the set growing without limit, so a
        # long-lived bot does not keep one entry per message it ever saw. Shrunk
        # here so the test does not run a thousand turns to prove it.
        monkeypatch.setattr(app_module, '_REMEMBERED_EVENTS', 3)
        bot = build(agent)
        for index in range(4):
            await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id=f'Ev{index}')
        assert len(slack_client.method_calls('chat_postMessage')) == 4
        # `Ev0` has fallen out of the window, so it is no longer recognised.
        await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev0')
        assert len(slack_client.method_calls('chat_postMessage')) == 5
        # `Ev3` is still inside it.
        await bot.handle_message(message(), slack_client, bot_user_id='U0BOT', event_id='Ev3')
        assert len(slack_client.method_calls('chat_postMessage')) == 5


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

    async def test_a_mention_runs(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[None, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['app_mention'](
            event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'}, body={}
        )
        assert slack_client.method_calls('chat_postMessage')

    async def test_the_workspace_comes_from_the_listener_context(
        self,
        bolt: Callable[[], FakeBoltApp],
        slack_agent_with_store: tuple[SlackBot[None], InMemoryConversationStore],
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

    async def test_the_thread_is_bound_around_the_run(
        self, bolt: Callable[[], FakeBoltApp], slack_client: FakeSlackClient
    ) -> None:
        seen: list[SlackThread | None] = []

        def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(current_thread())
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(FunctionModel(record))

        build(agent)
        await bolt().events['app_mention'](event=message(), client=slack_client, context={}, body={})
        assert seen == [SlackThread(channel_id='C123', thread_ts='1700000000.000001', user_id='U0ASKER', team_id='T1')]
        # Unbound once the turn is over, so a later run elsewhere cannot inherit it.
        assert current_thread() is None


class TestFindingTheCapability:
    def test_a_slack_chat_nested_in_a_combined_capability_is_found(self, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        chat = SlackChat(client=slack_client, ask_user=True, interactions=interactions)
        agent: Agent[None, str] = Agent(TestModel(custom_output_text='done'), capabilities=[Planning(), chat])
        assert build(agent).resolve_prompt(block_id='nothing', value='Yes', user_id='U0ASKER') is False
        assert chat.resolve_interactions() is interactions

    def test_the_bots_token_reaches_a_capability_that_has_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Configuring the bot is enough; the capability does not need
        # SLACK_BOT_TOKEN set as well.
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        chat = SlackChat()
        build(Agent(TestModel(custom_output_text='done'), capabilities=[chat]))
        client = chat.resolve_client()
        assert isinstance(client, AsyncWebClient)
        assert client.token == 'xoxb-t'

    def test_the_capability_you_passed_in_is_not_written_to(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The bot's token is remembered beside the field, not put in it: a token
        # nobody configured should not turn up in an object the caller still holds.
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        chat = SlackChat()
        build(Agent(TestModel(custom_output_text='done'), capabilities=[chat]))
        assert chat.token is None

    def test_a_token_you_configured_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        chat = SlackChat(token='xoxb-mine')
        build(Agent(TestModel(custom_output_text='done'), capabilities=[chat]))
        client = chat.resolve_client()
        assert isinstance(client, AsyncWebClient)
        assert client.token == 'xoxb-mine'

    def test_two_slack_chats_warn_and_route_to_the_first(
        self, slack_client: FakeSlackClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent: Agent[None, str] = Agent(
            TestModel(custom_output_text='done'),
            capabilities=[SlackChat(client=slack_client), SlackChat(client=slack_client)],
        )
        build(agent)
        assert 'SlackChat capabilities' in caplog.text
