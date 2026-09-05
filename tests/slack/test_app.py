from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

import pydantic_ai_harness.slack as slack_package
import pydantic_ai_harness.slack._app as app_module
from pydantic_ai_harness.slack import (
    InMemoryConversationStore,
    SlackAgent,
    SlackInteractions,
    SlackThread,
)

from .conftest import FakeSlackClient

pytestmark = pytest.mark.anyio

Handler = Callable[..., Any]


@dataclass
class FakeBoltApp:
    """Records the listeners `SlackAgent` registers, without touching Slack."""

    token: str
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

    def factory(*, token: str) -> FakeBoltApp:
        app = FakeBoltApp(token=token)
        built.append(app)
        return app

    monkeypatch.setattr(app_module, 'AsyncApp', factory)
    return lambda: built[-1]


@pytest.fixture
def slack_agent_with_store(agent: Agent[SlackThread, str]) -> tuple[SlackAgent, InMemoryConversationStore]:
    store = InMemoryConversationStore()
    return build(agent, store=store), store


@pytest.fixture
def agent() -> Agent[SlackThread, str]:
    return Agent(TestModel(custom_output_text='done'), deps_type=SlackThread)


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


def build(agent: Agent[SlackThread, str], **kwargs: object) -> SlackAgent:
    defaults: dict[str, object] = {'bot_token': 'xoxb-t', 'app_token': 'xapp-t', 'allowed_user_ids': ['U0ASKER']}
    defaults.update(kwargs)
    return SlackAgent(agent, **defaults)  # pyright: ignore[reportArgumentType]


class TestLazyExport:
    def test_the_bolt_app_is_reachable_from_the_package(self) -> None:
        # Named on the package, imported only when asked for, so the rest of the
        # package stays usable without `slack-bolt`.
        assert slack_package.SlackAgent is SlackAgent

    def test_an_unknown_name_still_raises(self) -> None:
        with pytest.raises(AttributeError, match='has no attribute'):
            _ = slack_package.SlackBot  # pyright: ignore[reportAttributeAccessIssue]


class TestConfiguration:
    def test_reads_tokens_from_the_environment(
        self, agent: Agent[SlackThread, str], bolt: Callable[[], FakeBoltApp], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-env')
        monkeypatch.setenv('SLACK_APP_TOKEN', 'xapp-env')
        monkeypatch.setenv('SLACK_ALLOWED_USER_IDS', 'U1, U2 ,')
        slack_agent = SlackAgent(agent)
        assert bolt().token == 'xoxb-env'
        assert slack_agent._allowed_user_ids == frozenset({'U1', 'U2'})  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize('missing', ['SLACK_BOT_TOKEN', 'SLACK_APP_TOKEN'])
    def test_a_missing_token_is_refused(
        self,
        agent: Agent[SlackThread, str],
        monkeypatch: pytest.MonkeyPatch,
        missing: str,
    ) -> None:
        monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-env')
        monkeypatch.setenv('SLACK_APP_TOKEN', 'xapp-env')
        monkeypatch.delenv(missing)
        with pytest.raises(ValueError, match=missing):
            SlackAgent(agent)

    def test_a_string_allowlist_is_refused(self, agent: Agent[SlackThread, str]) -> None:
        with pytest.raises(ValueError, match='not a string'):
            build(agent, allowed_user_ids='U0ASKER')

    def test_an_empty_allowlist_warns(
        self,
        agent: Agent[SlackThread, str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv('SLACK_ALLOWED_USER_IDS', raising=False)
        build(agent, allowed_user_ids=None)
        assert 'anyone who can reach this bot' in caplog.text

    def test_registers_the_listeners_a_slack_agent_needs(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[SlackThread, str]
    ) -> None:
        build(agent)
        assert set(bolt().events) == {'app_mention', 'message'}
        assert len(bolt().actions) == 1


class TestHandleMessage:
    async def test_runs_the_agent_and_replies_in_the_thread(
        self, agent: Agent[SlackThread, str], slack_client: FakeSlackClient
    ) -> None:
        await build(agent).handle_message(message(), slack_client, bot_user_id='U0BOT')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert call.kwargs['text'] == 'done'
        assert call.kwargs['thread_ts'] == '1700000000.000001'

    async def test_a_reply_continues_the_thread_it_arrived_in(
        self, agent: Agent[SlackThread, str], slack_client: FakeSlackClient
    ) -> None:
        event = message(ts='1700000000.000009', thread_ts='1700000000.000001')
        await build(agent).handle_message(event, slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['thread_ts'] == '1700000000.000001'

    async def test_history_accumulates_across_turns_in_one_thread(
        self, slack_agent_with_store: tuple[SlackAgent, InMemoryConversationStore], slack_client: FakeSlackClient
    ) -> None:
        slack_agent, store = slack_agent_with_store
        await slack_agent.handle_message(message(), slack_client, bot_user_id='U0BOT')
        reply_in_same_thread = message(ts='1700000000.000002', thread_ts='1700000000.000001')
        await slack_agent.handle_message(reply_in_same_thread, slack_client, bot_user_id='U0BOT')
        assert len(await store.load('T1:C123:1700000000.000001')) == 4

    async def test_a_separate_thread_keeps_its_own_history(
        self, slack_agent_with_store: tuple[SlackAgent, InMemoryConversationStore], slack_client: FakeSlackClient
    ) -> None:
        slack_agent, store = slack_agent_with_store
        await slack_agent.handle_message(message(), slack_client, bot_user_id='U0BOT')
        await slack_agent.handle_message(message(ts='1700000000.000002'), slack_client, bot_user_id='U0BOT')
        assert len(await store.load('T1:C123:1700000000.000001')) == 2
        assert len(await store.load('T1:C123:1700000000.000002')) == 2

    async def test_only_the_bots_own_mention_is_stripped(
        self, agent: Agent[SlackThread, str], slack_client: FakeSlackClient
    ) -> None:
        captured: list[str] = []
        recording = Agent(TestModel(custom_output_text='done'), deps_type=SlackThread)

        @recording.instructions
        def remember(ctx: RunContext[SlackThread]) -> str:
            captured.append(ctx.prompt if isinstance(ctx.prompt, str) else '')
            return ''

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
        agent: Agent[SlackThread, str],
        slack_client: FakeSlackClient,
        event: dict[str, object],
    ) -> None:
        await build(agent).handle_message(event, slack_client, bot_user_id='U0BOT')
        assert slack_client.calls == []

    async def test_an_empty_answer_posts_nothing(self, slack_client: FakeSlackClient) -> None:
        quiet = Agent(TestModel(custom_output_text='   '), deps_type=SlackThread)
        await build(quiet).handle_message(message(), slack_client, bot_user_id='U0BOT')
        assert slack_client.calls == []

    async def test_a_long_answer_is_split_rather_than_dropped(self, slack_client: FakeSlackClient) -> None:
        chatty = Agent(TestModel(custom_output_text='x' * 7001), deps_type=SlackThread)
        await build(chatty).handle_message(message(), slack_client, bot_user_id='U0BOT')
        posts = slack_client.method_calls('chat_postMessage')
        assert [len(str(post.kwargs['text'])) for post in posts] == [3500, 3500, 1]

    async def test_a_second_message_waits_for_the_running_turn(
        self, agent: Agent[SlackThread, str], slack_client: FakeSlackClient
    ) -> None:
        # Two turns in one thread share a history. Running them at once means the
        # second loads the history the first has not finished writing.
        order: list[str] = []
        release = anyio.Event()
        slow = Agent(TestModel(custom_output_text='done'), deps_type=SlackThread)

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
        slack_agent_with_store: tuple[SlackAgent, InMemoryConversationStore],
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
        broken = Agent(TestModel(custom_output_text='ok'), deps_type=SlackThread)

        @broken.instructions
        def explode() -> str:
            raise RuntimeError('no instructions today')

        await build(broken, error_reply='It broke.').handle_message(message(), slack_client, bot_user_id='U0BOT')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['text'] == 'It broke.'
        assert 'Slack agent run failed' in caplog.text


class TestListeners:
    async def test_a_direct_message_runs_without_a_mention(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[SlackThread, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['message'](
            event=message(text='ship it', channel_type='im'), client=slack_client, context={'bot_user_id': 'U0BOT'}
        )
        assert slack_client.method_calls('chat_postMessage')

    async def test_a_channel_message_without_a_mention_is_left_alone(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[SlackThread, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['message'](
            event=message(text='ship it', channel_type='channel'), client=slack_client, context={}
        )
        assert slack_client.calls == []

    async def test_a_mention_runs(
        self, bolt: Callable[[], FakeBoltApp], agent: Agent[SlackThread, str], slack_client: FakeSlackClient
    ) -> None:
        build(agent)
        await bolt().events['app_mention'](event=message(), client=slack_client, context={'bot_user_id': 'U0BOT'})
        assert slack_client.method_calls('chat_postMessage')

    async def test_the_workspace_comes_from_the_listener_context(
        self,
        bolt: Callable[[], FakeBoltApp],
        slack_agent_with_store: tuple[SlackAgent, InMemoryConversationStore],
        slack_client: FakeSlackClient,
    ) -> None:
        # Bolt puts the workspace on the context; the event body may not carry it,
        # and without it two workspaces share one thread's history.
        _, store = slack_agent_with_store
        event = message()
        del event['team']
        await bolt().events['app_mention'](event=event, client=slack_client, context={'team_id': 'T9'})
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
        self, agent: Agent[SlackThread, str], bolt: Callable[[], FakeBoltApp]
    ) -> None:
        resolved: list[tuple[str, str, str]] = []

        class RecordingInteractions(SlackInteractions):
            def resolve(self, *, block_id: str, value: str, user_id: str) -> bool:
                resolved.append((block_id, value, user_id))
                return True

        build(agent, interactions=RecordingInteractions())
        acked: list[bool] = []

        async def ack() -> None:
            acked.append(True)

        await bolt().actions[0](ack=ack, body=self._body())
        assert acked == [True]
        assert resolved == [('T1:C123:1.1#1', 'Yes', 'U0ASKER')]

    def test_clicks_are_dropped_without_a_prompt_registry(self, agent: Agent[SlackThread, str]) -> None:
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
        self, agent: Agent[SlackThread, str], bolt: Callable[[], FakeBoltApp], body: Mapping[str, object]
    ) -> None:
        interactions = SlackInteractions()
        build(agent, interactions=interactions)
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
        agent: Agent[SlackThread, str],
        socket_handler: list[tuple[object, str]],
    ) -> None:
        slack_agent = build(agent)
        await slack_agent.start()
        assert socket_handler == [(slack_agent.app, 'xapp-t')]

    def test_run_is_the_blocking_entry_point(
        self,
        agent: Agent[SlackThread, str],
        socket_handler: list[tuple[object, str]],
    ) -> None:
        build(agent).run()
        assert len(socket_handler) == 1
