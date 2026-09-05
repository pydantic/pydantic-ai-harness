from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.slack import (
    MAX_MESSAGE_CHARS,
    PlanStep,
    SlackChatToolset,
    SlackInteractions,
    SlackThread,
)

from .conftest import FakeSlackClient, prompt_block_id

pytestmark = pytest.mark.anyio


def context(run_id: str = 'run-1') -> RunContext[None]:
    # Deps stay None throughout: these tools read the thread from how the toolset
    # was built, never from the agent's deps.
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id)


def tool_names(toolset: SlackChatToolset[None]) -> set[str]:
    return set(toolset.tools)


def _plan_id(answer: str) -> str:
    """The plan id `post_plan` told the model to send back."""
    found = re.search(r"plan_id='([^']+)'", answer)
    assert found is not None, answer
    return found.group(1)


class TestRegistration:
    def test_ask_user_and_upload_file_are_opt_in(self, slack_client: FakeSlackClient) -> None:
        assert tool_names(SlackChatToolset(slack_client)) == {'post_message', 'post_plan', 'set_status'}

    def test_ask_user_appears_with_an_interactions_registry(self, slack_client: FakeSlackClient) -> None:
        assert 'ask_user' in tool_names(SlackChatToolset(slack_client, interactions=SlackInteractions()))

    def test_upload_file_appears_with_a_file_root(self, slack_client: FakeSlackClient, tmp_path: Path) -> None:
        assert 'upload_file' in tool_names(SlackChatToolset(slack_client, file_root=tmp_path))


class TestPostMessage:
    async def test_posts_into_the_thread(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        await SlackChatToolset(slack_client, thread=thread).post_message(context(), 'found the bad migration')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert call.kwargs == {
            'channel': 'C123',
            'thread_ts': thread.thread_ts,
            'text': 'found the bad migration',
            'blocks': None,
        }

    async def test_asks_the_model_to_split_an_oversized_message(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        with pytest.raises(ModelRetry, match='Send it as several shorter messages'):
            await SlackChatToolset(slack_client, thread=thread).post_message(context(), 'x' * (MAX_MESSAGE_CHARS + 1))

    async def test_rejects_an_empty_message(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        with pytest.raises(ModelRetry, match='cannot be empty'):
            await SlackChatToolset(slack_client, thread=thread).post_message(context(), '   ')

    async def test_reaches_slack_through_a_normal_agent_run(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # No deps at all: the toolset knows where it is talking without them.
        agent = Agent(
            TestModel(call_tools=['post_message']),
            toolsets=[SlackChatToolset(slack_client, thread=thread)],
        )
        await agent.run('go')
        assert slack_client.method_calls('chat_postMessage')


class TestPostPlan:
    async def test_the_returned_plan_id_edits_the_same_message(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        toolset = SlackChatToolset(slack_client, thread=thread)
        steps = [PlanStep(text='read the logs'), PlanStep(text='open a PR')]
        posted = await toolset.post_plan(context(), steps)
        assert slack_client.next_ts in posted

        steps[0].status = 'done'
        steps[1].status = 'running'
        await toolset.post_plan(context(), steps, plan_id=_plan_id(posted))

        assert len(slack_client.method_calls('chat_postMessage')) == 1
        update = slack_client.method_calls('chat_update')[0]
        assert update.kwargs['ts'] == slack_client.next_ts
        assert str(update.kwargs['text']).splitlines() == [
            ':white_check_mark: read the logs',
            ':hourglass_flowing_sand: open a PR',
        ]

    async def test_a_plan_without_an_id_posts_a_new_message(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        toolset = SlackChatToolset(slack_client, thread=thread)
        await toolset.post_plan(context(), [PlanStep(text='a')])
        await toolset.post_plan(context(), [PlanStep(text='b')])
        assert len(slack_client.method_calls('chat_postMessage')) == 2

    async def test_renders_every_status(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        steps = [
            PlanStep(text='a', status='pending'),
            PlanStep(text='b', status='running'),
            PlanStep(text='c', status='done'),
            PlanStep(text='d', status='failed'),
        ]
        await SlackChatToolset(slack_client, thread=thread).post_plan(context(), steps)
        text = str(slack_client.method_calls('chat_postMessage')[0].kwargs['text'])
        assert text.count(':white_circle:') == 1
        assert text.count(':x:') == 1

    async def test_tells_the_model_to_repost_when_slack_gave_no_id(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.post_response = {'ok': True}
        answer = await SlackChatToolset(slack_client, thread=thread).post_plan(context(), [PlanStep(text='a')])
        assert 'post a fresh plan' in answer

    async def test_rejects_an_empty_plan(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        with pytest.raises(ModelRetry, match='at least one step'):
            await SlackChatToolset(slack_client, thread=thread).post_plan(context(), [])

    async def test_rejects_an_overlong_plan(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        with pytest.raises(ModelRetry, match='20 steps or fewer'):
            await SlackChatToolset(slack_client, thread=thread).post_plan(context(), [PlanStep(text='a')] * 21)

    async def test_rejects_a_plan_slack_would_not_accept(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        with pytest.raises(ModelRetry, match='Slack takes at most'):
            await SlackChatToolset(slack_client, thread=thread).post_plan(context(), [PlanStep(text='x' * 400)] * 10)

    @pytest.mark.parametrize(
        'plan_id',
        ['1700000000.000100', 'made-up', 'e.abc', 'é.abc'],
        ids=['a real timestamp', 'nonsense', 'ascii junk', 'non-ascii junk'],
    )
    async def test_refuses_a_plan_id_this_thread_did_not_issue(
        self, thread: SlackThread, slack_client: FakeSlackClient, plan_id: str
    ) -> None:
        # A raw Slack timestamp is what a model would read off another message in
        # the channel. Accepting it would let a plan overwrite someone's prompt.
        with pytest.raises(ModelRetry, match='not a plan you posted here'):
            await SlackChatToolset(slack_client, thread=thread).post_plan(
                context(), [PlanStep(text='a')], plan_id=plan_id
            )
        assert slack_client.method_calls('chat_update') == []

    async def test_refuses_a_plan_id_from_an_earlier_turn(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # The id stays in the transcript, so the next turn can see it. It should
        # post its own checklist rather than edit the one before it.
        toolset = SlackChatToolset(slack_client, thread=thread)
        posted = await toolset.post_plan(context(run_id='run-1'), [PlanStep(text='a')])
        with pytest.raises(ModelRetry, match='not a plan you posted here'):
            await toolset.post_plan(context(run_id='run-2'), [PlanStep(text='a')], plan_id=_plan_id(posted))

    async def test_refuses_a_plan_id_issued_for_another_thread(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # One toolset, so one signing key: only the thread differs between the
        # plan being issued and the plan id being sent back.
        talking_to = [replace(thread, channel_id='C999')]
        toolset = SlackChatToolset(slack_client, thread=lambda ctx: talking_to[0])
        posted = await toolset.post_plan(context(), [PlanStep(text='a')])
        talking_to[0] = thread
        with pytest.raises(ModelRetry, match='not a plan you posted here'):
            await toolset.post_plan(context(), [PlanStep(text='a')], plan_id=_plan_id(posted))


class TestSetStatus:
    async def test_sets_the_status(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        assert (
            await SlackChatToolset(slack_client, thread=thread).set_status(context(), 'reading logs') == 'Status set.'
        )
        assert slack_client.method_calls('assistant_threads_setStatus')[0].kwargs['status'] == 'reading logs'

    async def test_a_channel_without_status_support_does_not_fail_the_turn(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.status_error = RuntimeError('not_allowed_in_channel')
        assert 'not available' in await SlackChatToolset(slack_client, thread=thread).set_status(
            context(), 'reading logs'
        )

    async def test_a_broken_status_call_is_logged_rather_than_hidden(
        self, thread: SlackThread, slack_client: FakeSlackClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The same answer covers a bad token and a rate limit, so the reason has
        # to be somewhere an operator can find it.
        caplog.set_level('INFO')
        slack_client.recorder.status_error = RuntimeError('invalid_auth')
        await SlackChatToolset(slack_client, thread=thread).set_status(context(), 'reading logs')
        assert 'Could not set the Slack status' in caplog.text
        assert 'invalid_auth' in caplog.text


class TestAskUser:
    async def test_unusable_options_come_back_as_something_the_model_can_fix(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        toolset = SlackChatToolset(slack_client, thread=thread, interactions=SlackInteractions(timeout_seconds=0.01))
        with pytest.raises(ModelRetry, match='must be unique'):
            await toolset.ask_user(context(), 'Pick', ['A', 'A'])

    async def test_returns_the_answer(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        toolset = SlackChatToolset(slack_client, thread=thread, interactions=interactions)
        answers: list[str] = []

        async with anyio.create_task_group() as tg:

            async def ask() -> None:
                answers.append(await toolset.ask_user(context(), 'Staging or prod?', ['staging', 'prod']))

            tg.start_soon(ask)
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            interactions.resolve(block_id=prompt_block_id(slack_client), value='staging', user_id='U0ASKER')

        assert answers == ['staging']

    async def test_tells_the_model_to_carry_on_when_nobody_answers(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        toolset = SlackChatToolset(slack_client, thread=thread, interactions=SlackInteractions(timeout_seconds=0.01))
        assert 'carry on' in await toolset.ask_user(context(), 'Ship it?', ['Yes', 'No'])


class TestUploadFile:
    async def test_uploads_a_file_inside_the_root(
        self, thread: SlackThread, slack_client: FakeSlackClient, tmp_path: Path
    ) -> None:
        (tmp_path / 'report.csv').write_text('a,b\n')
        toolset = SlackChatToolset(slack_client, thread=thread, file_root=tmp_path)
        assert await toolset.upload_file(context(), 'report.csv', title='Report') == 'Sent report.csv.'
        call = slack_client.method_calls('files_upload_v2')[0]
        assert call.kwargs['file'] == str((tmp_path / 'report.csv').resolve())
        assert call.kwargs['thread_ts'] == thread.thread_ts

    @pytest.mark.parametrize('path', ['../escape.csv', '/etc/passwd'])
    async def test_refuses_a_path_outside_the_root(
        self, thread: SlackThread, slack_client: FakeSlackClient, tmp_path: Path, path: str
    ) -> None:
        toolset = SlackChatToolset(slack_client, thread=thread, file_root=tmp_path)
        with pytest.raises(ModelRetry, match='outside the directory'):
            await toolset.upload_file(context(), path)

    async def test_refuses_a_path_that_is_not_a_file(
        self, thread: SlackThread, slack_client: FakeSlackClient, tmp_path: Path
    ) -> None:
        (tmp_path / 'subdir').mkdir()
        toolset = SlackChatToolset(slack_client, thread=thread, file_root=tmp_path)
        with pytest.raises(ModelRetry, match='no file at subdir'):
            await toolset.upload_file(context(), 'subdir')


class TestChannels:
    async def test_one_channel_is_where_messages_go(self, slack_client: FakeSlackClient) -> None:
        # No thread anywhere: this is the agent that only reports into a channel.
        toolset = SlackChatToolset(slack_client, channels=['#alerts'])
        await toolset.post_message(context(), 'the nightly job failed')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert call.kwargs['channel'] == '#alerts'
        assert call.kwargs['thread_ts'] is None

    def test_a_single_channel_as_a_string_is_refused(self, slack_client: FakeSlackClient) -> None:
        # A string is a Sequence[str], so this type checks and would otherwise
        # become seven channels called '#', 'a', 'l', and so on.
        with pytest.raises(ValueError, match=r"channels=\['#alerts'\]"):
            SlackChatToolset(slack_client, channels='#alerts')  # pyright: ignore[reportArgumentType]

    async def test_the_model_picks_when_several_are_listed(self, slack_client: FakeSlackClient) -> None:
        toolset = SlackChatToolset(slack_client, channels=['#alerts', '#eng'])
        await toolset.post_message(context(), 'deploying', channel='#eng')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['channel'] == '#eng'

    async def test_a_channel_that_was_not_listed_is_refused(self, slack_client: FakeSlackClient) -> None:
        toolset = SlackChatToolset(slack_client, channels=['#alerts'])
        with pytest.raises(ModelRetry, match='You can post to: #alerts'):
            await toolset.post_message(context(), 'hello', channel='#general')
        assert slack_client.method_calls('chat_postMessage') == []

    async def test_naming_a_channel_needs_a_list_to_name_it_from(self, slack_client: FakeSlackClient) -> None:
        toolset = SlackChatToolset(slack_client)
        with pytest.raises(ModelRetry, match='Leave channel unset'):
            await toolset.post_message(context(), 'hello', channel='#general')

    async def test_several_channels_and_no_thread_asks_which(self, slack_client: FakeSlackClient) -> None:
        toolset = SlackChatToolset(slack_client, channels=['#alerts', '#eng'])
        with pytest.raises(ModelRetry, match='Say which channel'):
            await toolset.post_message(context(), 'hello')

    async def test_the_thread_wins_over_a_configured_channel(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # Answering someone goes back to them, not to the channel the agent also
        # reports into.
        toolset = SlackChatToolset(slack_client, channels=['#alerts'], thread=thread)
        await toolset.post_message(context(), 'here is your answer')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert (call.kwargs['channel'], call.kwargs['thread_ts']) == ('C123', thread.thread_ts)

    async def test_with_nowhere_to_post_the_model_is_told_to_answer_normally(
        self, slack_client: FakeSlackClient
    ) -> None:
        with pytest.raises(ModelRetry, match='Answer normally'):
            await SlackChatToolset(slack_client).post_message(context(), 'hello')


class TestBoundThread:
    async def test_a_bound_thread_is_where_an_unconfigured_toolset_posts(
        self, bound_thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # This is the path `SlackBot` takes: nothing configured on the toolset,
        # the thread bound around the run.
        await SlackChatToolset(slack_client).post_message(context(), 'working on it')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert (call.kwargs['channel'], call.kwargs['thread_ts']) == ('C123', bound_thread.thread_ts)

    async def test_status_needs_a_thread_to_set_it_on(self, slack_client: FakeSlackClient) -> None:
        toolset = SlackChatToolset(slack_client, channels=['#alerts'])
        assert 'not available' in await toolset.set_status(context(), 'reading logs')
        assert slack_client.method_calls('assistant_threads_setStatus') == []
