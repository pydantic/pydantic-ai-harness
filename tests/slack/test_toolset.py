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


def context(thread: SlackThread, run_id: str = 'run-1') -> RunContext[SlackThread]:
    return RunContext(deps=thread, model=TestModel(), usage=RunUsage(), run_id=run_id)


def tool_names(toolset: SlackChatToolset) -> set[str]:
    return set(toolset.tools)


def _plan_id(answer: str) -> str:
    """The plan id `post_plan` told the model to send back."""
    found = re.search(r"plan_id='([^']+)'", answer)
    assert found is not None, answer
    return found.group(1)


class TestRegistration:
    def test_ask_user_and_upload_file_are_opt_in(self) -> None:
        assert tool_names(SlackChatToolset()) == {'post_message', 'post_plan', 'set_status'}

    def test_ask_user_appears_with_an_interactions_registry(self) -> None:
        assert 'ask_user' in tool_names(SlackChatToolset(interactions=SlackInteractions()))

    def test_upload_file_appears_with_a_file_root(self, tmp_path: Path) -> None:
        assert 'upload_file' in tool_names(SlackChatToolset(file_root=tmp_path))


class TestPostMessage:
    async def test_posts_into_the_thread(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        await SlackChatToolset().post_message(context(thread), 'found the bad migration')
        call = slack_client.method_calls('chat_postMessage')[0]
        assert call.kwargs == {
            'channel': 'C123',
            'thread_ts': thread.thread_ts,
            'text': 'found the bad migration',
            'blocks': None,
        }

    async def test_asks_the_model_to_split_an_oversized_message(self, thread: SlackThread) -> None:
        with pytest.raises(ModelRetry, match='Send it as several shorter messages'):
            await SlackChatToolset().post_message(context(thread), 'x' * (MAX_MESSAGE_CHARS + 1))

    async def test_rejects_an_empty_message(self, thread: SlackThread) -> None:
        with pytest.raises(ModelRetry, match='cannot be empty'):
            await SlackChatToolset().post_message(context(thread), '   ')

    async def test_reaches_slack_through_a_normal_agent_run(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        agent = Agent(
            TestModel(call_tools=['post_message']),
            deps_type=SlackThread,
            toolsets=[SlackChatToolset()],
        )
        await agent.run('go', deps=thread)
        assert slack_client.method_calls('chat_postMessage')


class TestPostPlan:
    async def test_the_returned_plan_id_edits_the_same_message(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        toolset = SlackChatToolset()
        steps = [PlanStep(text='read the logs'), PlanStep(text='open a PR')]
        posted = await toolset.post_plan(context(thread), steps)
        assert slack_client.next_ts in posted

        steps[0].status = 'done'
        steps[1].status = 'running'
        await toolset.post_plan(context(thread), steps, plan_id=_plan_id(posted))

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
        toolset = SlackChatToolset()
        await toolset.post_plan(context(thread), [PlanStep(text='a')])
        await toolset.post_plan(context(thread), [PlanStep(text='b')])
        assert len(slack_client.method_calls('chat_postMessage')) == 2

    async def test_renders_every_status(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        steps = [
            PlanStep(text='a', status='pending'),
            PlanStep(text='b', status='running'),
            PlanStep(text='c', status='done'),
            PlanStep(text='d', status='failed'),
        ]
        await SlackChatToolset().post_plan(context(thread), steps)
        text = str(slack_client.method_calls('chat_postMessage')[0].kwargs['text'])
        assert text.count(':white_circle:') == 1
        assert text.count(':x:') == 1

    async def test_tells_the_model_to_repost_when_slack_gave_no_id(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.post_response = {'ok': True}
        answer = await SlackChatToolset().post_plan(context(thread), [PlanStep(text='a')])
        assert 'post a fresh plan' in answer

    async def test_rejects_an_empty_plan(self, thread: SlackThread) -> None:
        with pytest.raises(ModelRetry, match='at least one step'):
            await SlackChatToolset().post_plan(context(thread), [])

    async def test_rejects_an_overlong_plan(self, thread: SlackThread) -> None:
        with pytest.raises(ModelRetry, match='20 steps or fewer'):
            await SlackChatToolset().post_plan(context(thread), [PlanStep(text='a')] * 21)

    async def test_rejects_a_plan_slack_would_not_accept(self, thread: SlackThread) -> None:
        with pytest.raises(ModelRetry, match='Slack takes at most'):
            await SlackChatToolset().post_plan(context(thread), [PlanStep(text='x' * 400)] * 10)

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
            await SlackChatToolset().post_plan(context(thread), [PlanStep(text='a')], plan_id=plan_id)
        assert slack_client.method_calls('chat_update') == []

    async def test_refuses_a_plan_id_from_an_earlier_turn(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # The id stays in the transcript, so the next turn can see it. It should
        # post its own checklist rather than edit the one before it.
        toolset = SlackChatToolset()
        posted = await toolset.post_plan(context(thread, run_id='run-1'), [PlanStep(text='a')])
        with pytest.raises(ModelRetry, match='not a plan you posted here'):
            await toolset.post_plan(context(thread, run_id='run-2'), [PlanStep(text='a')], plan_id=_plan_id(posted))

    async def test_refuses_a_plan_id_issued_for_another_thread(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        toolset = SlackChatToolset()
        elsewhere = replace(thread, channel_id='C999')
        posted = await toolset.post_plan(context(elsewhere), [PlanStep(text='a')])
        with pytest.raises(ModelRetry, match='not a plan you posted here'):
            await toolset.post_plan(context(thread), [PlanStep(text='a')], plan_id=_plan_id(posted))


class TestSetStatus:
    async def test_sets_the_status(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        assert await SlackChatToolset().set_status(context(thread), 'reading logs') == 'Status set.'
        assert slack_client.method_calls('assistant_threads_setStatus')[0].kwargs['status'] == 'reading logs'

    async def test_a_channel_without_status_support_does_not_fail_the_turn(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.status_error = RuntimeError('not_allowed_in_channel')
        assert 'not available' in await SlackChatToolset().set_status(context(thread), 'reading logs')

    async def test_a_broken_status_call_is_logged_rather_than_hidden(
        self, thread: SlackThread, slack_client: FakeSlackClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The same answer covers a bad token and a rate limit, so the reason has
        # to be somewhere an operator can find it.
        caplog.set_level('INFO')
        slack_client.recorder.status_error = RuntimeError('invalid_auth')
        await SlackChatToolset().set_status(context(thread), 'reading logs')
        assert 'Could not set the Slack status' in caplog.text
        assert 'invalid_auth' in caplog.text


class TestAskUser:
    async def test_unusable_options_come_back_as_something_the_model_can_fix(self, thread: SlackThread) -> None:
        toolset = SlackChatToolset(interactions=SlackInteractions(timeout_seconds=0.01))
        with pytest.raises(ModelRetry, match='must be unique'):
            await toolset.ask_user(context(thread), 'Pick', ['A', 'A'])

    async def test_returns_the_answer(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        toolset = SlackChatToolset(interactions=interactions)
        answers: list[str] = []

        async with anyio.create_task_group() as tg:

            async def ask() -> None:
                answers.append(await toolset.ask_user(context(thread), 'Staging or prod?', ['staging', 'prod']))

            tg.start_soon(ask)
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            interactions.resolve(block_id=prompt_block_id(slack_client), value='staging', user_id='U0ASKER')

        assert answers == ['staging']

    async def test_tells_the_model_to_carry_on_when_nobody_answers(self, thread: SlackThread) -> None:
        toolset = SlackChatToolset(interactions=SlackInteractions(timeout_seconds=0.01))
        assert 'carry on' in await toolset.ask_user(context(thread), 'Ship it?', ['Yes', 'No'])


class TestUploadFile:
    async def test_uploads_a_file_inside_the_root(
        self, thread: SlackThread, slack_client: FakeSlackClient, tmp_path: Path
    ) -> None:
        (tmp_path / 'report.csv').write_text('a,b\n')
        toolset = SlackChatToolset(file_root=tmp_path)
        assert await toolset.upload_file(context(thread), 'report.csv', title='Report') == 'Sent report.csv.'
        call = slack_client.method_calls('files_upload_v2')[0]
        assert call.kwargs['file'] == str((tmp_path / 'report.csv').resolve())
        assert call.kwargs['thread_ts'] == thread.thread_ts

    @pytest.mark.parametrize('path', ['../escape.csv', '/etc/passwd'])
    async def test_refuses_a_path_outside_the_root(self, thread: SlackThread, tmp_path: Path, path: str) -> None:
        toolset = SlackChatToolset(file_root=tmp_path)
        with pytest.raises(ModelRetry, match='outside the directory'):
            await toolset.upload_file(context(thread), path)

    async def test_refuses_a_path_that_is_not_a_file(self, thread: SlackThread, tmp_path: Path) -> None:
        (tmp_path / 'subdir').mkdir()
        toolset = SlackChatToolset(file_root=tmp_path)
        with pytest.raises(ModelRetry, match='no file at subdir'):
            await toolset.upload_file(context(thread), 'subdir')
