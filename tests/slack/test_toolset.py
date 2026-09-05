from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.slack import PlanStep, SlackChatToolset, SlackInteractions, SlackThread

from .conftest import FakeSlackClient, prompt_block_id

pytestmark = pytest.mark.anyio


def context(thread: SlackThread) -> RunContext[SlackThread]:
    return RunContext(deps=thread, model=TestModel(), usage=RunUsage())


def tool_names(toolset: SlackChatToolset) -> set[str]:
    return set(toolset.tools)


class TestRegistration:
    def test_ask_user_and_upload_file_are_opt_in(self) -> None:
        assert tool_names(SlackChatToolset()) == {'post_message', 'post_plan', 'set_status'}

    def test_ask_user_appears_with_an_interactions_registry(self) -> None:
        assert 'ask_user' in tool_names(SlackChatToolset(interactions=SlackInteractions()))

    def test_upload_file_appears_with_a_file_root(self, tmp_path: Path) -> None:
        assert 'upload_file' in tool_names(SlackChatToolset(file_root=tmp_path))

    def test_rejects_a_non_positive_message_limit(self) -> None:
        with pytest.raises(ValueError, match='max_message_chars must be positive'):
            SlackChatToolset(max_message_chars=0)


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
        toolset = SlackChatToolset(max_message_chars=10)
        with pytest.raises(ModelRetry, match='Send it as several shorter messages'):
            await toolset.post_message(context(thread), 'x' * 11)

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
    async def test_posts_once_then_edits_the_same_message(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        toolset = SlackChatToolset()
        steps = [PlanStep(text='read the logs'), PlanStep(text='open a PR')]
        assert await toolset.post_plan(context(thread), steps) == 'Plan posted.'

        steps[0].status = 'done'
        steps[1].status = 'running'
        assert await toolset.post_plan(context(thread), steps) == 'Plan updated.'

        assert len(slack_client.method_calls('chat_postMessage')) == 1
        update = slack_client.method_calls('chat_update')[0]
        assert update.kwargs['ts'] == slack_client.next_ts
        assert str(update.kwargs['text']).splitlines() == [
            ':white_check_mark: read the logs',
            ':hourglass_flowing_sand: open a PR',
        ]

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

    async def test_reposts_when_slack_gave_no_timestamp(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.post_response = {'ok': True}
        toolset = SlackChatToolset()
        await toolset.post_plan(context(thread), [PlanStep(text='a')])
        await toolset.post_plan(context(thread), [PlanStep(text='a')])
        assert len(slack_client.method_calls('chat_postMessage')) == 2

    async def test_rejects_an_empty_plan(self, thread: SlackThread) -> None:
        with pytest.raises(ModelRetry, match='at least one step'):
            await SlackChatToolset().post_plan(context(thread), [])

    async def test_rejects_an_overlong_plan(self, thread: SlackThread) -> None:
        with pytest.raises(ModelRetry, match='20 steps or fewer'):
            await SlackChatToolset().post_plan(context(thread), [PlanStep(text='a')] * 21)


class TestSetStatus:
    async def test_sets_the_status(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        assert await SlackChatToolset().set_status(context(thread), 'reading logs') == 'Status set.'
        assert slack_client.method_calls('assistant_threads_setStatus')[0].kwargs['status'] == 'reading logs'

    async def test_a_channel_without_status_support_does_not_fail_the_turn(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.status_error = RuntimeError('not_allowed_in_channel')
        assert 'not available' in await SlackChatToolset().set_status(context(thread), 'reading logs')


class TestAskUser:
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
