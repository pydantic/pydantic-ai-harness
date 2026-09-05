from __future__ import annotations

import anyio
import pytest
from pydantic_ai import Agent, ToolDenied
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.slack import APPROVE, DENY, SlackApprovals, SlackInteractions, SlackThread

from .conftest import FakeSlackClient, prompt_block_id

pytestmark = pytest.mark.anyio


def context(thread: SlackThread) -> RunContext[SlackThread]:
    return RunContext(deps=thread, model=TestModel(), usage=RunUsage())


def requests_for(*calls: ToolCallPart) -> DeferredToolRequests:
    return DeferredToolRequests(approvals=list(calls))


async def click(interactions: SlackInteractions, client: FakeSlackClient, value: str, index: int = 0) -> None:
    while len(client.method_calls('chat_postMessage')) <= index:
        await anyio.sleep(0)
    assert interactions.resolve(block_id=prompt_block_id(client, index), value=value, user_id='U0ASKER')


class TestSlackApprovals:
    async def test_approving_returns_true(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals(interactions)
        call = ToolCallPart(tool_name='merge_pr', args={'number': 7}, tool_call_id='c1')
        results: list[object] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(thread), requests_for(call))
                results.append(built.approvals['c1'])

            tg.start_soon(decide)
            await click(interactions, slack_client, APPROVE)

        assert results == [True]

    async def test_denying_returns_a_reason_the_model_can_read(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals(interactions)
        call = ToolCallPart(tool_name='merge_pr', args={'number': 7}, tool_call_id='c1')
        results: list[object] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(thread), requests_for(call))
                results.append(built.approvals['c1'])

            tg.start_soon(decide)
            await click(interactions, slack_client, DENY)

        denied = results[0]
        assert isinstance(denied, ToolDenied)
        assert 'denied this action' in denied.message

    async def test_an_unanswered_prompt_denies(self, thread: SlackThread) -> None:
        approvals = SlackApprovals(SlackInteractions(timeout_seconds=0.01))
        call = ToolCallPart(tool_name='delete_database', args={}, tool_call_id='c1')
        built = await approvals(context(thread), requests_for(call))
        denied = built.approvals['c1']
        assert isinstance(denied, ToolDenied)
        assert 'in time' in denied.message

    async def test_the_prompt_shows_the_tool_and_its_arguments(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        approvals = SlackApprovals(SlackInteractions(timeout_seconds=0.01))
        call = ToolCallPart(tool_name='merge_pr', args={'number': 7}, tool_call_id='c1')
        await approvals(context(thread), requests_for(call))
        text = str(slack_client.method_calls('chat_postMessage')[0].kwargs['text'])
        assert '`merge_pr`' in text
        assert '"number": 7' in text

    @pytest.mark.parametrize(
        'args,shows_detail',
        [
            ({}, False),
            ('not json at all', True),
            ({'note': 'x' * 500}, True),
        ],
    )
    async def test_every_shape_of_arguments_still_gets_a_prompt(
        self,
        thread: SlackThread,
        slack_client: FakeSlackClient,
        args: object,
        shows_detail: bool,
    ) -> None:
        approvals = SlackApprovals(SlackInteractions(timeout_seconds=0.01))
        call = ToolCallPart(tool_name='act', args=args, tool_call_id='c1')  # pyright: ignore[reportArgumentType]
        await approvals(context(thread), requests_for(call))
        text = str(slack_client.method_calls('chat_postMessage')[0].kwargs['text'])
        assert text.startswith('Run `act`?')
        assert ('```' in text) is shows_detail

    async def test_arguments_too_long_to_show_are_denied_without_asking(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        approvals = SlackApprovals(SlackInteractions(timeout_seconds=0.01))
        call = ToolCallPart(tool_name='write_file', args={'body': 'x' * 4000}, tool_call_id='c1')
        built = await approvals(context(thread), requests_for(call))
        denied = built.approvals['c1']
        assert isinstance(denied, ToolDenied)
        assert 'nobody could review the whole call' in denied.message
        assert slack_client.method_calls('chat_postMessage') == []

    async def test_the_tool_name_and_fences_count_toward_the_limit(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # The arguments alone fit. What Slack has to render does not.
        approvals = SlackApprovals(SlackInteractions(timeout_seconds=0.01))
        call = ToolCallPart(tool_name='x' * 200, args={'body': 'y' * 2800}, tool_call_id='c1')
        built = await approvals(context(thread), requests_for(call))
        denied = built.approvals['c1']
        assert isinstance(denied, ToolDenied)
        assert 'nobody could review the whole call' in denied.message
        assert slack_client.method_calls('chat_postMessage') == []

    def test_a_reviewer_id_passed_as_a_string_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='not a string'):
            SlackApprovals(SlackInteractions(), allowed_user_ids='U0REVIEWER')

    async def test_each_pending_call_gets_its_own_prompt(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals(interactions)
        calls = (
            ToolCallPart(tool_name='merge_pr', args={}, tool_call_id='c1'),
            ToolCallPart(tool_name='deploy', args={}, tool_call_id='c2'),
        )
        results: list[dict[str, object]] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(thread), requests_for(*calls))
                results.append(dict(built.approvals))

            tg.start_soon(decide)
            await click(interactions, slack_client, APPROVE, index=0)
            await click(interactions, slack_client, DENY, index=1)

        assert results[0]['c1'] is True
        assert isinstance(results[0]['c2'], ToolDenied)

    async def test_a_reviewer_group_answers_instead_of_the_asker(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals(interactions, allowed_user_ids=['U0REVIEWER'])
        call = ToolCallPart(tool_name='merge_pr', args={}, tool_call_id='c1')
        results: list[object] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(thread), requests_for(call))
                results.append(built.approvals['c1'])

            tg.start_soon(decide)
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            block_id = prompt_block_id(slack_client)
            assert interactions.resolve(block_id=block_id, value=APPROVE, user_id='U0ASKER') is False
            assert interactions.resolve(block_id=block_id, value=APPROVE, user_id='U0REVIEWER') is True

        assert results == [True]


class TestThroughAnAgent:
    async def test_a_tool_marked_for_approval_asks_before_running(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        agent = Agent(
            TestModel(call_tools=['merge_pr']),
            deps_type=SlackThread,
            capabilities=[HandleDeferredToolCalls(handler=SlackApprovals(interactions))],
        )
        merged: list[str] = []

        @agent.tool_plain(requires_approval=True)
        def merge_pr(number: int) -> str:
            merged.append(f'merged {number}')
            return 'merged'

        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: agent.run('merge it', deps=thread))
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            assert merged == []
            assert interactions.resolve(block_id=prompt_block_id(slack_client), value=APPROVE, user_id='U0ASKER')

        assert merged == ['merged 0']

    async def test_denying_stops_the_tool_from_running(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        agent = Agent(
            TestModel(call_tools=['merge_pr']),
            deps_type=SlackThread,
            capabilities=[HandleDeferredToolCalls(handler=SlackApprovals(interactions))],
        )
        merged: list[str] = []

        @agent.tool_plain(requires_approval=True)
        def merge_pr(number: int) -> str:  # pragma: no cover - the denial is what stops this running
            merged.append('merged')
            return 'merged'

        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: agent.run('merge it', deps=thread))
            await click(interactions, slack_client, DENY)

        assert merged == []
