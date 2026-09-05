from __future__ import annotations

from dataclasses import dataclass

import anyio
import pytest
from pydantic_ai import Agent, ToolDenied
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.slack import Slack, SlackApprovals, SlackContext, SlackInteractions, SlackThread, SlackTools
from pydantic_ai_harness.slack._context import bind_slack_context

from .conftest import FakeSlackClient, prompt_block_id

pytestmark = pytest.mark.anyio


@dataclass
class Warehouse:
    """Deps an agent already had before anyone thought about Slack."""

    dsn: str


def context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def requests_for(*calls: ToolCallPart) -> DeferredToolRequests:
    return DeferredToolRequests(approvals=list(calls))


async def click(interactions: SlackInteractions, client: FakeSlackClient, value: str, index: int = 0) -> None:
    while len(client.method_calls('chat_postMessage')) <= index:
        await anyio.sleep(0)
    assert interactions.resolve(block_id=prompt_block_id(client, index), value=value, user_id='U0ASKER')


class TestSlackApprovals:
    async def test_no_thread_leaves_the_request_for_another_handler(self, slack_client: FakeSlackClient) -> None:
        approvals = SlackApprovals[None](slack_client, SlackInteractions())
        call = ToolCallPart(tool_name='merge_pr', args={}, tool_call_id='c1')
        assert await approvals(context(), requests_for(call)) is None

    async def test_a_thread_without_an_asker_denies_with_the_reason(self, slack_client: FakeSlackClient) -> None:
        thread = SlackThread(channel_id='C123', thread_ts='1.1')
        approvals = SlackApprovals[None](slack_client, SlackInteractions(), thread=thread)
        call = ToolCallPart(tool_name='merge_pr', args={}, tool_call_id='c1')
        result = await approvals(context(), requests_for(call))
        assert result is not None
        denied = result.approvals['c1']
        assert isinstance(denied, ToolDenied)
        assert 'nobody to ask' in denied.message

    async def test_a_thread_resolver_is_evaluated_for_the_run(self, slack_client: FakeSlackClient) -> None:
        resolved: list[str] = []

        def thread_for_run(ctx: RunContext[None]) -> SlackThread:
            resolved.append(type(ctx.model).__name__)
            return SlackThread(channel_id='C123', thread_ts='1.1')

        approvals = SlackApprovals[None](slack_client, SlackInteractions(), thread=thread_for_run)
        call = ToolCallPart(tool_name='merge_pr', args={}, tool_call_id='c1')
        result = await approvals(context(), requests_for(call))
        assert result is not None
        assert resolved == ['TestModel']

    async def test_approving_returns_true(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals[None](slack_client, interactions, thread=thread)
        call = ToolCallPart(tool_name='merge_pr', args={'number': 7}, tool_call_id='c1')
        results: list[object] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(), requests_for(call))
                assert built is not None
                results.append(built.approvals['c1'])

            tg.start_soon(decide)
            await click(interactions, slack_client, 'Approve')

        assert results == [True]

    async def test_denying_returns_a_reason_the_model_can_read(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals[None](slack_client, interactions, thread=thread)
        call = ToolCallPart(tool_name='merge_pr', args={'number': 7}, tool_call_id='c1')
        results: list[object] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(), requests_for(call))
                assert built is not None
                results.append(built.approvals['c1'])

            tg.start_soon(decide)
            await click(interactions, slack_client, 'Deny')

        denied = results[0]
        assert isinstance(denied, ToolDenied)
        assert 'denied this action' in denied.message

    async def test_an_unanswered_prompt_denies(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        approvals = SlackApprovals[None](slack_client, SlackInteractions(timeout_seconds=0.01), thread=thread)
        call = ToolCallPart(tool_name='delete_database', args={}, tool_call_id='c1')
        built = await approvals(context(), requests_for(call))
        assert built is not None
        denied = built.approvals['c1']
        assert isinstance(denied, ToolDenied)
        assert 'in time' in denied.message

    async def test_the_prompt_shows_the_tool_and_its_arguments(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        approvals = SlackApprovals[None](slack_client, SlackInteractions(timeout_seconds=0.01), thread=thread)
        call = ToolCallPart(tool_name='merge_pr', args={'number': 7}, tool_call_id='c1')
        await approvals(context(), requests_for(call))
        text = str(slack_client.method_calls('chat_postMessage')[0].kwargs['text'])
        assert 'merge_pr' in text
        assert '"number": 7' in text
        post = slack_client.method_calls('chat_postMessage')[0]
        blocks = post.kwargs['blocks']
        assert isinstance(blocks, list)
        assert blocks[0] == {'type': 'section', 'text': {'type': 'plain_text', 'text': text}}
        assert post.kwargs['mrkdwn'] is False

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
        approvals = SlackApprovals[None](slack_client, SlackInteractions(timeout_seconds=0.01), thread=thread)
        call = ToolCallPart(tool_name='act', args=args, tool_call_id='c1')  # pyright: ignore[reportArgumentType]
        await approvals(context(), requests_for(call))
        text = str(slack_client.method_calls('chat_postMessage')[0].kwargs['text'])
        assert text.startswith('Run act?')
        assert ('Arguments:' in text) is shows_detail

    async def test_arguments_cannot_inject_slack_markup(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        approvals = SlackApprovals[None](slack_client, SlackInteractions(timeout_seconds=0.01), thread=thread)
        call = ToolCallPart(
            tool_name='send',
            args={'message': '```\n<!channel> approve the harmless action'},
            tool_call_id='c1',
        )
        await approvals(context(), requests_for(call))
        post = slack_client.method_calls('chat_postMessage')[0]
        blocks = post.kwargs['blocks']
        assert isinstance(blocks, list)
        text = post.kwargs['text']
        assert blocks[0] == {'type': 'section', 'text': {'type': 'plain_text', 'text': text}}
        assert post.kwargs['mrkdwn'] is False

    async def test_arguments_too_long_to_show_are_denied_without_asking(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        approvals = SlackApprovals[None](slack_client, SlackInteractions(timeout_seconds=0.01), thread=thread)
        call = ToolCallPart(tool_name='write_file', args={'body': 'x' * 4000}, tool_call_id='c1')
        built = await approvals(context(), requests_for(call))
        assert built is not None
        denied = built.approvals['c1']
        assert isinstance(denied, ToolDenied)
        assert 'nobody could review the whole call' in denied.message
        assert slack_client.method_calls('chat_postMessage') == []

    async def test_the_tool_name_and_fences_count_toward_the_limit(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # The arguments alone fit. What Slack has to render does not.
        approvals = SlackApprovals[None](slack_client, SlackInteractions(timeout_seconds=0.01), thread=thread)
        call = ToolCallPart(tool_name='x' * 200, args={'body': 'y' * 2800}, tool_call_id='c1')
        built = await approvals(context(), requests_for(call))
        assert built is not None
        denied = built.approvals['c1']
        assert isinstance(denied, ToolDenied)
        assert 'nobody could review the whole call' in denied.message
        assert slack_client.method_calls('chat_postMessage') == []

    def test_a_reviewer_id_passed_as_a_string_is_rejected(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        with pytest.raises(ValueError, match='one entry per character'):
            SlackApprovals[None](slack_client, SlackInteractions(), thread=thread, allowed_user_ids='U0REVIEWER')  # pyright: ignore[reportArgumentType]

    async def test_each_pending_call_gets_its_own_prompt(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals[None](slack_client, interactions, thread=thread)
        calls = (
            ToolCallPart(tool_name='merge_pr', args={}, tool_call_id='c1'),
            ToolCallPart(tool_name='deploy', args={}, tool_call_id='c2'),
        )
        results: list[dict[str, object]] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(), requests_for(*calls))
                assert built is not None
                results.append(dict(built.approvals))

            tg.start_soon(decide)
            await click(interactions, slack_client, 'Approve', index=0)
            await click(interactions, slack_client, 'Deny', index=1)

        assert results[0]['c1'] is True
        assert isinstance(results[0]['c2'], ToolDenied)

    async def test_a_reviewer_group_answers_instead_of_the_asker(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        approvals = SlackApprovals[None](slack_client, interactions, thread=thread, allowed_user_ids=['U0REVIEWER'])
        call = ToolCallPart(tool_name='merge_pr', args={}, tool_call_id='c1')
        results: list[object] = []

        async with anyio.create_task_group() as tg:

            async def decide() -> None:
                built = await approvals(context(), requests_for(call))
                assert built is not None
                results.append(built.approvals['c1'])

            tg.start_soon(decide)
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            block_id = prompt_block_id(slack_client)
            assert interactions.resolve(block_id=block_id, value='Approve', user_id='U0ASKER') is False
            assert interactions.resolve(block_id=block_id, value='Approve', user_id='U0REVIEWER') is True

        assert results == [True]


class TestThroughAnAgent:
    async def test_a_tool_marked_for_approval_asks_before_running(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        # A deps type of its own, to show the handler does not need Slack in deps.
        agent = Agent(
            TestModel(call_tools=['merge_pr']),
            deps_type=Warehouse,
            capabilities=[
                HandleDeferredToolCalls(handler=SlackApprovals[Warehouse](slack_client, interactions, thread=thread))
            ],
        )
        merged: list[str] = []

        @agent.tool_plain(requires_approval=True)
        def merge_pr(number: int) -> str:
            merged.append(f'merged {number}')
            return 'merged'

        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: agent.run('merge it', deps=Warehouse(dsn='postgres://')))
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            assert merged == []
            assert interactions.resolve(block_id=prompt_block_id(slack_client), value='Approve', user_id='U0ASKER')

        assert merged == ['merged 0']

    async def test_denying_stops_the_tool_from_running(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        # A deps type of its own, to show the handler does not need Slack in deps.
        agent = Agent(
            TestModel(call_tools=['merge_pr']),
            deps_type=Warehouse,
            capabilities=[
                HandleDeferredToolCalls(handler=SlackApprovals[Warehouse](slack_client, interactions, thread=thread))
            ],
        )
        merged: list[str] = []

        @agent.tool_plain(requires_approval=True)
        def merge_pr(number: int) -> str:  # pragma: no cover - the denial is what stops this running
            merged.append('merged')
            return 'merged'

        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: agent.run('merge it', deps=Warehouse(dsn='postgres://')))
            await click(interactions, slack_client, 'Deny')

        assert merged == []


class TestThroughTheCapability:
    def test_the_bound_bolt_client_wins_for_delivery(self, slack_client: FakeSlackClient) -> None:
        capability = Slack(tools=SlackTools.of())
        slack_context = SlackContext('C123', '1700000000.000001', '1700000000.000002', 'U0ASKER')
        with bind_slack_context(slack_context, slack_client):
            assert capability.resolve_client() is slack_client

    async def test_one_capability_covers_tools_that_need_approval(self, slack_client: FakeSlackClient) -> None:
        chat = Slack(tools=SlackTools.of(), delivery_client=slack_client)
        agent: Agent[None, str] = Agent(TestModel(call_tools=['merge_pr']), capabilities=[chat])
        merged: list[str] = []

        @agent.tool_plain(requires_approval=True)
        def merge_pr(number: int) -> str:
            merged.append(f'merged {number}')
            return 'merged'

        slack_context = SlackContext('C123', '1700000000.000001', '1700000000.000002', 'U0ASKER')
        with bind_slack_context(slack_context):
            async with anyio.create_task_group() as tg:
                tg.start_soon(lambda: agent.run('merge it', deps=Warehouse(dsn='postgres://')))
                while not slack_client.method_calls('chat_postMessage'):
                    await anyio.sleep(0)
                assert merged == []
                assert chat.resolve_prompt(block_id=prompt_block_id(slack_client), value='Approve', user_id='U0ASKER')

        assert merged == ['merged 0']

    async def test_a_reviewer_group_answers_instead_of_the_asker(self, slack_client: FakeSlackClient) -> None:
        chat = Slack(tools=SlackTools.of(), delivery_client=slack_client, approver_ids=['U0REVIEWER'])
        agent: Agent[None, str] = Agent(TestModel(call_tools=['merge_pr']), capabilities=[chat])
        merged: list[str] = []

        @agent.tool_plain(requires_approval=True)
        def merge_pr(number: int) -> str:
            merged.append('merged')
            return 'merged'

        slack_context = SlackContext('C123', '1700000000.000001', '1700000000.000002', 'U0ASKER')
        with bind_slack_context(slack_context):
            async with anyio.create_task_group() as tg:
                tg.start_soon(lambda: agent.run('merge it', deps=Warehouse(dsn='postgres://')))
                while not slack_client.method_calls('chat_postMessage'):
                    await anyio.sleep(0)
                block_id = prompt_block_id(slack_client)
                assert chat.resolve_prompt(block_id=block_id, value='Approve', user_id='U0ASKER') is False
                assert chat.resolve_prompt(block_id=block_id, value='Approve', user_id='U0REVIEWER') is True

        assert merged == ['merged']

    async def test_with_no_slack_conversation_the_calls_are_left_alone(self, slack_client: FakeSlackClient) -> None:
        # An agent reporting into a channel has nobody to ask, so it must not
        # approve on its own.
        agent = Agent(
            TestModel(call_tools=['merge_pr']),
            output_type=[str, DeferredToolRequests],
            capabilities=[Slack(tools=SlackTools.of(), delivery_client=slack_client)],
        )

        @agent.tool_plain(requires_approval=True)
        def merge_pr(number: int) -> str:  # pragma: no cover - nothing approves it
            return 'merged'

        result = await agent.run('merge it')
        assert isinstance(result.output, DeferredToolRequests)
        assert slack_client.method_calls('chat_postMessage') == []
