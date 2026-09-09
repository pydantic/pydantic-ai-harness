"""Events emitted by the SubAgents capability."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    CapabilityEvent,
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_ai_harness.subagents import (
    MAX_EVENT_TEXT_CHARS,
    DelegationEndEvent,
    DelegationStartEvent,
    SubAgent,
    SubAgents,
    SubAgentToolset,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class Listener(AbstractCapability[object]):
    """Records every delegation event in order."""

    events: list[CapabilityEvent] = field(default_factory=list[CapabilityEvent])

    @on_event(DelegationStartEvent, DelegationEndEvent)
    async def _on_delegation(self, ctx: RunContext[object], event: DelegationStartEvent | DelegationEndEvent) -> None:
        self.events.append(event)


def _delegations(calls: Sequence[Sequence[dict[str, str]]]) -> FunctionModel:
    """A parent model that issues each step's delegations together, then replies with text.

    Streams, because a listener capability on the parent makes core stream the run.
    """
    step = {'n': 0}

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        step['n'] += 1
        if step['n'] <= len(calls):
            yield {
                i: DeltaToolCall(name='delegate_task', json_args=json.dumps(args), tool_call_id=f'c{step["n"]}_{i}')
                for i, args in enumerate(calls[step['n'] - 1])
            }
        else:
            yield 'all done'

    return FunctionModel(stream_function=stream)


def _delegate_once(task: str = 'do it', **extra: str) -> FunctionModel:
    return _delegations([[{'agent_name': 'worker', 'task': task, **extra}]])


def _worker(output: str = 'W') -> Agent[object, str]:
    return Agent(TestModel(custom_output_text=output), name='worker')


async def _run(
    parent_model: FunctionModel, sub_agents: SubAgents[object], *, listener: Listener | None = None
) -> tuple[Listener, str]:
    listener = listener or Listener()
    result = await Agent(parent_model, capabilities=[sub_agents, listener]).run('go')
    return listener, result.output


def _pair(listener: Listener) -> tuple[DelegationStartEvent, DelegationEndEvent]:
    start, end = listener.events
    assert isinstance(start, DelegationStartEvent)
    assert isinstance(end, DelegationEndEvent)
    return start, end


class TestDelegationEvents:
    async def test_start_then_end_for_one_delegation(self) -> None:
        listener, output = await _run(_delegate_once(), SubAgents(agents=[SubAgent(_worker())]))

        assert output == 'all done'
        start, end = _pair(listener)
        assert start == DelegationStartEvent(
            agent_name='worker',
            task='do it',
            truncated=False,
            model=None,
            inherits_tools=False,
            capability_id='sub_agents',
            tool_call_id='c1_0',
            tool_name='delegate_task',
        )
        assert end.agent_name == 'worker'
        assert end.outcome == 'ok'
        assert end.output == 'W'
        assert end.truncated is False
        assert end.usage is None  # shared with the parent, so not separable
        assert end.duration_seconds >= 0
        assert end.tool_call_id == start.tool_call_id

    async def test_inherits_tools_and_menu_key_are_reported(self) -> None:
        listener, _ = await _run(
            _delegate_once(model='fast'),
            SubAgents(agents=[SubAgent(_worker())], models={'fast': TestModel()}, inherit_tools=True),
        )

        start, _ = _pair(listener)
        assert start.model == 'fast'
        assert start.inherits_tools is True

    async def test_restricted_delegate_reports_its_default_key(self) -> None:
        listener, _ = await _run(
            _delegate_once(),
            SubAgents(agents=[SubAgent(_worker(), models=['fast'])], models={'fast': TestModel(), 'deep': TestModel()}),
        )

        start, _ = _pair(listener)
        assert start.model == 'fast'

    async def test_own_accounting_reports_child_usage(self) -> None:
        listener, _ = await _run(
            _delegate_once(), SubAgents(agents=[SubAgent(_worker(), usage_limits=UsageLimits(request_limit=5))])
        )

        _, end = _pair(listener)
        assert isinstance(end.usage, RunUsage)
        assert end.usage.requests == 1

    async def test_unforwarded_usage_reports_child_usage(self) -> None:
        listener, _ = await _run(_delegate_once(), SubAgents(agents=[SubAgent(_worker())], forward_usage=False))

        _, end = _pair(listener)
        assert isinstance(end.usage, RunUsage)
        assert end.usage.requests == 1

    async def test_task_and_output_are_bounded(self) -> None:
        long_task = 't' * (MAX_EVENT_TEXT_CHARS + 1)
        long_output = 'o' * (MAX_EVENT_TEXT_CHARS + 1)
        listener, _ = await _run(_delegate_once(task=long_task), SubAgents(agents=[SubAgent(_worker(long_output))]))

        start, end = _pair(listener)
        assert (start.task, start.truncated) == ('t' * MAX_EVENT_TEXT_CHARS, True)
        assert (end.output, end.truncated) == ('o' * MAX_EVENT_TEXT_CHARS, True)

    async def test_parallel_delegations_pair_by_tool_call_id(self) -> None:
        listener, _ = await _run(
            _delegations([[{'agent_name': 'worker', 'task': 'a'}, {'agent_name': 'worker', 'task': 'b'}]]),
            SubAgents(agents=[SubAgent(_worker())]),
        )

        starts = [event for event in listener.events if isinstance(event, DelegationStartEvent)]
        ends = [event for event in listener.events if isinstance(event, DelegationEndEvent)]
        assert {start.tool_call_id for start in starts} == {'c1_0', 'c1_1'}
        assert {end.tool_call_id for end in ends} == {'c1_0', 'c1_1'}


class TestOutcomes:
    async def test_timeout(self) -> None:
        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            await asyncio.sleep(1)
            return ModelResponse(parts=[TextPart('late')])  # pragma: no cover - cancelled by the timeout

        worker = Agent(FunctionModel(slow), name='worker')
        listener, _ = await _run(_delegate_once(), SubAgents(agents=[SubAgent(worker, timeout_seconds=0.01)]))

        _, end = _pair(listener)
        assert end.outcome == 'timeout'
        assert "Sub-agent 'worker' exceeded its 0.01s time budget" in end.output

    async def test_budget(self) -> None:
        def worker_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('noop', {}, tool_call_id='n1')])
            return ModelResponse(parts=[TextPart('done')])  # pragma: no cover - blocked by the request budget

        worker = Agent(FunctionModel(worker_fn), name='worker')

        @worker.tool_plain
        def noop() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'x'

        listener, _ = await _run(
            _delegate_once(), SubAgents(agents=[SubAgent(worker, usage_limits=UsageLimits(request_limit=1))])
        )

        _, end = _pair(listener)
        assert end.outcome == 'budget'
        assert "Sub-agent 'worker' reached its usage budget" in end.output
        assert isinstance(end.usage, RunUsage)

    async def test_failed_with_on_failure_carries_the_steer(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UnexpectedModelBehavior('kaboom')

        worker = Agent(FunctionModel(boom), name='worker')
        listener, _ = await _run(_delegate_once(), SubAgents(agents=[SubAgent(worker, on_failure='use what you have')]))

        _, end = _pair(listener)
        assert end.outcome == 'failed'
        assert end.output == 'use what you have'

    async def test_failed_without_on_failure_carries_the_retry(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UnexpectedModelBehavior('kaboom')

        worker = Agent(FunctionModel(boom), name='worker')
        listener = Listener()
        result = await Agent(_delegate_once(), capabilities=[SubAgents(agents=[SubAgent(worker)]), listener]).run('go')

        _, end = _pair(listener)
        assert end.outcome == 'failed'
        assert end.output == "Sub-agent 'worker' failed: kaboom"
        retries = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
        ]
        assert [retry.content for retry in retries] == [end.output]

    async def test_contained(self) -> None:
        def crash(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelAPIError('m', 'provider down')

        worker = Agent(FunctionModel(crash), name='worker')
        listener, _ = await _run(_delegate_once(), SubAgents(agents=[SubAgent(worker, contain_errors=True)]))

        _, end = _pair(listener)
        assert end.outcome == 'contained'
        assert end.output.startswith("Sub-agent 'worker' crashed: ModelAPIError: provider down")

    async def test_propagating_crash_ends_without_an_end_event(self) -> None:
        def crash(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelAPIError('m', 'provider down')

        worker = Agent(FunctionModel(crash), name='worker')
        listener = Listener()
        with pytest.raises(ModelAPIError):
            await Agent(_delegate_once(), capabilities=[SubAgents(agents=[SubAgent(worker)]), listener]).run('go')

        assert [type(event) for event in listener.events] == [DelegationStartEvent]


class TestNothingEmitted:
    async def test_refused_delegations_emit_nothing(self) -> None:
        listener, _ = await _run(
            _delegations(
                [
                    [{'agent_name': 'ghost', 'task': 't'}],
                    [{'agent_name': 'worker', 'task': 't'}],
                    [{'agent_name': 'worker', 'task': 't'}],
                ]
            ),
            SubAgents(agents=[SubAgent(_worker(), max_calls=1)]),
        )

        # The unknown sub-agent and the over-budget call never start; one delegation ran.
        assert [type(event) for event in listener.events] == [DelegationStartEvent, DelegationEndEvent]

    async def test_directly_registered_toolset_emits_nothing(self) -> None:
        toolset: SubAgentToolset[object] = SubAgentToolset(
            agents={'worker': SubAgent(_worker())},
            forward_usage=True,
            inherit_tools=False,
            shared_capabilities=[],
            event_stream_handler=None,
            tool_name='delegate_task',
            tool_retries=1,
            contain_errors=False,
            call_counts={},
        )
        listener = Listener()
        result = await Agent(_delegate_once(), toolsets=[toolset], capabilities=[listener]).run('go')

        assert result.output == 'all done'
        assert listener.events == []
