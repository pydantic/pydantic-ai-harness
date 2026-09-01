"""Tests for `CodeMode(speculate=...)`: early launch of sandbox calls from streamed `run_code` deltas.

Behavioral, through `Agent(..., capabilities=[CodeMode(...)])` with a streaming `FunctionModel`:
the model streams the `run_code` arguments in small chunks with an await point between them, so
speculative tasks get scheduled while "generation" is still in flight, and the tools themselves
record when and how often they ran.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.code_mode import (
    CodeMode,
    CodeModeToolset,
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallLaunchedEvent,
    SpeculativeCallMissedEvent,
    SpeculativeCallSettledEvent,
    SpeculativeCodeUpdateEvent,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


@dataclass
class ToolLog:
    """Observations the fake tools record for assertions."""

    calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """(tool name, argument) per execution, in start order."""

    streaming_done: bool = False
    """Set by the model stream after its final chunk; tools read it at start."""

    started_during_stream: int = 0
    """How many tool executions began before the model finished streaming."""


def padded(code: str) -> str:
    """Append filler statements so earlier statements close with plenty of stream still to go.

    Against a real provider the gap between a closed statement and end-of-generation is the
    remaining decode; here the filler plays that role, giving launched tasks event-loop passes
    (one per chunk) to reach the tool body while the stream is still "generating".
    """
    filler = ''.join(f'\npad_{i} = {i}' for i in range(20))
    return f'{code}{filler}\n"ok"'


def build_agent(
    log: ToolLog,
    code: str,
    capability: CodeMode[None],
    chunk_size: int = 16,
    extra_capabilities: Sequence[AbstractCapability[None]] = (),
    raw_args: str | None = None,
) -> Agent[None, str]:
    """Agent whose model streams one `run_code` call for `code` in `chunk_size` pieces."""

    async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) > 1:
            # Two chunks so the text part produces a delta for an index the watcher never registered.
            yield 'do'
            yield 'ne'
            return
        args = raw_args if raw_args is not None else json.dumps({'code': code})
        log.streaming_done = False
        # Name-only first delta: the started part carries no arguments yet.
        yield {1: DeltaToolCall(name='run_code')}
        for offset in range(0, len(args), chunk_size):
            yield {1: DeltaToolCall(json_args=args[offset : offset + chunk_size])}
            # Yield to the event loop so launched speculation tasks actually make progress
            # mid-stream, the way tool latency overlaps decode against a real provider.
            await asyncio.sleep(0)
        log.streaming_done = True

    def call_code(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Non-streamed twin of `stream_code`: used when speculation is off and runs stay non-streaming."""
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        log.streaming_done = True
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code})])

    model = FunctionModel(call_code, stream_function=stream_code)
    agent: Agent[None, str] = Agent(model, deps_type=type(None), capabilities=[capability, *extra_capabilities])

    @agent.tool_plain
    async def search(query: str) -> str:
        """Return a canned result for `query`."""
        if not log.streaming_done:
            log.started_during_stream += 1
        log.calls.append(('search', query))
        await asyncio.sleep(0)
        return f'result:{query}'

    @agent.tool_plain
    async def side_effect(payload: str) -> str:
        """A tool deliberately kept off the allowlist."""
        if not log.streaming_done:  # pragma: no cover - would mean speculation launched it
            log.started_during_stream += 1
        log.calls.append(('side_effect', payload))
        return f'wrote:{payload}'

    @agent.tool_plain
    async def boom(payload: str) -> str:
        """A tool that always fails."""
        log.calls.append(('boom', payload))
        raise RuntimeError('kaboom')

    @agent.tool_plain
    async def approval_gate(value: int) -> str:
        """A tool that requires approval."""
        raise ApprovalRequired()

    @agent.tool_plain
    async def with_metadata(query: str) -> ToolReturn[str]:
        """A tool that returns a `ToolReturn` carrying metadata."""
        log.calls.append(('with_metadata', query))
        return ToolReturn(return_value=f'meta:{query}', metadata={'speculated': True})

    return agent


class TestSpeculation:
    async def test_literal_call_launches_during_stream_and_is_adopted(self):
        """A literal-args call streams past, launches early, and the snippet claims its result."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, padded('a = await search(query="alpha")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 1
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 0

    async def test_disabled_by_default_runs_cold(self):
        """Without `speculate`, nothing launches early and behavior is unchanged."""
        log = ToolLog()
        capability = CodeMode[None]()
        agent = build_agent(log, padded('a = await search(query="alpha")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 0
        assert capability.speculation_stats.launched == 0

    async def test_non_allowlisted_tool_never_launches_early(self):
        """Only allowlisted tools speculate; others run cold even with literal arguments."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('b = await side_effect(payload="x")\na = await search(query="alpha")\nprint(a, b)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert sorted(log.calls) == [('search', 'alpha'), ('side_effect', 'x')]
        assert log.started_during_stream == 1
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_variable_arguments_are_not_speculated(self):
        """A call whose argument is a variable has no literal identity to launch early."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, padded('q = "alpha"\na = await search(query=q)\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 0
        assert capability.speculation_stats.launched == 0

    async def test_unclaimed_branch_launch_is_evicted(self):
        """A literal call in an untaken branch launches, is never claimed, and gets cancelled."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('if False:\n    a = await search(query="never")\nb = await search(query="alpha")\nprint(b)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        # The wrong-branch launch may or may not reach the tool body before eviction, but the
        # snippet's own dispatch ran exactly once, with the taken branch's argument.
        assert log.calls.count(('search', 'alpha')) == 1
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 1

    async def test_both_arms_of_a_conditional_launch(self):
        """Literal calls in both arms launch; the taken arm claims, the untaken arm is evicted."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded(
            'if False:\n    a = await search(query="alpha")\nelse:\n    a = await search(query="beta")\nprint(a)'
        )
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        # The taken arm's dispatch adopted its launch: the tool body ran once for beta.
        assert log.calls.count(('search', 'beta')) == 1
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 1

    async def test_identical_call_in_both_arms_launches_per_occurrence(self):
        """Occurrence-exact multiplicity is syntactic: exclusive arms still launch twice.

        The dedupe in the streaming scan counts demand per occurrence and compares against
        launches already in flight from other parts; it does not reason about branch
        exclusivity within a part. Whichever arm runs claims one launch, the other launch
        is evicted at commit.
        """
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded(
            'if False:\n    a = await search(query="same")\nelse:\n    a = await search(query="same")\nprint(a)'
        )
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 1

    async def test_loop_body_literal_call_launches_once(self):
        """A literal call in a loop launches per occurrence, not per iteration.

        Iteration one claims the single launch; iteration two finds the queue empty and
        runs cold as a miss. Reachability is predicted, trip counts are not.
        """
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded(
            'results = []\nfor i in [1, 2]:\n    r = await search(query="fixed")\n    results.append(r)\nprint(results)'
        )
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        # Tool body ran twice: once inside the adopted launch, once cold for iteration two.
        assert log.calls.count(('search', 'fixed')) == 2
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 0

    async def test_repeated_identical_calls_launch_and_adopt_per_occurrence(self):
        """N identical dispatches claim N launches; nondeterministic results never collapse."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('a = await search(query="alpha")\nb = await search(query="alpha")\nprint(a, b)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha'), ('search', 'alpha')]
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 2

    async def test_broken_code_stream_speculates_nothing(self):
        """A snippet that never parses launches nothing and leaves the retry path untouched."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, 'a = await search(query="alpha"\nprint(a)', capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 0


def build_run_context(deps: None) -> RunContext[None]:
    """Build a `RunContext` for invoking the capability's public hooks directly.

    Mirrors the helper in `test_code_mode.py`.
    """
    return RunContext[None](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        pending_messages=[],
    )


class _PlainEventStream:
    """An async iterable of events that is not an async generator, so it has no `aclose`."""

    def __init__(self, events: Sequence[AgentStreamEvent]) -> None:
        self._events = list(events)

    def __aiter__(self) -> _PlainEventStream:
        return self

    async def __anext__(self) -> AgentStreamEvent:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class TestSpeculationEdgeCases:
    async def test_inactive_under_temporal_durability(self):
        """Under Temporal, nothing launches early and dispatches simply miss the store."""

        class TemporalDurability(AbstractCapability[None]):
            in_durable_context = True

        TemporalDurability.__module__ = 'pydantic_ai.durable_exec.temporal'
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        code = padded('a = await search(query="alpha")\nprint(a)')
        agent = build_agent(log, code, capability, extra_capabilities=[TemporalDurability()])

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('search', 'alpha')]
        assert log.started_during_stream == 0
        assert capability.speculation_stats.launched == 0

    async def test_speculated_tool_error_surfaces_at_claim(self):
        """A failed launch delivers its error where the cold call would have raised it."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['boom'])
        agent = build_agent(log, padded('a = await boom(payload="x")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('boom', 'x')]
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_denied_speculated_call_records_denial(self):
        """A handler denial reached through a launch is recorded and raised like a cold denial."""

        async def deny_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            return DeferredToolResults(
                approvals={call.tool_call_id: ToolDenied(message='nope') for call in requests.approvals}
            )

        log = ToolLog()
        capability = CodeMode[None](speculate=['approval_gate'])
        code = padded('a = await approval_gate(value=1)\nprint(a)')
        agent = build_agent(log, code, capability, extra_capabilities=[HandleDeferredToolCalls(handler=deny_all)])

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_unhandled_approval_in_speculated_call_becomes_user_error(self):
        """With no handler capability, a launch hitting `ApprovalRequired` mirrors the cold error."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['approval_gate'])
        agent = build_agent(log, padded('a = await approval_gate(value=1)\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_tool_return_metadata_survives_adoption(self):
        """A `ToolReturn`-returning tool keeps its metadata on the adopted nested return part."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['with_metadata'])
        agent = build_agent(log, padded('a = await with_metadata(query="m")\nprint(a)'), capability)

        result = await agent.run('go')

        assert result.output == 'done'
        assert log.calls == [('with_metadata', 'm')]
        assert capability.speculation_stats.adopted == 1

    async def test_malformed_argument_stream_launches_nothing(self):
        """Arguments that never decode as JSON produce no launches and no watcher crash."""
        log = ToolLog()
        capability = CodeMode[None](speculate=['search'])
        agent = build_agent(log, '', capability, raw_args='this is not json at all')

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 0

    async def test_rekeyed_part_id_still_claims_by_exact_arguments(self):
        """A launch recorded under the streamed part id is claimable under a different execution id.

        Some providers re-key a tool call between its streamed part and its executed form;
        strict per-part lookup would turn every launch into a silent miss. The purity promise
        makes an exact `(function, arguments)` match from another watch interchangeable.
        """

        def search(query: str) -> str:
            """Return a canned result."""
            return f'result:{query}'

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['search'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(search)]))
        assert isinstance(toolset, CodeModeToolset)
        tools = await toolset.get_tools(ctx)

        code = 'a = await search(query="alpha")\nprint(a)\na'
        events: list[AgentStreamEvent] = [
            PartStartEvent(index=0, part=ToolCallPart(tool_name='run_code', args={}, tool_call_id='streamed-id')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': code})),
        ]
        async for _ in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(events)):
            pass
        assert run_capability.speculation_stats.launched == 1

        exec_ctx = dataclasses.replace(ctx, tool_call_id='rekeyed-id', tool_name='run_code')
        exec_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(exec_ctx)
        async with toolset:
            result = await toolset.call_tool('run_code', {'code': code}, exec_ctx, tools['run_code'])

        assert isinstance(result, ToolReturn)
        assert "'result': 'result:alpha'" in repr(result.return_value)
        assert run_capability.speculation_stats.adopted == 1

    async def test_failed_snippet_keeps_launches_for_the_retry(self):
        """A snippet that dies before dispatching keeps its launches; the retry claims them.

        Syntax and type errors fail the feed before any call dispatches. Evicting there
        turned every launch of a failed attempt into waste, and retries relaunched work
        that was already running. The retry's fresh part id adopts the surviving launches
        through the cross-watch claim, and the streaming dedupe keeps the retry from
        double-launching them.
        """
        log = ToolLog()
        attempts: list[int] = []
        capability = CodeMode[None](speculate=['search'])
        good = padded('a = await search(query="alpha")\nprint(a)')

        async def stream_attempts(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior_calls = sum(
                1 for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)
            )
            if prior_calls >= 2:
                yield 'done'
                return
            attempts.append(prior_calls)
            if prior_calls == 0:
                # Same literal call, then a name Monty's type check rejects.
                code = 'a = await search(query="alpha")\nundefined_name'
            else:
                code = good
            args = json.dumps({'code': code})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                await asyncio.sleep(0)

        def search(query: str) -> str:
            """Return a canned result."""
            log.calls.append(('search', query))
            return f'result:{query}'

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_attempts),
            deps_type=type(None),
            capabilities=[capability],
            tools=[Tool(search)],
        )

        result = await agent.run('go')

        assert result.output == 'done'
        assert len(attempts) == 2
        # One launch total: the failed attempt's launch survived, the retry deduped
        # against it and claimed it. The tool body ran once.
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1
        assert log.calls == [('search', 'alpha')]

    async def test_execution_prefetch_runs_sequential_awaits_concurrently(self):
        """Literal calls the stream never saw all launch before Monty starts executing.

        The tools deadlock unless both are started before either finishes, so the test
        passes only when the prefetch actually runs them concurrently.
        """
        ping_started = asyncio.Event()
        pong_started = asyncio.Event()

        async def ping() -> str:
            """Wait for pong to start, then return."""
            ping_started.set()
            await asyncio.wait_for(pong_started.wait(), timeout=5)
            return 'ping'

        async def pong() -> str:
            """Wait for ping to start, then return."""
            pong_started.set()
            await asyncio.wait_for(ping_started.wait(), timeout=5)
            return 'pong'

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['ping', 'pong'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(ping), Tool(pong)]))
        assert isinstance(toolset, CodeModeToolset)
        tools = await toolset.get_tools(ctx)

        code = 'a = await ping()\nb = await pong()\na + b'
        exec_ctx = dataclasses.replace(ctx, tool_call_id='exec-1', tool_name='run_code')
        exec_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(exec_ctx)
        async with toolset:
            result = await toolset.call_tool('run_code', {'code': code}, exec_ctx, tools['run_code'])

        assert isinstance(result, ToolReturn)
        assert repr(result.return_value) == "'pingpong'"
        assert run_capability.speculation_stats.launched == 2
        assert run_capability.speculation_stats.adopted == 2
        assert toolset.speculation is not None
        launch_events = [e for e in toolset.speculation.pending_events if isinstance(e, SpeculativeCallLaunchedEvent)]
        assert [e.phase for e in launch_events] == ['execution', 'execution']

    async def test_execution_prefetch_guards_no_step_and_the_launch_cap(self):
        """Prefetch is inert before a step stashes ingredients, and the per-part cap holds."""

        def search(query: str) -> str:
            """Return a canned result."""
            return f'result:{query}'  # pragma: no cover - launches are cancelled before the body runs

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['search'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(search)]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.speculation is not None

        toolset.speculation.prelaunch_for_execution('p0', 'a = await search(query="x")\nprint(a)', ctx)
        assert run_capability.speculation_stats.launched == 0

        await toolset.get_tools(ctx)
        # The def body and the lambda hold eligible-looking calls that must not launch:
        # the AST extractor skips container statements and container children.
        skip_shapes = 'def helper():\n    return search(query="never")\nfn = lambda: search(query="never")\n'
        big = skip_shapes + '\n'.join(f'x{i} = await search(query="q{i}")' for i in range(33)) + '\nprint(1)'
        toolset.speculation.prelaunch_for_execution('p1', big, ctx)
        assert run_capability.speculation_stats.launched == 32

        await toolset.speculation.evict_part('p1')

    async def test_execution_prefetch_launches_only_the_deficit(self):
        """Streamed launches count against the prefetch, so FIFO multiplicity stays exact."""

        def search(query: str) -> str:
            """Return a canned result."""
            return f'result:{query}'

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['search'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(search)]))
        assert isinstance(toolset, CodeModeToolset)
        tools = await toolset.get_tools(ctx)

        streamed = 'a = await search(query="alpha")\nprint(a)\n'
        events: list[AgentStreamEvent] = [
            PartStartEvent(index=0, part=ToolCallPart(tool_name='run_code', args={}, tool_call_id='c1')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': streamed})),
        ]
        async for _ in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(events)):
            pass
        assert run_capability.speculation_stats.launched == 1

        code = 'a = await search(query="alpha")\nb = await search(query="alpha")\nb'
        exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        exec_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(exec_ctx)
        async with toolset:
            await toolset.call_tool('run_code', {'code': code}, exec_ctx, tools['run_code'])

        # Two identical dispatches, one streamed launch: the prefetch adds exactly one more.
        assert run_capability.speculation_stats.launched == 2
        assert run_capability.speculation_stats.adopted == 2

    async def test_part_end_launches_the_statements_streaming_held_back(self):
        """A snippet whose scans never fired still launches when the arguments finish.

        Text-level extraction launches a call as soon as its closing paren streams, but
        scans are newline-batched: a call whose paren closes in a later delta on the same
        line is not seen mid-stream, so without the final scan it would never launch and
        its dispatch would go cold.
        """

        def search(query: str) -> str:
            """Return a canned result."""
            return f'result:{query}'  # pragma: no cover - launches are cancelled before the body runs

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['search'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(search)]))
        assert isinstance(toolset, CodeModeToolset)
        await toolset.get_tools(ctx)

        part = ToolCallPart(tool_name='run_code', args='', tool_call_id='c1')
        events: list[AgentStreamEvent] = [
            PartStartEvent(index=0, part=part),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"code": "a = await search(query=')),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='\\"alpha\\")"}')),
        ]
        async for _ in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(events)):
            pass
        assert run_capability.speculation_stats.launched == 0

        end_events: list[AgentStreamEvent] = [PartEndEvent(index=0, part=part)]
        seen = [
            event async for event in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(end_events))
        ]

        assert run_capability.speculation_stats.launched == 1
        launches = [e for e in seen if isinstance(e, SpeculativeCallLaunchedEvent)]
        assert [e.wrapped_tool_name for e in launches] == ['search']
        updates = [e for e in seen if isinstance(e, SpeculativeCodeUpdateEvent)]
        assert updates[-1].closed_statements == 1

    async def test_watcher_handles_dict_args_odd_indexes_and_caps_launches(self):
        """Dict-argument deltas, unknown part indexes, plain (non-generator) streams, and the
        per-part launch cap, driven through the capability's public hooks."""

        def search(query: str) -> str:
            """Return a canned result."""
            return f'result:{query}'

        # Launched tasks may be cancelled before any reaches the tool body (that's the eviction
        # contract), so the body's coverage cannot depend on task scheduling; cover it directly.
        assert search(query='direct') == 'result:direct'

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['search'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(search)]))
        assert isinstance(toolset, CodeModeToolset)

        # Before any step stashed dispatch ingredients, streamed parts are watched but nothing
        # can launch. A string-args part start exercises the initial-args accumulation path.
        pre_events: list[AgentStreamEvent] = [
            PartStartEvent(
                index=0, part=ToolCallPart(tool_name='run_code', args='{"code": "y = 1\\n', tool_call_id='c0')
            ),
        ]
        seen = [
            event async for event in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(pre_events))
        ]
        assert seen == pre_events
        assert run_capability.speculation_stats.launched == 0

        await toolset.get_tools(ctx)

        code = (
            'def helper():\n    return search(query="inner")\n'
            'if True:\n    def nested():\n        return search(query="nested")\n'
            'if False:\n    search("positional")\n'
            'if False:\n    search(**{"query": "z"})\n'
            + ''.join(f'x{i} = search(query="q")\n' for i in range(34))
            + 'done = 1\n'
        )
        events: list[AgentStreamEvent] = [
            PartStartEvent(index=0, part=TextPart(content='hi')),
            PartStartEvent(index=1, part=ToolCallPart(tool_name='run_code', args={}, tool_call_id='c1')),
            PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta=None, tool_call_id='c1')),
            PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta={'code': code})),
            PartDeltaEvent(index=9, delta=ToolCallPartDelta(args_delta='{}')),
        ]

        seen = [event async for event in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(events))]

        # Wrapped events pass through in order; the wrapper interleaves its own
        # speculation events (yield-based delivery) between them.
        assert [e for e in seen if not isinstance(e, CapabilityEvent)] == events
        # 34 identical literal calls stream past; the def bodies (top-level and nested), the
        # positional call, and the double-star call are ineligible, and the per-part cap stops
        # launches at the limit.
        assert run_capability.speculation_stats.launched == 32

        with pytest.raises(RuntimeError, match='synthetic run failure'):
            await run_capability.on_run_error(ctx, error=RuntimeError('synthetic run failure'))
        assert run_capability.speculation_stats.evicted == 32

    async def test_run_error_without_speculation_passes_through(self):
        """`on_run_error` re-raises untouched when speculation was never enabled."""
        capability = CodeMode[None]()
        with pytest.raises(RuntimeError, match='plain failure'):
            await capability.on_run_error(build_run_context(None), error=RuntimeError('plain failure'))


def event_collector(events: list[CapabilityEvent]):
    """An `event_stream_handler` that keeps the capability events, for `agent.run(...)`."""

    async def collect(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            if isinstance(event, CapabilityEvent):
                events.append(event)

    return collect


def full_stream_collector(events: list[AgentStreamEvent]):
    """An `event_stream_handler` that keeps every stream event, for interleaving assertions."""

    async def collect(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    return collect


class TestSpeculationEvents:
    async def test_happy_path_emits_updates_launch_settle_and_claim(self):
        """The full lifecycle reaches the run's event stream with correlated ids and line spans."""
        log = ToolLog()
        events: list[CapabilityEvent] = []
        capability = CodeMode[None](speculate=['search'])
        code = padded('a = await search(query="alpha")\nprint(a)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go', event_stream_handler=event_collector(events))

        assert result.output == 'done'
        updates = [e for e in events if isinstance(e, SpeculativeCodeUpdateEvent)]
        launches = [e for e in events if isinstance(e, SpeculativeCallLaunchedEvent)]
        settles = [e for e in events if isinstance(e, SpeculativeCallSettledEvent)]
        claims = [e for e in events if isinstance(e, SpeculativeCallClaimedEvent)]
        assert len(updates) > 1
        assert updates[-1].code.startswith('a = await search')
        closed_counts = [e.closed_statements for e in updates]
        assert closed_counts == sorted(closed_counts)
        assert [e.kind for e in launches] == ['code_mode.speculative_call_launched']
        launch = launches[0]
        assert launch.sandbox_function == 'search'
        assert launch.wrapped_tool_name == 'search'
        assert launch.arguments == {'query': 'alpha'}
        assert (launch.line_start, launch.line_end) == (1, 1)
        assert launch.tool_call_id is not None
        assert [e.outcome for e in settles] == ['ready']
        assert settles[0].launch_id == launch.launch_id
        (claim,) = claims
        assert claim.launch_id == launch.launch_id
        assert claim.ready_at_claim is True
        assert claim.elapsed_ms >= 0
        assert not [e for e in events if isinstance(e, (SpeculativeCallMissedEvent, SpeculativeCallEvictedEvent))]

    async def test_stream_events_interleave_with_argument_deltas(self):
        """Updates and launches arrive while the `run_code` part is still streaming.

        Pins the yield-based delivery: emitting through `ctx.emit_event` buffers stream-phase
        events until the model request's node stream ends, which showed up in the CLI as the
        whole snippet appearing only after its arguments finished streaming.
        """
        log = ToolLog()
        events: list[AgentStreamEvent] = []
        capability = CodeMode[None](speculate=['search'])
        code = padded('a = await search(query="alpha")\nprint(a)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go', event_stream_handler=full_stream_collector(events))

        assert result.output == 'done'
        part_end_at = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, PartEndEvent) and isinstance(e.part, ToolCallPart) and e.part.tool_name == 'run_code'
        )
        first_update_at = next(i for i, e in enumerate(events) if isinstance(e, SpeculativeCodeUpdateEvent))
        first_launch_at = next(i for i, e in enumerate(events) if isinstance(e, SpeculativeCallLaunchedEvent))
        assert first_update_at < part_end_at
        assert first_launch_at < part_end_at

    async def test_eligible_cold_dispatch_emits_miss(self):
        """A variable-argument dispatch of an eligible tool reports a miss, and no launch."""
        log = ToolLog()
        events: list[CapabilityEvent] = []
        capability = CodeMode[None](speculate=['search'])
        code = padded('q = "alpha"\na = await search(query=q)\nprint(a)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go', event_stream_handler=event_collector(events))

        assert result.output == 'done'
        misses = [e for e in events if isinstance(e, SpeculativeCallMissedEvent)]
        assert [e.wrapped_tool_name for e in misses] == ['search']
        assert misses[0].nested_tool_call_id
        assert not [e for e in events if isinstance(e, SpeculativeCallLaunchedEvent)]

    async def test_unclaimed_branch_launch_emits_eviction(self):
        """The wrong-branch launch reports an eviction alongside the taken branch's claim."""
        log = ToolLog()
        events: list[CapabilityEvent] = []
        capability = CodeMode[None](speculate=['search'])
        code = padded('if False:\n    a = await search(query="never")\nb = await search(query="alpha")\nprint(b)')
        agent = build_agent(log, code, capability)

        result = await agent.run('go', event_stream_handler=event_collector(events))

        assert result.output == 'done'
        evictions = [e for e in events if isinstance(e, SpeculativeCallEvictedEvent)]
        claims = [e for e in events if isinstance(e, SpeculativeCallClaimedEvent)]
        assert len(claims) == 1
        assert [e.state in ('pending', 'ready') for e in evictions] == [True]
        assert evictions[0].launch_id != claims[0].launch_id

    async def test_failed_launch_settles_failed_and_still_claims(self):
        """A failing tool's launch settles as `failed`; adoption still reports the claim."""
        log = ToolLog()
        events: list[CapabilityEvent] = []
        capability = CodeMode[None](speculate=['boom'])
        agent = build_agent(log, padded('a = await boom(payload="x")\nprint(a)'), capability)

        result = await agent.run('go', event_stream_handler=event_collector(events))

        assert result.output == 'done'
        settles = [e for e in events if isinstance(e, SpeculativeCallSettledEvent)]
        claims = [e for e in events if isinstance(e, SpeculativeCallClaimedEvent)]
        assert [e.outcome for e in settles] == ['failed']
        assert len(claims) == 1


class TestDeclaredSpeculation:
    """`speculate='declared'`: tools carry their own evidence instead of a user allowlist."""

    async def test_tool_metadata_and_mcp_annotations_grant_eligibility(self):
        """Eligibility comes from the tool definitions the run actually presents."""

        def free(query: str) -> str:
            """Declared side-effect free."""
            return query  # pragma: no cover - eligibility test only

        def idem(query: str) -> str:
            """Declared idempotent."""
            return query  # pragma: no cover - eligibility test only

        def readonly(query: str) -> str:
            """Carries an MCP readOnlyHint annotation."""
            return query  # pragma: no cover - eligibility test only

        def malformed(query: str) -> str:
            """Annotations that are not a mapping present no evidence."""
            return query  # pragma: no cover - eligibility test only

        def hintless(query: str) -> str:
            """Annotations without safety hints present no evidence."""
            return query  # pragma: no cover - eligibility test only

        def plain(query: str) -> str:
            """No metadata at all."""
            return query  # pragma: no cover - eligibility test only

        def ordered(query: str) -> str:
            """Declared safe but sequential; ordering wins."""
            return query  # pragma: no cover - eligibility test only

        toolset = FunctionToolset[None](
            tools=[
                Tool(free, metadata={'read_only': True}),
                Tool(idem, metadata={'idempotent': True}),
                Tool(readonly, metadata={'annotations': {'readOnlyHint': True, 'title': 'Read'}}),
                Tool(malformed, metadata={'annotations': 'not-a-mapping'}),
                Tool(hintless, metadata={'annotations': {'title': 'No hints'}}),
                Tool(plain),
                Tool(ordered, sequential=True, metadata={'read_only': True}),
            ]
        )
        ctx = build_run_context(None)
        capability = CodeMode[None](speculate='declared')
        run_capability = await capability.for_run(ctx)
        wrapper = run_capability.get_wrapper_toolset(toolset)
        assert isinstance(wrapper, CodeModeToolset)
        await wrapper.get_tools(ctx)

        speculation = wrapper.speculation
        assert speculation is not None
        assert speculation.eligible('free')
        assert speculation.eligible('idem')
        assert speculation.eligible('readonly')
        assert not speculation.eligible('malformed')
        assert not speculation.eligible('hintless')
        assert not speculation.eligible('plain')
        assert not speculation.eligible('ordered')

    async def test_declared_tool_speculates_at_agent_level(self):
        """A metadata-declared tool launches from the stream; an undeclared one stays cold."""
        starts: list[str] = []
        streaming_done = False

        async def search(query: str) -> str:
            """Declared side-effect free."""
            starts.append(f'search:{"stream" if not streaming_done else "exec"}')
            await asyncio.sleep(0)
            return f'result:{query}'

        async def fetch(url: str) -> str:
            """No declaration; must not launch early."""
            starts.append(f'fetch:{"stream" if not streaming_done else "exec"}')
            return f'page:{url}'

        code = padded('a = await search(query="alpha")\nb = await fetch(url="u")\nprint(a, b)')

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            nonlocal streaming_done
            if len(messages) > 1:
                yield 'done'
                return
            streaming_done = False
            args = json.dumps({'code': code})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                await asyncio.sleep(0)
            streaming_done = True

        capability = CodeMode[None](speculate='declared')
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
            tools=[Tool(search, metadata={'read_only': True}), Tool(fetch)],
        )

        result = await agent.run('go')

        assert result.output == 'done'
        assert starts == ['search:stream', 'fetch:exec']
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_branch_launches_overlap_a_blocking_earlier_statement(self):
        """Both branch arms are already in flight while an earlier statement still executes.

        The screenshot scenario: a blocking first statement, then a conditional whose arms
        each call an allowlisted tool. The gate refuses to return until both arm calls have
        started, so the run deadlocks unless the engine looked ahead past the blocking
        statement and launched both arms.
        """
        starts: list[str] = []
        both_started = asyncio.Event()

        async def search(query: str) -> str:
            """Return a canned result."""
            starts.append(query)
            if len(starts) >= 2:
                both_started.set()
            await asyncio.sleep(0)
            return f'result:{query}'

        async def gate(x: int) -> str:
            """Block until both branch arms have started executing."""
            await asyncio.wait_for(both_started.wait(), timeout=5)
            return 'open'

        code = padded(
            'g = await gate(x=1)\n'
            'if g == "open":\n'
            '    a = await search(query="alpha")\n'
            'else:\n'
            '    a = await search(query="beta")\n'
            'print(a)'
        )

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': code})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                await asyncio.sleep(0)

        capability = CodeMode[None](speculate=['search'])
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
            tools=[Tool(search), Tool(gate)],
        )

        result = await agent.run('go')

        assert result.output == 'done'
        assert sorted(starts) == ['alpha', 'beta']
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 1
        parts = [
            p
            for m in result.all_messages()
            for p in getattr(m, 'parts', [])
            if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
        ]
        metadata: dict[str, Any] = parts[-1].metadata
        assert 'eager' not in metadata
        speculation_meta: dict[str, Any] = metadata['speculation']
        assert speculation_meta['hits'] == 1
        assert speculation_meta['misses'] == 0
        assert speculation_meta['wasted'] == 1
        assert speculation_meta['hidden_ms'] >= 0

    async def test_a_bare_string_that_is_not_declared_is_rejected(self):
        """`speculate='search'` is a likely typo for a one-element list, not a mode."""
        with pytest.raises(UserError, match='one-element list'):
            CodeMode[None](speculate='search')


class TestTierComposition:
    """`eager=True` with `speculate=[...]`: the REPL frontier advances while launches run ahead."""

    async def test_eager_execution_and_speculative_launches_compose(self):
        """Deadlock-unless-both: mid-stream execution and past-the-blocker launches.

        The model refuses to finish streaming until the blocking first statement has
        started executing in the REPL (only eager can do that), and the blocking statement
        refuses to return until both branch arms' calls have started (only speculative
        lookahead past a blocking statement can do that). The taken arm then claims its
        launch from an eagerly fed fragment.
        """
        gate_started = asyncio.Event()
        release_gate = asyncio.Event()
        starts: list[str] = []

        async def gate(x: int) -> str:
            """Block until both branch arms are in flight."""
            gate_started.set()
            await asyncio.wait_for(release_gate.wait(), timeout=5)
            return 'open'

        async def search(query: str) -> str:
            """Return a canned result, releasing the gate once both arms started."""
            starts.append(query)
            if len(starts) >= 2:
                release_gate.set()
            await asyncio.sleep(0)
            return f'result:{query}'

        code = (
            'g = await gate(x=1)\n'
            'if g == "open":\n'
            '    a = await search(query="alpha")\n'
            'else:\n'
            '    a = await search(query="beta")\n'
            'print(a)\n'
            '"ok"'
        )

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': code})
            chunks = [args[offset : offset + 16] for offset in range(0, len(args), 16)]
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in chunks[:-1]:
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            await asyncio.wait_for(gate_started.wait(), timeout=5)
            yield {1: DeltaToolCall(json_args=chunks[-1])}

        capability = CodeMode[None](eager=True, speculate=['search'])
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
            tools=[Tool(search), Tool(gate)],
        )

        result = await agent.run('go')

        assert result.output == 'done'
        assert sorted(starts) == ['alpha', 'beta']
        assert capability.speculation_stats.launched == 2
        assert capability.speculation_stats.adopted == 1
        assert capability.speculation_stats.evicted == 1
        parts = [
            p
            for m in result.all_messages()
            for p in getattr(m, 'parts', [])
            if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
        ]
        metadata: dict[str, Any] = parts[-1].metadata
        eager_meta: dict[str, Any] = metadata['eager']
        assert eager_meta['statements'] >= 1
        assert eager_meta['executed_ms'] >= 0
        speculation_meta: dict[str, Any] = metadata['speculation']
        assert speculation_meta['hits'] == 1
        assert speculation_meta['misses'] == 0
        assert speculation_meta['wasted'] == 1
        assert speculation_meta['hidden_ms'] >= 0


class TestCallLevelLaunch:
    """Launches fire when a call's text completes, not when its statement closes."""

    async def test_call_launches_while_its_statement_is_still_streaming(self):
        """Deadlock-unless-call-level: the stream stalls until the open `if` arm's call starts.

        The `if` statement never closes before the model's last chunks (closure requires a
        later top-level statement), so statement-level launching would deadlock here: the
        model refuses to finish generating until the call inside the open arm is running.
        """
        started = asyncio.Event()

        async def search(query: str) -> str:
            """Return a canned result."""
            started.set()
            return f'result:{query}'

        code = 'if 1 == 1:\n    a = await search(query="alpha")\nelse:\n    a = "cold"\nprint(a)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': code})
            hold_from = args.index('else')
            yield {1: DeltaToolCall(name='run_code')}
            yield {1: DeltaToolCall(json_args=args[:hold_from])}
            await asyncio.wait_for(started.wait(), timeout=5)
            yield {1: DeltaToolCall(json_args=args[hold_from:])}

        capability = CodeMode[None](speculate=['search'])
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
            tools=[Tool(search)],
        )

        result = await agent.run('go')

        assert result.output == 'done'
        assert capability.speculation_stats.launched == 1
        assert capability.speculation_stats.adopted == 1

    async def test_text_extraction_skips_non_calls_and_non_literals(self):
        """The raw-text extractor launches only genuine literal-keyword calls.

        Attribute access, `def` parameter lists, unclosed spans, positional or non-literal
        arguments are all skipped; nested container literals and escaped quotes inside
        argument strings are matched through.
        """
        starts: list[str] = []

        async def search(query: str | dict[str, list[int]]) -> str:
            """Return a canned result."""
            starts.append(repr(query))
            return 'r'

        ctx = build_run_context(None)
        capability = CodeMode[None](speculate=['search'])
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(search)]))
        assert isinstance(toolset, CodeModeToolset)
        await toolset.get_tools(ctx)

        code = (
            'client.search(query="attribute")\n'
            'def search(query="param"): pass\n'
            "s = 'search(query=\"unclosed\\'\n"
            'bad = "search(,)"\n'
            'ref = await search(query=variable)\n'
            'pos = await search("positional")\n'
            'a = await search(query="a\\"b")\n'
            'b = await search(query={"k": [1, 2]})\n'
            'c = 1\n'
        )
        part = ToolCallPart(tool_name='run_code', args={}, tool_call_id='c1')
        events: list[AgentStreamEvent] = [
            PartStartEvent(index=0, part=part),
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': code})),
        ]
        async for _ in run_capability.wrap_run_event_stream(ctx, stream=_PlainEventStream(events)):
            pass

        await asyncio.sleep(0.05)
        assert run_capability.speculation_stats.launched == 2
        assert sorted(starts) == ["'a\"b'", "{'k': [1, 2]}"]
