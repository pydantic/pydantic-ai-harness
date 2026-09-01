"""Tests for `CodeMode(eager=True)`: streamed `run_code` statements execute as they close.

Behavioral, through `Agent(..., capabilities=[CodeMode(eager=True)])` with a streaming
`FunctionModel`. The central proof is a stream that refuses to finish until the first
statement's tool call has executed: only mid-stream execution can unblock it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import types
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from itertools import zip_longest
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.code_mode import CodeMode, CodeModeToolset, EagerPrefixCommittedEvent

pytestmark = pytest.mark.anyio


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


@pytest.fixture
def anyio_backend() -> str:
    """Eager pumps schedule with `asyncio.ensure_future`; the trio backend does not apply."""
    return 'asyncio'


async def _plain_stream(events: Sequence[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
    for event in events:
        yield event


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


def _prior_run_code_calls(messages: list[ModelMessage]) -> int:
    return sum(1 for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart))


def _stream_json_args(code: str, chunk_size: int = 16) -> list[str]:
    args = json.dumps({'code': code})
    return [args[offset : offset + chunk_size] for offset in range(0, len(args), chunk_size)]


def _run_code_return_content(messages: list[ModelMessage]) -> object:
    contents = [
        p.content
        for m in messages
        for p in getattr(m, 'parts', [])
        if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
    ]
    assert contents, 'no run_code ToolReturnPart in history'
    return contents[-1]


def _run_code_return_metadata(messages: list[ModelMessage]) -> dict[str, Any]:
    parts = [
        p
        for m in messages
        for p in getattr(m, 'parts', [])
        if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
    ]
    assert parts, 'no run_code ToolReturnPart in history'
    metadata: dict[str, Any] = parts[-1].metadata
    return metadata


class TestEagerExecution:
    async def test_statements_execute_while_the_model_is_still_streaming(self):
        """The stream stalls until the first statement's tool call has run.

        Deadlock-unless-eager: the model refuses to emit the final chunks until `search`
        has executed, which only eager mid-stream feeding can achieve.
        """
        first_call = asyncio.Event()
        calls: list[str] = []

        async def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            first_call.set()
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nb = await search(query="beta")\nprint(a)\nprint(b)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            chunks = _stream_json_args(code)
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in chunks[:-1]:
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            await asyncio.wait_for(first_call.wait(), timeout=5)
            yield {1: DeltaToolCall(json_args=chunks[-1])}

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        events: list[CapabilityEvent] = []

        async def collect(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                if isinstance(event, CapabilityEvent):
                    events.append(event)

        result = await agent.run('go', event_stream_handler=collect)

        assert result.output == 'done'
        assert calls == ['alpha', 'beta']
        content = _run_code_return_content(result.all_messages())
        assert content == {'output': 'result:alpha\nresult:beta\n', 'result': 'ok'}
        commits = [event for event in events if isinstance(event, EagerPrefixCommittedEvent)]
        assert len(commits) == 1
        # The first three statements closed and fed before the held-back final chunk;
        # `print(b)` and the provisional `"ok"` ran as the dispatch tail.
        assert commits[0].statements == 3
        assert commits[0].executed_ms > 0
        assert commits[0].waited_ms >= 0
        eager_meta: dict[str, Any] = _run_code_return_metadata(result.all_messages())['eager']
        assert eager_meta['statements'] == 3
        assert eager_meta['executed_ms'] > 0

    async def test_failed_statement_surfaces_and_earlier_state_persists(self):
        """A statement that fails mid-stream stops the pump; the retry sees prior assignments.

        Identical to the non-eager failure contract: assignments before the failing line
        persist, the error arrives as the `run_code` result, and the tool that already ran
        is not re-executed by the retry.
        """
        calls: list[str] = []

        def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            return f'result:{query}'

        bad = 'a = await search(query="alpha")\nboom = 1 // 0\nprint(a)\nprint(boom)\n"x"'
        good = 'print(a)\n"recovered"'

        async def stream_attempts(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior = _prior_run_code_calls(messages)
            if prior >= 2:
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(bad if prior == 0 else good):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_attempts),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']
        content = _run_code_return_content(result.all_messages())
        assert content == {'output': 'result:alpha\n', 'result': 'recovered'}

    async def test_restart_discards_the_pumped_prefix(self):
        """`restart: true` resets the session; the streamed prefix's state is gone."""
        calls: list[str] = []

        def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nprint(a)\n"ok"'

        async def stream_restart(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            args = json.dumps({'code': code, 'restart': True})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_restart),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        # The pumped prefix ran once, then restart re-ran the full snippet from scratch.
        assert calls == ['alpha', 'alpha']

    async def test_diverged_code_resets_the_session(self):
        """Execution with a different prefix than the pump ran raises a retry and resets."""
        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None
        tools = await toolset.get_tools(ctx)

        exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        exec_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(exec_ctx)
        stream_ctx = dataclasses.replace(ctx)
        stream_ctx.tool_manager = exec_ctx.tool_manager
        async with toolset:
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': 'x = 1\ny = 2\nprint(x)'}, tool_call_id='c1'),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_PlainEventStream(events)):
                pass
            with pytest.raises(ModelRetry, match='no longer matches'):
                await toolset.call_tool('run_code', {'code': 'z = 9\nw = 8\nprint(z)'}, exec_ctx, tools['run_code'])
            # The diverged part was consumed; nothing is left to adopt.
            assert toolset.eager.pop_watch('cX', 'anything') is None

    async def test_take_tolerates_rewritten_ids_and_rejects_foreign_code(self):
        """A re-keyed execution adopts the part whose executed prefix matches its code."""
        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None
        await toolset.get_tools(ctx)

        stream_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        stream_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(stream_ctx)
        async with toolset:
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': 'x = 1\ny = 2\nprint(x)'}, tool_call_id='c1'),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_plain_stream(events)):
                pass
            taken = eager.pop_watch('provider-rewrote-this', 'x = 1\ny = 2\nprint(x)')
            assert taken is not None and taken.tool_call_id == 'c1'
            await eager.drain(taken)

    async def test_observe_edges_and_idle_close(self):
        """Non-run_code parts, unknown ids, dict merges, and broken partial JSON are inert."""
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None

        # Other tools are ignored outright.
        await eager.observe(
            PartStartEvent(index=0, part=ToolCallPart(tool_name='other', args={}, tool_call_id='o1')), ctx
        )
        # String args at part start participate; a single statement stays provisional (no feed).
        await eager.observe(
            PartStartEvent(
                index=1, part=ToolCallPart(tool_name='run_code', args='{"code": "x = 1"}', tool_call_id='c1')
            ),
            ctx,
        )
        # A delta for an id the watcher never saw is dropped.
        await eager.observe(PartDeltaEvent(index=9, delta=ToolCallPartDelta(args_delta='x', tool_call_id='zzz')), ctx)
        # Dict deltas merge without disturbing the code; the newline gate skips a re-parse.
        await eager.observe(
            PartStartEvent(index=2, part=ToolCallPart(tool_name='run_code', args={'code': 'q = 7'}, tool_call_id='c2')),
            ctx,
        )
        await eager.observe(
            PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta={'restart': False}, tool_call_id='c2')), ctx
        )
        # With two live parts, an id-less delta matches nothing.
        await eager.observe(PartDeltaEvent(index=3, delta=ToolCallPartDelta(args_delta='x')), ctx)
        # Broken partial JSON has no code yet.
        await eager.observe(
            PartStartEvent(index=4, part=ToolCallPart(tool_name='run_code', args='{"cod', tool_call_id='c3')), ctx
        )
        # Arguments that decode to a non-mapping have no code either.
        await eager.observe(
            PartStartEvent(index=7, part=ToolCallPart(tool_name='run_code', args='[', tool_call_id='c6')), ctx
        )
        assert eager.pop_watch('c6', 'anything') is not None
        # Completed lines that do not parse close nothing: an open bracket resolves later.
        await eager.observe(
            PartStartEvent(
                index=8,
                part=ToolCallPart(tool_name='run_code', args={'code': 'x = (\n1,\n'}, tool_call_id='c7'),
            ),
            ctx,
        )
        assert eager.pop_watch('c7', 'x = (\n1,\n2)') is not None
        # A `None` args delta changes nothing.
        await eager.observe(PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta=None, tool_call_id='c2')), ctx)
        # Clear the empty-prefix parts, then verify a foreign execution matches nothing
        # against a part whose executed prefix is real.
        assert eager.pop_watch('c1', 'x = 1') is not None
        assert eager.pop_watch('c2', 'q = 7') is not None
        assert eager.pop_watch('c3', '') is not None
        await eager.observe(
            PartStartEvent(
                index=5,
                part=ToolCallPart(tool_name='run_code', args={'code': 'm = 1\nn = 2\nprint(m)'}, tool_call_id='c4'),
            ),
            ctx,
        )
        assert eager.pop_watch('cZ', 'entirely different program') is None
        taken = eager.pop_watch('c4', 'm = 1\nn = 2\nprint(m)')
        assert taken is not None
        await eager.drain(taken)
        # Close skips a part that never grew a pump.
        await eager.observe(
            PartStartEvent(index=6, part=ToolCallPart(tool_name='run_code', args={'code': 'k = 1'}, tool_call_id='c5')),
            ctx,
        )
        await eager.close()

    async def test_inactive_under_temporal_durability(self):
        """Under Temporal, nothing feeds early; execution runs the snippet whole."""

        class TemporalDurability(AbstractCapability[None]):
            in_durable_context = True

        TemporalDurability.__module__ = 'pydantic_ai.durable_exec.temporal'
        calls: list[str] = []
        stream_finished = asyncio.Event()

        def search(query: str) -> str:
            """Return a canned result, refusing to run before the stream completes."""
            assert stream_finished.is_set(), 'search executed before the model stream completed'
            calls.append(query)
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nb = 1\nprint(a)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            stream_finished.set()

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability, TemporalDurability()],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']

    async def test_on_run_error_closes_eager_state(self):
        """The error hook tears the pump state down and re-raises."""
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        assert run_capability is not capability
        with pytest.raises(RuntimeError, match='boom'):
            await run_capability.on_run_error(ctx, error=RuntimeError('boom'))
        # Without eager enabled there is no pump state; the hook just re-raises.
        plain = CodeMode[None]()
        with pytest.raises(RuntimeError, match='boom'):
            await plain.on_run_error(ctx, error=RuntimeError('boom'))

    async def test_close_cancels_a_blocked_pump(self):
        """Run-end close cancels an in-flight fragment feed without hanging."""
        release = asyncio.Event()
        started = asyncio.Event()

        async def slow(query: str) -> str:
            """Wait until released."""
            started.set()
            await release.wait()
            return query  # pragma: no cover - cancelled before completion

        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(slow)]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None
        await toolset.get_tools(ctx)

        stream_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        stream_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(stream_ctx)
        async with toolset:
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name='run_code',
                        args={'code': 'a = await slow(query="x")\nb = 1\nprint(b)'},
                        tool_call_id='c1',
                    ),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_plain_stream(events)):
                pass
            await asyncio.wait_for(started.wait(), timeout=5)
            await toolset.eager.close()


class TestEagerHardening:
    """Pins for the streaming pipeline's guard rails: budgets, ordering, lifecycle, scanner edges."""

    async def test_call_budget_spans_fragments_and_tail(self):
        """`max_tool_calls` bounds the whole `run_code` call, not each eager fragment."""
        calls: list[str] = []

        def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            return f'result:{query}'

        greedy = 'a = await search(query="alpha")\nb = await search(query="beta")\nprint(a)\n"x"'
        frugal = '"recovered"'

        async def stream_attempts(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior = _prior_run_code_calls(messages)
            if prior >= 2:
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(greedy if prior == 0 else frugal):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True, max_tool_calls=1)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_attempts),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']

    async def test_semicolon_line_keeps_the_final_expression(self):
        """Statements sharing a line never feed early; the final expression survives."""
        calls: list[str] = []

        def charge(x: int) -> int:
            """Return the charge."""
            calls.append(str(x))
            return x

        # The semicolon line is followed by more lines, so the scanner sees its statements
        # close; feeding any of them would drag the whole line (and the call) along.
        code = 'q = 1; pre = await charge(x=1); mid = 3\nfinal = 2\nprint(final)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(charge)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['1']
        content = _run_code_return_content(result.all_messages())
        assert content == {'output': '2\n', 'result': 'ok'}

    async def test_restart_streamed_before_code_feeds_nothing(self):
        """A `restart` key seen in the arguments halts feeding before any statement runs."""
        calls: list[str] = []

        def search(query: str) -> str:
            """Return a canned result."""
            calls.append(query)
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nprint(a)\n"ok"'

        async def stream_restart(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            args = json.dumps({'restart': True, 'code': code})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_restart),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        # Only the dispatch ran the snippet; nothing fed during streaming.
        assert calls == ['alpha']

    async def test_two_streamed_parts_never_execute_concurrently(self):
        """Fragments and tails from different `run_code` parts serialize against the session."""
        depth = 0
        max_depth = 0

        async def probe(q: str) -> str:
            """Track concurrent executions."""
            nonlocal depth, max_depth
            depth += 1
            max_depth = max(max_depth, depth)
            await asyncio.sleep(0.01)
            depth -= 1
            return q

        def snippet(tag: str) -> str:
            return f'a = await probe(q="{tag}")\nb = 1\nprint(b)\n"{tag}"'

        async def stream_two(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            one = _stream_json_args(snippet('one'))
            two = _stream_json_args(snippet('two'))
            yield {1: DeltaToolCall(name='run_code'), 2: DeltaToolCall(name='run_code')}
            for chunk_one, chunk_two in zip_longest(one, two, fillvalue=''):
                yield {1: DeltaToolCall(json_args=chunk_one), 2: DeltaToolCall(json_args=chunk_two)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_two),
            deps_type=type(None),
            capabilities=[capability],
        )
        agent.tool_plain(probe)

        result = await agent.run('go')

        assert result.output == 'done'
        assert max_depth == 1

    async def test_dict_replacement_rescans_and_null_bytes_are_inert(self):
        """A dict delta that rewrites `code` re-scans; NUL bytes defer to the dispatch."""
        calls: list[str] = []

        async def probe(q: str) -> str:
            """Record the call."""
            calls.append(q)
            return q

        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(probe)]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None
        await toolset.get_tools(ctx)

        stream_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        stream_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(stream_ctx)
        async with toolset:
            # v1 does not parse (open bracket); v2 replaces the code with the same newline
            # count, which a newline-counting gate would skip.
            await eager.observe(
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': 'a = (\nz = 1\n'}, tool_call_id='c1'),
                ),
                stream_ctx,
            )
            fixed = 'a = await probe(q="r")\nz = 1\n'
            await eager.observe(
                PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': fixed}, tool_call_id='c1')),
                stream_ctx,
            )
            taken = eager.pop_watch('c1', fixed)
            assert taken is not None
            await eager.drain(taken)
            assert calls == ['r']

            # NUL bytes make `ast.parse` raise `ValueError`; the watcher must stay inert.
            await eager.observe(
                PartStartEvent(
                    index=1,
                    part=ToolCallPart(
                        tool_name='run_code', args={'code': 'x = "a"\x00\nprint(x)\n'}, tool_call_id='c2'
                    ),
                ),
                stream_ctx,
            )
            assert eager.pop_watch('c2', 'anything') is not None

    async def test_rekey_fallback_ignores_unfed_parts(self):
        """A part that fed nothing is never adopted by an unrelated execution."""
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None

        await eager.observe(
            PartStartEvent(
                index=0, part=ToolCallPart(tool_name='run_code', args={'code': 'solo = 1'}, tool_call_id='c1')
            ),
            ctx,
        )
        assert eager.pop_watch('rewritten-id', 'entirely different program') is None
        assert eager.pop_watch('c1', 'solo = 1') is not None

    async def test_run_error_cancels_an_executing_parts_pump(self):
        """A pump whose part already reached execution is still cancelled on run failure."""
        release = asyncio.Event()
        started = asyncio.Event()

        async def slow(query: str) -> str:
            """Wait until released."""
            started.set()
            await release.wait()
            return query  # pragma: no cover - cancelled before completion

        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(slow)]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None
        await toolset.get_tools(ctx)

        code = 'a = await slow(query="x")\nb = 1\nprint(b)'
        stream_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        stream_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(stream_ctx)
        async with toolset:
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1'),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_plain_stream(events)):
                pass
            await asyncio.wait_for(started.wait(), timeout=5)
            # The dispatch popped the part; the run then fails while the pump is mid-feed.
            taken = toolset.eager.pop_watch('c1', code)
            assert taken is not None and taken.pump is not None
            with pytest.raises(RuntimeError, match='boom'):
                await run_capability.on_run_error(ctx, error=RuntimeError('boom'))
            assert taken.pump.done()

    async def test_non_asyncio_backend_leaves_watcher_inactive(self, monkeypatch: pytest.MonkeyPatch):
        """Without an asyncio loop (Trio), the stream passes through unwatched."""

        def no_loop() -> None:
            raise RuntimeError('no running event loop')

        monkeypatch.setattr(
            'pydantic_ai_harness.code_mode._capability.asyncio', types.SimpleNamespace(get_running_loop=no_loop)
        )
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None

        events = [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='run_code', args={'code': 'x = 1\ny = 2\nprint(x)'}, tool_call_id='c1'),
            ),
        ]
        seen = [event async for event in run_capability.wrap_run_event_stream(ctx, stream=_plain_stream(events))]
        assert len(seen) == 1
        assert toolset.eager.pop_watch('c1', 'x = 1\ny = 2\nprint(x)') is None

    async def test_fragment_output_is_capped(self, monkeypatch: pytest.MonkeyPatch):
        """Accumulated fragment prints stop growing at the cap instead of filling host memory."""
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset._EAGER_OUTPUT_CAP', 8)
        code = 'print("aaaaaaaaaa")\nprint("bbbbbbbbbb")\nprint("cc")\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability],
        )

        result = await agent.run('go')

        assert result.output == 'done'
        content = _run_code_return_content(result.all_messages())
        assert isinstance(content, dict)
        output: object = content.get('output')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        assert isinstance(output, str) and '[eager output truncated]' in output

    async def test_failed_fragment_retry_carries_prior_output(self):
        """Prints from fragments that succeeded before the failure reach the retry message."""
        attempts: list[str] = []

        async def stream_attempts(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior = _prior_run_code_calls(messages)
            if prior >= 2:
                yield 'done'
                return
            code = 'print("diagnostic")\nboom = 1 // 0\nx = 1\n"x"' if prior == 0 else '"recovered"'
            attempts.append(code)
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_attempts),
            deps_type=type(None),
            capabilities=[capability],
        )

        result = await agent.run('go')

        assert result.output == 'done'
        assert len(attempts) == 2
        retry_texts = [
            p.model_response()
            for m in result.all_messages()
            for p in getattr(m, 'parts', [])
            if isinstance(p, RetryPromptPart)
        ]
        assert any('[stdout before error]' in t and 'diagnostic' in t for t in retry_texts)

    async def test_step_rebind_feeds_through_the_new_wrapper(self):
        """After `for_run_step` swaps the wrapped toolset, fragments see the new tools."""
        calls: list[str] = []

        async def probe(q: str) -> str:
            """Record the call."""
            calls.append(q)
            return q

        class _StepSwap(FunctionToolset[None]):
            swap_to: FunctionToolset[None] | None = None

            async def for_run_step(self, ctx: RunContext[None]) -> FunctionToolset[None]:
                return self.swap_to if self.swap_to is not None else self

        base = _StepSwap(tools=[])
        base.swap_to = FunctionToolset[None](tools=[Tool(probe)])

        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(base)
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None

        stream_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        async with toolset:
            stepped = await toolset.for_run_step(stream_ctx)
            assert isinstance(stepped, CodeModeToolset)
            stream_ctx.tool_manager = await ToolManager(toolset=stepped).for_run_step(stream_ctx)
            await stepped.get_tools(stream_ctx)
            code = 'a = await probe(q="s")\nz = 1\nprint(z)'
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1'),
                ),
            ]
            async for _ in run_capability.wrap_run_event_stream(stream_ctx, stream=_plain_stream(events)):
                pass
            assert stepped.eager is not None
            taken = stepped.eager.pop_watch('c1', code)
            assert taken is not None
            await stepped.eager.drain(taken)
            assert taken.error is None
            assert calls == ['s']


class TestEagerRewriteAndDurability:
    """Pins for the second-round bot findings: rewritten prefixes and non-Temporal durability."""

    async def test_rewritten_executed_prefix_halts_and_resets(self):
        """A dict delta that rewrites already-executed code cannot forge the divergence check."""
        ctx = build_run_context(None)
        ctx.tool_manager = None
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        assert toolset.eager is not None
        tools = await toolset.get_tools(ctx)

        exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        exec_ctx.tool_manager = await ToolManager(toolset=toolset).for_run_step(exec_ctx)
        stream_ctx = dataclasses.replace(ctx)
        stream_ctx.tool_manager = exec_ctx.tool_manager
        async with toolset:
            original = 'a = 1\nb = 2\nc'
            rewritten = 'a = 9\nb = 2\nc'
            await toolset.eager.observe(
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'code': original}, tool_call_id='c1'),
                ),
                stream_ctx,
            )
            await toolset.eager.observe(
                PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': rewritten}, tool_call_id='c1')),
                stream_ctx,
            )
            # The executed prefix is `a = 1`; the rewritten code no longer starts with it.
            with pytest.raises(ModelRetry, match='no longer matches'):
                await toolset.call_tool('run_code', {'code': rewritten}, exec_ctx, tools['run_code'])

    async def test_inactive_under_dbos_durability(self):
        """Any durable executor deactivates eager, not only Temporal."""

        class DbosDurability(AbstractCapability[None]):
            in_durable_context = True

        DbosDurability.__module__ = 'pydantic_ai.durable_exec.dbos'
        calls: list[str] = []
        stream_finished = asyncio.Event()

        def search(query: str) -> str:
            """Return a canned result, refusing to run before the stream completes."""
            assert stream_finished.is_set(), 'search executed before the model stream completed'
            calls.append(query)
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nb = 1\nprint(a)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _prior_run_code_calls(messages):
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in _stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            stream_finished.set()

        capability = CodeMode[None](eager=True)
        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[capability, DbosDurability()],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']

    async def test_oversized_prefix_is_not_parsed_on_the_host(self):
        """A snippet past the scan cap feeds nothing; the sandbox parser handles it whole."""
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None

        big = 'x = 1\n' * 60_000
        await eager.observe(
            PartStartEvent(index=0, part=ToolCallPart(tool_name='run_code', args={'code': big}, tool_call_id='c1')),
            ctx,
        )
        taken = eager.pop_watch('c1', big)
        assert taken is not None
        assert taken.feed_count == 0 and not taken.queue

    async def test_oversized_streamed_arguments_halt_the_watch(self):
        """Past the scan cap the watch stops decoding, feeding, and accumulating."""
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None

        big = '{"code": "' + 'x = 1\\n' * 60_000
        # Streamed past the cap by a delta: the watch halts and later deltas are dropped.
        await eager.observe(
            PartStartEvent(index=0, part=ToolCallPart(tool_name='run_code', args='', tool_call_id='c1')), ctx
        )
        await eager.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta=big, tool_call_id='c1')), ctx)
        await eager.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='junk', tool_call_id='c1')), ctx)
        taken = eager.pop_watch('c1', 'anything')
        assert taken is not None and taken.halted and taken.feed_count == 0
        assert 'junk' not in taken.args_text
        # Arriving oversized in one part-start event: the decoder refuses it outright.
        await eager.observe(
            PartStartEvent(index=1, part=ToolCallPart(tool_name='run_code', args=big, tool_call_id='c2')), ctx
        )
        assert eager.pop_watch('c2', 'anything') is not None

    async def test_carriage_return_lines_defer_to_dispatch(self):
        """Lone-CR line separators feed nothing; the snippet runs whole at dispatch.

        The AST counts `\r` as a line terminator but the executed-prefix slicing is
        `\n`-based, so scanning such code would feed misaligned slices. Staying inert
        costs eagerness, never correctness.
        """
        ctx = build_run_context(None)
        capability = CodeMode[None](eager=True)
        run_capability = await capability.for_run(ctx)
        toolset = run_capability.get_wrapper_toolset(FunctionToolset[None](tools=[]))
        assert isinstance(toolset, CodeModeToolset)
        eager = toolset.eager
        assert eager is not None

        code = 'a = 1\rb = 2\rc = 3'
        await eager.observe(
            PartStartEvent(index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')),
            ctx,
        )
        taken = eager.pop_watch('c1', code)
        assert taken is not None
        assert taken.feed_count == 0 and not taken.queue
