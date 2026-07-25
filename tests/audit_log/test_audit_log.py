"""Tests for the `AuditLog` capability driven through `Agent.run`."""

from __future__ import annotations

import json

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.audit_log import (
    AuditLog,
    InMemoryAuditSink,
    RunAuditRecord,
    ToolCallRecord,
)
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class RecordingSink:
    """An `AuditSink` that keeps every record in public lists for whole-store assertions."""

    def __init__(self) -> None:
        self.tool_calls: list[ToolCallRecord] = []
        self.runs: list[RunAuditRecord] = []

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)

    async def record_run(self, record: RunAuditRecord) -> None:
        self.runs.append(record)

    async def list_tool_calls(self, *, run_id: str) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.run_id == run_id]

    async def get_run(self, *, run_id: str) -> RunAuditRecord | None:
        return next((r for r in self.runs if r.run_id == run_id), None)


def _ctx(*, run_id: str | None, conversation_id: str | None = None) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id=run_id, conversation_id=conversation_id)


class TestAuditLogThroughAgent:
    async def test_records_tool_call_and_run(self):
        sink = InMemoryAuditSink()
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=sink, agent_name='librarian')])

        @agent.tool_plain
        def lookup(query: str) -> str:
            return f'found: {query}'

        result = await agent.run('go')
        (call,) = await sink.list_tool_calls(run_id=result.run_id)
        assert call.tool_name == 'lookup'
        assert json.loads(call.arguments) == {'query': 'a'}
        assert call.result == 'found: a'
        assert call.error is None
        assert call.agent_name == 'librarian'
        assert call.started_at <= call.ended_at  # type: ignore[operator]
        assert call.started_at.tzinfo is not None  # timestamps are timezone-aware UTC

        run = await sink.get_run(run_id=result.run_id)
        assert run is not None
        assert run.outcome == 'completed'
        assert run.total_tokens == run.input_tokens + run.output_tokens  # type: ignore[operator]
        assert run.total_tokens and run.total_tokens > 0

    async def test_redacts_secret_arguments(self):
        sink = InMemoryAuditSink()
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=sink)])

        @agent.tool_plain
        def connect(host: str, api_key: str) -> str:
            return 'connected'

        result = await agent.run('go')
        (call,) = await sink.list_tool_calls(run_id=result.run_id)
        assert json.loads(call.arguments) == {'host': 'a', 'api_key': '***'}

    async def test_custom_redactor(self):
        sink = InMemoryAuditSink()
        agent = Agent(
            TestModel(), capabilities=[AuditLog(sink=sink, redactor=lambda k, v: 'HIDDEN' if k == 'host' else v)]
        )

        @agent.tool_plain
        def connect(host: str) -> str:
            return 'connected'

        result = await agent.run('go')
        (call,) = await sink.list_tool_calls(run_id=result.run_id)
        assert json.loads(call.arguments) == {'host': 'HIDDEN'}

    async def test_tool_error_and_run_failure_are_recorded(self):
        sink = RecordingSink()
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=sink)])

        @agent.tool_plain
        def boom() -> str:
            raise ValueError('kaboom')

        with pytest.raises(ValueError, match='kaboom'):
            await agent.run('go')

        assert len(sink.tool_calls) == 1
        assert sink.tool_calls[0].result is None
        assert 'kaboom' in (sink.tool_calls[0].error or '')
        assert len(sink.runs) == 1
        assert sink.runs[0].outcome == 'failed'
        assert 'kaboom' in (sink.runs[0].error or '')
        # Exercise the sink read path for the failed run.
        assert await sink.get_run(run_id=sink.runs[0].run_id) is not None
        assert len(await sink.list_tool_calls(run_id=sink.tool_calls[0].run_id)) == 1

    async def test_value_bounds_apply(self):
        sink = InMemoryAuditSink()
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=sink, max_value_chars=20)])

        @agent.tool_plain
        def big() -> str:
            return 'z' * 500

        result = await agent.run('go')
        (call,) = await sink.list_tool_calls(run_id=result.run_id)
        assert call.result is not None and len(call.result) == 20

    async def test_conversation_id_is_recorded(self):
        sink = InMemoryAuditSink()
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=sink)])

        @agent.tool_plain
        def noop() -> str:
            return 'ok'

        result = await agent.run('go', conversation_id='conv-42')
        (call,) = await sink.list_tool_calls(run_id=result.run_id)
        assert call.conversation_id == 'conv-42'
        run = await sink.get_run(run_id=result.run_id)
        assert run is not None and run.conversation_id == 'conv-42'


class TestPerRunIsolation:
    async def test_sink_resolver_routes_by_deps(self):
        sinks = {'a': InMemoryAuditSink(), 'b': InMemoryAuditSink()}
        agent = Agent(
            TestModel(),
            deps_type=str,
            capabilities=[AuditLog[str](sink_resolver=lambda ctx: sinks[ctx.deps])],
        )

        @agent.tool_plain
        def noop() -> str:
            return 'ok'

        ra = await agent.run('go', deps='a')
        rb = await agent.run('go', deps='b')

        assert len(await sinks['a'].list_tool_calls(run_id=ra.run_id)) == 1
        assert await sinks['a'].get_run(run_id=rb.run_id) is None
        assert len(await sinks['b'].list_tool_calls(run_id=rb.run_id)) == 1

    async def test_parent_run_id_links_delegate_to_orchestrator(self):
        sink = RecordingSink()

        delegate = Agent(TestModel(), name='delegate', capabilities=[AuditLog(sink=sink, agent_name='delegate')])

        @delegate.tool_plain
        def leaf() -> str:
            return 'leaf'

        orchestrator = Agent(TestModel(), name='orch', capabilities=[AuditLog(sink=sink, agent_name='orch')])

        @orchestrator.tool_plain
        async def delegate_to_child() -> str:
            return (await delegate.run('sub')).output

        result = await orchestrator.run('top')

        orch_run = next(r for r in sink.runs if r.agent_name == 'orch')
        delegate_run = next(r for r in sink.runs if r.agent_name == 'delegate')
        assert orch_run.parent_run_id is None
        assert delegate_run.parent_run_id == result.run_id
        assert delegate_run.run_id != result.run_id

        leaf_call = next(c for c in sink.tool_calls if c.tool_name == 'leaf')
        assert leaf_call.parent_run_id == result.run_id


class TestComposition:
    async def test_composes_with_step_persistence(self):
        audit = InMemoryAuditSink()
        steps = InMemoryStepStore()
        agent = Agent(
            TestModel(),
            capabilities=[AuditLog(sink=audit), StepPersistence(store=steps)],
        )

        @agent.tool_plain
        def noop() -> str:
            return 'ok'

        result = await agent.run('go')
        assert len(await audit.list_tool_calls(run_id=result.run_id)) == 1
        assert len(await steps.list_events(run_id=result.run_id)) > 0


class TestDirectHookEdges:
    """Lower-level cases that are awkward to force through a full agent run."""

    async def test_run_id_falls_back_when_context_lacks_one(self):
        # `Agent.run` always resolves `run_id`; the fallback covers a hand-built context.
        sink = RecordingSink()
        cap = AuditLog(sink=sink)
        await cap.after_tool_execute(
            _ctx(run_id=None),
            call=ToolCallPart(tool_name='lookup', args={'q': 1}, tool_call_id='c1'),
            tool_def=ToolDefinition(name='lookup'),
            args={'q': 1},
            result='ok',
        )
        (call,) = sink.tool_calls
        assert len(call.run_id) == 32
        # No `before_tool_execute` ran, so `started_at` fell back to record time.
        assert call.started_at <= call.ended_at  # type: ignore[operator]
