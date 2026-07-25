"""Tests for the `AuditLog` capability driven through `Agent.run`."""

from __future__ import annotations

import json
import logging

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
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


class _FailingSink:
    """An `AuditSink` whose writes always raise, standing in for sink I/O failure (full disk,
    permission loss). `list_tool_calls`/`get_run` are never exercised through this double; they
    only exist to satisfy the `AuditSink` protocol shape.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error if error is not None else RuntimeError('sink write failed')

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        raise self.error

    async def record_run(self, record: RunAuditRecord) -> None:
        raise self.error

    async def list_tool_calls(self, *, run_id: str) -> list[ToolCallRecord]:
        return []  # pragma: no cover

    async def get_run(self, *, run_id: str) -> RunAuditRecord | None:
        return None  # pragma: no cover


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

    async def test_no_redaction_by_default(self):
        # `AuditLog` ships no default secret-detection policy: with no
        # `redactor` configured, every argument value -- including one named
        # `api_key` -- is recorded as given. Redaction policy is the
        # consumer's; see `test_custom_redactor` for how to opt in.
        sink = InMemoryAuditSink()
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=sink)])

        @agent.tool_plain
        def connect(host: str, api_key: str) -> str:
            return 'connected'

        result = await agent.run('go')
        (call,) = await sink.list_tool_calls(run_id=result.run_id)
        assert json.loads(call.arguments) == {'host': 'a', 'api_key': 'a'}

    async def test_custom_redactor(self):
        # A consumer opts into a redaction policy by passing their own
        # `redactor` -- there is nothing to override, since there is no default.
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

    async def test_result_is_recorded_faithfully_without_a_size_cap(self):
        # `AuditLog` has no size cap of its own: a large result is recorded
        # in full. A consumer who wants one bounds it before returning it
        # from the tool.
        sink = InMemoryAuditSink()
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=sink)])

        @agent.tool_plain
        def big() -> str:
            return 'z' * 5000

        result = await agent.run('go')
        (call,) = await sink.list_tool_calls(run_id=result.run_id)
        assert call.result == 'z' * 5000

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

    async def test_run_id_fallback_is_shared_across_records_in_one_run(self):
        # `for_run` resolves the per-run clone once, the same way `Agent.run`
        # does; every hook call against that clone must join on the same
        # fallback run_id instead of minting a fresh one each time.
        sink = RecordingSink()
        run_cap = await AuditLog(sink=sink).for_run(_ctx(run_id=None))

        await run_cap.after_tool_execute(
            _ctx(run_id=None),
            call=ToolCallPart(tool_name='lookup', args={'q': 1}, tool_call_id='c1'),
            tool_def=ToolDefinition(name='lookup'),
            args={'q': 1},
            result='ok',
        )
        await run_cap.after_tool_execute(
            _ctx(run_id=None),
            call=ToolCallPart(tool_name='lookup', args={'q': 2}, tool_call_id='c2'),
            tool_def=ToolDefinition(name='lookup'),
            args={'q': 2},
            result='ok',
        )
        await run_cap.after_run(_ctx(run_id=None), result=AgentRunResult(output='ok'))

        run_ids = {c.run_id for c in sink.tool_calls} | {r.run_id for r in sink.runs}
        assert len(run_ids) == 1


class _UnstringifiableResult:
    """A tool result whose `__str__` raises -- audit-record construction must not propagate that."""

    def __str__(self) -> str:
        raise RuntimeError('__str__ is broken')


class _UnreprableError(Exception):
    """An exception whose `__repr__` raises -- audit-record construction must not replace it."""

    def __repr__(self) -> str:
        raise RuntimeError('__repr__ is broken')


class _UnreprableViaBaseException(Exception):
    """An exception whose `__repr__` raises `KeyboardInterrupt` -- a `BaseException` that is not
    an `Exception`. `on_run_error`/`on_tool_execute_error` accept any `BaseException`, so the
    conversion guard must catch more than just `Exception` or this escapes and replaces the run's
    real failure with an unrelated `KeyboardInterrupt`.
    """

    def __repr__(self) -> str:
        raise KeyboardInterrupt


class TestDefensiveRecordConstruction:
    """A broken `__str__`/`__repr__` on a tool result or error must not change the run outcome.

    `str(result)` and `repr(error)` happen while building the audit record,
    after the tool has already succeeded or failed. A conversion that itself
    raises must not turn a successful call into a failure, or replace the
    tool's real exception with one raised while merely describing it.
    """

    async def test_result_with_broken_str_leaves_the_returned_result_unchanged(self):
        sink = RecordingSink()
        cap = AuditLog(sink=sink)
        outcome = _UnstringifiableResult()

        returned = await cap.after_tool_execute(
            _ctx(run_id='r1'),
            call=ToolCallPart(tool_name='lookup', args={}, tool_call_id='c1'),
            tool_def=ToolDefinition(name='lookup'),
            args={},
            result=outcome,
        )

        assert returned is outcome  # the successful call still returns its actual result, unaltered
        (call,) = sink.tool_calls
        assert call.result == '<unrepresentable>'
        assert call.error is None

    async def test_error_with_broken_repr_still_reraises_the_original_error(self):
        sink = RecordingSink()
        cap = AuditLog(sink=sink)
        original = _UnreprableError('original tool failure')

        with pytest.raises(_UnreprableError) as exc_info:
            await cap.on_tool_execute_error(
                _ctx(run_id='r1'),
                call=ToolCallPart(tool_name='boom', args={}, tool_call_id='c1'),
                tool_def=ToolDefinition(name='boom'),
                args={},
                error=original,
            )

        assert exc_info.value is original  # the original exception, not one raised while describing it
        (call,) = sink.tool_calls
        assert call.result is None
        assert call.error == '<unrepresentable>'

    async def test_run_error_whose_repr_raises_a_non_exception_base_exception_still_reraises_the_original(
        self, caplog: pytest.LogCaptureFixture
    ):
        # `on_run_error` accepts any `BaseException`. Here the run's own error
        # is an ordinary `Exception` subclass, but its `__repr__` -- invoked
        # only while building the audit record -- raises `KeyboardInterrupt`,
        # which is *not* an `Exception`. That must not escape the conversion
        # guard and replace the run's real failure with an unrelated
        # `KeyboardInterrupt`.
        sink = RecordingSink()
        cap = AuditLog(sink=sink)
        original = _UnreprableViaBaseException('the run actually failed this way')

        with caplog.at_level(logging.WARNING):
            with pytest.raises(_UnreprableViaBaseException) as exc_info:
                await cap.on_run_error(_ctx(run_id='r1'), error=original)

        assert exc_info.value is original  # the original error, not the KeyboardInterrupt raised while describing it
        (run,) = sink.runs
        assert run.outcome == 'failed'
        assert run.error == '<unrepresentable>'
        assert 'failed to convert' in caplog.text  # the conversion failure is logged, not just swallowed


class TestSinkFailureIsolation:
    """A sink write failure (full disk, permission loss) must never alter the run outcome or
    replace the tool's/run's own error -- see `_capability.py`'s `_record_tool_call` and
    `_record_run`. Covers both the success path (the hook must still return/report the real
    outcome) and the error path (the hook must still re-raise the *original* exception, not the
    sink's) for both the tool-call record and the run record.
    """

    async def test_tool_call_sink_failure_leaves_the_returned_result_unchanged(self, caplog: pytest.LogCaptureFixture):
        cap = AuditLog(sink=_FailingSink())

        with caplog.at_level(logging.WARNING):
            returned = await cap.after_tool_execute(
                _ctx(run_id='r1'),
                call=ToolCallPart(tool_name='lookup', args={}, tool_call_id='c1'),
                tool_def=ToolDefinition(name='lookup'),
                args={},
                result='the real result',
            )

        assert returned == 'the real result'  # the sink's failure never touches the tool's own result
        assert 'failed to record tool call' in caplog.text  # observable, not silently dropped

    async def test_tool_call_sink_failure_on_error_path_still_reraises_the_original_error(
        self, caplog: pytest.LogCaptureFixture
    ):
        cap = AuditLog(sink=_FailingSink(RuntimeError('sink write failed')))
        original = ValueError('the tool actually failed this way')

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError) as exc_info:
                await cap.on_tool_execute_error(
                    _ctx(run_id='r1'),
                    call=ToolCallPart(tool_name='boom', args={}, tool_call_id='c1'),
                    tool_def=ToolDefinition(name='boom'),
                    args={},
                    error=original,
                )

        assert exc_info.value is original  # the tool's own exception, never replaced by the sink's
        assert 'failed to record tool call' in caplog.text

    async def test_run_sink_failure_does_not_raise_on_a_successful_run(self, caplog: pytest.LogCaptureFixture):
        cap = AuditLog(sink=_FailingSink())

        with caplog.at_level(logging.WARNING):
            result = await cap.after_run(_ctx(run_id='r1'), result=AgentRunResult(output='ok'))

        assert result.output == 'ok'  # the run still reports success despite the sink failure
        assert 'failed to record run' in caplog.text

    async def test_run_sink_failure_on_error_path_still_reraises_the_original_error(
        self, caplog: pytest.LogCaptureFixture
    ):
        cap = AuditLog(sink=_FailingSink())
        original = TypeError('the run actually failed this way')

        with caplog.at_level(logging.WARNING):
            with pytest.raises(TypeError) as exc_info:
                await cap.on_run_error(_ctx(run_id='r1'), error=original)

        assert exc_info.value is original  # the run's own exception, never replaced by the sink's
        assert 'failed to record run' in caplog.text

    async def test_full_agent_run_succeeds_despite_every_sink_write_failing(self, caplog: pytest.LogCaptureFixture):
        agent = Agent(TestModel(), capabilities=[AuditLog(sink=_FailingSink())])

        @agent.tool_plain
        def lookup(query: str) -> str:
            return f'found: {query}'

        with caplog.at_level(logging.WARNING):
            result = await agent.run('go')

        assert result.output  # the run completed normally end to end; a broken sink never surfaces
        assert 'failed to record tool call' in caplog.text
        assert 'failed to record run' in caplog.text
