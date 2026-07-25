"""The `AuditLog` capability: durable audit records of tool calls and runs to a pluggable sink."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from pydantic_ai_harness.audit_log._context import current_run_id
from pydantic_ai_harness.audit_log._redact import Redactor, bound_text, identity_redactor, redact_arguments
from pydantic_ai_harness.audit_log._sink import AuditSink, InMemoryAuditSink
from pydantic_ai_harness.audit_log._types import RunAuditRecord, RunOutcome, ToolCallRecord

_UNREPRESENTABLE = '<unrepresentable>'


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_convert(convert: Callable[[], str]) -> str:
    """Run a `str()`/`repr()` conversion, substituting a placeholder if it raises.

    A tool's result or an exception can have a `__str__`/`__repr__` that
    itself raises. `after_tool_execute`, `on_tool_execute_error`, and
    `on_run_error` all promise to return or re-raise the underlying value
    unchanged, so a broken conversion must not propagate from audit-record
    construction and turn a successful call into a failure, or replace the
    original error with a different one raised while merely describing it.
    """
    try:
        return convert()
    except Exception:
        return _UNREPRESENTABLE


@dataclass
class AuditLog(AbstractCapability[AgentDepsT]):
    """Record a structured audit trail of tool calls and run outcomes.

    Writes a `ToolCallRecord` per tool call (arguments, result or error,
    timing, and lineage) and a `RunAuditRecord` per run (outcome plus token
    usage) to a pluggable `AuditSink`. Where OpenTelemetry emits spans for
    observability and `step_persistence` records boundary events for resume,
    this captures the *content* of each call as a durable record to query for
    audit and compliance, attribute cost against, or mine into eval datasets.
    It composes alongside `step_persistence`: boundaries there, content here.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.audit_log import AuditLog, InMemoryAuditSink

    sink = InMemoryAuditSink()
    agent = Agent('openai:gpt-5', capabilities=[AuditLog(sink=sink, agent_name='librarian')])
    result = await agent.run('look up the release notes')
    calls = await sink.list_tool_calls(run_id=result.run_id)
    ```

    `sink` is the extension point: `InMemoryAuditSink` and `JsonlAuditSink` are
    trivial, dependency-free references, not a persistent backend. Implement
    `AuditSink` against SQLite, Postgres, Mongo, or a warehouse for your own
    deployment. Redaction is a pluggable `redactor` over each argument,
    defaulting to no redaction -- pass your own `Redactor` for your
    secret-handling policy. Values are bounded to `max_value_chars`. Pass a
    `sink_resolver` for a per-run backend keyed on `RunContext` (tenant/scope),
    mirroring `Memory.store_resolver`.
    """

    sink: AuditSink = field(default_factory=InMemoryAuditSink)
    """Backend the records are written to. The default persists for the process lifetime."""

    sink_resolver: Callable[[RunContext[AgentDepsT]], AuditSink] | None = None
    """Optional per-run sink resolver, keyed on the run context. Resolver failures propagate."""

    redactor: Redactor = identity_redactor
    """Per-argument redaction policy `(arg_name, value) -> value`. Defaults to no redaction."""

    max_value_chars: int = 2000
    """Character bound applied to each argument value, and to the full result/error text."""

    agent_name: str | None = None
    """Logical agent name stamped on every record for cross-run attribution."""

    _parent_run_id: str | None = field(default=None, init=False, repr=False, compare=False)
    _started_at: datetime = field(default_factory=_utcnow, init=False, repr=False, compare=False)
    _pending: dict[str, datetime] = field(default_factory=dict[str, datetime], init=False, repr=False, compare=False)
    _run_id_fallback: str | None = field(default=None, init=False, repr=False, compare=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AuditLog[AgentDepsT]:
        """Clone with per-run state: the resolved sink, the parent-run link, and a fresh start time.

        Reads the parent id from the contextvar set by any enclosing
        `AuditLog.wrap_run` before this run overwrites it, so a delegate's
        records point at the run that spawned them. Also materializes the
        `run_id` fallback once for the run, so every record from a `ctx`
        without a `run_id` still joins on the same generated id instead of
        each hook call minting its own.
        """
        sink = self.sink_resolver(ctx) if self.sink_resolver is not None else self.sink
        clone = replace(self, sink=sink)
        clone._parent_run_id = current_run_id.get()
        clone._run_id_fallback = ctx.run_id or uuid4().hex
        return clone

    def _run_id(self, ctx: RunContext[AgentDepsT]) -> str:
        return ctx.run_id or self._run_id_fallback or uuid4().hex

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Publish this run's id on the contextvar so nested delegates can record it as their parent."""
        token = current_run_id.set(self._run_id(ctx))
        try:
            return await handler()
        finally:
            current_run_id.reset(token)

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Stamp the tool call's start time so the record carries its duration."""
        self._pending[call.tool_call_id] = _utcnow()
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Record the completed tool call and return its result unchanged."""
        await self._record_tool_call(ctx, call=call, tool_def=tool_def, args=args, result=result, error=None)
        return result

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        """Record the failed tool call, then re-raise so behavior is unchanged."""
        await self._record_tool_call(ctx, call=call, tool_def=tool_def, args=args, result=None, error=error)
        raise error

    async def _record_tool_call(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
        error: Exception | None,
    ) -> None:
        record = ToolCallRecord(
            run_id=self._run_id(ctx),
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
            arguments=redact_arguments(args, redactor=self.redactor, max_chars=self.max_value_chars),
            result=None if error is not None else bound_text(_safe_convert(lambda: str(result)), self.max_value_chars),
            error=None if error is None else bound_text(_safe_convert(lambda: repr(error)), self.max_value_chars),
            started_at=self._pending.pop(call.tool_call_id, _utcnow()),
            ended_at=_utcnow(),
            conversation_id=ctx.conversation_id,
            parent_run_id=self._parent_run_id,
            agent_name=self.agent_name,
        )
        await self.sink.record_tool_call(record)

    async def after_run(self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        """Record the run as completed with its final token usage, then return it unchanged."""
        await self._record_run(ctx, outcome='completed', error=None)
        return result

    async def on_run_error(self, ctx: RunContext[AgentDepsT], *, error: BaseException) -> AgentRunResult[Any]:
        """Record the run as failed with its error, then re-raise."""
        await self._record_run(ctx, outcome='failed', error=error)
        raise error

    async def _record_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        outcome: RunOutcome,
        error: BaseException | None,
    ) -> None:
        usage = ctx.usage
        record = RunAuditRecord(
            run_id=self._run_id(ctx),
            outcome=outcome,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            error=None if error is None else bound_text(_safe_convert(lambda: repr(error)), self.max_value_chars),
            conversation_id=ctx.conversation_id,
            parent_run_id=self._parent_run_id,
            agent_name=self.agent_name,
            started_at=self._started_at,
            ended_at=_utcnow(),
        )
        await self.sink.record_run(record)
