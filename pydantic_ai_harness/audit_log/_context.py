"""Async-context state that links a nested run to the run that spawned it."""

from __future__ import annotations

from contextvars import ContextVar

current_run_id: ContextVar[str | None] = ContextVar(
    'pydantic_ai_harness.audit_log.current_run_id',
    default=None,
)
"""Async-context-local id of the active `AuditLog` run.

Set by `AuditLog.wrap_run` for the duration of a run and read by a nested
`AuditLog`'s `for_run` to fill `parent_run_id`. When an orchestrator's tool
synchronously calls a delegate's `Agent.run(...)`, the delegate's records point
at the orchestrator's `run_id` without manual threading.

`RunContext` carries `run_id` and `conversation_id` but not a parent link, and
pydantic-ai's own cross-run signals are single-slot (the inner
`Instrumentation.wrap_run` overwrites them before a nested capability runs), so
a capability-local `ContextVar` is the seam that survives the nesting. Mirrors
the mechanism `step_persistence` uses for the same lineage.
"""
