"""Record types for the audit log.

The two records are name-spaced (`ToolCallRecord`, `RunAuditRecord`) to avoid a
collision with `step_persistence`'s `RunRecord`, and carry the same
conversation/run/parent identity stack so the two capabilities line up when
composed on one agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

RunOutcome = Literal['completed', 'failed']
"""Terminal outcome of an agent run recorded in a `RunAuditRecord`."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(kw_only=True)
class ToolCallRecord:
    """The audit record for one tool call.

    `arguments` is the JSON of the validated call arguments, passed through
    the capability's `redactor` first -- content a trace span emits
    ephemerally and `step_persistence` omits by design. Exactly one of
    `result` (success) or `error` (the tool raised) is set. `arguments`,
    `result`, and `error` are all recorded faithfully, with no size cap; for
    `arguments`, a `redactor` that also shortens a value is how a consumer
    limits size.

    `conversation_id`, `parent_run_id`, and `agent_name` mirror pydantic-ai's
    identity stack so audit records join against runs and against
    `step_persistence` events.
    """

    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: str
    result: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None


@dataclass(kw_only=True)
class RunAuditRecord:
    """The audit record for one agent run.

    `outcome` is `completed` on success and `failed` when the run raised, with
    `error` carrying the exception text, recorded faithfully, in the failure
    case. Token counts are read from `RunContext.usage`; `total_tokens` is its
    `input_tokens + output_tokens` sum.
    """

    run_id: str
    outcome: RunOutcome
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime = field(default_factory=_utcnow)
