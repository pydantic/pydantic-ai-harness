"""Tool-call error recovery for Pydantic AI agents.

Turns failures into graceful degradation without hiding bugs.

Governing principle: recovery splits the user surface (made quieter) from the
operator surface (made louder). It is never *silence* -- a recovered failure always
stays visible in the log. Genuine bugs are never disguised as success; only
expected/operational failures are smoothed over.

Public API:

- `RecoveryOutcome` -- the verdict for a failed tool call (retry / inform / fallback / propagate).
- `RecoveryPolicy` -- classify + render + log.
- `make_default_classify` / `default_classify` -- a conservative built-in classifier.
- `ToolErrorRecovery` -- recover from tool execution errors.
"""

from pydantic_ai_harness.tool_error_recovery._capability import ToolErrorRecovery
from pydantic_ai_harness.tool_error_recovery._outcome import (
    DEFAULT_BUG_TYPES,
    DEFAULT_TRANSIENT_TYPES,
    RecoveryOutcome,
)
from pydantic_ai_harness.tool_error_recovery._policy import (
    ErrorFormatter,
    RecoveryClassifier,
    RecoveryPolicy,
    default_classify,
    make_default_classify,
)

__all__ = [
    'DEFAULT_BUG_TYPES',
    'DEFAULT_TRANSIENT_TYPES',
    'ErrorFormatter',
    'RecoveryClassifier',
    'RecoveryOutcome',
    'RecoveryPolicy',
    'ToolErrorRecovery',
    'default_classify',
    'make_default_classify',
]
