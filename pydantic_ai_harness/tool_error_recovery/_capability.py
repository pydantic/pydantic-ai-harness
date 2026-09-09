"""The capability class: `ToolErrorRecovery`."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import anyio
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from pydantic_ai_harness.tool_error_recovery._outcome import RecoveryOutcome
from pydantic_ai_harness.tool_error_recovery._policy import RecoveryPolicy

# Control-flow signals that MUST always propagate untouched: recovery must never turn a
# retry / approval / deferred / terminal-failure signal into something else. At the tool
# boundary a tool-raised `ModelRetry` arrives as `ToolRetryError` and a tool-raised
# `ToolFailed` (a terminal "report this failure to the model" signal, outcome='failed', no
# retry) arrives as `ToolFailedError`; both raw and wrapped forms are listed.
_CONTROL_FLOW: tuple[type[BaseException], ...] = (
    SkipToolExecution,
    CallDeferred,
    ApprovalRequired,
    ModelRetry,
    ToolRetryError,
    ToolFailed,
    ToolFailedError,
)


@dataclass
class ToolErrorRecovery(AbstractCapability[AgentDepsT]):
    """Recover from tool execution errors: retry, inform, fallback, or propagate.

    The `policy` decides the outcome from the error and the tool call, so a rule can key
    on the error type, the tool name, or both. `max_recoveries` / `per_tool_recoveries`
    bound how many failures are smoothed over per run before the error surfaces.

    All logic lives in one hook (`wrap_tool_execute`), so counters have a single path and
    control-flow signals are re-raised before any recovery.
    """

    policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    """How errors are classified, rendered, and logged."""

    max_recoveries: int | None = None
    """Per-run cap on smoothed-over failures across all tools; `None` = unlimited."""

    per_tool_recoveries: Mapping[str, int] | None = None
    """Per-tool caps by tool name, on top of `max_recoveries`."""

    _used_total: int = field(default=0, init=False, repr=False)
    _used_by_tool: dict[str, int] = field(default_factory=dict[str, int], init=False, repr=False)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Opting out of spec construction, though the budgets and the policy's data fields
        # would serialize: `classify` is the decision this capability exists for, and a spec
        # cannot express it. Half a policy in YAML reads like a configured one.
        return None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ToolErrorRecovery[AgentDepsT]:
        """Fresh per-run counters: `replace` copies all init fields and re-initializes the rest."""
        return replace(self)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        attempt = 0
        while True:
            try:
                return await handler(args)
            except _CONTROL_FLOW:
                raise  # never intercept control flow, whatever the classifier says
            except Exception as exc:  # the recovery boundary
                with self._chain_callable_bug('classify', call, exc):
                    outcome = await self.policy.run_classify(ctx, call, exc)
                if outcome is None or outcome.action == 'propagate':
                    self._log(logging.ERROR, 'propagate', call, exc, None)
                    raise
                if outcome.action == 'retry' and attempt + 1 < outcome.max_attempts:
                    self._log(logging.DEBUG, 'retry', call, exc, None)
                    await self._sleep_backoff(outcome, attempt, call, exc)
                    attempt += 1
                    continue
                terminal = (outcome.on_exhausted if outcome.action == 'retry' else outcome) or RecoveryOutcome.inform()
                if terminal.action == 'propagate' or not self._spend(call):
                    # budget checked BEFORE recovering; label the actual reason so operator
                    # dashboards can tell a policy decision from an exhausted budget
                    reason = 'propagate' if terminal.action == 'propagate' else 'budget-exhausted'
                    self._log(logging.ERROR, reason, call, exc, None)
                    raise
                if terminal.action == 'fallback':
                    self._log(logging.WARNING, 'fallback', call, exc, terminal.label)
                    return terminal.value
                # render before the WARNING, which would otherwise claim a recovery that never landed
                with self._chain_callable_bug('format_error', call, exc):
                    text = self.policy.render(call, exc, terminal.label, expose_message=terminal.expose_message)
                self._log(logging.WARNING, 'inform', call, exc, terminal.label)
                raise ToolFailed(text) from exc  # inform

    # --- helpers for the single recovery path above ---

    def _spend(self, call: ToolCallPart) -> bool:
        """Spend one recovery from the budget. Returns False (do not recover) if exhausted.

        Counts final recoveries, not retry attempts. No `await` between check and
        increment -> atomic under cooperative asyncio, even with parallel tool calls.
        """
        if self.max_recoveries is not None and self._used_total >= self.max_recoveries:
            return False
        if self.per_tool_recoveries is not None:
            limit = self.per_tool_recoveries.get(call.tool_name)
            if limit is not None and self._used_by_tool.get(call.tool_name, 0) >= limit:
                return False
        self._used_total += 1
        self._used_by_tool[call.tool_name] = self._used_by_tool.get(call.tool_name, 0) + 1
        return True

    async def _sleep_backoff(self, outcome: RecoveryOutcome, attempt: int, call: ToolCallPart, exc: Exception) -> None:
        with self._chain_callable_bug('backoff', call, exc):
            delay = outcome.backoff(attempt) if callable(outcome.backoff) else outcome.backoff * (2**attempt)
        if delay > 0:
            await anyio.sleep(delay)

    @contextmanager
    def _chain_callable_bug(self, which: str, call: ToolCallPart, exc: Exception) -> Generator[None, None, None]:
        """Re-raise a bug from a user callable, chained to the failure it was handling.

        Nothing `classify`/`backoff`/`format_error` raise is intended: a verdict is returned, not raised.
        `__cause__` is explicit because the task group around tool calls overwrites `__context__`.
        """
        try:
            yield
        except Exception as bug:
            self._log_callable_bug(which, call, exc, bug)
            raise bug from exc

    def _log_callable_bug(self, which: str, call: ToolCallPart, exc: Exception, bug: Exception) -> None:
        """Log a callable's bug together with the failure it was handling.

        `exc_info` carries the original, whose traceback would otherwise be lost; the bug's comes with the crash.
        `recovery_error` stays the original: the incident is the tool failure, `recovery_bug` the broken callable.
        """
        self.policy.logger.log(
            logging.ERROR,
            'tool %r %s raised %s: %s -- while handling %s: %s',
            call.tool_name,
            which,
            type(bug).__name__,
            bug,
            type(exc).__name__,
            exc,
            extra={
                'recovery_scope': 'tool',
                'recovery_tool': call.tool_name,
                'recovery_action': f'{which}-failed',
                'recovery_error': type(exc).__name__,
                'recovery_bug': type(bug).__name__,
                'recovery_label': None,
            },
            exc_info=exc,
        )

    def _log(self, level: int, action: str, call: ToolCallPart, exc: BaseException, label: str | None) -> None:
        # The log is the only place a recovered failure stays visible to the operator.
        self.policy.logger.log(
            level,
            'tool %r %s: %s: %s',
            call.tool_name,
            action,
            type(exc).__name__,
            exc,
            extra={
                'recovery_scope': 'tool',
                'recovery_tool': call.tool_name,
                'recovery_action': action,
                'recovery_error': type(exc).__name__,
                'recovery_label': label,
            },
            exc_info=exc if level >= logging.ERROR else None,
        )
