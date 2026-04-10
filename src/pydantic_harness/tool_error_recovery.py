"""Tool error recovery capability for PydanticAI agents.

Catches unhandled tool execution errors and applies a configurable recovery
strategy so the agent run can continue instead of crashing.

Strategies:
    - ``inform`` (default): Return a descriptive error message to the model
      so it can adapt its approach.
    - ``retry``: Retry the failed tool call up to *N* times before falling
      back to ``inform``.
    - ``fallback``: Return a static fallback value on error.

Per-tool strategies can be configured via ``tool_strategies``, with a
``default_strategy`` applied to any tools not listed.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_harness import ToolErrorRecovery

    agent = Agent(
        'openai:gpt-4.1',
        capabilities=[
            ToolErrorRecovery(
                default_strategy='inform',
                tool_strategies={
                    'flaky_api': ('retry', 3),
                    'optional_lookup': ('fallback', None),
                },
            ),
        ],
    )
    ```
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import anyio
from pydantic_ai.capabilities.abstract import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

# ---------------------------------------------------------------------------
# Strategy types
# ---------------------------------------------------------------------------

InformStrategy = str  # Literal['inform'], but we use str for 3.10 compat
RetryStrategy = tuple[str, int] | tuple[str, int, float, tuple[type[Exception], ...]]
"""A retry strategy tuple.

Short form: ``('retry', max_retries)``
Full form: ``('retry', max_retries, retry_delay, retryable_exceptions)``
"""
FallbackStrategy = tuple[str, Any]  # ('fallback', value)

Strategy = InformStrategy | RetryStrategy | FallbackStrategy
"""A recovery strategy.

- ``'inform'``: Return the error message to the model.
- ``('retry', N)``: Retry up to *N* times, then fall back to ``inform``.
- ``('retry', N, delay, exceptions)``: Retry with exponential backoff and exception filter.
- ``('fallback', value)``: Return *value* on error.
"""


def _validate_strategy(strategy: Strategy, label: str = 'strategy') -> None:
    """Raise ``ValueError`` if *strategy* is not a well-formed :data:`Strategy`.

    Note: accepts ``Strategy`` at the type level but performs full runtime
    validation (including shape checks) because strategies can come from
    untyped sources like YAML specs.
    """
    if isinstance(strategy, str):
        if strategy != 'inform':
            raise ValueError(f"Invalid {label}: string strategy must be 'inform', got {strategy!r}")
        return
    if not isinstance(strategy, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(f'Invalid {label}: expected a string or tuple, got {strategy!r}')
    if len(strategy) < 2:
        raise ValueError(f'Invalid {label}: expected a tuple of length 2 or 4, got {strategy!r}')
    kind = strategy[0]
    if kind == 'retry':
        if len(strategy) == 2:
            _, max_retries = strategy
            if not isinstance(max_retries, int) or max_retries < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f'Invalid {label}: retry max_retries must be a positive integer, got {max_retries!r}')
        elif len(strategy) == 4:
            _, max_retries, delay, exc_types = strategy  # type: ignore[misc]
            if not isinstance(max_retries, int) or max_retries < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f'Invalid {label}: retry max_retries must be a positive integer, got {max_retries!r}')
            if not isinstance(delay, (int, float)) or delay < 0:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f'Invalid {label}: retry_delay must be a non-negative number, got {delay!r}')
            if not isinstance(exc_types, tuple) or not all(  # pyright: ignore[reportUnnecessaryIsInstance]
                isinstance(t, type) and issubclass(t, Exception)  # pyright: ignore[reportUnnecessaryIsInstance]
                for t in exc_types
            ):
                raise ValueError(
                    f'Invalid {label}: retryable_exceptions must be a tuple of Exception subclasses, got {exc_types!r}'
                )
        else:
            raise ValueError(f'Invalid {label}: retry strategy must be a 2-tuple or 4-tuple, got {strategy!r}')
    elif kind == 'fallback':
        if len(strategy) != 2:
            raise ValueError(f'Invalid {label}: fallback strategy must be a 2-tuple, got {strategy!r}')
    else:
        raise ValueError(f"Invalid {label}: tuple strategy kind must be 'retry' or 'fallback', got {kind!r}")


def _format_error(tool_name: str, error: Exception, *, include_traceback: bool) -> str:
    """Build a human-readable error string for the model."""
    parts = [f'Error in tool `{tool_name}` ({type(error).__name__}): {error}']
    if include_traceback:
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        parts.append('Traceback:\n' + ''.join(tb))
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def retry(
    max_retries: int = 3,
    *,
    retry_delay: float = 0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> RetryStrategy:
    """Create a retry strategy.

    Args:
        max_retries: Maximum number of retry attempts before falling back to ``inform``.
        retry_delay: Base delay in seconds for exponential backoff between retries.
            When > 0, waits ``retry_delay * 2 ** attempt`` seconds before each retry.
            Defaults to 0 (no delay).
        retryable_exceptions: Exception types eligible for retry. If the raised
            exception is not an instance of any of these types, it is not retried
            and the inform strategy is used immediately. Defaults to ``(Exception,)``
            (all exceptions are retryable).
    """
    if not isinstance(max_retries, int) or max_retries < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(f'max_retries must be a positive integer, got {max_retries!r}')
    if not isinstance(retry_delay, (int, float)) or retry_delay < 0:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(f'retry_delay must be a non-negative number, got {retry_delay!r}')
    if not isinstance(retryable_exceptions, tuple) or not all(  # pyright: ignore[reportUnnecessaryIsInstance]
        isinstance(t, type) and issubclass(t, Exception)  # pyright: ignore[reportUnnecessaryIsInstance]
        for t in retryable_exceptions
    ):
        raise ValueError(f'retryable_exceptions must be a tuple of Exception subclasses, got {retryable_exceptions!r}')
    # Use the short 2-tuple form when defaults are used, for backward compat.
    if retry_delay == 0 and retryable_exceptions == (Exception,):
        return ('retry', max_retries)
    return ('retry', max_retries, retry_delay, retryable_exceptions)


def fallback(value: Any = None) -> FallbackStrategy:
    """Create a fallback strategy.

    Args:
        value: The value to return when the tool fails.
    """
    return ('fallback', value)


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


@dataclass
class ToolErrorRecovery(AbstractCapability[Any]):
    """Catch tool execution errors and recover gracefully.

    Instead of letting unhandled exceptions crash the agent run, this
    capability intercepts failures via the ``on_tool_execute_error`` hook
    and applies a configurable strategy per tool.

    Strategies:
        - ``'inform'`` (default) -- Return a descriptive error message to the
          model so it can adjust its approach.
        - ``('retry', N)`` -- Retry the tool call up to *N* times. If all
          retries fail, fall back to ``inform``.
        - ``('fallback', value)`` -- Return a static value on error.

    Per-tool configuration is available via ``tool_strategies``.  Any tool
    not listed uses ``default_strategy``.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness import ToolErrorRecovery

        agent = Agent(
            'openai:gpt-4.1',
            capabilities=[
                ToolErrorRecovery(
                    tool_strategies={
                        'flaky_api': ('retry', 3),
                        'optional_lookup': ('fallback', None),
                    },
                ),
            ],
        )
    """

    default_strategy: Strategy = 'inform'
    """Strategy applied to tools not listed in ``tool_strategies``."""

    tool_strategies: dict[str, Strategy] = field(default_factory=lambda: dict[str, Strategy]())
    """Per-tool strategy overrides.  Keys are tool names."""

    include_traceback: bool = False
    """Whether to include the Python traceback in error messages sent to the model.

    Useful for debugging but may waste tokens in production.
    """

    max_total_errors: int | None = None
    """Maximum number of total errors across all tools before recovery stops.

    When set, the capability tracks every error that occurs (including errors
    consumed by retry attempts).  Once the budget is exhausted, subsequent errors
    propagate as-is instead of being recovered.  ``None`` (default) means no limit.
    """

    # --- Per-run state (populated by ``for_run``) ---

    _retry_counts: dict[str, int] = field(default_factory=lambda: dict[str, int](), repr=False)
    """Tracks per-tool retry counts within a single run.  Keys are ``tool_name``."""

    _total_errors: int = field(default=0, repr=False)
    """Total errors observed in this run (for ``max_total_errors`` budget)."""

    def __post_init__(self) -> None:
        """Validate strategies at construction time."""
        _validate_strategy(self.default_strategy, 'default_strategy')
        for name, strat in self.tool_strategies.items():
            _validate_strategy(strat, f'tool_strategies[{name!r}]')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return serialization name for spec construction."""
        return 'ToolErrorRecovery'

    async def for_run(self, ctx: RunContext[Any]) -> ToolErrorRecovery:
        """Return a fresh instance with empty retry counts for each agent run."""
        return ToolErrorRecovery(
            default_strategy=self.default_strategy,
            tool_strategies=self.tool_strategies,
            include_traceback=self.include_traceback,
            max_total_errors=self.max_total_errors,
        )

    def _get_strategy(self, tool_name: str) -> Strategy:
        """Look up the strategy for a given tool."""
        return self.tool_strategies.get(tool_name, self.default_strategy)

    def _budget_exhausted(self) -> bool:
        """Return ``True`` if the error budget has been spent."""
        return self.max_total_errors is not None and self._total_errors > self.max_total_errors

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Wrap tool execution to implement retry logic.

        The ``retry`` strategy requires re-invoking the tool, which can only
        be done from ``wrap_tool_execute`` (``on_tool_execute_error`` fires
        after the tool has already failed and cannot re-invoke it).
        """
        strategy = self._get_strategy(call.tool_name)
        if not (isinstance(strategy, tuple) and strategy[0] == 'retry'):
            # Non-retry strategies are handled by on_tool_execute_error.
            return await handler(args)

        max_retries: int = strategy[1]
        retry_delay: float = strategy[2] if len(strategy) == 4 else 0  # type: ignore[misc]
        retryable_exceptions: tuple[type[Exception], ...] = (
            strategy[3] if len(strategy) == 4 else (Exception,)  # type: ignore[misc]
        )
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                result = await handler(args)
                # Success -- reset retry count for this tool.
                self._retry_counts.pop(call.tool_name, None)
                return result
            except Exception as exc:
                last_error = exc
                self._total_errors += 1
                self._retry_counts[call.tool_name] = attempt + 1

                # If the exception isn't retryable, stop immediately.
                if not isinstance(exc, retryable_exceptions):
                    return _format_error(call.tool_name, exc, include_traceback=self.include_traceback)

                # If the error budget is exhausted, let the error propagate.
                if self._budget_exhausted():
                    raise

                if attempt < max_retries:
                    if retry_delay > 0:
                        await anyio.sleep(retry_delay * (2**attempt))
                    continue
                # All retries exhausted -- fall through to inform.
                return _format_error(call.tool_name, exc, include_traceback=self.include_traceback)

        # Unreachable, but satisfies the type checker.
        raise last_error  # type: ignore[misc] # pragma: no cover

    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        """Handle tool execution errors for non-retry strategies.

        For ``retry`` strategies, errors are handled by ``wrap_tool_execute``
        and this hook is not reached (the wrapper catches exceptions before
        they propagate to the hook).
        """
        self._total_errors += 1

        # If the error budget is exhausted, let the error propagate.
        if self._budget_exhausted():
            raise error

        strategy = self._get_strategy(call.tool_name)

        if isinstance(strategy, tuple) and strategy[0] == 'fallback':
            return strategy[1]

        # Default: 'inform' strategy.
        return _format_error(call.tool_name, error, include_traceback=self.include_traceback)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    'ToolErrorRecovery',
    'Strategy',
    'retry',
    'fallback',
]
