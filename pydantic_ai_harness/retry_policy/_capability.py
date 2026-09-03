"""Retry policy capability: configurable retry with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import RunContext

logger = logging.getLogger(__name__)


def _default_retryable_status_codes() -> tuple[int, ...]:
    """HTTP status codes that are typically retryable."""
    return (429, 500, 502, 503, 504)


def _default_retryable_exceptions() -> tuple[type[Exception], ...]:
    """Exception types that are typically retryable."""
    return (TimeoutError, ConnectionError, OSError)


def _is_retryable_http_error(exc: Exception, status_codes: tuple[int, ...]) -> bool:
    """Check if an exception represents a retryable HTTP error."""
    status_code = getattr(exc, 'status_code', None)
    if status_code is not None:
        return status_code in status_codes

    response = getattr(exc, 'response', None)
    if response is not None:
        status_code = getattr(response, 'status_code', None)
        if status_code is not None:
            return status_code in status_codes

    return False


@dataclass
class RetryPolicy(AbstractCapability[AgentDepsT]):
    """Configurable retry logic with exponential backoff for tool calls.

    Handles transient failures (rate limits, timeouts, provider errors) automatically
    without requiring manual retry loops. Wraps tool execution with configurable
    retry strategies, logging each attempt and surfacing final failures gracefully.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import RetryPolicy

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[
            RetryPolicy(
                max_retries=3,
                backoff_factor=0.5,  # 0.5s, 1s, 2s
            )
        ],
    )
    ```
    """

    max_retries: int = 3
    """Maximum number of retry attempts for each tool call. 0 means no retries."""

    backoff_factor: float = 0.5
    """Base delay in seconds for exponential backoff. Actual delay = backoff_factor * 2^attempt."""

    max_backoff: float = 30.0
    """Maximum delay in seconds between retries."""

    retryable_status_codes: tuple[int, ...] = field(
        default_factory=_default_retryable_status_codes
    )
    """HTTP status codes that trigger a retry."""

    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=_default_retryable_exceptions
    )
    """Exception types that trigger a retry."""

    tool_overrides: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Per-tool retry configuration overrides.

    Keys are tool names, values are dicts with RetryPolicy field names.
    Example: {'web_search': {'max_retries': 2, 'backoff_factor': 1.0}}
    """

    on_retry: Callable[[str, int, Exception], None] | None = None
    """Optional callback invoked on each retry attempt.

    Arguments: (tool_name, attempt_number, exception)
    """

    on_failure: Callable[[str, Exception], None] | None = None
    """Optional callback invoked when all retries are exhausted.

    Arguments: (tool_name, final_exception)
    """

    def get_tool_config(self, tool_name: str) -> dict[str, Any]:
        """Get retry config for a specific tool, falling back to defaults."""
        return self.tool_overrides.get(tool_name, {})

    def _get_retryable_status_codes(self, tool_name: str) -> tuple[int, ...]:
        """Get retryable status codes for a specific tool."""
        config = self.get_tool_config(tool_name)
        return config.get('retryable_status_codes', self.retryable_status_codes)

    def _get_retryable_exceptions(self, tool_name: str) -> tuple[type[Exception], ...]:
        """Get retryable exceptions for a specific tool."""
        config = self.get_tool_config(tool_name)
        return config.get('retryable_exceptions', self.retryable_exceptions)

    def should_retry(self, exc: Exception, tool_name: str) -> bool:
        """Determine if an exception is retryable."""
        retryable_exceptions = self._get_retryable_exceptions(tool_name)
        retryable_status_codes = self._get_retryable_status_codes(tool_name)

        if isinstance(exc, retryable_exceptions):
            return True

        if _is_retryable_http_error(exc, retryable_status_codes):
            return True

        error_type = getattr(exc, 'error_type', None)
        if error_type in ('rate_limit', 'timeout', 'server_error'):
            return True

        return False

    def calculate_delay(self, attempt: int, tool_name: str) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random

        config = self.get_tool_config(tool_name)
        backoff_factor = config.get('backoff_factor', self.backoff_factor)
        max_backoff = config.get('max_backoff', self.max_backoff)

        delay = backoff_factor * (2 ** attempt)
        delay = min(delay, max_backoff)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0.01, delay + jitter)

    def get_max_retries(self, tool_name: str) -> int:
        """Get max retries for a specific tool."""
        config = self.get_tool_config(tool_name)
        return config.get('max_retries', self.max_retries)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: Any,
        args: Any,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Wrap tool execution with retry logic."""
        tool_name = call.tool_name
        max_retries = self.get_max_retries(tool_name)
        last_exception: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return await handler(args)
            except Exception as exc:
                last_exception = exc

                if not self.should_retry(exc, tool_name):
                    raise

                if attempt < max_retries:
                    delay = self.calculate_delay(attempt, tool_name)
                    if self.on_retry:
                        self.on_retry(tool_name, attempt + 1, exc)
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for tool '{tool_name}' "
                        f"after {delay:.2f}s: {exc}"
                    )
                    await asyncio.sleep(delay)
                else:
                    if self.on_failure:
                        self.on_failure(tool_name, exc)
                    raise

        raise last_exception  # type: ignore[misc]
