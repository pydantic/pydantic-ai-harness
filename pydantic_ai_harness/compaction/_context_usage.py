"""`ContextUsageMonitor` -- report how full the context is, for a host application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW, resolve_context_window
from pydantic_ai_harness.compaction._shared import estimate_token_count

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass(frozen=True)
class ContextUsage:
    """A single reading of how full the context is."""

    used_tokens: int
    """Estimated tokens in the history about to be sent."""

    window_tokens: int
    """Context window the reading is measured against."""

    resolved: bool
    """Whether `window_tokens` is the model's real window or the fallback.

    A gauge can render an unresolved window differently -- the percentage is a guess when
    the model is not in the pricing registry.
    """

    @property
    def fraction(self) -> float:
        """`used_tokens` as a fraction of the window."""
        return self.used_tokens / self.window_tokens


@dataclass
class ContextUsageMonitor(AbstractCapability[AgentDepsT]):
    """Report context usage to the application before each model request.

    A compaction strategy knows when to act but says nothing about how close the run is to
    the limit, so an application that wants to show "context: 73%" has to re-count the
    history itself and guess the denominator. This capability does neither: it reuses the
    same estimator the strategies use and the model's real context window.

    It only observes -- it never edits the history.

    Order matters: register it *after* a compaction capability to see the compacted history,
    or before it to see what triggered the compaction.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import ContextUsageMonitor, SummarizingCompaction

        agent = Agent(
            'anthropic:claude-sonnet-4-6',
            capabilities=[
                SummarizingCompaction(max_fraction=0.9, keep_messages=20),
                ContextUsageMonitor(on_usage=lambda usage: print(f'{usage.fraction:.0%}')),
            ],
        )
        ```
    """

    on_usage: Callable[[ContextUsage], None]
    """Called with a fresh reading before every model request."""

    context_window: int | None = None
    """Window override in tokens. `None` resolves it from the run's model."""

    fallback_context_window: int = DEFAULT_CONTEXT_WINDOW
    """Window assumed when the run's model is not in the pricing registry."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer, matching the one your compaction strategy uses."""

    _window: int = field(default=DEFAULT_CONTEXT_WINDOW, init=False, repr=False, compare=False)
    _resolved: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.context_window is not None and self.context_window < 1:
            raise ValueError('context_window must be positive.')
        if self.fallback_context_window < 1:
            raise ValueError('fallback_context_window must be positive.')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ContextUsageMonitor[AgentDepsT]:
        """Resolve the window against this run's model."""
        run: ContextUsageMonitor[AgentDepsT] = replace(self)
        if self.context_window is not None:
            run._window = self.context_window
            run._resolved = True
            return run
        window = resolve_context_window(ctx.model)
        run._window = window if window is not None else self.fallback_context_window
        run._resolved = window is not None
        return run

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Measure the pending history and hand the reading to `on_usage`."""
        messages: list[ModelMessage] = list(request_context.messages)
        self.on_usage(
            ContextUsage(
                used_tokens=estimate_token_count(messages, self.tokenizer),
                window_tokens=self._window,
                resolved=self._resolved,
            )
        )
        return request_context
