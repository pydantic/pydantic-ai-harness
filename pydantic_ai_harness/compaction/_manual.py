"""`compact_now` -- run a compaction strategy outside an agent run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from pydantic_ai._run_context import AgentDepsT
    from pydantic_ai.models import Model

    from pydantic_ai_harness.compaction._shared import CompactionStrategy


@runtime_checkable
class SupportsFocus(Protocol):
    """A strategy whose output can be steered toward a topic.

    Only strategies that *write* something -- a summary -- can be focused; the ones that drop
    or blank content by rule have nothing to steer, so `compact_now` ignores a focus they
    cannot honour rather than rejecting it.
    """

    def with_focus(self, focus: str) -> Any:
        """Return a copy of this strategy that prioritizes `focus`."""
        ...  # pragma: no cover


async def compact_now(
    strategy: CompactionStrategy[AgentDepsT],
    messages: list[ModelMessage],
    *,
    model: Model | str,
    focus: str | None = None,
    deps: AgentDepsT | None = None,
    usage: RunUsage | None = None,
) -> list[ModelMessage]:
    """Compact `messages` with `strategy`, without an agent run in progress.

    A strategy's `compact` takes a `RunContext` because it may need the run's model and usage
    -- but an application holding a conversation *between* runs has neither, which is exactly
    when a user-invoked `/compact` happens. This builds a throwaway context so the same
    strategy the agent uses can be driven from a command handler, and the compacted history
    handed back to the next `agent.run(message_history=...)`.

    Unlike the automatic path this always runs the strategy: an explicit request should not be
    subject to the threshold check.

    Args:
        strategy: The strategy to run. Any `CompactionStrategy` works.
        messages: History to compact. Not mutated; the compacted list is returned.
        model: Model the strategy should use, needed by summarizing strategies that call one.
        focus: What the summary should prioritize. Ignored by strategies that cannot honour it.
        deps: Dependencies to expose on the throwaway context, for a strategy that reads them.
        usage: Usage to accumulate into, so a summarization call can be billed to your own
            counter. A fresh `RunUsage` is used when omitted.

    Example:
        ```python {test="skip"}
        from pydantic_ai_harness.compaction import SummarizingCompaction, compact_now

        strategy = SummarizingCompaction(max_fraction=0.9, keep_messages=20)
        history = await compact_now(
            strategy,
            history,
            model='anthropic:claude-sonnet-4-6',
            focus='the auth refactor, not the earlier CSS work',
        )
        ```
    """
    from pydantic_ai.models import infer_model

    if focus is not None and isinstance(strategy, SupportsFocus):
        strategy = strategy.with_focus(focus)

    ctx: RunContext[Any] = RunContext(
        deps=deps,
        model=infer_model(model) if isinstance(model, str) else model,
        usage=usage if usage is not None else RunUsage(),
    )
    return await strategy.compact(messages, ctx)
