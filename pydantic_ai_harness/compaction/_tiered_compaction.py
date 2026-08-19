"""`TieredCompaction` -- escalation orchestrator over a sequence of strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    SupportsFocus,
    compact_with_span,
    context_for_request,
    estimate_context_tokens,
    estimate_token_count,
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import AbstractModel, ModelRequestContext, ModelRequestParameters


@dataclass
class TieredCompaction(AbstractCapability[AgentDepsT]):
    """Escalation orchestrator over a sequence of compaction strategies.

    Runs each tier in order, re-measuring the token count after each, and stops as soon as
    the conversation fits ``target_tokens``.  Order tiers cheap-to-expensive (e.g. clear
    tool results, deduplicate reads, then summarize) so the expensive summarization tier is
    only reached when the cheap passes cannot reclaim enough.

    Each tier's own trigger is bypassed -- `TieredCompaction` drives the tiers directly via
    their ``compact`` method and decides when to stop.

    By default the gate and the stop share the one ``target_tokens`` budget. Setting
    ``trigger_tokens`` above the target adds hysteresis: the history is left untouched (and
    the provider's prompt cache undisturbed) until it crosses the trigger, then one event
    escalates all the way down to the target -- see `trigger_tokens`.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import (
            ClearToolResults,
            SummarizingCompaction,
            TieredCompaction,
        )

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[TieredCompaction(
                tiers=[
                    ClearToolResults(max_tokens=1),
                    SummarizingCompaction(model='openai:gpt-4o-mini', max_messages=1),
                ],
                target_tokens=100_000,
            )],
        )
        ```
    """

    tiers: Sequence[CompactionStrategy[AgentDepsT]]
    """Strategies to apply in order, cheap-to-expensive.  The last is typically a summarizer."""

    target_tokens: int | None = None
    """Stop escalating once the estimated token count is at or below this value.

    Mutually exclusive with `target_fraction`; exactly one of the two must be set.
    """

    target_fraction: float | None = field(default=None, kw_only=True)
    """Target expressed as a fraction of the model's context window, resolved per request.

    Use this instead of `target_tokens` when the same agent runs on models with
    different windows. Mutually exclusive with `target_tokens`.
    """

    trigger_tokens: int | None = field(default=None, kw_only=True)
    """Compact only once the estimated context exceeds this value; ``None`` gates on the target.

    A trigger separate from (above) the target gives the escalation hysteresis. Gating on the
    one target budget compacts on every request the history sits above it, and each rewrite
    invalidates the provider's prompt cache from the first changed message onward -- so a
    conversation hovering at the target pays a cache bust per request. With a trigger above
    the target, the history is left byte-identical until it crosses the trigger, then ONE
    event escalates all the way down to the target: each cache bust buys
    ``trigger - target`` tokens of headroom before the next. Mutually exclusive with
    `trigger_fraction`; set it at or above the target (a trigger below the target just gates
    like the target -- escalation still stops there).
    """

    trigger_fraction: float | None = field(default=None, kw_only=True)
    """Trigger expressed as a fraction of the model's context window, resolved per request.

    Use this instead of `trigger_tokens` when the same agent runs on models with
    different windows. Mutually exclusive with `trigger_tokens`.
    """

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens. `None` resolves it from the request's model.

    Unlike `fallback_context_window`, this applies whether or not resolution succeeds. Reach
    for it when the registry is confidently wrong: a beta- or tier-gated window it records as
    the maximum, or a self-hosted endpoint whose model id describes someone else's
    deployment. Only consulted alongside `target_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the model is not in the pricing registry.

    Only consulted alongside `target_fraction`. Supply the real number for a deployment the
    registry cannot resolve."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError('tiers must not be empty.')
        if self.target_tokens is None and self.target_fraction is None:
            raise ValueError('One of target_tokens or target_fraction must be set.')
        validate_token_trigger(
            self.target_tokens,
            self.target_fraction,
            self.fallback_context_window,
            self.context_window,
            tokens_name='target_tokens',
            fraction_name='target_fraction',
        )
        validate_token_trigger(
            self.trigger_tokens,
            self.trigger_fraction,
            self.fallback_context_window,
            self.context_window,
            tokens_name='trigger_tokens',
            fraction_name='trigger_fraction',
        )

    def with_focus(self, focus: str) -> TieredCompaction[AgentDepsT]:
        """Return a copy whose focus-capable tiers prioritize `focus`.

        A tiered strategy is focusable when any of its tiers is: the summarizing tier writes
        the prose, so the hint has to reach it rather than stopping at this wrapper. Tiers that
        cannot honour a focus are passed through unchanged.
        """
        return replace(
            self,
            tiers=[tier.with_focus(focus) if isinstance(tier, SupportsFocus) else tier for tier in self.tiers],
        )

    def _target(self, model: AbstractModel | str) -> int | None:
        """Absolute token target, resolved against *model* when expressed as a fraction.

        Returns `None` when *model* is realtime (an `AbstractModel` that is not a request-response
        `Model`): a realtime session never compacts, so this only guards the widened
        `RunContext.model` type. `__post_init__` requires one of `target_tokens` /
        `target_fraction`, so a request-response `Model` never resolves to `None`. #585
        """
        return resolve_token_trigger(
            self.target_tokens, self.target_fraction, model, self.fallback_context_window, self.context_window
        )

    def _trigger(self, model: AbstractModel | str) -> int | None:
        """Absolute gate threshold: the trigger when one is configured, else the target.

        The default keeps the original single-budget behavior -- gate and stop on the same
        target -- so setting a trigger is purely opt-in hysteresis.
        """
        if self.trigger_tokens is None and self.trigger_fraction is None:
            return self._target(model)
        return resolve_token_trigger(
            self.trigger_tokens, self.trigger_fraction, model, self.fallback_context_window, self.context_window
        )

    async def _escalate(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        target: int,
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> list[ModelMessage]:
        """Apply tiers in order until the history fits *target* or tiers run out."""
        original = messages
        # The usage anchor describes the request as it was sent, so a tier's rewrite of older
        # messages is invisible to re-anchoring. Subtract the estimated tokens of the text a
        # tier removed from the anchored baseline instead: the reclaim is measured only on
        # message text, so the baseline's fixed overhead (tool definitions, instructions) is
        # never treated as compacted away. On content the estimator undercounts, this
        # understates the reclaim and escalates a tier early -- the cheap direction.
        baseline = estimate_context_tokens(messages, self.tokenizer, model_request_parameters=model_request_parameters)
        heuristic_baseline = estimate_token_count(messages, self.tokenizer)
        estimate = baseline
        for tier in self.tiers:
            if estimate <= target:
                break
            messages = await tier.compact(messages, ctx)
            # Before the next stop decision, so escalation measures the history it would return.
            messages = reinject_pinned(original, messages)
            reclaimed = heuristic_baseline - estimate_token_count(messages, self.tokenizer)
            estimate = max(baseline - reclaimed, 0)
        return messages

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Apply tiers in order until the history fits the target or tiers run out."""
        target = self._target(ctx.model)
        if target is None:
            # A realtime model has no token target to escalate toward; leave the history as-is.
            return messages
        return await self._escalate(messages, ctx, target)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Escalate through the tiers when the conversation exceeds the trigger (or target)."""
        messages: list[ModelMessage] = list(request_context.messages)
        # The tiers get the request's context, not the run's: a tier that resolves a model --
        # a summarizing one, or a `TieredCompaction` nested inside this one -- has to reach the
        # same conclusion this gate did.
        request_ctx = context_for_request(ctx, request_context)
        # Resolved once, so the gate and the escalation loop cannot disagree about the target.
        target = self._target(request_ctx.model)
        # The gate checks the trigger; the escalation loop still runs down to the target, so
        # with a trigger above the target one event buys `trigger - target` of headroom.
        trigger = self._trigger(request_ctx.model)
        if (
            target is None
            or trigger is None
            or (
                estimate_context_tokens(
                    messages,
                    self.tokenizer,
                    model_request_parameters=request_context.model_request_parameters,
                )
                <= trigger
            )
        ):
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='TieredCompaction',
            messages=messages,
            compact=lambda: self._escalate(messages, request_ctx, target, request_context.model_request_parameters),
            tokenizer=self.tokenizer,
        )
        record_compaction_reclaim(
            request_context,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        request_context.messages = compacted
        return request_context
