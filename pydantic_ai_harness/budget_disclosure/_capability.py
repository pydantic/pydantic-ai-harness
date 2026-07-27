"""`BudgetDisclosure`: tell the model how much of its usage budget is left.

`UsageLimits` enforces a run's budget, but the model never sees it: it gets stopped
mid-step by a `UsageLimitExceeded` instead of pacing itself as the budget runs down. This
capability contributes one short line of remaining-budget state to each model request so the
model can adjust strategy (with little left, land current results rather than start new work).

The line is contributed through the capability `get_instructions` channel, so it is
*ephemeral*: instructions are rebuilt for every request and are never stored as a message
part in the run history. That is the cache-safe channel. A remaining-budget number changes on
every request by construction, so persisting it into the history would move the cacheable
prefix each turn and re-charge the whole conversation. Instructions sit in the system-prompt
region (for providers where instructions are the system-prompt tail, they sit *after* the
cached prefix), so a changing budget line there does not disturb the cached tools + history
prefix. The line is kept short and the numbers coarse (rounded) regardless, so it costs little
and reads as a stable-shaped status line rather than churning detail.

The run's `UsageLimits` is not reachable from a capability: it is passed to `Agent.run(...,
usage_limits=...)` and lives on the agent graph's run deps, not on the `RunContext` a
capability sees. So the budget is configured on the capability directly via `limits`; pass the
same `UsageLimits` you pass to the run. This mirrors `compaction.LimitWarner`, which is
configured with its own `max_*` ceilings for the same reason.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

BudgetDimension = Literal['requests', 'tool_calls', 'input_tokens', 'output_tokens', 'total_tokens']
"""A usage dimension that can be limited and therefore disclosed."""

BudgetFormatter: TypeAlias = Callable[[RunContext[AgentDepsT], Mapping[BudgetDimension, int]], str | None]
"""Signature of a `format` override: receives the run context and the remaining budget per
active dimension (raw, unrounded), and returns the line to contribute (or `None` to contribute
nothing this request)."""

# The order dimensions appear in the default line. Token dimensions lead (they are the figure
# the model paces against most), counts trail.
_DIMENSION_ORDER: tuple[BudgetDimension, ...] = (
    'total_tokens',
    'input_tokens',
    'output_tokens',
    'requests',
    'tool_calls',
)

_TOKEN_DIMENSIONS: frozenset[BudgetDimension] = frozenset({'total_tokens', 'input_tokens', 'output_tokens'})

_LIMIT_ATTR: Mapping[BudgetDimension, str] = {
    'requests': 'request_limit',
    'tool_calls': 'tool_calls_limit',
    'input_tokens': 'input_tokens_limit',
    'output_tokens': 'output_tokens_limit',
    'total_tokens': 'total_tokens_limit',
}

_USAGE_ATTR: Mapping[BudgetDimension, str] = {
    'requests': 'requests',
    'tool_calls': 'tool_calls',
    'input_tokens': 'input_tokens',
    'output_tokens': 'output_tokens',
    'total_tokens': 'total_tokens',
}

_LABELS: Mapping[BudgetDimension, str] = {
    'requests': 'requests',
    'tool_calls': 'tool calls',
    'input_tokens': 'input tokens',
    'output_tokens': 'output tokens',
    'total_tokens': 'tokens',
}

_GUIDANCE = 'Pace your work; if nearly exhausted, prioritize delivering current results over starting new work.'


@dataclass
class BudgetDisclosure(AbstractCapability[AgentDepsT]):
    """Contribute a remaining-usage-budget line to each model request.

    Attach it to an agent that runs under a `UsageLimits` budget. On each request it reads the
    run's accumulated `ctx.usage`, computes how much of each configured limit is left, and
    contributes one short line through the capability instructions channel, for example:

        Budget remaining: ~38k tokens, 7 requests. Pace your work; if nearly exhausted,
        prioritize delivering current results over starting new work.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.usage import UsageLimits
    from pydantic_ai_harness.budget_disclosure import BudgetDisclosure

    limits = UsageLimits(request_limit=20, total_tokens_limit=200_000)
    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[BudgetDisclosure(limits=limits)])
    await agent.run('...', usage_limits=limits)
    ```

    The line is ephemeral per-request instruction text, never a stored message part, so it does
    not move the cacheable prefix as the numbers change (see the module docstring). It reads the
    same `UsageLimits` you pass to the run, because that object is not otherwise reachable from a
    capability. When no `limits` are configured, or none of the disclosed dimensions has a limit
    set, the capability contributes nothing.
    """

    limits: UsageLimits | None = None
    """The run's usage budget. Pass the same `UsageLimits` you pass to `Agent.run(...,
    usage_limits=...)`; a capability cannot read it from the run otherwise. `None` (the default)
    means the capability contributes nothing."""

    disclose: Collection[BudgetDimension] | None = None
    """Which limited dimensions to disclose. `None` (the default) discloses every dimension whose
    limit is actually set on `limits`. An explicit collection must only name dimensions whose limit
    is set."""

    start_at: float = 0.0
    """Fraction of a limit (0..1) that must be consumed before disclosure begins. `0.0` (the
    default) always discloses. `0.5` discloses only once any one disclosed dimension is at least
    half consumed, so short runs that never approach the budget stay silent."""

    round_tokens_to: int = 1000
    """Granularity that remaining token counts are rounded to for display (nearest multiple).
    Coarse numbers keep the line short and stop it from implying false precision. Request and
    tool-call counts are small and shown exactly."""

    format: BudgetFormatter[AgentDepsT] | None = None
    """Optional override for the disclosed line. Receives the run context and the remaining budget
    per active dimension (raw, unrounded); returns the line, or `None` to contribute nothing this
    request. When set, `round_tokens_to` and the default wording do not apply."""

    _disclose: tuple[BudgetDimension, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_at <= 1.0:
            raise ValueError('start_at must be between 0.0 and 1.0 (inclusive).')
        if self.round_tokens_to <= 0:
            raise ValueError('round_tokens_to must be positive.')

        if self.disclose is None:
            self._disclose = None
            return

        given = set(self.disclose)
        if not given:
            raise ValueError('disclose must not be empty; pass None to disclose every set limit.')
        unknown = given - set(_DIMENSION_ORDER)
        if unknown:
            raise ValueError(f'disclose contains unknown dimensions: {sorted(unknown)}.')
        if self.limits is not None:
            missing = [d for d in _DIMENSION_ORDER if d in given and getattr(self.limits, _LIMIT_ATTR[d]) is None]
            if missing:
                raise ValueError(f'disclose names dimensions whose limit is not set: {missing}.')
        # Re-order and de-duplicate against the canonical order for a stable line.
        self._disclose = tuple(d for d in _DIMENSION_ORDER if d in given)

    def _active_dimensions(self) -> tuple[BudgetDimension, ...]:
        """Dimensions this capability will disclose: the configured set, else every set limit."""
        limits = self.limits
        if limits is None:
            return ()
        if self._disclose is not None:
            return self._disclose
        return tuple(d for d in _DIMENSION_ORDER if getattr(limits, _LIMIT_ATTR[d]) is not None)

    def _remaining(self, usage: RunUsage, limits: UsageLimits) -> Mapping[BudgetDimension, int] | None:
        """Remaining budget per active dimension, or `None` if nothing should be disclosed yet."""
        remaining: dict[BudgetDimension, int] = {}
        peak_fraction = 0.0
        for dimension in self._active_dimensions():
            limit: int = getattr(limits, _LIMIT_ATTR[dimension])
            used: int = getattr(usage, _USAGE_ATTR[dimension])
            # A zero limit is fully consumed by definition; avoid dividing by zero.
            fraction = 1.0 if limit == 0 else used / limit
            peak_fraction = max(peak_fraction, fraction)
            remaining[dimension] = max(0, limit - used)
        if peak_fraction < self.start_at:
            return None
        return remaining

    def _humanize_tokens(self, value: int) -> str:
        rounded = round(value / self.round_tokens_to) * self.round_tokens_to
        if rounded >= 1000 and rounded % 1000 == 0:
            return f'{rounded // 1000}k'
        return str(rounded)

    def _default_line(self, remaining: Mapping[BudgetDimension, int]) -> str:
        parts: list[str] = []
        for dimension in _DIMENSION_ORDER:
            if dimension not in remaining:
                continue
            value = remaining[dimension]
            if dimension in _TOKEN_DIMENSIONS:
                parts.append(f'~{self._humanize_tokens(value)} {_LABELS[dimension]}')
            else:
                parts.append(f'{value} {_LABELS[dimension]}')
        return f'Budget remaining: {", ".join(parts)}. {_GUIDANCE}'

    async def _instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Render the remaining-budget line for the current request, or `None` to contribute nothing."""
        limits = self.limits
        if limits is None:  # pragma: no cover - get_instructions only returns this callable when limits is set
            return None
        remaining = self._remaining(ctx.usage, limits)
        if remaining is None:
            return None
        if self.format is not None:
            return self.format(ctx, remaining)
        return self._default_line(remaining)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Contribute the per-request budget line, or nothing when no limits are disclosed."""
        if not self._active_dimensions():
            return None
        return self._instructions
