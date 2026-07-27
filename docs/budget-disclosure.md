---
title: Budget Disclosure
description: Tell the model how much of its usage budget is left, so it can pace itself.
---

# Budget Disclosure

Tell the model how much of its usage budget is left, so it can pace itself.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.budget_disclosure import BudgetDisclosure
> ```

Budget Disclosure is a released, non-experimental capability. Pydantic AI Harness is still on 0.x releases, so the API may change between minor releases. See the repository [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

`UsageLimits` enforces a run's budget, but the model never sees it. It gets stopped mid-step
by a `UsageLimitExceeded` instead of adjusting as the budget runs down. `BudgetDisclosure`
reads the run's accumulated usage on each request and contributes one short line of
remaining-budget state, so the model can change strategy: with little left, land current
results rather than start new work.

```
Budget remaining: ~38k tokens, 7 requests. Pace your work; if nearly exhausted, prioritize
delivering current results over starting new work.
```

## Cache safety

The line is contributed through the capability `get_instructions` channel, so it is
*ephemeral*: instructions are rebuilt for every request and are never stored as a message part
in the run history. A remaining-budget number changes on every request by construction, so
persisting it into the history would move the cacheable prefix each turn and re-charge the whole
conversation. Instructions sit in the system-prompt region (for providers where instructions are
the system-prompt tail, they sit after the cached tools + history prefix), so a changing budget
line there does not disturb that prefix. The numbers are rounded and the line kept short
regardless, so it costs little and reads as a stable-shaped status line rather than churning
detail.

## Minimal usage

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.budget_disclosure import BudgetDisclosure

limits = UsageLimits(request_limit=20, total_tokens_limit=200_000)
agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[BudgetDisclosure(limits=limits)])
await agent.run('...', usage_limits=limits)
```

The run's `UsageLimits` is not reachable from a capability: it is passed to `Agent.run(...,
usage_limits=...)` and lives on the agent graph's run deps, not on the `RunContext` a capability
sees. So pass the same `UsageLimits` to the capability's `limits`. This mirrors
`compaction.LimitWarner`, which is configured with its own `max_*` ceilings for the same reason.

When no `limits` are configured, or none of the disclosed dimensions has a limit set, the
capability contributes nothing.

## Options

- `limits` (default `None`): the run's `UsageLimits`. `None` discloses nothing.
- `disclose` (default `None`): which limited dimensions to disclose, from `requests`,
  `tool_calls`, `input_tokens`, `output_tokens`, `total_tokens`. `None` discloses every dimension
  whose limit is set on `limits`. An explicit collection must only name dimensions whose limit is
  set.
- `start_at` (default `0.0`): fraction of a limit (0..1) that must be consumed before disclosure
  begins. `0.0` always discloses; `0.5` discloses only once any one disclosed dimension is at
  least half consumed, so short runs that never approach the budget stay silent.
- `round_tokens_to` (default `1000`): granularity that remaining token counts are rounded to for
  display. Request and tool-call counts are small and shown exactly.
- `format` (default `None`): override the line. Receives the run context and the remaining budget
  per active dimension (raw, unrounded); returns the line, or `None` to contribute nothing this
  request. When set, `round_tokens_to` and the default wording do not apply.

## Relationship to tool budgets

This capability *discloses* remaining budget; it does not enforce it. It complements
enforcement capabilities (`ToolBudget`, #168) and the underlying `UsageLimits`: those stop the
run when a ceiling is hit, this one lets the model see the ceiling coming and pace toward it.

## Scope

- **Stateless.** It reads `ctx.usage` and the static `limits` each request; there is no per-run
  state, so one instance can be reused across many runs.
- **Disclosure only.** It never changes tool availability, model settings, or message history.

## Further reading

- [`pydantic_ai_harness.budget_disclosure` source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/budget_disclosure/)
- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
