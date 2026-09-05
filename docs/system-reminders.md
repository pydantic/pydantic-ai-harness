---
title: System Reminders
description: Re-inject behavioral guidance mid-run -- on a cadence or reactively -- to counter instruction fade, without invalidating the prompt cache.
---

# System Reminders

`SystemReminders` re-states targeted behavioral guidance partway through a run -- on a fixed cadence or reactively from a condition -- to counter the instruction fade that sets in over many turns, without ever invalidating the prompt cache.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/system_reminders/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Long multi-turn runs suffer instruction fade: after many tool-use turns the model progressively ignores the guidance it was given at the start. A single start-of-session system prompt is not enough for extended work. The fix is to re-state targeted guidance mid-run -- on a fixed cadence, or reactively when a condition is detected.

## The solution

`SystemReminders` injects reminders on each model request, either statically (`Reminder`, on a cadence) or dynamically (a callable that reads the run context). Reminders are appended to the **tail** of the request as an ephemeral `UserPromptPart`:

- The injection runs *after* the durable history is persisted, so the reminder reaches the model but is never written to `message_history`. No reminders accumulate across turns.
- When one or more tagged static reminders fire, the first tagged reminder places its stable opening tag before a `CachePoint`, so its mutable body stays outside the cache. This also gives providers that map each `UserPromptPart` separately content for the cache point to attach to.
- Raw static reminders (`tag=None`) and dynamic reminders preserve their text and order. When no tagged static reminder fires, no cache point is added, because adding a hidden prefix would change their content contract.

Injecting into the system prompt (or any persisted part) instead would sit at the front of the request, so every reminder would bust the cached prefix and stale reminders would pile up in history. This capability avoids both.

## Usage

Construct an `Agent` with `SystemReminders(...)` in its `capabilities`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import Reminder

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        SystemReminders(
            reminders=[Reminder('Stay focused on the original request.', interval=5)],
        )
    ],
)

result = agent.run_sync('Refactor the auth module and add tests.')
print(result.output)
```

## Static reminders

A `Reminder` fires on a cadence within a run:

| Field | Purpose |
|---|---|
| `content` | The reminder text. |
| `interval` | Fire every N model requests (`interval=3` fires on the 3rd, 6th, ...). |
| `first_after` | Request number of the first fire, then every `interval` after. `None` = first multiple of `interval` (plain modulo). |
| `trigger` | Predicate over `RunContext`. When set, fires only when it returns `True` *and* the cadence matches. |
| `max_fires` | Cap the number of fires per run. `None` = no limit. |
| `tag` | Wrap the content in `<tag>\ncontent\n</tag>`. Defaults to `'system-reminder'`; set `None` for raw content. |

The default `tag='system-reminder'` wraps every reminder in `<system-reminder>...</system-reminder>`, following Claude Code's convention so the model reads it as an out-of-band steering note rather than user text.

The `tag` wrapping applies only to static `Reminder` content. Dynamic callables (including `GoalReanchor` and `LLMReminder`) inject their returned text raw and own their own formatting.

## Dynamic reminders

A dynamic reminder is any callable `(RunContext) -> str | None` (sync or async), evaluated on every model request. Return a string to inject, or `None` to skip. This is the general seam for conditions that need run state -- token budget, post-compaction, mode switches -- without hardcoded detectors:

```python
from pydantic_ai_harness import SystemReminders

SystemReminders(
    dynamic_reminders=[
        lambda ctx: 'Wrap up soon.' if ctx.run_step > 20 else None,
    ],
)
```

### `GoalReanchor` -- zero-cost goal anchoring

`GoalReanchor` re-states the run's first user request as the anchor and asks the model to check its next action advances it. No model call, no dependencies:

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import GoalReanchor

SystemReminders(dynamic_reminders=[GoalReanchor()])
```

### `LLMReminder` -- model-generated nudges

`LLMReminder` has a model summarize a compact transcript (original goal + recent activity) into a short stay-on-task nudge. It requires an explicit `model` -- there is no default model id -- and falls back to `GoalReanchor` text on any error, so a failed generation never blocks the run:

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import LLMReminder

SystemReminders(dynamic_reminders=[LLMReminder(model='anthropic:claude-haiku-4-5')])
```

Dynamic reminders have no cadence of their own -- they run on every model request. `LLMReminder` therefore issues one extra model call per turn (its usage is threaded onto the parent run via `ctx.usage`, so it shows up in `result.usage()`). The nested call also runs under the parent's `usage_limits` with one request held back for the model request it precedes, so the reminder cannot push a run past its `request_limit`; once the budget is that tight the generation is skipped and `GoalReanchor` text is used instead. Because the fallback is silent, a persistently misconfigured `model` (bad id, missing key) looks like normal operation. To bound the cost, gate it behind a cadence with an async wrapper:

```python
_llm = LLMReminder(model='anthropic:claude-haiku-4-5')

async def every_tenth(ctx):
    return await _llm(ctx) if ctx.run_step % 10 == 0 else None

SystemReminders(dynamic_reminders=[every_tenth])
```

Under a durability engine, a `LLMReminder` listed directly in `dynamic_reminders` is a journaled
capability operation: replay restores the recorded reminder instead of repeating the model call,
and a generation error is recorded as the `GoalReanchor` fallback rather than inheriting the
engine's retry policy, so a best-effort reminder cannot stall the run. `SystemReminders` carries the
stable default `id='system_reminders'`, so durable recovery works without configuration.

Only a direct entry takes that route. Two shapes do not:

- A wrapper like `every_tenth` above calls `LLMReminder` from orchestration context, where engines
  that forbid I/O can fail the call outright.
- An `LLMReminder` subclass that overrides `__call__` runs that override directly, so it cannot be
  journaled either.

Without a durability engine, generation runs directly in all three cases, with the same fallback to
`GoalReanchor` on error.

## Configuration

```python
from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import Reminder

SystemReminders(
    reminders=[Reminder('...', interval=5)],
    dynamic_reminders=[],       # callables evaluated every request
    cache_ttl='5m',             # TTL for the cache breakpoint after a fired tagged opening tag ('5m' | '1h')
    on_fire=None,               # optional callback invoked with each rendered reminder
)
```

Per-run state (the request counter and per-reminder fire counts) is isolated via `for_run`, so concurrent runs on the same agent never share fire state.

## Caching guarantee

Reminders are never injected into the system prompt or instructions. They live only in the ephemeral request tail, so across turns:

- the durable history grows append-only and is replayed byte-identically, so the whole prefix stays eligible for a cache hit (subject to the provider's cache TTL -- a gap longer than `cache_ttl` expires the entry even under an unchanged prefix);
- a fired tagged static reminder adds its `CachePoint` only to the per-request copy, so it cannot invalidate anything and isn't persisted.

`CachePoint` is supported on Anthropic, Amazon Bedrock (Converse API), and OpenRouter (Anthropic and Gemini models); on providers without prompt caching it's simply ignored (nothing to bust). When one or more tagged static reminders fire, the first tagged reminder puts its stable opening tag before the cache point, which keeps the cache point valid for providers that map each `UserPromptPart` independently. Raw static reminders (`tag=None`) and dynamic reminders preserve their text and order; when no tagged static reminder fires, the tail has no cache point.

## Composition

- [Planning](planning.md) uses the same ephemeral-tail mechanism to surface the plan. Both compose in one agent, and neither reminder is persisted. `Planning` adds its cache breakpoint; `SystemReminders` adds one only when a tagged static reminder fires. Anthropic allows 4 (3 with automatic caching), and core trims the excess oldest-first, so stacking several tail-injecting capabilities alongside `anthropic_cache_instructions` / `anthropic_cache_tool_definitions` can evict an older breakpoint. Two capabilities plus the defaults stay within budget.
- Loop detection (detect-and-interrupt with a durable nudge) is a separate concern. `SystemReminders` is cadence/condition steering that stays ephemeral; a dynamic reminder can read loop state from your deps if you want to steer on it.

The tail reminder is only appended when the last message in the request is a `ModelRequest` and at least one reminder fires, so a turn where nothing fires adds nothing to the request. Provider-resume turns (where the request tail is a suspended `ModelResponse` that is echoed back verbatim) are skipped and do not consume a cadence slot.

## Not spec-serializable

`SystemReminders.get_serialization_name()` returns `None`: reminders take arbitrary callables, which cannot be serialized to an [agent spec](/ai/core-concepts/agent-spec/).

## Further reading

- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Hooks](/ai/core-concepts/hooks/) -- `wrap_model_request` is the ephemeral injection point used here
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- [Planning](planning.md) -- another prompt-cache-aware harness capability

## API reference

::: pydantic_ai_harness.system_reminders.SystemReminders

::: pydantic_ai_harness.system_reminders.Reminder

::: pydantic_ai_harness.system_reminders.GoalReanchor

::: pydantic_ai_harness.system_reminders.LLMReminder
