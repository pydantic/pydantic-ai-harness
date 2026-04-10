# AdaptiveReasoning Capability

Closes #84

## Summary

A capability that dynamically adjusts model thinking effort per agent step,
using the `get_model_settings()` callable mechanism from Pydantic AI's
capabilities abstraction. This reduces token usage on simple steps (file reads,
straightforward follow-ups) while preserving deep reasoning for complex
decisions (first step task understanding, error recovery).

## Design

### Approach: capability with dynamic `get_model_settings`

As noted by @DouweM in #84, this is cleanly implemented as a capability whose
`get_model_settings()` returns a callable receiving `RunContext`. The callable
inspects `ctx.run_step` and `ctx.messages` to select an effort level, then
returns `ModelSettings(thinking=...)`.

This leverages two existing Pydantic AI primitives:
1. **Dynamic model settings** (callable `get_model_settings`) -- resolved per
   model request with the current `RunContext`
2. **Unified `thinking` setting** -- maps to provider-specific parameters
   (Claude thinking budget, OpenAI reasoning_effort, etc.)

### Effort levels

Three coarse levels (`'low'`, `'medium'`, `'high'`) mapped directly to
`ThinkingEffort` values. These are a subset of the full `ThinkingEffort` scale
(`'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`) chosen to match the
research literature (Ares uses three tiers) and keep the API simple.

### Built-in heuristic (`default_effort_fn`)

Rules evaluated in order:
1. First step (`run_step <= 1`): `'high'` -- understand the task
2. After tool errors (retry prompts in latest request): `'high'` -- reason about failures
3. Later steps without errors: `'low'` -- simple follow-ups incorporating tool results

### Custom effort function

Users can supply `effort_fn: Callable[[RunContext], Literal['low', 'medium', 'high']]`
to override the built-in heuristic with domain-specific logic.

## Files

| File | Purpose |
|------|---------|
| `src/pydantic_harness/adaptive_reasoning.py` | `AdaptiveReasoning` capability, `default_effort_fn`, `EffortLevel` type alias |
| `src/pydantic_harness/__init__.py` | Re-export `AdaptiveReasoning` |
| `tests/test_adaptive_reasoning.py` | 18 tests covering helper, heuristic, capability, and custom fn |

## Not included (future work)

- **`ModelRoutedEffort`**: Small model (Haiku-class) predicting effort from history (the Ares approach). This is a natural follow-up but requires model call infrastructure.
- **`PhaseBasedEffort`**: High for planning, medium for execution, high for verification. Requires a phase detection mechanism.
- **Provider-specific token budgets**: Mapping effort levels to concrete `budget_tokens` values per provider. The current implementation uses the unified `thinking` setting which handles this portably.
