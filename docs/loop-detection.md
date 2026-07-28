---
title: Loop Detection
description: Detect when an agent is stuck in a repeated-action loop and intervene.
---

# Loop Detection

Detect when an agent is stuck in a repeated-action loop and intervene.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.loop_detection import LoopDetection
> ```

Loop Detection is a released, non-experimental capability. Pydantic AI Harness is still on 0.x releases, so the API may change between minor releases. See the repository [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## The problem

An autonomous agent that gets stuck does not stop -- it repeats the same action until it
exhausts its step or token budget: re-reading a missing file, re-running a command that keeps
failing, thrashing between two edits, or narrating what it will do without doing it. Five
major coding harnesses (Gemini CLI, OpenHands, Roo, Crush, goose) each grew some form of loop
detection to break out of this. `LoopDetection` is that guardrail as a composable capability.

## Minimal usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.loop_detection import LoopDetection

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[LoopDetection()])
await agent.run('...')  # a stuck loop nudges the model to change approach
```

## Detection tiers

| Tier | Signal | Default |
|------|--------|---------|
| Exact repetition | The same `(tool_name, canonical_args)` fingerprint occurs `repeat_threshold` times within a sliding `window` of recent calls. Catches loops that interleave an occasional different call. | 5 within 10 |
| Error cycle | The same tool call returns a byte-identical result on `error_cycle_threshold` consecutive executions. Coding-agent tools usually report failure as an ordinary error-shaped result rather than by raising, so an identical repeated result is the signature of a call that is not making progress. | 3 |
| Alternation | Two distinct call fingerprints alternate A-B-A-B for `alternation_cycles` full cycles (a two-step thrash, e.g. edit then re-read). | 3 cycles |
| Monologue | `monologue_threshold` consecutive model responses carry no tool call and near-identical text (normalized prefix match): the model is narrating instead of acting. | 3 |

Arguments are canonicalized (JSON with sorted keys) before fingerprinting, so `{"a": 1, "b": 2}`
and `{"b": 2, "a": 1}` count as the same call. After a tier fires, its counters reset, so the
same loop has to rebuild before it fires again rather than triggering on every later step.

## Action on detection: `on_loop`

- `'nudge'` (default): enqueue a harness-marked message that the model sees on its next
  request, e.g. *"You appear to be repeating the same action (`read_file` called 5 times with
  identical arguments). Change approach, or state plainly what is blocking you."* The nudge
  uses Pydantic AI's pending-message queue, so it is delivered as a real user turn on the next
  model request (and redirects the run into one more request if the agent would otherwise
  stop).
- `'error'`: raise `LoopDetectedError` to abort the run. The structured `LoopDetected` is on
  `error.detected`.
- a callable `(LoopDetected) -> None | Awaitable[None]`: called with the structured detection.
  It may be sync or async; raise from it to abort, or use it to log, record, or enqueue custom
  steering.

```python
from pydantic_ai_harness.loop_detection import LoopDetected, LoopDetection

def on_loop(detected: LoopDetected) -> None:
    print(detected.tier, detected.tool_name, detected.count)

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[LoopDetection(on_loop=on_loop)])
```

`LoopDetected` carries `tier`, `tool_name` (`None` for a monologue), `count`, `window`, the
canonical `fingerprints` involved, and the rendered `message`.

## Options

- `repeat_threshold` (default `5`), `window` (default `10`): tier 1 sliding-window counting.
- `error_cycle_threshold` (default `3`): tier 2a consecutive identical results.
- `alternation_cycles` (default `3`): tier 2b full A-B cycles.
- `monologue_threshold` (default `3`), `monologue_prefix_chars` (default `200`): tier 2c
  consecutive near-identical text-only responses and how many leading normalized characters
  two responses must share to count as near-identical.
- `on_loop` (default `'nudge'`): see above.

## Composition

- The capability only implements `for_run`, `after_model_request`, and `after_tool_execute`;
  it adds no tools, instructions, or model settings, so it composes with any toolset,
  `ToolSearch` setup, or other capability without interference.
- Per-run state (the sliding window and counters) is materialized in `for_run`, so one
  `LoopDetection` instance can be reused across many `Agent.run` calls -- concurrent runs never
  share counters.

## Observability

On detection the capability adds a `loop_detection.detected` event (with `loop.tier`,
`loop.tool_name`, `loop.count`, `loop.window` attributes) to the active OpenTelemetry span,
so a loop shows up in a Logfire/OTel trace whether or not `on_loop` aborts the run. When no
span is recording the event is a no-op.

## Scope

- **Signal-based, not semantic.** It catches structural loops (repetition, cycles,
  monologues), not a model that is making slow but real progress. Keep the thresholds
  conservative so a legitimately repeated action is not flagged.
- **No LLM-judge tier.** Detecting subtler "spinning" with a small model is intentionally out
  of scope until harness settles a small-model-roles convention.
- **No pause/approval integration.** The actions are nudge, error, or callback; wiring a loop
  into an approval/pause flow is left to the caller's callback.

## Further reading

- [`pydantic_ai_harness.loop_detection` source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/loop_detection/)
- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Pydantic AI hooks](/ai/core-concepts/hooks/) -- `after_tool_execute` and `after_model_request` (observe) and `for_run` (per-run state) are the surfaces used here
