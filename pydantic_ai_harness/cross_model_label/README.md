# Cross-Model Label

Tell the model when earlier assistant turns came from a different model.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.cross_model_label import CrossModelHistoryLabel
> ```

Cross-Model Label is a released, non-experimental capability. Pydantic AI Harness is still on 0.x releases, so the API may change between minor releases. See the repository [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

When a run continues a history whose assistant turns were produced by a *different* model than
the one now serving -- a `FallbackModel` failover, a model swap between runs, an A/B handoff, a
takeover -- the serving model otherwise reads those turns as its own. It defends claims it never
made and keeps commitments it cannot verify. `CrossModelHistoryLabel` detects the mismatch and
contributes one short line naming the other model, so the serving model treats the earlier turns
as inherited context.

```
Note: assistant responses before this point were produced by a different model (gpt-5.2).
Treat their claims and commitments as inherited context, not your own output.
```

## Cache safety and provenance

The line is contributed through the capability `get_instructions` channel, so it is *ephemeral*:
instructions are rebuilt for every request and are never stored as a message part in the run
history. That is the cache-safe channel (a note that changes with the history must not move the
cached message prefix), and it is the correct provenance channel besides: a note *about* the
history must not itself become history that a later model reads back as fact. A test asserts the
line is never a persisted message part.

## Minimal usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.cross_model_label import CrossModelHistoryLabel

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[CrossModelHistoryLabel()])
# `history` was produced by a different model earlier:
await agent.run('continue', message_history=history)
```

## Family-level comparison

Identity is compared at the *family* level by default: `gpt-5.2-mini` and `gpt-5.2` are the same
family and do not trigger the label, while `gpt-5.2` and `claude-sonnet-4-5` do. A smaller sibling
of the same model does not carry a foreign voice, so it is not worth a note.

Pydantic AI profiles do not expose a first-class family key, so the default resolver derives one
heuristically from the model name: it strips a leading provider segment, a trailing dated snapshot
(`-2024-08-06`, `-20241022`), trailing alias markers (`-latest`, `-preview`), and size-tier
suffixes (`-mini`, `-nano`, `-small`, `-lite`, `-tiny`). This is best-effort, not an authoritative
taxonomy. When you need exact control, set `granularity='exact'` (compare full normalized names) or
pass a `(model_name, provider_name) -> key` callable. The default resolver is exported as
`model_family` so a callable can wrap it.

## `threshold`: handoff nudge vs provenance banner

- `'recent'` (default) fires only when the immediately preceding response (the most recent one
  carrying a `model_name`) is a different family. It acts as a one-shot handoff nudge: it fires on
  the first request after the model changes, then goes quiet once the serving model has added its
  own turn (its own turn becomes the most recent).
- A float in `(0, 1]` fires when at least that fraction of all prior responses (that carry a
  `model_name`) are a different family. It acts as a persistent provenance banner: it keeps firing
  for as long as the history stays majority-foreign. When several other families are present it
  names the most common one (ties broken by most recent).

## FallbackModel

Under a `FallbackModel` the serving member is not known until it answers, so the current identity
is taken from the most recent response's `model_name`, which records who actually served (Pydantic
AI core [#6338](https://github.com/pydantic/pydantic-ai/pull/6338)). Before any response, it falls
back to the wrapper's first candidate. So a mid-history failover -- early turns from candidate A,
later turns from candidate B -- is detected against B, the model now serving.

## Options

- `granularity` (default `'family'`): `'family'`, `'exact'`, or a `(model_name, provider_name) ->
  key` callable.
- `threshold` (default `'recent'`): `'recent'`, or a float in `(0, 1]`.
- `format` (default `None`): override the line. Receives the run context and a `CrossModelHistory`
  summary (current family, other family, raw other name, differing/known counts); returns the line,
  or `None` to contribute nothing this request.

## Scope

- **Stateless.** It reads only `ctx.model` and `ctx.messages` each request; one instance is reusable
  across runs.
- **Detect and disclose only.** It never mutates the message history, tool availability, or model
  settings. Unlabeled prior responses (no `model_name`) are skipped, never guessed.
- **One line at most.** It contributes a single line per request, or nothing.

## Further reading

- [`pydantic_ai_harness.cross_model_label` source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/cross_model_label/)
