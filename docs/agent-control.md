---
title: Agent Control
description: Drive a Pydantic AI agent's instructions, model, model settings, and tool definitions from one Logfire-managed variable, so you can change its behavior without redeploying.
---

# Agent Control

`AgentControl` backs a whole agent's configuration -- its instructions, its model, that model's
settings, and the LLM-facing definitions of its tools -- with a single
[Logfire-managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
named `agent__<name>`. That is the same variable Logfire's Agent Control UI edits, so the agent
becomes tunable from Logfire -- versioned, labelled, and rolled out -- without touching code. It's a
Pydantic AI [capability](index.md), so you wire it in through the `capabilities=` parameter on
`Agent`.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire/)

Install the `logfire` extra:

```bash
uv add "pydantic-ai-harness[logfire]"
```

## The problem it solves

An agent's behavior is spread across knobs that all live in code: the instructions, the model and its
sampling settings, and the tool descriptions that make up half of what the model actually sees.
Tuning any of them takes a redeploy per tweak. Tuning them *together* -- a new prompt that only works
with a smarter model, a reworded tool description that the prompt now refers to -- takes coordinated
redeploys, and there is no single thing to roll back when the combination misbehaves.

`AgentControl` makes the whole configuration one versioned unit. A prompt change and the model change
it depends on land atomically, roll out together, and roll back together.

## Usage

Pass nothing and the variable name comes from the agent's own `name`, normalized exactly the way the
Logfire UI normalizes an observed agent's name -- so the variable this resolves is the variable the UI
creates when you click "control" on an agent you can already see in your traces:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import AgentControl

logfire.configure()
logfire.instrument_pydantic_ai()


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f'The weather in {city} is sunny.'


agent = Agent(
    'openai:gpt-5',
    name='checkout_assistant',
    tools=[get_weather],
    capabilities=[
        AgentControl(  # -> agent__checkout_assistant
            instructions='You are a concise checkout assistant.',
            label='production',
        )
    ],
)

result = agent.run_sync('Refund my last order.')
print(result.output)
```

Resolving managed values needs a `LOGFIRE_API_KEY` with the `project:read_variables` scope, and
publishing the code baseline also needs `project:write_variables` -- both different credentials from
the write token that sends spans. Instrumentation is worth adding even if you have it elsewhere:
without spans, neither the config's adoption nor the version that produced a given run makes it back
to the Logfire UI.

Pinning `label='production'` is the recommended default, for the same
[prompt-cache reasons](managed-prompt.md#prompt-cache-trade-off) as a managed prompt.

## One rule: a managed value is a patch

The variable holds an `AgentConfig`. Every section that is **present** is managed from Logfire; every
**absent** section keeps whatever the code does. Removing a section in Logfire is therefore a
deliberate revert to code, not a gap.

The four sections read differently because a patch's unit is whatever the thing it patches is keyed
by, and the four are keyed differently:

| Section | Patched per | Can add something new? |
| --- | --- | --- |
| `model` | the whole value | no |
| `settings` | each setting | -- |
| `tool_definitions` | each tool, by its code-side `name` | no -- an entry naming a tool that doesn't exist is inert |
| `instructions` | each block, by its `id` | **yes** -- an entry with no `id` adds a block |

Instructions are the only section that can add, and what makes them the exception is what rules the
others out: text is the only thing a remote value can supply whole. A tool needs an implementation
and a model needs a provider, so for those a managed value can only ever patch what the code already
declares -- which is also what keeps a published config from being able to make an agent do something
its code cannot.

Together those look like this:

```json
{
  "instructions": [
    "Always confirm the order total.",
    {"id": "agent", "instructions": "You are a concise checkout assistant."},
    {"id": "toolset:legacy_crm", "instructions": null}
  ],
  "model": "openai:gpt-5",
  "settings": {"temperature": 0.4, "max_tokens": 2048, "thinking": "high"},
  "tool_definitions": [
    {
      "name": "get_weather",
      "new_name": "lookup_weather",
      "description": "Look up the current weather for a city.",
      "parameter_descriptions": {"city": "City name, e.g. 'London'"}
    }
  ]
}
```

A missing, invalid, or unreachable remote value degrades to exactly the agent you wrote -- never a
crashed run. Values resolve **once per run**, and the resolved label and version ride as baggage on
every span of that run, so a trace always shows which version produced which behavior.

An agent with no model of its own is the exception, because there is nothing to degrade *to*: with
`Agent(capabilities=[AgentControl(...)])` and no valid managed `model`, the run raises rather than
guessing one. Keep a model in code if you want a Logfire outage to be a non-event; leave it out only
when the managed value is meant to be the single source of truth for which model runs.

## Instructions add, or take over

Instructions are the one section that is a **composition point** rather than a field. Pydantic AI
assembles them from every source that contributes one: the agent's own literal, `@agent.instructions`
functions injecting today's date or the signed-in user, each toolset and MCP server, each capability,
a skill catalog. A managed section that replaced all of that would silence every one of them.

So an entry does one of two things, and which one it does is the difference between a prompt that
reads well and one sent to the model twice.

**No `id` adds a block.** A bare string is the shorthand:

```python
from pydantic_ai_harness.logfire import AgentConfig

AgentConfig(instructions='Always confirm the order total.')
```

**An `id` swaps out a block that already exists** -- replacing its text, or dropping it with `None`.
The block has to be one whose text the agent wrote down, not one it computes per request; see below:

```python
from pydantic_ai_harness.logfire import AgentConfig, InstructionBlock

AgentConfig(
    instructions=[
        'Always confirm the order total.',
        InstructionBlock(id='agent', instructions='You are a refund specialist.'),
        InstructionBlock(id='toolset:legacy_crm'),
    ]
)
```

The keys are Pydantic AI's
[`InstructionPart.id`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.InstructionPart.id):

A **source key** addresses everything that source contributes; appending a segment addresses one
part the author named within it:

| `id` | Addresses |
| --- | --- |
| `agent` | the agent's own literal `instructions` |
| `toolset:<id>` | everything a toolset with that `id` contributes |
| `capability:<id>` | everything a capability with that `id` contributes |
| `agent:<name>` | one part the agent named |
| `toolset:<id>:<name>` | one part that toolset named |
| `capability:<id>:<name>` | one part that capability named |

A code author names a part either where a function is registered
(`@agent.instructions(name=...)`, `@capability.instructions(name=...)`) or on the text itself, by writing
it as an `InstructionPart` with a `name` — in `Agent(instructions=...)`, `Capability(instructions=...)`,
or a `get_instructions()` return. They name the part relative to what they own (`limits`) and
Pydantic AI qualifies it against the source key (`toolset:weather:limits`), so a code author never
repeats their own identity and no part can claim a top-level key.

Parts Pydantic AI cannot key cannot be addressed at all: a callable passed to
`Agent(instructions=...)` with no `name`, anything from `run(instructions=...)`, and anything from a
toolset or capability that has no `id` of its own — without a source key there is nothing for a name
to hang off. Give the toolset or capability an `id`, or name the part, if you want it reachable. An
`id` that matches nothing here is inert rather than an error, so one config can be applied across
services that don't all install the same toolsets.

**A block the agent recomputes per request is not addressable either**, keyed or not. Replacing it
would pin whatever it evaluated to once; dropping it would remove the computation. Either way the
block stops doing what it was written to do, and the managed value carries nothing that says which
was meant — so both are ignored, with a warning. This covers every instruction *function*: one
registered with `@agent.instructions`, `@toolset.instructions` or `@capability.instructions`, even
when it returns fixed text. Write text you mean to manage remotely as text — `Agent(instructions=...)`,
`Capability(instructions=...)`, a `get_instructions()` string — and it stays addressable. Which side
of this line a block falls on is published in its baseline `dynamic` flag, so the Logfire editor can
show a block without offering to change it.

!!! warning "Don't re-add the prompt you already have"
    Copying an agent's observed system prompt into a managed value as *added* text -- lifted out of a
    trace, or the whole baseline snapshot -- while the same text stays in `Agent(instructions=...)`
    sends every block of it to the model **twice**, and freezes any dynamic block (`Today is
    2026-07-29.`) at whatever it said when you copied it. Take the text over by `id`, or move it to
    the capability. Don't add it back.

### Where a managed base prompt belongs

| Where | Managed from Logfire? | What a published config does to it |
| --- | --- | --- |
| `Agent(instructions=...)` | Only by `id` | Adds to it by default; an `agent` entry replaces or drops it. |
| `AgentControl(instructions=...)` | It *is* the code-side default | **Supersedes** it -- the capability contributes the published value or the default, never both. |
| The published `instructions` in Logfire | Yes | It is the value. |

`AgentControl(instructions=...)` is shorthand for `default=AgentConfig(instructions=...)`; passing
both raises a `UserError` rather than picking one for you. An `agent` override is the sharper tool,
but the capability default is still the better home for a prompt you *intend* to manage: it
supersedes rather than shadows, so there is only ever one copy of the text and no way for the two to
drift apart.

### Composition and prompt caching

Added blocks compose the way a capability's instructions always have. Pydantic AI groups static
instruction text ahead of dynamic text -- so a provider can cache the stable prefix -- and preserves
source order within each group, which puts added blocks after the agent's own literal and its
`@agent.instructions` text and before dynamic toolset instructions.

An override leaves its block's position and its `dynamic` flag untouched. Replacing a dynamic block
does not make it static, so **no override moves the cacheable prefix**. It does pin that block's text
for as long as the override is published, which is what the baseline's `dynamic` flag exists to warn
you about before you publish it.

## Tool definitions are overlays, never code

Each entry names a tool by its code-side `name` and patches only what the model is shown:

```python
from pydantic_ai_harness.logfire import AgentConfig, ToolDefinitionOverride

AgentConfig(
    tool_definitions=[
        ToolDefinitionOverride(
            name='get_weather',
            new_name='lookup_weather',
            description='Look up the current weather for a city.',
            parameter_descriptions={'city': "City name, e.g. 'London'"},
        )
    ]
)
```

The tool itself -- its implementation and the *structure* of its parameter schema -- stays exactly as
written, so a remote value can never drift from the validator the tool actually runs against.
Parameter names, types, and requiredness are code-owned; only their description strings are patchable.

`new_name` changes the name the model is shown, and a call to the renamed tool routes back to the
original implementation, with `ctx.tool_name` still the original name. A rename that collides with a
name another tool already advertises is dropped with a warning -- the tool keeps its original name and
the entry's other patches still apply -- rather than breaking the run. An entry naming a tool that no
longer exists is inert; that's the drift case, and the Logfire UI is where it becomes visible.

## Model and settings precedence

`model` is a first-class field rather than a setting, because Pydantic AI keeps the model id separate
from `ModelSettings` (which has no `model` key). It is sourced during model selection, so it lands
with the precedence you'd expect:

```
run(model=...)  >  managed model  >  Agent(model=...)
```

A fully model-less `Agent(None, name='...')` can therefore be driven entirely from Logfire.

`settings` keys are the canonical, cross-framework ones -- they match
[`ModelSettings`](https://ai.pydantic.dev/api/settings/) -- with a nested `provider_options` escape
hatch, where `provider_options.openai.reasoning_effort` lowers to the `openai_reasoning_effort` model
setting and a provider-specific value wins over its canonical counterpart. They merge **over** the
agent's constructor `model_settings` and **under** per-run `model_settings=`, so run arguments always
win.

The only thing above a managed value in either list is the call site, and that is the whole rule: a
published value overrides what the code says, and a `run(...)` argument overrides the published one
for that single run. Nothing in between changes it -- another capability contributing a model or
settings of its own does not outrank the config the Logfire UI is showing you, whichever order the
capabilities were registered in.

## Degradation is per value, never per config

An `AgentConfig` that fails validation falls back to the code-defined agent *whole*, through Logfire's
resolution fallback. That makes strictness expensive: one unfamiliar enum value would silently
un-manage the instructions, the model, and every tool override alongside it. So nothing the SDK fails
to understand costs more than the part that contains it:

- **Unknown keys** are ignored, at every level, so a value written by a newer Logfire UI degrades to
  the sections this SDK understands.
- **A value it can't act on** -- a `thinking` effort level or `service_tier` a newer Pydantic AI
  accepts, a tool entry with no `name`, an instruction entry with empty text -- drops that one
  setting, that one tool, that one block.
- **Malformed sections degrade locally.** A wrong settings or tool-definitions container drops that
  section; malformed settings keys drop independently; valid sibling sections still apply.

Every drop warns once per process, naming the offending value, because a config resolves on every
single run and a per-run warning would bury the signal in its own repetition.

Each instruction block is limited to 65,536 characters. An oversized bare `instructions` value drops
that section; an oversized list entry drops only itself. Valid sibling entries and other config
sections continue to apply.

`''` is never accepted where a string carries meaning. Omission and `null` already mean "leave this to
code", so an empty string is only ever a half-filled field -- and `"model": ""` is not "no model":
Pydantic AI raises `Unknown model:` on every request the agent makes, and the config around it is
valid, so nothing downstream would catch it.

## Auto-create and the code-side baseline

When the backing variable doesn't exist in Logfire yet, it is created in the background on first use,
with this contract's JSON schema (exported as `AGENT_CONFIG_JSON_SCHEMA`) and a description -- so the
Logfire UI becomes the editing surface with no manual create step. Creation happens off the run's
thread, is attempted at most once per process per variable, and never blocks or fails the run. Because
the variable is persistent and visible to everyone with access to the project, the outcome is reported
back into that project: a log record on success, and a log record plus a `UserWarning` on failure. Opt
out with `auto_create=False`.

Inside a Temporal, DBOS, or other Pydantic AI durable workflow or flow, managed config remains
readable. Auto-create and baseline publishing are skipped with one warning because background threads
and remote writes are not replay-safe there. Create the variable in the Logfire UI, or run the SDK
outside the workflow once to perform the write-backs.

The variable's `example` is an `AgentConfig`-shaped snapshot of the code-side agent, which is
what the UI's editor and Logfire's optimizer diff managed values against. Instructions are snapshotted
**per block**, each with the `id` that addresses it and its `dynamic` flag -- which is what makes a
per-block override offerable in the first place. The joined prompt a trace records has no seams in it,
so a baseline built from telemetry alone could only ever be copied wholesale.

The snapshot means what the agent would do with `AgentControl` removed. The capability records the
code model and settings before replacing them and the original tool definitions before applying
renames or descriptions. It excludes its own instruction contribution by capability id. This avoids
trying to reverse managed behavior out of an already assembled request.

`AgentControl` publishes that snapshot in the background after the first model request that exposes a
baseline. It compares against the provider's current definition first, so an unchanged baseline
causes no write; a changed baseline is attempted at most once per process. The targeted update copies the
complete current definition and changes only `example`, preserving its labels, rollout, overrides,
schema, and other metadata.

The snapshot comes from the model request that triggers publishing, so for
instructions or a toolset that vary with `deps`, run input, or the step within a run, it is one
point-in-time sample rather than a description of the agent. An agent that never reaches a model
request never publishes a baseline.

Publishing is on by default because `example` is documentation and is never resolved or applied to a
run. A stale or failed publish cannot change agent behavior. A missing `project:write_variables`
scope, network error, or missing variable warns once per process and otherwise leaves the run alone.
Pass `publish_baseline=False` to disable publishing, for example when the process intentionally uses a
read-only variables token. `auto_create=False` remains a separate opt-out for creating a missing
variable.

## Naming and sharing

Omitting `name` derives the variable from the agent's `name`. That normalization is lossy --
`checkout-assistant`, `Checkout Assistant`, and `checkout_assistant` all land on
`agent__checkout_assistant`, including across different services reporting to the same project -- so
two agents that differ only in punctuation share one managed config. Pass an explicit `name` to keep
them apart, or to deliberately share one config across agents. `targeting_key`, `attributes`, and
`render_template` work exactly as they do for [`ManagedPrompt`](managed-prompt.md#targeting).

## How it composes

- **Resolves once per run.** A label flip that lands mid-run is picked up by the next run, not this
  one -- the trade-off for a run-stable config and a single baggage scope across all child spans.
- **Callable targeting is consistent with model selection.** Static or callable `targeting_key` and
  `attributes` are evaluated for model selection once, and the run reuses that exact resolution.
  An unknown managed model warns and keeps the code model.
- **Overrides apply at the last moment.** Instruction overrides are applied in `before_model_request`,
  the one point at which every contribution exists, which is what lets an entry reach a toolset's or
  an MCP server's text. They land on
  [`instruction_parts`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.ModelRequestParameters.instruction_parts),
  the source of truth for what the model is sent, so message history and traces show the rewritten
  prompt too.
- **Adoption is reported.** For the run's duration, `logfire.managed.applied_sections` baggage names
  the sections that were applied (e.g. `instructions,settings`), which the Logfire UI reads to tell a
  wired-up agent from one whose config resolves but isn't applied.
- **Concurrency-safe.** Resolution is isolated per run via a context variable, so one capability
  instance is safe to share across concurrent runs.
- **Inspectable mid-run.** `AgentControl.resolved` exposes the active run's `ResolvedVariable`
  (`value`, `label`, `version`, `reason`); it is `None` outside a run.

## API reference

::: pydantic_ai_harness.logfire.AgentControl

::: pydantic_ai_harness.logfire.AgentConfig

::: pydantic_ai_harness.logfire.InstructionBlock

::: pydantic_ai_harness.logfire.ToolDefinitionOverride

::: pydantic_ai_harness.logfire.AgentConfigSettings
