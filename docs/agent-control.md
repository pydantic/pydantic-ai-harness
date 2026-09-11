---
title: Agent Control
description: Change a Pydantic AI agent's instructions, model, model settings, and tool descriptions from Logfire, without a redeploy.
---

# Agent Control

Agent Control is the Logfire feature for changing how a running agent behaves -- its instructions, the
model it runs on, that model's settings, and the tool descriptions the model reads -- from Logfire
instead of from code. Changes are versioned, labelled, and rolled out like anything else you publish
there, and they take effect without a redeploy.

`AgentControl` is the Pydantic AI [capability](index.md) that connects your agent to it. You add it,
and the agent you already wrote becomes editable from Logfire.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire/)

Install the `logfire` extra:

```bash
pip/uv-add "pydantic-ai-harness[logfire]"
```

## The problem it solves

An agent's behavior is spread across knobs that all live in code: the instructions, the model and its
sampling settings, and the tool descriptions that make up half of what the model actually sees.
Tuning any one of them takes a redeploy. Tuning them *together* -- a new prompt that only works with a
smarter model, a reworded tool description the prompt now refers to -- takes coordinated redeploys,
and leaves nothing single to roll back when the combination misbehaves.

Agent Control makes the whole configuration one versioned unit that lands, rolls out, and rolls back
together.

## Usage

Write the agent exactly as you would anyway, give it an explicit `name`, and add the capability:

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
    'anthropic:claude-fable-5-1',
    name='checkout_assistant',
    instructions='You are a concise checkout assistant.',
    tools=[get_weather],
    capabilities=[AgentControl(label='production')],
)

result = agent.run_sync('Refund my last order.')
print(result.output)
```

That's the whole setup. The agent's `name` is how Logfire matches it to the agent you can already see
in your traces, and the first run puts it on the Agent Control page with nothing to create by hand.

Agent Control needs a `LOGFIRE_API_KEY` with the `project:read_variables` and
`project:write_variables` scopes -- a different credential from the write token that sends spans.
Instrumentation is worth keeping even if you have it elsewhere: without spans, neither the version
that produced a given run nor whether the agent is picking its config up at all makes it back to
Logfire.

Pinning `label='production'` is the recommended default, for the same
[prompt-cache reasons](managed-prompt.md#prompt-cache-trade-off) as a managed prompt.

## What you can change from Logfire

| | What Logfire can do |
| --- | --- |
| **Instructions** | Rewrite or remove any block of the prompt, and add new ones |
| **Model** | Run the agent on a different model |
| **Model settings** | Change settings individually -- `temperature`, `max_tokens`, `thinking`, ... |
| **Tools** | Rename a tool and reword its description and its parameters' descriptions, for the model's eyes only |

Anything you don't change in Logfire keeps doing what the code says, and removing a change there puts
that piece back the way the code has it.

The settings are the canonical ones every framework has a knob for, under the names
[`ModelSettings`](https://ai.pydantic.dev/api/settings/) gives them. Provider-specific settings
(`openai_reasoning_effort`, `extra_headers`) stay in code, where they always were.

What a config may hold is the same contract in every Agent Control SDK, and it lives in one place:
the `logfire.agent_control` package that ships with the `logfire` extra. `AgentControl` reads it
from there, so what the Logfire UI offers, what a TypeScript service reads, and what this agent
applies cannot drift apart. Import `AgentConfig` from that package when you want to publish a value
from code rather than from the UI.

A tool's implementation and the shape of its arguments stay code-owned: Logfire edits what the model
is *told* about a tool, never what it *does*. A renamed tool still routes to the same function, and
`ctx.tool_name` is still the original name, so nothing in code has to know about the rename. Logfire
shows which toolset each tool came from, and a change can be narrowed to that toolset's tool of a
given name -- so two services sharing one config, each with its own `search`, can be told apart.

Your call site still wins. A `run(model=...)` or `run(model_settings=...)` argument overrides the
published value for that one run, and the published value overrides what the agent was constructed
with. Nothing in between changes it: another capability contributing a model or settings of its own
does not outrank the config the Logfire UI is showing you.

## Which parts of the prompt are editable

Everything else is one value, but instructions are assembled from many places -- your agent's text,
`@agent.instructions` functions injecting today's date or the signed-in user, each toolset and MCP
server, each capability, a skill catalog. So Logfire edits them **one block at a time**, and a block
it can identify is a block it can offer to change.

Most blocks already have an identity. Your agent's own text is `agent`; a toolset's is
`toolset:<id>`; a capability's is `capability:<id>`. Two cases are worth knowing about, and Logfire
marks both in the editor so you can see which of your blocks they are:

**A block computed on every request is shown but not editable.** That's any instruction *function* --
`@agent.instructions` and its toolset and capability equivalents -- even one that returns fixed text.
Rewriting it would freeze whatever it happened to return once, and removing it would delete the
computation.

Only the *fact* of such a block goes to Logfire, never its text. An instruction function reads the
run -- a tenant, a signed-in user, a retrieved document -- and what Logfire is told is the block's id
and that it is recomputed, which is all the editor needs to show it and to not offer to change it.

So if one function today produces a prompt that is mostly fixed with a dynamic piece in it, split it
in two: the fixed part as text, the varying part as its own function. The fixed half becomes editable
and the varying half stays where it belongs.

```python {test="skip"}
agent = Agent(
    'anthropic:claude-fable-5-1',
    name='checkout_assistant',
    instructions='You are a concise checkout assistant.',  # editable in Logfire
    capabilities=[AgentControl()],
)


@agent.instructions(name='today')  # shown, never edited, never sent
def today(_ctx: RunContext[None]) -> str:
    return f'Today is {date.today()}.'
```

**A block with nothing to identify it isn't offered at all.** That's text from a toolset or capability
with no `id` of its own, an unnamed callable in `Agent(instructions=...)`, or anything passed to
`run(instructions=...)`. Give the toolset or capability an `id` and its text becomes editable.

To make one block of a larger prompt separately editable, give that part a name:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.messages import InstructionPart

from pydantic_ai_harness.logfire import AgentControl

agent = Agent(
    'anthropic:claude-fable-5-1',
    name='checkout_assistant',
    instructions=[
        'You are a concise checkout assistant.',
        InstructionPart(content='Always confirm the order total before refunding.', name='refunds'),
    ],
    capabilities=[AgentControl()],
)
```

Logfire now offers those as two blocks, `agent` and `agent:refunds`, so the refund rule can be
reworded without touching the rest of the prompt. The same works wherever instructions are written:
`@agent.instructions(name=...)`, `Capability(instructions=...)`, a toolset's `get_instructions()`. A
name is relative to whatever owns it, and Pydantic AI qualifies it with the source, so
`name='limits'` on a toolset becomes `toolset:weather:limits` and no two can collide.

!!! warning "Take a prompt over, don't paste it back in"
    Logfire can also *add* instruction blocks, which is the one thing that would let the same text
    reach the model twice. Copying your agent's prompt out of a trace and publishing it as an added
    block, while that text is still in the code, sends every line of it twice -- and freezes anything
    per-request (`Today is 2026-07-29.`) at whatever it said when you copied. Edit the existing
    blocks instead.

## When something doesn't apply

If Logfire is unreachable, has nothing published, or has something this SDK is too old to understand,
the agent runs exactly as written -- a published config can never crash a run. The one exception is an
agent with no model of its own: `Agent(None, ...)` has nothing to fall back to, so it raises rather
than guessing. Keep a model in code if you want a Logfire outage to be a non-event.

Nothing this SDK can't act on costs more than the piece that contains it. A setting it doesn't
recognize drops that setting; a malformed tool entry drops that tool; a block it can't apply drops
that block. The rest of the config still applies, and each drop warns once per process naming what it
skipped, rather than once per run.

A published change can also be perfectly valid and still reach nothing *here*. `on_unmatched` decides
what that costs:

```python {test="skip"}
AgentControl(label='production', on_unmatched='error')
```

That covers an instruction block this deployment never assembles (or only computes per request), a
tool no toolset advertises, a parameter a patched tool does not have, a rename another tool already
answers to, a setting the contract has no field for, and a `timeout` no request could be given.

- `'warn'` (the default) emits a `UserWarning` once per process, at the point the change would have
  applied, so a change Logfire shows and the agent isn't making is visible without stopping anything.
- `'error'` raises `UserError` with the same message there, for a deployment that would rather stop
  than run with part of its published config unapplied.
- `'ignore'` applies nothing and says nothing.

Warning is the default because tool availability is dynamic: one config is applied across services
that need not all install the same toolsets, and a toolset can advertise different tools from one
request to the next, so a change that reaches nothing now is not necessarily wrong.

Changes are picked up **once per run**. Publishing mid-run takes effect on the next run, which is what
lets every span of a run agree on the version that produced it.

## Registration and the baseline

You never have to create anything in Logfire by hand. The two write-backs that make that true both
run in the background, off the run's thread, and neither can fail or slow a run.

**Registering the agent.** On a run where Logfire has no config for this agent yet, `AgentControl`
creates one, seeded with the code baseline below. It first confirms with Logfire that the agent
really is unknown -- "there is no config" and "there is no config *for you*" are different answers,
and only the first should create anything. It is attempted **once per process per agent**, so a
failed attempt does not retry in a loop. Because what it creates is visible to everyone with access
to the project, the outcome is reported there: a log record on success, a log record and a
`UserWarning` on failure. `auto_create=False` opts out.

**The baseline.** Logfire's editor shows published values as changes *to something*, and that
something is a snapshot of what your agent does in code: its prompt block by block, its model, its
settings, its tool definitions. It is documentation -- never resolved, never applied to a run -- so a
stale or failed snapshot cannot change what your agent does.

It is captured on the first model request of the process that has one to capture, which means an
agent that never reaches a model never publishes, and a prompt or toolset that varies with `deps`,
the run's input, or the step within a run is a point-in-time sample. Publishing is attempted once per
process per agent, is a no-op when the snapshot already matches what Logfire holds, and writes only
the baseline -- your published config, labels, and rollout are preserved. `publish_baseline=False`
opts out, for instance when the process deliberately holds a read-only token.

**Inside a durable workflow** (Temporal, DBOS, ...) published config still applies, but both
write-backs are skipped with one warning: they write from background threads, which is not
replay-safe. Run the agent outside the workflow once to get it registered.

## Which agent in Logfire it controls

The agent's `name`, which you have to **set explicitly** -- `AgentControl` refuses an agent that has
none. Pydantic AI would otherwise infer one from the Python variable the agent was assigned to, and
an agent's config is not something a local rename should be able to move. The refusal happens where
the capability is wired up, not mid-run:

```python {test="skip"}
Agent('anthropic:claude-fable-5-1', capabilities=[AgentControl()])
# UserError: `AgentControl` without an explicit `name` reads the agent's `name`, and this agent has none.
```

The name is normalized the way Logfire normalizes an agent's name in your traces, and that is lossy:
`checkout-assistant`, `Checkout Assistant`, and `checkout_assistant` are one agent to Logfire,
including across services reporting to the same project, so two agents differing only in punctuation
share one config. Pass an explicit `name` to `AgentControl` to decouple it from the agent's name
entirely -- to keep two such agents apart, or to deliberately point several at one config.

`targeting_key`, `attributes`, and `render_template` work exactly as they do for
[`ManagedPrompt`](managed-prompt.md#targeting). `AgentControl.resolved` exposes the config the
current run picked up (`value`, `label`, `version`, `reason`), and is `None` outside a run.

## API reference

::: pydantic_ai_harness.logfire.AgentControl

The config a variable holds -- `AgentConfig` and the models it is built from -- is documented with the
contract itself, in [`logfire.agent_control`](https://github.com/pydantic/logfire/tree/main/logfire/agent_control).
