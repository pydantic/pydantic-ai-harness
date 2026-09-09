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
uv add "pydantic-ai-harness[logfire]"
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

Write the agent exactly as you would anyway, give it a `name`, and add the capability:

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
| **Tools** | Rename a tool and reword its description and its parameters', for the model's eyes only |

Anything you don't change in Logfire keeps doing what the code says, and removing a change there puts
that piece back the way the code has it.

A tool's implementation and the shape of its arguments stay code-owned: Logfire edits what the model
is *told* about a tool, never what it *does*. A renamed tool still routes to the same function, and
`ctx.tool_name` is still the original name, so nothing in code has to know about the rename.

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
computation. If you want that text editable, write it as text rather than as a function.

**A block with nothing to identify it isn't offered at all.** That's text from a toolset or capability
with no `id` of its own, an unnamed callable in `Agent(instructions=...)`, or anything passed to
`run(instructions=...)`. Give the toolset or capability an `id` and its text becomes editable.

To make one block of a larger prompt separately editable, give that part a name:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.messages import InstructionPart

from pydantic_ai_harness.logfire import AgentControl

agent = Agent(
    'openai:gpt-5',
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

Changes are picked up **once per run**. Publishing mid-run takes effect on the next run, which is what
lets every span of a run agree on the version that produced it.

## What Logfire compares against

The editor shows published values as changes *to something*, and that something is a snapshot of what
your agent does in code -- its prompt block by block, its model, its settings, its tool definitions.
`AgentControl` sends that snapshot in the background, on a run that reaches the model, and only when
it has changed. It is documentation: never resolved, never applied to a run, so a stale or failed
snapshot cannot change what your agent does.

Because it comes from one request, a prompt or toolset that varies with `deps`, the run's input, or
the step within a run is captured as it was at that moment.

Inside a Temporal, DBOS, or other Pydantic AI durable workflow, published config still applies, but
the first-run registration and the snapshot are skipped with a warning: both write from background
threads, which is not replay-safe. Set the agent up by running it outside the workflow once. Pass
`auto_create=False` or `publish_baseline=False` to turn either off anywhere else -- for instance when
the process deliberately holds a read-only token.

## Which agent in Logfire it controls

By default, the one matching your agent's `name`, normalized the way Logfire normalizes an agent's
name in your traces. That normalization is lossy -- `checkout-assistant`, `Checkout Assistant`, and
`checkout_assistant` are one agent to Logfire, including across services reporting to the same
project -- so two agents differing only in punctuation share one config. Pass an explicit `name` to
`AgentControl` to keep them apart, or to deliberately point several agents at one config.

`targeting_key`, `attributes`, and `render_template` work exactly as they do for
[`ManagedPrompt`](managed-prompt.md#targeting). `AgentControl.resolved` exposes the config the
current run picked up (`value`, `label`, `version`, `reason`), and is `None` outside a run.

## API reference

::: pydantic_ai_harness.logfire.AgentControl

::: pydantic_ai_harness.logfire.AgentConfig

::: pydantic_ai_harness.logfire.InstructionBlock

::: pydantic_ai_harness.logfire.ToolDefinitionOverride

::: pydantic_ai_harness.logfire.AgentConfigSettings
