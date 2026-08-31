---
title: Absurd Durability
description: Make a Pydantic AI agent run durable on Absurd, a Postgres-based durable-execution engine -- model requests, MCP calls, and function tool calls are checkpointed into steps so a crashed worker resumes mid-run.
---

# Absurd Durability

`AbsurdDurability` makes an agent run durable on [Absurd](https://github.com/earendil-works/absurd), a
Postgres-based durable-execution engine by Armin Ronacher (Python SDK `absurd-sdk`). Attach the
capability and call `agent.run()` inside an Absurd task handler: every model request, MCP call, and
function tool call is checkpointed into an Absurd step (`ctx.step(...)`), so if a worker crashes
part-way through a run it resumes from the last completed step instead of restarting. A completed
step is served from its checkpoint on replay instead of being recomputed, so tokens are not re-spent
on work that already finished. A step is checkpointed after it runs, so a crash between a tool's side
effect and its checkpoint re-runs the tool on recovery: keep tool side effects idempotent. Outside a
task the capability is transparent and the run is a normal, non-durable agent run.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/absurd/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
uv add "pydantic-ai-harness[absurd]"
```

Absurd stores its state in Postgres. Once per database, install the Absurd schema and create a
queue. The schema SQL and the queue helpers ship with the upstream project; see the
[Absurd repository](https://github.com/earendil-works/absurd) for the schema file and setup steps.

```python {test="skip"}
from absurd_sdk import AsyncAbsurd

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
await absurd.create_queue()
```

## Quick start

Construct the agent with the capability, register a task handler that runs it, then split the work
across a producer that spawns tasks and a worker that executes them. The agent needs a `name`; it
prefixes every checkpoint step.

```python {test="skip"}
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import Agent
from pydantic_ai_harness.absurd import AbsurdDurability

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
agent = Agent('openai:gpt-5', name='analyst', capabilities=[AbsurdDurability()])


@absurd.register_task(name='analyse')
async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
    assert isinstance(params, dict)
    result = await agent.run(params['prompt'])
    return {'output': result.output}


# Producer: enqueue a task.
await absurd.spawn('analyse', {'prompt': 'Summarize the Q3 report.'})

# Worker: claim and run tasks (in its own process). `start_worker` polls continuously.
await absurd.start_worker()
```

The task handler runs inside an `AsyncTaskContext`, which is how the capability knows to checkpoint.
Call `agent.run()` (async) from an async handler; a synchronous `TaskContext` raises a `UserError`
because an agent run cannot be awaited from one.

## What gets checkpointed, and what replay means

Each of these is wrapped in its own `ctx.step(...)`, so once it completes its result is checkpointed
and served from the checkpoint on replay rather than being recomputed:

- **model requests** -- `{name}__model.request`, `.request_stream`, and `.cancel_suspended_response`
  (the request's model id is appended as a `.{model_id}` suffix when it differs from the agent's
  default);
- **function tool calls** -- `{name}__function_toolset__{id}.call_tool:{tool_name}`;
- **MCP I/O** -- `{name}__mcp_server__{id}.get_tools`, `.get_instructions`, and `.call_tool`;
- **dynamic toolset resolution and calls** -- `{name}__dynamic_toolset__{id}.get_tools` and
  `.call_tool:{tool_name}`. A `DynamicToolset` built at construction time is re-resolved inside the
  step (its factory runs there, not in task code), so its resolution and inner tool calls are
  checkpointed and not re-run on recovery;
- **event-stream handler calls** -- `{name}__event_stream_handler`, when an `event_stream_handler`
  is set.

Replay means: after a crash, Absurd re-runs the task handler from the top. Plain Python in the
handler body runs again, but each checkpointed step returns its stored result instead of re-issuing
the model request or re-calling the tool. A `ModelRetry`, `ApprovalRequired`, or `CallDeferred`
raised by a tool crosses the checkpoint as a serialized value, so on replay the same outcome is
reproduced without re-running the tool.

Calling `agent.run()` more than once in a single task handler works: a step name that recurs (a
second run's model request, or the same tool called twice in one response) is disambiguated by
Absurd's encounter-order counter (`{name}#2`, `{name}#3`, ...), so each occurrence keeps its own
checkpoint and lines up on replay.

## Constraints

- The agent needs a `name` (or pass `name=` to `AbsurdDurability`); it prefixes every step.
- Leaf toolsets that execute their own tools (function toolsets, MCP servers) need a unique `id`,
  which identifies their steps within the task. A `DynamicToolset` also needs an `id`; it is
  supported when built at construction time (its factory and inner calls are checkpointed), but
  cannot be added per-run via `run(toolsets=...)`.
- The agent's `name` and a toolset's `id` are part of every step name, so they should not be
  changed once the durable agent has been deployed to production: a rename orphans the checkpoints
  of in-flight tasks, which resume under the old names and re-run their steps from the start.
- A model id, including a `models=` key or an id passed to `run(model=...)`, is folded into the
  model step name, so it must not contain `#`, the character Absurd uses to disambiguate repeated
  step names. A `#` in a model id is rejected with a `UserError`.
- A checkpointed tool's return value is stored in Postgres as JSON, so it must be JSON-serializable.
- Concurrent runs with the same capability name in one Absurd task context are rejected because
  encounter-order step names would let the runs claim each other's checkpoints. Await one run
  before starting another, or give each run a distinct capability `name` or task context.
- The executing toolsets are fixed when the agent is constructed. Passing an executing toolset
  per-run via `run(toolsets=...)` inside a task raises a `UserError`, because a runtime toolset has
  no registered steps and would re-run its side effects on recovery. Non-executing toolsets such as
  `ExternalToolset` are allowed at runtime.
- Streaming inside a task is a replay, not a live wire: the model stream is consumed and captured
  inside the step, and the run-side stream replays the captured events.
- An `event_stream_handler` runs live inside the model-request step, and its call is itself
  checkpointed. The handler may run more than once if the run recovers before that step is
  checkpointed, so keep its side effects idempotent.
- Do not use `run_sync` inside a task handler. The handler is async; use `await agent.run(...)`.

## Parallel execution

`parallel_execution_mode` defaults to `'sequential'`. Set it to `'parallel_ordered_events'` to run
tool calls concurrently while emitting their result events in model-call order. Plain `'parallel'`
is excluded because completion-order event delivery can assign repeated event-handler step names
to different calls on replay. Outside an Absurd task, the agent's configured execution mode is left
unchanged.

## Per-tool opt-out

A function tool can opt out of checkpointing with `metadata={'absurd': False}`, which runs it inline
inside the task rather than in a step. Reach for this for a cheap, side-effect-free tool where a
checkpoint would add more overhead than it saves.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability

tools = FunctionToolset(id='math')


@tools.tool_plain(metadata={'absurd': False})
def add(a: int, b: int) -> int:
    return a + b


agent = Agent('openai:gpt-5', name='calc', toolsets=[tools], capabilities=[AbsurdDurability()])
```

`False` is the only supported value for the `absurd` metadata key. A step takes no per-tool options,
so a mapping (`metadata={'absurd': {...}}`) has nothing to apply and raises a `UserError` rather than
being dropped. The opt-out and this rule apply to a tool from a dynamic toolset as well: `False` runs
it inline (its listing stays checkpointed), and a mapping raises.

MCP tools cannot opt out: they perform I/O and so are always checkpointed. Setting
`metadata={'absurd': False}` on an MCP tool raises a `UserError`.

## CodeMode composition

`AbsurdDurability` composes with the harness [Code Mode](code-mode.md) capability, and checkpointing
reaches inside the sandbox. `CodeMode` dispatches every tool call the generated code makes through
its wrapped toolset, over a `ToolManager` built on that toolset. Because `AbsurdDurability` registers
as the innermost capability, its leaf-toolset swap runs first, so the toolset `CodeMode` wraps
already holds the durable wrappers. A `search(...)` call made from inside `run_code` therefore runs
in its own `{name}__function_toolset__{id}.call_tool:search` step and is served from the checkpoint
on replay, the same as a direct tool call.

The `run_code` tool call itself is not a checkpointed step (the code-execution boundary is a wrapper
toolset, not a durable leaf), so its Python body re-runs on replay. Each checkpointed step that body
reaches, including the inner tool call, short-circuits to its stored result, so the tool function
does not run a second time.

## Checkpoint format compatibility

The step names and the checkpoint payload shapes are byte-compatible with the `pydantic-ai-absurd`
package (the standalone Absurd integration by Marcelo Trylesinski), so a run started under one
package can resume under the other. Treat the step names and payload shapes as a stable persistence
format.

## Relation to Step Persistence

`AbsurdDurability` and the [Step Persistence](step-persistence.md) capability solve different
problems and compose. Absurd gives crash-resume *within* a single run: a worker that dies mid-run
picks up from the last completed step (steps are at-least-once, so keep side effects idempotent).
Step Persistence records step events and
continuation snapshots *across* runs, so a run can be resumed, forked, or replayed as a separate
invocation later. Use Absurd for durability against crashes during a run, and Step Persistence to
persist and resume runs as first-class records.

## Further reading

- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Absurd](https://github.com/earendil-works/absurd)

## API reference

::: pydantic_ai_harness.absurd.AbsurdDurability
