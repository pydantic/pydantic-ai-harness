---
title: E2B Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated E2B cloud sandbox instead of on your machine."
---

# E2B Sandbox

Run your agent's commands and file edits in an isolated [E2B](https://e2b.dev) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/e2b_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[e2b]"
```

Then set `E2B_API_KEY` to your E2B API key.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. The sandbox is created the first time a tool uses it, and it keeps running, and billing, after the run ends; see [Clean up](#clean-up).

A sandbox lives for 1 hour by default. When that runs out it pauses, and the next run resumes it. On E2B's Pro plan you can pass up to `E2BSandbox(sandbox_timeout=86400)`.

E2B's default template does not include ripgrep (`rg`). `Coder` can search without it, but installing `rg` makes searches faster. Build a reusable [template](https://e2b.dev/docs/template/quickstart) once, outside an agent run (building can take a minute):

```text
from e2b import AsyncTemplate, Template
await AsyncTemplate.build(Template().from_base_image().apt_install(['ripgrep']), 'my-rg-template')
agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(template='my-rg-template'), Coder()])
```

The build snippet is illustrative and not part of the runnable agent examples below.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. A paused sandbox resumes. If the sandbox has been killed, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.e2b_sandbox import E2BSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, store the run's workspace ref and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `E2B_API_KEY` too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`template` and `allow_internet_access` only shape a new sandbox. `sandbox_timeout` also applies when you reattach, and `working_dir` and `env` apply to every command.

Already have an `e2b.AsyncSandbox`? Pass `workspace=E2BSandboxBackend(workspace=sandbox)` to a run, with `E2BSandboxBackend` from `pydantic_ai_harness.e2b_sandbox`. `E2BSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The sandbox keeps running, and billing, after the run ends. Pydantic AI never kills it. Kill it with the ref you stored:

```python {names="defined"}
import e2b
from pydantic_ai.workspaces import WorkspaceRef


async def kill_sandbox(ref: WorkspaceRef) -> None:
    await e2b.AsyncSandbox.kill(ref.id)
```

This kills a paused sandbox too, without resuming it. A sandbox you don't kill is paused when its `sandbox_timeout` runs out. See [E2B's sandbox lifecycle](https://docs.e2b.dev/sandbox).

A failed run returns no result, so there is no ref to store. To terminate its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await kill_sandbox(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[E2BSandbox(), Coder(), hooks])
```

## Configuration

| Option | What it does |
| --- | --- |
| `template` | E2B template name or ID for a new sandbox. Default: E2B's `base`. An unknown template fails on first use. |
| `sandbox_timeout` | Seconds the sandbox lives before E2B pauses it, set on create and on reattach. Default: `3_600` (1 hour, the most E2B's Hobby plan allows). |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: the sandbox's own. |
| `env` | Environment variables every command gets. Commands default to `LC_ALL=C.UTF-8` (override it with `env`); images without that locale fall back to the C locale. Nothing from your machine's environment reaches the sandbox. |

## API reference

::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend
