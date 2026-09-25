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

A sandbox lives for 24 hours, E2B's maximum, and is then paused; reattaching resumes it. To lower that, pass for example `E2BSandbox(sandbox_timeout=3600)`, the most E2B's Hobby plan allows.

`Coder` searches with ripgrep (`rg`) when the sandbox has it and with its built-in search otherwise. For faster searches, pass an E2B [template](https://e2b.dev/docs/template/quickstart) with ripgrep installed as `E2BSandbox(template='<name>')`.

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
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend


async def kill_sandbox(ref: WorkspaceRef) -> None:
    sandbox = await E2BSandboxBackend(ref=ref).get_client()
    await sandbox.kill()
```

`get_client()` returns the `e2b.AsyncSandbox`. A sandbox you don't kill is paused when its `sandbox_timeout` runs out. See [E2B's sandbox lifecycle](https://docs.e2b.dev/sandbox).

## Configuration

| Option | What it does |
| --- | --- |
| `template` | E2B template name or ID for a new sandbox. Default: E2B's `base`. An unknown template fails on first use. |
| `sandbox_timeout` | Seconds the sandbox lives before E2B pauses it, set on create and on reattach. Default: `86_400` (24 hours, E2B's maximum). |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: the sandbox's own. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the sandbox. |

## API reference

::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend
