---
title: Modal Sandbox
description: "Run a Pydantic AI agent's commands and file edits in an isolated Modal cloud sandbox instead of on your machine."
---

# Modal Sandbox

Run your agent's commands and file edits in an isolated [Modal](https://modal.com) cloud sandbox instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[modal]"
```

```bash
py-cli modal token new
```

This saves your Modal credentials; in CI, set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` instead.

## Quick start

```python
import modal
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

# Coder uses git and ripgrep, which the default image doesn't have.
image = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')
agent = Agent('anthropic:claude-sonnet-5', capabilities=[ModalSandbox(image=image), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
print(result.output)
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. The sandbox is created the first time a tool uses it, and it keeps running, and billing, after the run ends; see [Clean up](#clean-up).

Modal stops a sandbox 5 minutes after it's created by default, even if it's still in use. For longer work, pass `ModalSandbox(sandbox_timeout=3600)`.

## Continue in the same sandbox

```python
followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If the sandbox has been terminated or has expired, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool: inside a tool, `ctx.workspace` is the sandbox.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.modal_sandbox import ModalSandbox
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-sonnet-5', capabilities=[ModalSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for everything `ctx.workspace` can do.

Modal can't stop a command once it starts: cancelling a run stops waiting, but the command runs on until its timeout, or until the sandbox ends if it has none. `Shell` sets a 30-second timeout for you; in your own tools, pass `timeout=` to `ctx.workspace.run()`.

If only your own tools use the sandbox, pass `ModalSandbox(warn_if_no_tools=False)` to silence the missing-tools warning.

## Reattach later

To come back to the sandbox without the message history, save its id and pass it back as `workspace=`:

```python
import modal
from pydantic_ai import Agent
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

image = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')
agent = Agent('anthropic:claude-sonnet-5', capabilities=[ModalSandbox(image=image), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref
assert ref is not None  # None only if no tool used the sandbox
sandbox_id = ref.id  # save this, e.g. in your database

later = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    workspace=WorkspaceRef(provider='modal', id=sandbox_id),
)
print(later.output)
```

The id holds no credentials, so the process that reattaches needs your Modal credentials too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`image`, `app_name`, `create_app_if_missing`, `sandbox_timeout`, and `idle_timeout` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach.

Already have a `modal.Sandbox`? Pass `workspace=ModalSandboxBackend(workspace=sandbox)` to a run, with `ModalSandboxBackend` from `pydantic_ai_harness.modal_sandbox`. `ModalSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The sandbox keeps running, and billing, after the run ends. Pydantic AI never terminates it. Terminate it with the id you saved:

```python
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend


async def terminate_sandbox(sandbox_id: str) -> None:
    backend = ModalSandboxBackend(ref=WorkspaceRef(provider='modal', id=sandbox_id))
    sandbox = await backend.get_client()
    await sandbox.terminate.aio()
```

`get_client()` returns the `modal.Sandbox`. A sandbox you don't terminate ends when its `sandbox_timeout` runs out, or after `idle_timeout` seconds without activity. See [Modal's timeouts](https://modal.com/docs/guide/sandbox#timeouts).

## Configuration

| Option | What it does |
| --- | --- |
| `image` | Image for a new sandbox: a registry tag or a `modal.Image`. Default: `'python:3.12-slim'`. |
| `app_name` | Modal app a new sandbox belongs to. Default: `'pydantic-ai-harness'`. |
| `create_app_if_missing` | Create that app if it doesn't exist. Default: `True`. |
| `sandbox_timeout` | Seconds a new sandbox lives before Modal stops it. Default: Modal's, 5 minutes. |
| `idle_timeout` | Seconds without activity before Modal stops a new sandbox. Default: none. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: the image's. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the sandbox. |
| `warn_if_no_tools` | Warn when the agent has no `Shell` or `FileSystem` tool. Default: `True`. |

## Upgrading from the previous `ModalSandbox`

The previous `ModalSandbox` registered its own `run_command`, `read_file`, `write_file`, and `list_directory` tools and terminated its sandbox when the run ended. Now it only supplies the sandbox, so add `Coder()`, or `Shell()` and `FileSystem()`, as shown above. Old arguments and imports fail with an error that names the replacement.

### What changed in the lifecycle

- A run no longer terminates the sandbox. It runs until you terminate it or a timeout ends it (see [Clean up](#clean-up)).
- A run that continues a `message_history` reattaches to the previous run's sandbox. Pass `workspace='new'` for a fresh one.
- Reattaching to an expired or terminated sandbox raises `WorkspaceUnavailableError`. No empty replacement is created.

### Migration table

| Previous API | Now |
| --- | --- |
| `image`, `app_name`, `create_app_if_missing`, `env` | Unchanged. `image` also takes a `modal.Image`. |
| `sandbox_timeout` | Unchanged. When unset, Modal's default (5 minutes) applies. |
| `workdir` | Renamed `working_dir`. `workdir=` still works, with a deprecation warning. |
| `sandbox_id` | Removed. Use `agent.run(..., workspace=WorkspaceRef(provider='modal', id=sandbox_id))`. |
| `session`, `ModalSandboxSession` | Removed. Use `agent.run(..., workspace=ModalSandboxBackend(workspace=<modal.Sandbox>))`. |
| `default_command_timeout` | Removed. Use `Shell(default_timeout=...)`. |
| `max_command_timeout` | Removed. Use `sandbox_timeout`, which bounds every command. |
| `max_output_bytes`, `max_output_lines` | Removed. Use `Shell(max_output_chars=...)` or `ToolOutputLimits`. |
| `max_read_bytes` | Removed. Use `FileSystem(max_read_lines=..., max_read_chars=...)`. |
| `instructions` | Removed. Use the agent's `instructions`. |
| `run_command` tool | Removed. Use `Shell()`. |
| `read_file`, `write_file`, `list_directory` tools | Removed. Use `FileSystem()`. |
| `ModalSandboxExecResult` | Removed. Use `pydantic_ai.workspaces.CommandResult`. |
| `ModalSandboxError` | Removed. Catch `pydantic_ai.workspaces.WorkspaceError`. |
| `ModalSandboxTerminalError`, `ModalSandboxUnavailableError`, `ModalSandboxAuthError` | Removed. Catch `pydantic_ai.workspaces.WorkspaceUnavailableError`. |

## API reference

::: pydantic_ai_harness.modal_sandbox.ModalSandbox

::: pydantic_ai_harness.modal_sandbox.ModalSandboxBackend
