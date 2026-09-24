# Daytona Sandbox

Run an agent's commands and file operations in a [Daytona](https://www.daytona.io) sandbox. `DaytonaSandbox` supplies the run's `ctx.workspace`; the tools you add, such as `Coder`, `Shell`, and `FileSystem`, act in it.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/daytona_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[daytona]"
```

pip:

```bash
pip install "pydantic-ai-harness[daytona]"
```

Set `DAYTONA_API_KEY` for authentication. The integration uses Daytona SDK 0.198.x.

## A coding agent in a sandbox

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[DaytonaSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
print(result.output)
```

`Coder`'s file and shell tools run in the sandbox, not on your machine. Daytona's default snapshot includes Python and the `git` and ripgrep (`rg`) that `Coder` needs; pass `snapshot=` to start from your own, with both installed.

Constructing the capability makes no Daytona requests. The first workspace operation of a run creates the sandbox and records its reference; a run that never touches its workspace creates nothing.

## Continue in the same sandbox

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[DaytonaSandbox(), Coder()])
first = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai.')
second = agent.run_sync('Now run its unit tests for capabilities.', message_history=first.all_messages())
```

The second run finds the sandbox's reference in the message history and attaches to it, starting it if Daytona stopped it in the meantime. A run with no history or reference creates a fresh sandbox. If the referenced sandbox was deleted, the run fails with `WorkspaceUnavailableError` rather than continuing in an empty replacement.

## Choose the tools

`Coder` bundles file tools, a shell, and context management. For a narrower agent, add `Shell` and `FileSystem` on their own, or write tools against `ctx.workspace`:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-sonnet-5', capabilities=[DaytonaSandbox(), Shell(), FileSystem()])

@agent.tool
async def run_python(ctx: RunContext[None], code: str) -> str:
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

Shell commands run under `/bin/sh -c` in the sandbox's shell environment, and every command starts in `working_dir` (the sandbox's default directory when unset); relative paths resolve against it. `env=` sets variables every command gets, and a command's own `env` is layered on top. Nothing is read from the agent process's environment.

## Reattach from a reference

Persist `result.workspace.ref` to come back to a sandbox outside message history, and pass it as `workspace=`:

```python
from pydantic_ai import Agent
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[DaytonaSandbox(), Coder()])

def resume(ref: WorkspaceRef) -> str:
    return agent.run_sync('Where did we leave off?', workspace=ref).output
```

`ref` is `None` until the sandbox has been created. An explicit reference takes precedence over the one in message history. To start over in a fresh sandbox while keeping the conversation, pass `workspace='new'`:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[DaytonaSandbox(), Coder()])

def start_over(history: list[ModelMessage]) -> str:
    return agent.run_sync('Try again from a clean checkout.', message_history=history, workspace='new').output
```

## Manage the sandbox

`DaytonaSandbox` does not stop or delete sandboxes. Use the typed `daytona.AsyncSandbox` from `get_client()` for that, or for any other SDK operation:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

agent = Agent('anthropic:claude-sonnet-5', capabilities=[DaytonaSandbox(), Coder()])

async def run_and_delete(prompt: str) -> str:
    result = await agent.run(prompt)
    backend = result.workspace.backend
    if isinstance(backend, DaytonaSandboxBackend) and backend.ref is not None:
        sandbox = await backend.get_client()
        await sandbox.delete()
    return result.output
```

A `DaytonaSandboxBackend` can also be built yourself and passed as `workspace=`: from a `ref=`, or around a native handle you already have with `DaytonaSandboxBackend(workspace=native)`.

Without `client=`, each run opens its own `AsyncDaytona` API client on first use and closes it when the run ends; `result.workspace` opens a new one if you use it afterwards. Pass `client=` to share one client across runs; you own closing it, for example with `async with AsyncDaytona() as client:`. The reference carries no credentials, so every process that reattaches needs its own `DAYTONA_API_KEY` or `client=`.

## Lifetime and cost

A sandbox keeps running, and billing, after the run ends. Daytona stops it after `auto_stop_interval` idle minutes (default 60; `0` disables it), archives a stopped sandbox after `auto_archive_interval` minutes (Daytona's default, 7 days, when unset), and deletes a stopped sandbox after `auto_delete_interval` minutes (default `-1`, never). A stopped sandbox keeps its disk, and attaching to it starts it again. Deleting sandboxes you are done with is the application's job, through `get_client()` or the Daytona dashboard. These settings, `snapshot`, and `network_block_all` apply when a sandbox is created, not to one attached by reference.

A command's `timeout` is enforced client-side; when it expires or the caller is cancelled, the backend deletes the command's session, which kills the command. Cancelling a run while the sandbox is being created can leave a sandbox the application never received a reference for; `auto_stop_interval` and `auto_delete_interval` bound what it costs.

The capability emits no telemetry spans of its own; core agent and tool spans cover the calls made through tools.

## API reference

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandbox

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend
