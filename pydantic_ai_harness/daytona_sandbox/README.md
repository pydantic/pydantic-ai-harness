# Daytona Sandbox

Run your agent's commands and file edits in an isolated [Daytona](https://www.daytona.io) cloud sandbox instead of on your machine.

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

Then set `DAYTONA_API_KEY` to an API key from your Daytona dashboard.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. The sandbox is created the first time a tool uses it, and it keeps running, and billing, after the run ends; see [Clean up](#clean-up).

Daytona's default snapshot includes Python and `git`. `Coder` searches with ripgrep (`rg`) when the sandbox has it and with its built-in search otherwise; installing it in a snapshot of your own (`snapshot=`) makes searches faster. A snapshot Daytona doesn't know fails on first use with a clear error.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If Daytona stopped the sandbox while it was idle, attaching in a new run starts it again. A held backend also retries command setup once after an idle stop. If the sandbox has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

## Reattach later

To come back to the sandbox without the message history, keep the run's workspace ref (in your database, say) and pass it back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `DAYTONA_API_KEY` too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`snapshot`, `auto_stop_interval`, and `network_block_all` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach.

Already have a `daytona.AsyncSandbox`? Pass `workspace=DaytonaSandboxBackend(workspace=sandbox)` to a run, with `DaytonaSandboxBackend` from `pydantic_ai_harness.daytona_sandbox`. `DaytonaSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The sandbox keeps running, and billing, after the run ends. Pydantic AI never stops or deletes it. Sandboxes created by this backend carry the `created-by=pydantic-ai` label; filter by that label when auditing your Daytona account. Delete one with the ref you kept:

```python {names="defined"}
from daytona import AsyncDaytona
from pydantic_ai.workspaces import WorkspaceRef


async def delete_sandbox(ref: WorkspaceRef) -> None:
    async with AsyncDaytona() as client:
        await (await client.get(ref.id)).delete()
```

Use the SDK directly to delete the stored ref without first starting a stopped sandbox.

By default, Daytona stops a sandbox after 15 idle minutes (a later run starts it again); set `auto_stop_interval=` to a smaller number of minutes to stop it sooner. A stopped sandbox keeps its disk, is archived after 7 days, and is never deleted. See [Daytona's SDK docs](https://www.daytona.io/docs/en/python-sdk/async/async-daytona/).

A failed run returns no result, so there is no ref to store. To terminate its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await delete_sandbox(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(), Coder(), hooks])
```

## What a timeout stops

A `run(timeout=...)` deadline starts after sandbox acquisition. Daytona deletes the command session to stop its foreground work, leaving the sandbox available for other commands; session deletion can take around 10 seconds. `WorkspaceTimeoutError` carries partial stdout and stderr. Cancellation also attempts to stop the session. If Daytona cannot confirm deletion, the command may still be running: inspect or delete the sandbox yourself. `timeout=None` removes the command deadline, not Daytona's idle auto-stop or transport limits.

## Configuration

| Option | What it does |
| --- | --- |
| `snapshot` | Daytona snapshot a new sandbox starts from. Default: Daytona's. |
| `auto_stop_interval` | Idle minutes before Daytona stops a new sandbox; `0` disables it. Default: Daytona's, 15 minutes. |
| `network_block_all` | Block DNS and non-allowlisted outbound traffic from a new sandbox. Daytona still allows package registries, GitHub, and AI APIs, so this is not an exfiltration boundary. See [Daytona's network limits](https://www.daytona.io/docs/en/network-limits/). Default: `False`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: `/home/daytona` for the non-root `daytona` user in Daytona's default image. Prefer relative paths or set `working_dir=` for portable code. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the sandbox. |
| `client` | An `AsyncDaytona` client to share across runs. You close it; `DaytonaSandbox` never does. |

## Durable execution

Under [Temporal](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/) or another durable engine, create one client when the worker starts and pass it as `client=`. Otherwise every activity opens its own client and never closes it.

```python {names="defined"}
from daytona import AsyncDaytona
from pydantic_ai import Agent
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox


async def run_worker() -> None:
    async with AsyncDaytona() as client:
        agent = Agent('anthropic:claude-opus-5-5', capabilities=[DaytonaSandbox(client=client)])
        ...  # wrap `agent` for Temporal and start the worker
```

`Coder`, `Shell`, and `FileSystem` work under DBOS, but not yet under Temporal or Prefect.

See [Workspaces: Durable execution](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution) for how workspaces work under durable engines.

## API reference

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandbox

::: pydantic_ai_harness.daytona_sandbox.DaytonaSandboxBackend
