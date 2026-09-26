# Sprites Sandbox

Run your agent's commands and file edits in a persistent [Fly.io Sprite](https://sprites.dev) instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/sprites_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[sprites]"
```

pip:

```bash
pip install "pydantic-ai-harness[sprites]"
```

Then set `SPRITE_TOKEN` to your Sprites API token.

## Quick start

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
```

`Coder`'s shell and file tools now run in a Sprite, not on your machine. The Sprite is created the first time a tool uses it. It has no lifetime limit: it sleeps when idle and keeps its files, and costs money, until you delete it; see [Clean up](#clean-up).

A new Sprite comes with git, Python, and Node.js ([preinstalled tools](https://docs.fly.io/sprites/working-with-sprites/)). Ripgrep (`rg`) is not preinstalled. For faster `Coder` searches, run `sudo apt-get update && sudo apt-get install ripgrep` once in the Sprite and store its ref to reuse it on later runs. Sprites retain installed packages.

## Continue in the same sandbox

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')

followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the Sprite in the message history and works in it, so the clone is still there. Without the history, a run starts a new Sprite. If the Sprite has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one. A command exiting 137 after confirmed Sprite deletion also raises this error; a SIGKILLed command in a live Sprite returns exit 137.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool that runs in the Sprite:

```python {names="defined"}
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the Sprite."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.

When a command times out, plain `&` children of that command end with it. A Sprite pauses processes between commands unless you run them as a [Sprites service](https://docs.sprites.dev/working-with-sprites/services/).

## What a timeout stops

A command timeout starts after the Sprite is ready. The backend closes that command's exec connection and asks Sprites to stop it after one second; it does not delete the Sprite. The command's process group, especially children of a shell, is not yet guaranteed to have stopped. If stopping is uncertain, inspect the Sprite or delete it explicitly. `timeout=None` removes the command deadline, not the Sprite's idle pause or transport limits.

## Reattach later

To come back to the Sprite without the message history, pass its ref back as `workspace=`:

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref  # store this, e.g. in your database

later = agent.run_sync('Which capability would you add next, and where would it live?', workspace=ref)
```

The ref holds no credentials, so the process that reattaches needs `SPRITE_TOKEN` too. Pass `workspace='new'` to start a fresh Sprite even when the message history names one.

`runtime` only shapes a new Sprite, and an unknown one raises a clear error on first use; `working_dir` and `env` apply to every command, including after you reattach. A selected command directory is checked before execution; a missing directory raises `FileNotFoundError`.

Already have a `sprites.AsyncSprite`? Pass `workspace=SpritesSandboxBackend(workspace=sprite)` to a run, with `SpritesSandboxBackend` from `pydantic_ai_harness.sprites_sandbox`. `SpritesSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The Sprite keeps its files and installed packages after the run ends. Pydantic AI never deletes it. It sleeps when idle and wakes on the next command: you pay for compute while it is active and for storage until you delete it. Delete it with the ref you stored:

```python {names="defined"}
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox


async def delete_sprite(ref: WorkspaceRef) -> None:
    await SpritesSandbox().destroy(ref)
```

`destroy(ref)` deletes by Sprite id without attaching or waking it. Use `backend(ref)` to construct a lazy backend for an existing Sprite. See [Sprite lifecycle](https://docs.sprites.dev/concepts/lifecycle/).

A failed run returns no result, so there is no ref to store. To terminate its sandbox, clean up in an `on_run_error` hook; `after_run` doesn't run when a run fails. If creation's reply was lost, the ref may identify a Sprite that is not yet visible to the API; a lookup failure does not prove it was never created:

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

hooks = Hooks()


@hooks.on.run_error
async def terminate_failed_run(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
    if ctx.workspace.ref is not None:
        await delete_sprite(ctx.workspace.ref)
    raise error


agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(), Coder(), hooks])
```

## Configuration

| Option | What it does |
| --- | --- |
| `runtime` | Runtime for a new Sprite. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. A default Sprite runs as non-root `sprite` in `/home/sprite`; use relative paths or set `working_dir=` for portable code. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the Sprite. |
| `client` | A `sprites.AsyncSpritesClient` to share across runs on one event loop, or to set its base URL or timeout. You close it; `SpritesSandbox` never does. |

## Durable execution

If the exec socket drops after connection but before an exit status arrives, the command may have run. This raises a non-retryable workspace error rather than replaying a potentially non-idempotent command. Connection failures before the socket opens remain retryable.

Under [Temporal](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/) or another durable engine, create one client when the worker starts and pass it as `client=`. Otherwise every activity opens its own client and never closes it.

```python {names="defined"}
import os

from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox
from sprites import AsyncSpritesClient


async def run_worker() -> None:
    async with AsyncSpritesClient(token=os.environ['SPRITE_TOKEN']) as client:
        agent = Agent('anthropic:claude-opus-5-5', capabilities=[SpritesSandbox(client=client), Coder()])
        ...  # wrap `agent` for Temporal and start the worker
```

[`Coder`](../coder/#durable-execution), [`Shell`](../shell/#durable-execution), and [`FileSystem`](../filesystem/#durable-execution) work under DBOS, Temporal, and Prefect; each page's durable execution section lists what they can't do yet.

See [Workspaces: Durable execution](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution) for how workspaces work under durable engines.

## API reference

::: pydantic_ai_harness.sprites_sandbox.SpritesSandbox

::: pydantic_ai_harness.sprites_sandbox.SpritesSandboxBackend
