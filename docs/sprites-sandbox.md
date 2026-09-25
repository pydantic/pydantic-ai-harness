---
title: Sprites Sandbox
description: "Run a Pydantic AI agent's commands and file edits in a persistent Fly.io Sprite instead of on your machine."
---

# Sprites Sandbox

Run your agent's commands and file edits in a persistent [Fly.io Sprite](https://sprites.dev) instead of on your machine.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/sprites_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[sprites]"
```

Then set `SPRITE_TOKEN` to your Sprites API token.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[SpritesSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
print(result.output)
```

`Coder`'s shell and file tools now run in a Sprite, not on your machine. The Sprite is created the first time a tool uses it, and it keeps its files, and costs money, after the run ends; see [Clean up](#clean-up).

A new Sprite comes with git, Python, and Node.js, but not the ripgrep `Coder` searches with ([preinstalled tools](https://docs.fly.io/sprites/working-with-sprites/)); the model installs it on first use, once per Sprite.

## Continue in the same sandbox

```python
followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the Sprite in the message history and works in it, so the clone is still there. Without the history, a run starts a new Sprite. If the Sprite has been deleted, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](shell.md) and [`FileSystem`](filesystem.md) instead of `Coder`, or write your own tool: inside a tool, `ctx.workspace` is the Sprite.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[SpritesSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the Sprite."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for everything `ctx.workspace` can do.

When a command times out, processes it started in the background (such as `server &`) keep running.

## Reattach later

To come back to the Sprite without the message history, save its id (the Sprite's name) and pass it back as `workspace=`:

```python
from pydantic_ai import Agent
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[SpritesSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref
assert ref is not None  # None only if no tool used the Sprite
sprite_name = ref.id  # save this, e.g. in your database

later = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    workspace=WorkspaceRef(provider='sprites', id=sprite_name),
)
print(later.output)
```

The id holds no credentials, so the process that reattaches needs `SPRITE_TOKEN` too. Pass `workspace='new'` to start a fresh Sprite even when the message history names one.

`runtime` only shapes a new Sprite; `working_dir` and `env` apply to every command, including after you reattach.

Already have a `sprites.AsyncSprite`? Pass `workspace=SpritesSandboxBackend(workspace=sprite)` to a run, with `SpritesSandboxBackend` from `pydantic_ai_harness.sprites_sandbox`. `SpritesSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The Sprite keeps its files and installed packages after the run ends. Pydantic AI never deletes it. It pauses when idle and resumes on the next command: you pay for compute while it is active and for storage until you delete it. Delete it with the name you saved:

```python
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.sprites_sandbox import SpritesSandboxBackend


async def delete_sprite(sprite_name: str) -> None:
    backend = SpritesSandboxBackend(ref=WorkspaceRef(provider='sprites', id=sprite_name))
    sprite = await backend.get_client()
    await sprite.delete()
    await backend.aclose()
```

`get_client()` returns the `sprites.AsyncSprite`, and `aclose()` closes the Sprites client the backend opened. If a run fails before it returns, you never get the Sprite's name: find leftovers with `AsyncSpritesClient.list_sprites()` and delete them with `delete_sprite(name)`. See [Sprite lifecycle](https://docs.sprites.dev/concepts/lifecycle/).

## Configuration

| Option | What it does |
| --- | --- |
| `runtime` | Runtime for a new Sprite. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: the Sprite's own. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the Sprite. |
| `client` | A `sprites.AsyncSpritesClient` to share across runs on one event loop, or to set its base URL or timeout. You close it; `SpritesSandbox` never does. |

## Durable execution

Under [Temporal](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/) or another durable engine, create one client when the worker starts and pass it as `client=`. Otherwise every activity opens its own client and never closes it.

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox
from sprites import AsyncSpritesClient


async def run_worker() -> None:
    async with AsyncSpritesClient(token=os.environ['SPRITE_TOKEN']) as client:
        agent = Agent('anthropic:claude-sonnet-5', capabilities=[SpritesSandbox(client=client), Coder()])
        ...  # wrap `agent` for Temporal and start the worker
```

See [Workspaces: Durable execution](https://pydantic.dev/docs/ai/core-concepts/workspace/#durable-execution) for how workspaces work under durable engines.

## API reference

::: pydantic_ai_harness.sprites_sandbox.SpritesSandbox

::: pydantic_ai_harness.sprites_sandbox.SpritesSandboxBackend
