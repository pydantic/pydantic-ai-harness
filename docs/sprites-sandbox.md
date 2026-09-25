---
title: Sprites Sandbox
description: Give a Pydantic AI agent a persistent Fly.io Sprite as its workspace for commands and files.
---

# Sprites Sandbox

Give an agent a persistent [Fly.io Sprite](https://sprites.dev) to work in. `SpritesSandbox` supplies the Sprite as the run's `ctx.workspace`, so the commands and file edits of `Coder`, `Shell`, and `FileSystem` happen in the Sprite, not on your machine.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/sprites_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip/uv-add "pydantic-ai-harness[sprites,anthropic]"
```

Set `SPRITE_TOKEN` to your Sprites API token; the capability reads it to create its client, since the SDK does not. The integration uses sprites-py 0.7.x.

## A coding agent in a Sprite

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

# A new Sprite comes with git, Python, and Node.js installed.
agent = Agent('anthropic:claude-sonnet-5', capabilities=[SpritesSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
print(result.output)
```

Construction makes no Sprites requests. The Sprite is created at the first workspace operation, such as the agent's first tool call; a run that never uses its workspace creates none.

`Coder`'s `grep` and `list_files` tools use ripgrep (`rg`), which is not on the [list of tools a new Sprite comes with](https://docs.fly.io/sprites/working-with-sprites/). If it is missing, the tool asks the model to install it, and because the Sprite persists, that happens once per Sprite.

Commands run under `sh -c` in the Sprite's shell environment. Nothing from your machine's environment reaches the Sprite: pass `env=` for variables every command should get, such as a token the agent needs. `working_dir=` sets the absolute directory commands start in and relative paths resolve against; it defaults to the Sprite's.

## Continue in the same Sprite

```python
followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the Sprite's reference in the message history and reattaches to it, so the clone is still there. If that Sprite has been deleted, the first workspace operation raises `WorkspaceUnavailableError`; no empty replacement is created.

## Choose the tools

`Coder` bundles the tools a coding agent needs. For a narrower agent, pick `Shell` and `FileSystem` yourself; both act in the Sprite:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[SpritesSandbox(), Shell(), FileSystem(read_only=True)],
)
```

Your own tools and hooks can use `ctx.workspace` directly (`run()`, `read_text()`, `write_text()`, and the other filesystem operations, which run as commands in the Sprite).

## Reattach from a reference

Persist `result.workspace.ref` to come back to the Sprite outside message history. It is `None` if the run never used its workspace.

```python
from pydantic_ai import Agent
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[SpritesSandbox(), Coder()])


async def resume(ref: WorkspaceRef) -> str:
    result = await agent.run('Run the test suite again.', workspace=ref)
    return result.output
```

An explicit reference takes precedence over message history. Pass `workspace='new'` to `agent.run()` to start a fresh Sprite even when the history names one. `runtime` only applies to a new Sprite; `working_dir` and `env` apply to every command, in a reattached Sprite too.

## Manage the Sprite yourself

For Sprites SDK operations, hold a `SpritesSandboxBackend` and call `await backend.get_client()` for the typed `sprites.AsyncSprite`. The first call creates or attaches to the Sprite; later calls return the same object. `SpritesSandboxBackend(workspace=native)` wraps a Sprite handle you already have; pass either that or `ref=`, not both.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandboxBackend

agent = Agent('anthropic:claude-sonnet-5', capabilities=[Coder()])


async def run_and_clean_up(prompt: str) -> str:
    backend = SpritesSandboxBackend()
    sprite = await backend.get_client()
    try:
        result = await agent.run(prompt, workspace=backend)
        return result.output
    finally:
        await sprite.delete()
        await backend.aclose()
```

A backend creates its own `AsyncSpritesClient` from `SPRITE_TOKEN` on first use. `SpritesSandbox` closes the client of the backend it supplied when the run ends; a backend you construct is yours to close with `aclose()`. Either reopens a client if it is used again, for example through `result.workspace`. Pass `client=` to `SpritesSandbox` or the backend to share one client across runs, or to set its base URL or timeout; a client you pass is never closed for you, and must be used on the event loop it was created on.

To delete the Sprite an agent run created, take the backend from `result.workspace.backend`; check that its `ref` is not `None` first, since `get_client()` would otherwise create a Sprite.

## Lifetime and cost

A Sprite persists after the agent run ends, with its files and installed packages. It pauses when idle and resumes on the next command: compute is billed while it is active, storage until it is deleted. Pydantic AI never deletes a Sprite; deleting it through the SDK when you are done is the application's job.

- Pass a finite `timeout` to bound a command. On a timeout or a cancelled call, the command is sent SIGKILL through the Sprites API, and `WorkspaceTimeoutError` carries the output produced so far. Background processes the command started, such as a `server &` or a `Shell` background job, may outlive it.
- A deleted Sprite and rejected credentials raise `WorkspaceUnavailableError`, which ends the run. Network and connection errors propagate unchanged, so a durable execution engine can retry them.
- The `WorkspaceRef` carries no credentials, so every worker that reattaches needs its own `SPRITE_TOKEN` or `client=`. See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for how a run selects and restores its workspace.

The capability emits no telemetry spans of its own; core's agent and tool spans cover the calls made through tools.

## API reference

::: pydantic_ai_harness.sprites_sandbox.SpritesSandbox

::: pydantic_ai_harness.sprites_sandbox.SpritesSandboxBackend
