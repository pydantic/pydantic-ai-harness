---
title: Daytona Sandbox
description: Supply a Daytona environment through ctx.sandbox.
---

# Daytona Sandbox

`DaytonaSandbox` supplies a Daytona sandbox as the run's `ctx.sandbox`. It owns
only provider connection and lifecycle behavior. Add tools or capabilities that
consume `ctx.sandbox` for the model-facing interface you want.

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/daytona_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and authenticate

```bash
uv add "pydantic-ai-harness[daytona]"
export DAYTONA_API_KEY=...
```

## Use with an agent

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[DaytonaSandbox(snapshot='base')],
)


@agent.tool
async def run_command(ctx: RunContext[None], argv: list[str]) -> str:
    result = await ctx.sandbox.run(argv, timeout=60)
    return result.stdout
```

The example deliberately uses argv, which the backend safely quotes into the
shell command Daytona accepts. Use `shell=True` only when the tool deliberately
exposes shell syntax.

## Lifecycle

Asking the capability for a sandbox does no I/O. It hands back a backend holding settings
plus, when there is one, the identity of a sandbox that already exists; the first command or
file operation creates or attaches, once.

An owned sandbox gets a Daytona-safe name derived from the conversation, so a follow-up run
continues in the same workspace and a durable retry attaches to the sandbox the first attempt
made rather than provisioning a second one. If creation races, a failed create is followed by
one attach to the winner.

Nothing here deletes a sandbox. A conversation can span many runs, so the end of a run is not
the end of the workspace; Daytona stops an idle sandbox after `auto_stop_minutes` and deletes it
immediately after. If it has already done so, the next run gets a fresh, empty sandbox under the
same name and the old files are gone — raise `auto_stop_minutes` when a conversation needs to
outlive it, or delete the sandbox yourself with `DaytonaSandboxBackend.delete_by_id`, which
resolves it by ID without starting it first.

Attach to a sandbox managed elsewhere by ID or name when the capability must not
own its lifetime:

```python
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

DaytonaSandbox(sandbox_id='existing', workdir='/workspace')
```

Creation-only settings cannot be combined with `sandbox_id`. Concurrent runs on the same
sandbox share its filesystem and process space.

## Direct backend use

`DaytonaSandboxBackend` implements Pydantic AI's `SandboxBackend` protocol and its optional
filesystem. Building one does no I/O; the first operation creates the sandbox:

```python
import anyio

from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend


async def main() -> None:
    backend = DaytonaSandboxBackend(snapshot='base', auto_stop_minutes=60)
    try:
        result = await backend.run(['python', '--version'], timeout=60)
        print(result.stdout)
    finally:
        await backend.close(terminate=True)


anyio.run(main)
```

Pass `ref=SandboxRef(sandbox_id=...)` to attach to one specific sandbox, or `name=...` to attach
by name and create it only if there is none. A `ref` whose sandbox is gone raises rather than
quietly providing an empty replacement.

`backend.sandbox` is the live Daytona sandbox, for anything Daytona-specific. You can only await
it, so no code path can reach a sandbox that has not been created yet.

## Process and output behavior

Daytona process sessions provide separate stdout and stderr callbacks. The
backend preserves that separation and joins each stream once when the complete
result is requested. Log collection and the final exit-status RPC use one deadline
measured from `start()`. A command deadline raises
`pydantic_ai.sandboxes.SandboxTimeoutError` with the stdout and stderr collected
before expiry. Timeout or caller cancellation attempts to delete the remote
process session before returning. A failed best-effort deletion does not replace
the command's original outcome; the sandbox lifetime remains the cleanup backstop.

Complete command output is buffered in memory. The backend does not add a second
presentation policy or claim that transport is bounded. Model-facing tools should
apply their own byte or line budget, and commands that can produce very large
output should bound it at the source.

The public error surface is deliberately narrow:

- `DaytonaSandboxError` reports transient Daytona operation failures and is a core
  `pydantic_ai.sandboxes.SandboxError`.
- `DaytonaSandboxAuthError` and `DaytonaSandboxUnavailableError` report dead or unauthorized
  Daytona environments and are core `pydantic_ai.sandboxes.SandboxUnavailableError` instances.
- `pydantic_ai.sandboxes.SandboxTimeoutError` for command deadlines, with the partial
  `stdout`, `stderr`, and enforced `timeout`.

Sandbox creation, connection, and process-session setup timeouts are provider
operation failures, not command deadlines, and raise `DaytonaSandboxError`.

Filesystem misses use the built-in `FileNotFoundError` contract.

## Configuration

```python
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

DaytonaSandbox(
    sandbox_id=None,
    snapshot=None,
    auto_stop_minutes=60,
    workdir=None,
    env=None,
    network_block_all=False,
)
```
