# Daytona Sandbox

`DaytonaSandbox` supplies a Daytona sandbox as the run's `ctx.sandbox`. It owns
only provider connection and lifecycle behavior. Add tools or capabilities that
consume `ctx.sandbox` for the model-facing interface you want.

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/daytona_sandbox/)

> [!NOTE]
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

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

Runs leave the sandbox available. A conversation can span many runs, so the end of a run is not
the end of the workspace; Daytona stops an idle sandbox after `auto_stop_minutes` and keeps its
disk until an explicit `destroy()` or another storage policy removes it.

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
        await backend.destroy()


anyio.run(main)
```

Import `SandboxRef` from `pydantic_ai.sandboxes` and pass `ref=SandboxRef(sandbox_id=...)` to attach to one specific sandbox, or `name=...` to attach
by name and create it only if there is none. A `ref` whose sandbox is gone raises rather than
quietly providing an empty replacement.

`await backend.sandbox` creates or attaches and returns the native Daytona sandbox
for provider-specific operations. `await backend.destroy()` deletes the referenced sandbox
without starting it, including one attached from elsewhere. `await backend.disconnect()` closes
the SDK client while leaving the remote sandbox unchanged. Both methods require callers to finish
in-flight operations first. `pause()` is available only for Daytona sandbox classes that support
VM pause, and `stop()` rejects sandboxes with `auto_delete_interval=0` because stopping deletes
their disk. Stopping preserves disk for the persistent sandboxes this backend creates.

The property remains awaitable after the native handle is cached. Await it before accessing SDK
methods; type checkers reject using the awaitable as the native sandbox.

`DaytonaSandboxBackend` inherits this behavior from `LazySandbox`. Its `create_or_attach()`
hook contains the Daytona-specific acquisition and identity handling; the shared helper owns
coordination and caching. See [writing sandbox backends](https://ai.pydantic.dev/sandbox/#supply-a-sandbox-from-a-capability) for the authoring contract.

## Process and output behavior

Daytona process sessions provide separate stdout and stderr callbacks. The
backend preserves that separation and joins each stream once when the complete
result is requested. Log collection and the final exit-status RPC use one deadline
measured from `run()`, including acquisition and session setup. A command deadline raises
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

Provider request timeouts raise `DaytonaSandboxError`. Expiry of the explicit
`run(timeout=...)` deadline raises `SandboxTimeoutError`, including during setup.

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
