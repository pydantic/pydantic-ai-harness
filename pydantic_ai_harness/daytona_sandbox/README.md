# Daytona Sandbox

`DaytonaSandbox` supplies a Daytona sandbox as the run's `ctx.sandbox`. It owns
only provider connection and lifecycle behavior. Add tools or capabilities that
consume `ctx.sandbox` for the model-facing interface you want.

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/daytona_sandbox/)

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

Owned acquisition derives a Daytona-safe name from the logical run ID. A durable
retry first reconnects by that name. If creation races, a failed create is
followed by one reconnect to the winner. The serialized `SandboxRef` contains the
provider and sandbox ID; later workers reconnect by ID and never create from
`get_sandbox`. Acquisition closes its SDK client after recording the ref, and
release opens a fresh client, resolves the sandbox by ID without starting it,
deletes it, and closes the client again.

An already missing sandbox counts as successfully released. Unexpected delete or
client-close failures are surfaced. Owned sandboxes use `auto_stop_minutes`
together with Daytona's immediate delete-after-stop setting as the server-side
backstop for cancellation during creation and other abandoned lifecycles.

Attach to a sandbox managed elsewhere by ID or name when the capability must not
own its lifetime:

```python
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

DaytonaSandbox(sandbox_id='existing', workdir='/workspace')
```

Creation-only settings cannot be combined with `sandbox_id`. Attached sandboxes
are not deleted at run end, and concurrent runs share their filesystem and
process space.

## Direct backend use

`DaytonaSandboxBackend` implements Pydantic AI's `SandboxBackend` protocol and
its filesystem and process-start opt-ins:

```python
import anyio

from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend


async def main() -> None:
    backend = await DaytonaSandboxBackend.create(
        snapshot='base',
        auto_stop_minutes=60,
    )
    try:
        result = await backend.run(['python', '--version'], timeout=60)
        print(result.stdout)
    finally:
        await backend.close(terminate=True)


anyio.run(main)
```

Use `connect(sandbox_id_or_name)` for a fresh handle to an existing sandbox; it
never provisions a replacement.

## Process and output behavior

Daytona process sessions provide separate stdout and stderr callbacks. The
backend preserves that separation and joins each stream once when the complete
result is requested. Log collection and the final exit-status RPC use one deadline
measured from `start()`. A command deadline raises
`pydantic_ai.sandboxes.SandboxTimeoutError` with the stdout and stderr collected
before expiry. Timeout or caller cancellation attempts to delete the remote
process session before returning. A failed best-effort deletion does not replace
the command's original outcome; the sandbox lifetime remains the cleanup backstop.

Complete command output is buffered in memory. Model-facing tools should apply their
own byte or line budget, and commands that can produce very large output should bound
it at the source.

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
