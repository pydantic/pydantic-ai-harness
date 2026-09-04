# Modal Sandbox

`ModalSandbox` supplies a Modal container as the run's `ctx.sandbox`. It owns
provider connection and lifecycle behavior. Add tools or capabilities that consume
`ctx.sandbox` for the model-facing interface you want.

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)

> [!NOTE]
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install and authenticate

```bash
uv add "pydantic-ai-harness[modal]"
modal token new
```

In deployed environments, set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

## Use with an agent

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ModalSandbox(image='python:3.12-slim')],
)


@agent.tool
async def run_command(ctx: RunContext[None], argv: list[str]) -> str:
    result = await ctx.sandbox.run(argv, timeout=60)
    return result.stdout
```

The example uses argv, which Modal executes directly. To expose shell syntax, call
`ctx.sandbox.run(command, shell=True, timeout=...)` from a tool whose schema makes
that behavior explicit.

## Lifecycle

An owned sandbox gets a deterministic name derived from the logical run ID. A retry
reconnects to the running sandbox with that name before creating one. Acquisition
detaches the initial handle and stores only the provider and sandbox ID in the
`SandboxRef`; later workers reconnect by ID. Release reconnects, terminates the
sandbox, and detaches. An already missing sandbox counts as released.

Attach to a sandbox managed elsewhere when the capability must not own its lifetime:

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

ModalSandbox(sandbox_id='sb-abc123')
```

Creation-only settings cannot be combined with `sandbox_id`. Attached sandboxes are
not terminated at run end, and concurrent runs share their filesystem and processes.
`sandbox_timeout` is Modal's server-side lifetime backstop. It also limits an orphan
if cancellation interrupts sandbox creation before the provider returns its ID.

## Direct backend use

`ModalSandboxBackend` implements Pydantic AI's `SandboxBackend` protocol and its
filesystem, process start, and streaming opt-ins:

```python
import anyio

from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend


async def main() -> None:
    backend = await ModalSandboxBackend.create(
        image='python:3.12-slim',
        sandbox_timeout=1800,
    )
    try:
        result = await backend.run(['python', '--version'], timeout=60)
        print(result.stdout)
    finally:
        await backend.close(terminate=True)


anyio.run(main)
```

Use `connect(sandbox_id)` or `connect_name(app_name, name)` for a running sandbox.
Neither method provisions a replacement.

## Limits and errors

Modal exposes no per-command kill operation, so `process.kill()` raises
`NotImplementedError`. Set `timeout=` when starting commands. Modal accepts whole
seconds, so fractional timeouts round up. Cancelling a wait does not kill the remote
command; it can continue until its command deadline or the sandbox lifetime ends.

Command output is returned in full. Tools that put output into model context should
apply their own byte or line limits.

- `ModalSandboxError` reports transient Modal operation failures and is a core
  `SandboxError`.
- `ModalSandboxAuthError` and `ModalSandboxUnavailableError` report dead or unauthorized
  Modal environments and are core `SandboxUnavailableError` instances, so the run does not retry them.
- Command deadlines raise core `SandboxTimeoutError`, including partial `stdout` and
  `stderr`.
- Missing filesystem paths raise `FileNotFoundError`.

## Configuration

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

ModalSandbox(
    image='python:3.12-slim',
    sandbox_id=None,
    app_name='pydantic-ai-harness',
    create_app_if_missing=True,
    sandbox_timeout=300,
    workdir=None,
    env=None,
)
```
