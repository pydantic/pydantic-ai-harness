---
title: Modal Sandbox
description: Run commands and access files in a Modal sandbox.
---

# Modal Sandbox

`ModalSandbox` supplies a Modal container as the run's `ctx.sandbox`. It owns
provider connection and lifecycle behavior. Add tools or capabilities that consume
`ctx.sandbox` for the model-facing interface you want.

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

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

Asking the capability for a sandbox does no I/O. It hands back a backend holding settings
plus, when there is one, the identity of a sandbox that already exists; the first command or
file operation creates or attaches, once.

An owned sandbox gets a deterministic name derived from the conversation, so a follow-up run
continues in the same workspace and a durable retry attaches to the sandbox the first attempt
made rather than provisioning a second one.

Nothing here terminates a sandbox. A conversation can span many runs, so the end of a run is
not the end of the workspace; Modal reaps a sandbox at `sandbox_timeout`. If Modal has
already reaped a conversation's sandbox, the next run gets a fresh, empty one under the same
name and the old files are gone -- raise `sandbox_timeout` when a conversation needs to outlive
it, or terminate it explicitly with `await backend.destroy()`.

`await backend.disconnect()` releases only the local connection and does not terminate the remote
sandbox. The backend retains its `ref`, so its next operation can attach to the same sandbox if
it is still available.
Both methods wait for acquisition to finish, but they do not wait for commands already running;
finish those commands before disconnecting or destroying the sandbox.

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

`ModalSandboxBackend` implements Pydantic AI's `SandboxBackend` protocol and its optional
filesystem. Building one does no I/O; the first operation creates the sandbox:

```python
import anyio

from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend


async def main() -> None:
    backend = ModalSandboxBackend(image='python:3.12-slim', sandbox_timeout=1800)
    try:
        result = await backend.run(['python', '--version'], timeout=60)
        print(result.stdout)
    finally:
        await backend.destroy()


anyio.run(main)
```

Pass `ref=SandboxRef(sandbox_id=...)` to attach to one specific sandbox, or `name=...` to
attach to a running sandbox with that name and create it only if there is none. A `ref` whose
sandbox is gone raises rather than quietly providing an empty replacement.

Use `await backend.sandbox` to access the live `modal.Sandbox` for provider-specific operations.
`sandbox_timeout` is a maximum lifetime, not an idle timeout.

The property remains awaitable after the native handle is cached. Await it before accessing SDK
methods; type checkers reject using the awaitable as the native sandbox.

`ModalSandboxBackend` inherits this behavior from `LazySandbox`. Its `create_or_attach()`
hook contains the Modal-specific acquisition and identity handling; the shared helper owns
coordination and caching. See [writing sandbox backends](https://ai.pydantic.dev/sandbox/#supply-a-sandbox-from-a-capability) for the authoring contract.

## Limits and errors

Modal exposes no per-command kill operation, so a command runs to its own deadline. Set
`timeout=` when starting commands. Modal accepts whole
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

This backend uses asyncio. After `disconnect()`, the same backend tries to reattach with its saved
`ref` on the next operation. If `destroy()` reports a cleanup failure, retry it on the same
backend; its saved `ref` remains available for that retry. A backend constructed with a saved
`ref` can destroy the remote sandbox before its first normal operation, without creating a new
one. Both methods treat an already-gone sandbox as successfully released.

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

## API reference

::: pydantic_ai_harness.modal_sandbox.ModalSandbox

::: pydantic_ai_harness.modal_sandbox.ModalSandboxBackend
