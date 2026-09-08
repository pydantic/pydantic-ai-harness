# Sprite Sandbox

`SpriteSandbox` supplies a persistent [Fly.io Sprite](https://sprites.dev) as
`ctx.sandbox`. Add tools that use the core sandbox methods to run commands and
work with files.

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/sprites/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](../../docs/index.md#version-policy).

## Install and authenticate

```bash
uv add "pydantic-ai-harness[sprites]" "pydantic-ai-slim[anthropic]"
export SPRITE_TOKEN=...
export ANTHROPIC_API_KEY=...
```

## Use with an agent

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.sprites import SpriteSandbox

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SpriteSandbox()])


@agent.tool
async def run_command(ctx: RunContext[None], command: str) -> str:
    result = await ctx.sandbox.run(command, shell=True, timeout=30)
    return result.stdout + result.stderr


@agent.tool
async def read_file(ctx: RunContext[None], path: str) -> str:
    return (await ctx.sandbox.read_file(path, limit=200)).text
```

The capability supplies the backend; it does not add provider-specific tools.
Construction and `get_sandbox()` do no I/O. The first operation creates or attaches
one Sprite, coordinated by core's `LazySandbox`. Later operations reuse it.

For a caller-owned SDK client, keep its lifetime around the agent run:

```python
from sprites import SpritesClient
from pydantic_ai import Agent
from pydantic_ai_harness.sprites import SpriteSandbox

async def run_agent() -> None:
    with SpritesClient(token='...', timeout=30) as client:
        agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SpriteSandbox(client=client)])
        result = await agent.run('List the files in the workspace.')
        print(result.output)
```

The supplied client owns its token, API endpoint, and HTTP timeout, and the backend never closes it.
The backend's `api_timeout` bounds lifecycle cleanup calls; command control connections use a separate fixed close bound.

## Identity and lifetime

By default, the conversation ID determines the Sprite name. Later runs in that
conversation attach to the same workspace. Different conversations get different
names. `SpriteSandbox(sprite_name='existing')` attaches only; it raises if the
Sprite is absent. An explicit run `sandbox=SandboxRef(sandbox_id='existing')`
takes precedence over the configured name. Persist that ref when continuity
must not fall back to a newly created workspace.

Ending a run does not delete or disconnect the Sprite. Sprites preserve their filesystem across
idle periods; application code owns permanent deletion. Do not treat a cached SDK
object as proof that remote compute is still available.

For a backend creating a Sprite by name, cancellation during first acquisition can leave the remote result unknown while `ref` remains unset. Retry `await backend.sandbox` to recover the same deterministic name; it attaches if creation succeeded and creates if the name is absent. `destroy()` has no remote target until `ref` is known, so it cannot clean up an uncertain acquisition.

## Backend behavior

Import `SpriteSandboxBackend` for direct use. Await `backend.sandbox` to obtain
the native `sprites.Sprite` for provider-specific features such as checkpoints.
`token`, `base_url`, and `api_timeout` configure its SDK client. `workdir` on the
capability (`working_dir` on the backend) must be an absolute path inside the Sprite.
`runtime` applies only when creating a Sprite.

Commands require Python 3, Bash for shell strings, and POSIX process groups.
Filesystem operations use core's shell fallback, so files and commands share the
same filesystem. The released SDK's filesystem adapter is not used.

Command deadlines include lazy acquisition. A remote supervisor kills the command
process group on completion, timeout, or cancellation; background children in that
group do not outlive the call. Processes that deliberately create a different
session can escape that group. If the cleanup request cannot reach the Sprite,
termination cannot be confirmed and a warning is logged. Cancellation before
remote startup leaves a cancellation marker so a delayed request cannot start the
command; if that request never arrives, or the command finishes just before cancellation arrives,
the marker directory may remain in `/tmp`.

The synchronous SDK client is used in worker threads for short HTTP requests. Commands use one
public asyncio `ControlConnection` per command, and the connection is closed after success,
error, or cancellation. The backend is asyncio-only. Cancelling its caller does not stop an
in-flight SDK request; stable names allow acquisition retries to reconnect.
Command stdout and stderr are buffered separately and are not truncated. Use
bounded commands or redirect large output to files and read a window through core.
The SDK uses a fixed 120-second creation HTTP timeout independently of `api_timeout`.

SDK authentication and missing-Sprite failures reported by SDK HTTP calls or control handshakes
become `SpriteSandboxAuthError` and `SpriteSandboxUnavailableError`, retaining their original cause.
Other command transport errors preserve their cause. Command nonzero exits remain ordinary results;
deadlines raise core `SandboxTimeoutError`.

## Explicit lifecycle

Use `backend.destroy()` when the remote Sprite should be deleted. It is a no-op before a
Sprite has a saved ref, including on an unused backend, and it deletes an attached ref directly
without creating or looking up another Sprite. An already missing Sprite counts as destroyed.
The ref remains available when deletion fails so the call can be retried. Successful destruction
clears the local handle and working-directory cache.

Use `backend.disconnect()` to close the backend's owned SDK client while leaving the remote Sprite
unchanged. It is a local detach operation: successful disconnect clears local caches and the saved
ref lets the next operation reattach. A caller-injected `client` is never closed. Both methods
require callers to finish in-flight commands first, and neither is called automatically when an
agent run ends.

## Provider references

The integration targets [sprites-py 0.6](https://github.com/superfly/sprites-py/tree/v0.6.0).
Its [exec API](https://sprites.dev/api/sprites/exec) allows commands to continue after
WebSocket disconnect, which is why the backend supplies remote process supervision.
See [lifecycle and persistence](https://docs.sprites.dev/concepts/lifecycle/) for
provider retention behavior.
