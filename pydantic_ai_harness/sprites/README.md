# Sprite Workspace

Run agent tools against files and processes in a persistent [Fly.io Sprite](https://sprites.dev). `SpriteWorkspace` supplies the run's `ctx.workspace`; your tools choose which operations the model can use.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/sprites/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[sprites]"
```

Set `SPRITE_TOKEN` for authentication. The integration uses sprites-py 0.6.x.

## Use a workspace

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.sprites import SpriteWorkspace

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SpriteWorkspace()])


@agent.tool
async def run_python(ctx: RunContext[None], code: str) -> str:
    result = await ctx.workspace.run(['python3', '-c', code], timeout=10)
    return result.stdout + result.stderr


async def main() -> None:
    first = await agent.run('Write the numbers 1 to 5 to /tmp/numbers.txt.')
    second = await agent.run(
        'Read /tmp/numbers.txt and calculate their sum.',
        message_history=first.all_messages(),
    )
    print(second.output)
```

Constructing the capability or backend makes no Sprites requests. First use creates a Sprite and records its reference; later operations on that backend reuse the SDK handle. A run that does not use its workspace creates no Sprite.

The second run recovers the reference from history and attaches on first use. Without a reference or history, each run creates a fresh Sprite when needed. An explicit reference takes precedence over history. A missing referenced Sprite fails instead of creating an empty replacement.

Tools can also call `ctx.workspace.read_text()`, `write_text()`, and filesystem operations, which run as commands inside the Sprite. The existing `Shell` and `FileSystem` capabilities operate on the agent process's host; adding `SpriteWorkspace` does not move those tools into the Sprite.

## References and the native SDK handle

Persist `result.workspace.ref` to reuse the Sprite outside message history. It is `None` until creation succeeds; supplying an existing reference exposes that reference immediately.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.sprites import SpriteWorkspace

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SpriteWorkspace()])


@agent.tool
async def read_file(ctx: RunContext[None], path: str) -> str:
    return await ctx.workspace.read_text(path)


async def resume(ref: WorkspaceRef) -> str:
    result = await agent.run('Read /tmp/numbers.txt.', workspace=ref)
    return result.output
```

Retain a `SpriteWorkspaceBackend` and await its `workspace` property to obtain the typed `sprites.Sprite`. An existing native handle can be supplied with `SpriteWorkspaceBackend(workspace=native)`. Supply either a native handle or `ref=`, not both; the caller retains ownership of an injected handle and its SDK client.

Without `client=`, the backend creates and owns its local `SpritesClient` on first acquisition from `token=` (or `SPRITE_TOKEN`). Retain that backend and call `disconnect()` in `finally`. This example also deletes the remote Sprite using the native SDK:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.sprites import SpriteWorkspaceBackend

agent = Agent('anthropic:claude-sonnet-4-6')


@agent.tool
async def working_directory(ctx: RunContext[None]) -> str:
    return await ctx.workspace.working_dir()


async def run_with_cleanup() -> str:
    backend = SpriteWorkspaceBackend()
    try:
        native = await backend.workspace
        try:
            result = await agent.run('What is the working directory?', workspace=backend)
            return result.output
        finally:
            native.delete()
    finally:
        await backend.disconnect()
```

To keep the remote Sprite, omit `native.delete()` and retain the outer `finally` that disconnects the backend. `disconnect()` closes only a backend-owned SDK client; it does not delete a Sprite or close a caller-supplied client. Finish in-flight commands before disconnecting.

## Lifetimes and durable execution

A Sprite persists after a run ends. It auto-suspends when idle and resumes on the next command; nothing deletes it automatically, so delete it explicitly with the native SDK when finished. Attaching by reference does not change its retention.

`runtime` and `workdir` configure a newly created Sprite; they do not reconfigure a Sprite attached by reference. Each command runs through its own asyncio control connection that the backend closes before returning. A control-connection disconnect does not stop the remote command, so the backend supervises the remote process group and cancels it on timeout or cancellation. Complete output is buffered, and timeout errors include output received so far.

The synchronous SDK acquires Sprites in a worker thread that a cancelled caller cannot abort, so a cancelled creation may leave a Sprite whose remote outcome is unknown to the caller. A command whose process detaches from its session group can outlive cancellation. Durable applications own creation coordination, reference persistence, and workspace restoration inside their activities. The capability does not make remote operations replay-safe or restore application wrappers automatically. Apply workspace policies when restoring the backend, then use `ctx.workspace` in tools.

The capability emits no additional telemetry spans. Core agent and tool spans cover calls made through tools; the Sprites SDK retains its own instrumentation behavior.

## API reference

::: pydantic_ai_harness.sprites.SpriteWorkspace

::: pydantic_ai_harness.sprites.SpriteWorkspaceBackend
