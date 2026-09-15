# Modal Workspace

Run agent tools against files and processes in a Modal container. `ModalWorkspace` supplies the run's `ctx.workspace`; your tools choose which workspace operations the model can use.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_workspace/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[modal]"
modal token new
```

Modal SDK 1.5.2 or later is required for its filesystem operations. Credentials can also be supplied through `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

## Use a workspace

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.modal_workspace import ModalWorkspace

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ModalWorkspace(image='python:3.12-slim')],
)

@agent.tool
async def run_python(ctx: RunContext[None], code: str) -> str:
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr

async def main() -> None:
    first = await agent.run('Write the numbers 1 to 5 to /tmp/numbers.txt.')
    second = await agent.run(
        'Read /tmp/numbers.txt and calculate their sum.',
        message_history=first.all_messages(),
    )
    print(second.output)
```

Construction makes no Modal requests. The first workspace operation creates the container and records its reference. Later operations on that backend reuse the same SDK handle. A run that does not use its workspace creates nothing.

The second run above recovers the reference from message history and attaches on first use. Starting a run without a reference or history creates a fresh workspace when needed. An explicit reference takes precedence over history. If the referenced container is gone, attachment fails; it does not create an empty replacement. `name=` is a creation option, not a lookup key.

Tools can also use `ctx.workspace.read_text()`, `write_text()`, and filesystem operations. The existing `Shell` and `FileSystem` capabilities operate on the agent process's host; adding `ModalWorkspace` does not move those tools into Modal.

## References and the native SDK handle

Persist `result.workspace.ref` when your application needs to reuse the container outside message history. It is `None` until creation succeeds; a backend supplied an existing reference exposes that reference immediately.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.modal_workspace import ModalWorkspace

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ModalWorkspace()])

@agent.tool
async def read_file(ctx: RunContext[None], path: str) -> str:
    return await ctx.workspace.read_text(path)

async def resume(ref: WorkspaceRef) -> str:
    result = await agent.run('Read /tmp/numbers.txt.', workspace=ref)
    return result.output
```

For SDK operations, retain a `ModalWorkspaceBackend` and await its `workspace` property to get the typed `modal.Sandbox`. You can also construct the backend with an existing native handle using `ModalWorkspaceBackend(workspace=native)`. Supply either a native handle or `ref=`, not both.

The application owns termination and SDK detachment. This example explicitly acquires a container and cleans it up after the run:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.modal_workspace import ModalWorkspaceBackend

agent = Agent('anthropic:claude-sonnet-4-6')

@agent.tool
async def working_directory(ctx: RunContext[None]) -> str:
    return await ctx.workspace.working_dir()

async def run_with_cleanup() -> str:
    backend = ModalWorkspaceBackend(image='python:3.12-slim')
    native = await backend.workspace
    try:
        result = await agent.run('What is the working directory?', workspace=backend)
        return result.output
    finally:
        try:
            await native.terminate.aio()
        finally:
            await native.detach.aio()
```

## Lifetimes and durable execution

`sandbox_timeout` is the lifetime of a newly created container, in seconds; it defaults to 300. Finishing an agent run does not terminate it. Creation settings such as the image, environment, and working directory do not reconfigure a container attached by reference.

Pass a finite `timeout` to bound a command. Modal applies whole seconds, so fractional execution timeouts round up. Result collection allows up to 30 additional seconds for the SDK to return output after that deadline. Timeout errors include output that has been collected.

Modal has no per-command kill operation. Cancelling a call stops waiting locally; the command may continue until its server deadline or the container's lifetime ends. Cancelling creation can leave a container whose ID the caller did not receive; its server lifetime still applies.

Durable applications own reference persistence, creation coordination, and restoring a workspace inside their activities. The capability does not make remote operations replay-safe or recreate application wrappers automatically. Apply your workspace policies when restoring the backend, then access it through `ctx.workspace` in tools.

The capability emits no additional telemetry spans. Core agent and tool spans cover the calls made through tools; provider-specific diagnostics remain available through Modal.

## API reference

::: pydantic_ai_harness.modal_workspace.ModalWorkspace

::: pydantic_ai_harness.modal_workspace.ModalWorkspaceBackend
