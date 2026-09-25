# E2B Sandbox

Run an agent's commands and file operations in an isolated [E2B](https://e2b.dev) sandbox. `E2BSandbox` supplies the run's `ctx.workspace`; the tools you add alongside it, such as `Coder`, `Shell`, and `FileSystem`, act in that sandbox.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/e2b_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[e2b]"
```

pip:

```bash
pip install "pydantic-ai-harness[e2b]"
```

Set `E2B_API_KEY` for authentication. The integration requires E2B SDK 2.48.0 or later.

## A coding agent in a sandbox

`Coder`'s search tools need ripgrep (`rg`), which E2B's default `base` template does not include (it does include `git`). Build a template that adds it once with E2B's template SDK:

```python
from e2b import Template

Template.build(Template().from_template('base').apt_install('ripgrep'), 'coder')
```

Then name that template:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[E2BSandbox(template='coder'), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
print(result.output)
```

The model's shell and file tools run in a fresh E2B sandbox built from that template. Nothing runs on the machine that hosts the agent. The sandbox is created the first time the run uses it; a run that never uses it creates nothing.

Continue the conversation in the same sandbox by passing the history. The run recovers the sandbox's reference from it and attaches on first use:

```python
followup = agent.run_sync('Now find where capability hooks are ordered.', message_history=result.all_messages())
```

Without history or a reference, each run creates a fresh sandbox. A reference to a sandbox that no longer exists fails with `WorkspaceUnavailableError` instead of handing the model an empty replacement.

Shell commands run under `sh -c` inside the sandbox's login shell environment, as they do on a local workspace.

## Choose the tools

`Coder` bundles the shell and file tools with context management. For a narrower surface, add `Shell` and `FileSystem` on their own, or write tools that call `ctx.workspace` directly:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.e2b_sandbox import E2BSandbox
from pydantic_ai_harness.filesystem import FileSystem

agent = Agent('anthropic:claude-sonnet-5', capabilities=[E2BSandbox(), FileSystem()])

@agent.tool
async def run_python(ctx: RunContext[None], code: str) -> str:
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

## Reattach from a reference

Persist `result.workspace.ref` to reuse the sandbox outside message history. It is `None` until creation succeeds. An explicit reference takes precedence over history:

```python
from pydantic_ai import Agent
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[E2BSandbox(template='coder'), Coder()])

async def resume(ref: WorkspaceRef) -> str:
    result = await agent.run('Run the test suite again.', workspace=ref)
    return result.output
```

Pass `workspace='new'` to start a fresh sandbox while keeping the conversation, ignoring the reference in `message_history`:

```python
from pydantic_ai.messages import ModelMessage


async def retry_on_clean_checkout(history: list[ModelMessage]) -> str:
    result = await agent.run('Try the same fix on a clean checkout.', message_history=history, workspace='new')
    return result.output
```

E2B resumes a paused sandbox when a run attaches to it.

## Manage the sandbox yourself

Retain an `E2BSandboxBackend` and call `await backend.get_client()` to get the typed `e2b.AsyncSandbox`. The first call creates or attaches to the sandbox; later calls return the same object. An existing native handle can be supplied with `E2BSandboxBackend(workspace=native)`. Supply either a native handle or `ref=`, not both.

This example creates a sandbox, uses it for one run, and kills it afterwards. The kill is shielded and bounded so it still runs when the surrounding task is cancelled:

```python
import anyio
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

agent = Agent('anthropic:claude-sonnet-5', capabilities=[Coder()])

async def run_with_cleanup() -> str:
    backend = E2BSandboxBackend(template='coder')
    native = await backend.get_client()
    try:
        result = await agent.run('Which Python version is installed?', workspace=backend)
        return result.output
    finally:
        with anyio.move_on_after(30, shield=True):
            await native.kill()
```

## Configuration

`env` sets environment variables every command gets, including on a sandbox attached by reference; a command's own `env` is layered on top, and nothing is read from the host environment. `template` picks the E2B template a new sandbox runs, and `allow_internet_access` controls its outbound network access; these configure creation and do not reconfigure an attached sandbox.

`working_dir` is the absolute directory commands start in and relative paths resolve against. E2B has no create-time working directory, so it applies per command, including on an attached sandbox; `None` uses the sandbox's own default. Invalid values for `working_dir` or `sandbox_timeout` raise `UserError` when `E2BSandbox` is constructed.

A command `timeout` covers the command alone: it starts once the sandbox is acquired, and creating a sandbox has its own bound. On timeout, cancellation, or a failure reading the result, the backend kills the command's process. A process the command started in the background outlives that kill until the sandbox ends. Commands buffer their complete output, and a timeout error carries the output collected so far.

Rejected credentials and a sandbox that is gone raise `WorkspaceUnavailableError`, which ends the run. A path problem raises the builtin file error (`FileNotFoundError`, `IsADirectoryError`, `NotADirectoryError`, `PermissionError`, or `FileExistsError`), and any other failed operation raises `WorkspaceError`; the model sees either as a failed tool call. Sandbox creation that does not finish within 120 seconds raises `TimeoutError`. Rate limits, a busy E2B service, and network failures propagate unchanged, so a durable execution engine can retry them.

## Lifetime and cost

A sandbox keeps running, and billing, after the agent run ends: Pydantic AI does not kill it. `sandbox_timeout` sets the lifetime of a newly created sandbox, after which E2B stops it; when it is unset, [E2B's default lifetime](https://docs.e2b.dev/sandbox) applies. That lifetime is the backstop for sandboxes you lose track of. Attaching by reference applies E2B's default lifetime from the time of attaching, which can extend a shorter remaining lifetime. Cancelling a run while E2B creates its sandbox waits for creation to finish, so `ref` still names the sandbox. To stop a sandbox sooner, kill it through `get_client()`, as above.

A run can reattach from a `WorkspaceRef` wherever it continues, including under a durable execution engine. The reference carries no credentials, so each worker needs its own `E2B_API_KEY`. See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for how a run selects and restores its workspace.

The capability emits no additional telemetry spans. Core agent and tool spans cover calls made through tools; provider-specific diagnostics remain available through E2B.

## API reference

::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend
