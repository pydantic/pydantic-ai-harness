# Modal Sandbox

Give an agent an isolated Modal container to work in. `ModalSandbox` supplies the container as the run's `ctx.workspace`, so the commands and file edits of `Coder`, `Shell`, and `FileSystem` happen in the sandbox, not on your machine.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install

uv:

```bash
uv add "pydantic-ai-harness[modal]"
uv run modal token new
```

pip:

```bash
pip install "pydantic-ai-harness[modal]"
modal token new
```

Modal SDK 1.5.2 or later is required for its filesystem operations. Credentials can also be supplied through `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

## A coding agent in a sandbox

```python
import modal
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

# `Coder` needs git and ripgrep, which the default image, `python:3.12-slim`, does not have.
image = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')
agent = Agent('anthropic:claude-sonnet-5', capabilities=[ModalSandbox(image=image), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
print(result.output)
```

Construction makes no Modal requests. The first workspace operation creates the sandbox; with `Coder` or `FileSystem`, that is the start of the run.

Commands run under `sh -c` in the sandbox's shell environment. Nothing from your machine's environment reaches the sandbox: pass `env=` for variables every command should get, such as a token the agent needs. `working_dir=` sets the absolute directory commands start in and relative paths resolve against; it defaults to the image's.

## Continue in the same sandbox

```python
followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox's reference in the message history and reattaches to it, so the clone is still there. If that sandbox has expired or been terminated, the first workspace operation raises `WorkspaceUnavailableError`; no empty replacement is created.

## Choose the tools

`Coder` bundles the tools a coding agent needs. For a narrower agent, pick `Shell` and `FileSystem` yourself; both act in the sandbox:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.modal_sandbox import ModalSandbox
from pydantic_ai_harness.shell import Shell

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[ModalSandbox(), Shell(), FileSystem(read_only=True)],
)
```

Your own tools and hooks can use `ctx.workspace` directly (`run()`, `read_text()`, `write_text()`, and the other filesystem operations). A run whose tools include none of `Shell`'s or `FileSystem`'s emits a `UserWarning` once per process, because the model then has no way to reach the sandbox; if that is intended, pass `ModalSandbox(warn_if_no_tools=False)`. The flag goes away in the stable harness release.

## Reattach from a reference

Persist `result.workspace.ref` to come back to the sandbox outside message history. It is `None` if the run never used its workspace.

```python
import modal
from pydantic_ai import Agent
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox

image = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')
agent = Agent('anthropic:claude-sonnet-5', capabilities=[ModalSandbox(image=image), Coder()])


async def resume(ref: WorkspaceRef) -> str:
    result = await agent.run('Run the test suite again.', workspace=ref)
    return result.output
```

An explicit reference takes precedence over message history. Pass `workspace='new'` to `agent.run()` to start a fresh sandbox even when the history names one. Settings on `ModalSandbox` that describe a new sandbox (`image`, `app_name`, `sandbox_timeout`, `idle_timeout`, `name`) do not change one you reattach to; `working_dir` and `env` apply to every command, in a reattached sandbox too.

## Manage the sandbox yourself

For Modal SDK operations, hold a `ModalSandboxBackend` and call `await backend.get_client()` for the typed `modal.Sandbox`. The first call creates or attaches to the sandbox; later calls return the same object. `ModalSandboxBackend(workspace=native)` wraps a sandbox you already have; pass either that or `ref=`, not both.

```python
import modal
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

agent = Agent('anthropic:claude-sonnet-5', capabilities=[Coder()])
image = modal.Image.debian_slim(python_version='3.12').apt_install('git', 'ripgrep')


async def run_and_clean_up(prompt: str) -> str:
    backend = ModalSandboxBackend(image=image)
    sandbox = await backend.get_client()
    try:
        result = await agent.run(prompt, workspace=backend)
        return result.output
    finally:
        await sandbox.terminate.aio()
```

To terminate the sandbox an agent run created, take the backend from `result.workspace.backend`; check that its `ref` is not `None` first, since `get_client()` would otherwise create a sandbox.

## Lifetime and cost

A sandbox keeps running, and billing, after the agent run ends: until you terminate it, until `sandbox_timeout` (default 300 seconds) runs out, or, if you set `idle_timeout`, after that many seconds without activity. Pydantic AI never terminates a sandbox; terminating it, and picking lifetimes that reap the ones you lose track of, is the application's job.

- Pass a finite `timeout` to bound a command. Modal counts whole seconds, so fractional timeouts round up; a timed-out command reports the output it produced.
- Modal has no per-command kill: cancelling a call stops the wait, but the command runs on until its timeout or the sandbox's end.
- An expired or terminated sandbox, and rejected credentials, raise `WorkspaceUnavailableError`, which ends the run. Modal's connection and rate-limit errors propagate unchanged, so a durable execution engine can retry them.
- The `WorkspaceRef` carries no credentials, so every worker that reattaches needs its own Modal configuration. See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for how a run selects and restores its workspace.

## Upgrading from the previous `ModalSandbox`

Earlier releases shipped a `ModalSandbox` that registered its own `run_command`, `read_file`, `write_file`, and `list_directory` tools and terminated its sandbox when the run ended. The capability now only supplies the sandbox as `ctx.workspace`: it registers no tools and adds no instructions. Passing one of the previous constructor arguments raises a `UserError`, and importing one of the removed names raises an `ImportError`; both name the replacement.

`ModalSandbox(image=...)` on its own still builds, but the model then has no tool that reaches the sandbox; add `Coder()`, or `Shell()` and `FileSystem()`, as shown above.

### What changed in the lifecycle

- A run no longer terminates the sandbox when it ends. The sandbox runs until you terminate it or a lifetime runs out (see [Lifetime and cost](#lifetime-and-cost)).
- A run that continues a `message_history` reattaches to the sandbox the previous run used. Pass `workspace='new'` to `agent.run()` to start a fresh sandbox instead.
- If the sandbox being reattached has expired or was terminated, the first workspace operation raises `WorkspaceUnavailableError`. No empty replacement is created.

### Migration table

| Previous API | Now |
| --- | --- |
| `image`, `app_name`, `create_app_if_missing` | Unchanged; they configure a newly created sandbox. `image` also takes a `modal.Image`. |
| `env` | Unchanged, and now also applied to every command of a reattached sandbox. |
| `sandbox_timeout` | Unchanged. It also bounds every command, since a command cannot outlive the sandbox. |
| `workdir` | Renamed `working_dir`, which also applies in a reattached sandbox. `workdir=` still works and emits a deprecation warning. |
| `sandbox_id` | Removed. Use `agent.run(..., workspace=WorkspaceRef(provider='modal', id=sandbox_id))`. |
| `session` | Removed. Pass `ModalSandboxBackend(workspace=<modal.Sandbox>)` as `workspace=` to `agent.run()`. |
| `default_command_timeout` | Removed. Use `Shell(default_timeout=...)`. |
| `max_command_timeout` | Removed, no replacement. `sandbox_timeout` bounds every command. |
| `max_output_bytes`, `max_output_lines` | Removed. Use `Shell(max_output_chars=...)`, or `ToolOutputLimits` for any tool. |
| `max_read_bytes` | Removed. Use `FileSystem(max_read_lines=..., max_read_chars=...)`. |
| `instructions` | Removed. Put guidance in the agent's `instructions`. |
| `run_command` tool | Removed. Add `Shell()`. |
| `read_file`, `write_file`, `list_directory` tools | Removed. Add `FileSystem()`. |
| `ModalSandboxSession` | Removed. Use `ModalSandboxBackend(workspace=<modal.Sandbox>)` passed as `workspace=`. |
| `ModalSandboxExecResult` | Removed. `backend.run(...)` returns `pydantic_ai.workspaces.CommandResult`. |
| `ModalSandboxError` | Removed. Catch `pydantic_ai.workspaces.WorkspaceError`. |
| `ModalSandboxTerminalError`, `ModalSandboxUnavailableError`, `ModalSandboxAuthError` | Removed. Catch `pydantic_ai.workspaces.WorkspaceUnavailableError`. |

## API reference

::: pydantic_ai_harness.modal_sandbox.ModalSandbox

::: pydantic_ai_harness.modal_sandbox.ModalSandboxBackend
