# E2B Sandbox

Run your agent's commands and file edits in an isolated [E2B](https://e2b.dev) cloud sandbox instead of on your machine.

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

Then set `E2B_API_KEY` to your E2B API key.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[E2BSandbox(), Coder()])
result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
print(result.output)
```

`Coder`'s shell and file tools now run in the sandbox, not on your machine. The sandbox is created the first time a tool uses it, and it keeps running, and billing, after the run ends; see [Clean up](#clean-up).

E2B stops a sandbox 5 minutes after it's created by default, even if it's still in use. For longer work, pass `E2BSandbox(sandbox_timeout=3600)`.

E2B's default template has `git` but not the ripgrep (`rg`) that `Coder` searches with, so the model installs it on first use in each sandbox. To skip that step, build a template with it once, as a separate script, and pass `E2BSandbox(template='coder')`:

```python
from e2b import Template

Template.build(Template().from_template('base').apt_install('ripgrep'), 'coder')
```

## Continue in the same sandbox

```python
followup = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    message_history=result.all_messages(),
)
```

The follow-up run finds the sandbox in the message history and works in it, so the clone is still there. Without the history, a run starts a new sandbox. If you paused the sandbox in E2B, it resumes. If the sandbox has been killed or has expired, the run raises `WorkspaceUnavailableError` instead of starting over in an empty one.

## Choose the tools

For a narrower agent, use [`Shell`](../shell/) and [`FileSystem`](../filesystem/) instead of `Coder`, or write your own tool: inside a tool, `ctx.workspace` is the sandbox.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.e2b_sandbox import E2BSandbox
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-sonnet-5', capabilities=[E2BSandbox(), Shell(), FileSystem()])


@agent.tool
async def run_python(ctx: RunContext, code: str) -> str:
    """Run a Python snippet in the sandbox."""
    result = await ctx.workspace.run(['python', '-c', code], timeout=10)
    return result.stdout + result.stderr
```

See [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for everything `ctx.workspace` can do.

## Reattach later

To come back to the sandbox without the message history, save its id and pass it back as `workspace=`:

```python
from pydantic_ai import Agent
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent('anthropic:claude-sonnet-5', capabilities=[E2BSandbox(), Coder()])

result = agent.run_sync('Clone https://github.com/pydantic/pydantic-ai and summarize how capabilities work.')
ref = result.workspace.ref
assert ref is not None  # None only if no tool used the sandbox
sandbox_id = ref.id  # save this, e.g. in your database

later = agent.run_sync(
    'Which capability would you add next, and where would it live?',
    workspace=WorkspaceRef(provider='e2b', id=sandbox_id),
)
print(later.output)
```

The id holds no credentials, so the process that reattaches needs `E2B_API_KEY` too. Pass `workspace='new'` to start a fresh sandbox even when the message history names one.

`template`, `sandbox_timeout`, and `allow_internet_access` only shape a new sandbox; `working_dir` and `env` apply to every command, including after you reattach.

Already have an `e2b.AsyncSandbox`? Pass `workspace=E2BSandboxBackend(workspace=sandbox)` to a run, with `E2BSandboxBackend` from `pydantic_ai_harness.e2b_sandbox`. `E2BSandbox`'s settings don't apply to it; pass `working_dir=` and `env=` to the backend.

## Clean up

The sandbox keeps running, and billing, after the run ends. Pydantic AI never kills it. Kill it with the id you saved:

```python
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend


async def kill_sandbox(sandbox_id: str) -> None:
    backend = E2BSandboxBackend(ref=WorkspaceRef(provider='e2b', id=sandbox_id))
    sandbox = await backend.get_client()
    await sandbox.kill()
```

`get_client()` returns the `e2b.AsyncSandbox`. A sandbox you don't kill stops when its `sandbox_timeout` runs out. See [E2B's sandbox lifecycle](https://docs.e2b.dev/sandbox).

## Configuration

| Option | What it does |
| --- | --- |
| `template` | E2B template for a new sandbox. Default: E2B's `base`. |
| `sandbox_timeout` | Seconds a new sandbox lives before E2B stops it. Default: E2B's, 5 minutes. |
| `allow_internet_access` | Whether a new sandbox can reach the internet. Default: `True`. |
| `working_dir` | Absolute directory commands start in and relative paths resolve against. Default: the sandbox's own. |
| `env` | Environment variables every command gets. Nothing from your machine's environment reaches the sandbox. |

## API reference

::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxBackend
