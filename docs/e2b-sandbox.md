---
title: E2B Sandbox
description: Supply an E2B microVM through ctx.sandbox.
---

# E2B Sandbox

`E2BSandbox` supplies an E2B microVM as the run's `ctx.sandbox`. It owns only
provider connection and lifecycle behavior. Add tools or capabilities that
consume `ctx.sandbox` for the model-facing interface you want.

[Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/e2b_sandbox/)

## Install and authenticate

```bash
uv add "pydantic-ai-harness[e2b]"
export E2B_API_KEY=...
```

## Use with an agent

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[E2BSandbox(template='base')],
)


@agent.tool
async def run_command(ctx: RunContext[None], argv: list[str]) -> str:
    result = await ctx.sandbox.run(argv, timeout=60)
    return result.stdout
```

The example deliberately uses argv. E2B ultimately executes a shell string, so
the backend quotes every argv element. If your tool needs shell syntax, call
`ctx.sandbox.run(command, shell=True, timeout=...)` and expose that power
explicitly in the tool schema.

## Lifecycle

Asking the capability for a sandbox does no I/O. It hands back a backend holding settings
plus, when there is one, the identity of a sandbox that already exists; the first command or
file operation creates or attaches, once.

An owned sandbox carries the conversation id in E2B metadata, so a follow-up run finds the
sandbox the previous one used and continues in the same workspace. The first use searches for
the oldest running or paused match before creating, which is also what makes a durable retry
attach rather than provision a second sandbox. After a create it checks again and kills the new
sandbox if another creator won the race.

The metadata key `pydantic-ai-conversation-id` is reserved for that identity. Other metadata is
preserved. E2B does not enforce metadata uniqueness, so the post-create canonicalization is
best-effort under control-plane propagation delay.

Nothing here kills a sandbox. A conversation can span many runs, so the end of a run is not the
end of the workspace; E2B reaps an idle sandbox at `sandbox_timeout`. If E2B has already reaped
a conversation's sandbox, the next run gets a fresh, empty one and the old files are gone —
raise `sandbox_timeout` when a conversation needs to outlive it, or kill the sandbox yourself
with `E2BSandboxBackend.kill_by_id`, which is bounded, shielded from cancellation, and safe to
retry.

Attach to a sandbox managed elsewhere by ID when the capability must not own its
lifetime:

```python
E2BSandbox(sandbox_id='sbx-abc123', workdir='/workspace')
```

Creation-only settings cannot be combined with `sandbox_id`. Concurrent runs on the same
sandbox share its filesystem and process space. E2B resumes a paused sandbox when connecting to
it, so attaching to one restarts it.

## Direct backend use

`E2BSandboxBackend` implements Pydantic AI's `SandboxBackend` protocol and its optional
filesystem. Building one does no I/O; the first operation creates the sandbox:

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

backend = E2BSandboxBackend(template='base', sandbox_timeout=1800)
try:
    result = await backend.run(['python', '--version'], timeout=60)
    print(result.stdout)
finally:
    await backend.close(terminate=True)
```

Pass `ref=SandboxRef(sandbox_id=...)` to attach to one specific sandbox, or `identity={...}` to
reuse the oldest sandbox carrying that metadata and create one only if there is none. A `ref`
whose sandbox is gone raises rather than quietly providing an empty replacement.

`backend.sandbox` is the live `e2b.AsyncSandbox`, for anything E2B-specific. You can only await
it, so no code path can reach a sandbox that has not been created yet.

## Limits and cancellation

E2B's SDK timeout stops consuming its event stream but does not stop the remote
command. The backend therefore enforces deadlines client-side and sends SIGKILL
when a command times out or the caller is cancelled. Background children may
outlive the killed command until the sandbox itself is killed.

Command results are buffered by E2B and returned in full. The backend does not
invent a model-output policy or pretend the transport is bounded. Tools that put
output into model context should enforce their own byte or line budget, and
commands that may produce very large output should bound it at the source.

The public error surface is deliberately narrow:

- `E2BSandboxError` reports transient E2B operation failures and is a core
  `pydantic_ai.sandboxes.SandboxError`.
- `E2BSandboxAuthError` and `E2BSandboxUnavailableError` report dead or unauthorized
  E2B environments and are core `pydantic_ai.sandboxes.SandboxUnavailableError` instances.
- `pydantic_ai.sandboxes.SandboxTimeoutError` for command deadlines, with the partial
  `stdout`, `stderr`, and enforced `timeout`.

Filesystem misses use the built-in `FileNotFoundError` contract.

## Configuration

```python
E2BSandbox(
    template=None,
    sandbox_id=None,
    sandbox_timeout=300,
    workdir=None,
    env=None,
    metadata=None,
    allow_internet_access=True,
)
```
