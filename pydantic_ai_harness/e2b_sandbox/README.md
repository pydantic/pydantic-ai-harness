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

For an owned sandbox, acquisition stores the logical run ID in E2B metadata. A
retry searches for the oldest running or paused match before creating. After a
create, it checks again and kills the new sandbox if another creator won the
race. The serialized `SandboxRef` contains only the provider and sandbox ID;
later workers reconnect by ID and never create from `get_sandbox`. Release sends
a bounded, cancellation-shielded kill directly by ID without reconnecting or
resuming a paused sandbox, so it works in a different worker and is safe to
retry. An already missing sandbox counts as successfully released.

The metadata key `pydantic-ai-run-id` is reserved for this lifecycle identity.
Other metadata is preserved. E2B does not enforce metadata uniqueness, so the
post-create canonicalization is best-effort under control-plane propagation
delay; `sandbox_timeout` remains the server-side cleanup backstop.

Attach to a sandbox managed elsewhere by ID when the capability must not own its
lifetime:

```python
E2BSandbox(sandbox_id='sbx-abc123', workdir='/workspace')
```

Creation-only settings cannot be combined with `sandbox_id`. Attached sandboxes
are not killed at run end, and concurrent runs share their filesystem and process
space. E2B resumes a paused sandbox when connecting to it. The SDK also extends
the sandbox's remaining lifetime to at least its 300-second default on connect,
even when no explicit timeout is passed.

## Direct backend use

`E2BSandboxBackend` implements Pydantic AI's `SandboxBackend` protocol and its
filesystem and process-start opt-ins:

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

backend = await E2BSandboxBackend.create(
    template='base',
    sandbox_timeout=1800,
)
try:
    result = await backend.run(['python', '--version'], timeout=60)
    print(result.stdout)
finally:
    await backend.close(terminate=True)
```

Use `connect(sandbox_id)` when you need a fresh handle to an existing sandbox;
it never provisions a replacement, but it can extend the existing sandbox's
remaining lifetime as described above.

## Limits and cancellation

E2B's SDK timeout stops consuming its event stream but does not stop the remote
command. The backend therefore enforces deadlines client-side and sends SIGKILL
when a command times out or the caller is cancelled. Background children may
outlive the killed command until the sandbox itself is killed.

Command results are buffered by E2B and returned in full. Tools should enforce
their own byte or line budget, and commands that may produce very large output
should bound it at the source.

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
