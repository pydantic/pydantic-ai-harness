# Belgie Sandbox

`BelgieSandbox` gives a Pydantic AI agent a restricted embedded Deno runtime for
JavaScript, TypeScript, and TSX. It adds one `run_typescript` tool without
removing or wrapping the agent's other tools.

Use it when a task is easier in the JavaScript ecosystem or needs browser-style
language APIs, but should not run model-authored code directly in the
application process. [Belgie](https://pypi.org/project/belgie/) bundles Deno, so
Node.js and Deno do not need to be installed separately.

## Quick start

Belgie supports Python 3.12-3.14. Install the optional extra:

```bash
uv add "pydantic-ai-harness[belgie]"
```

Add the capability to an agent:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.belgie_sandbox import BelgieSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[BelgieSandbox()],
)

result = agent.run_sync(
    'Use TypeScript to group ["ant", "ape", "bear"] by first letter.'
)
print(result.output)
```

The model writes a complete module:

```typescript
export default function run(): Record<string, string[]> {
  const words = ["ant", "ape", "bear"];
  return Object.groupBy(words, (word) => word[0]);
}
```

The exported function is called without arguments. It can be a default export
or a named `run` export, and its return value must be JSON-serializable.
`console.log` output is not captured.

## Tool and composition model

The capability is additive. An agent with `search`, `save`, and
`BelgieSandbox()` exposes all three existing tools plus `run_typescript`.
JavaScript inside the sandbox cannot call those other agent tools.

The distinct tool name also lets it compose with `CodeMode`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.belgie_sandbox import BelgieSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[CodeMode(), BelgieSandbox()],
)
```

`CodeMode` keeps `run_typescript` as a peer code-execution tool instead of
folding it into Python `run_code`. The tool carries
`code_arg_language=typescript` metadata for instrumentation and renderers.

Set `defer_loading=True` with no explicit `id` to use the stable
`belgie_sandbox` capability ID:

```python
BelgieSandbox(defer_loading=True)
```

The model receives the tool and its instructions only after loading the
capability.

## Default isolation

Every agent run gets a separate temporary Belgie `Environment` and Deno
`Runtime`. The runtime starts lazily on the first `run_typescript` call, so an
unused or deferred capability does not start a worker.

By default:

- npm, JSR, URL, and relative imports are disabled;
- runtime network access, including `fetch`, is denied;
- host environment variables, filesystem paths, subprocesses, writes, FFI, and
  system information are denied;
- the runtime can read only its temporary workspace;
- each call has a 30-second deadline and a 50 KiB JSON output limit;
- V8's old-generation heap is limited to 128 MiB.

An empty Deno `allow_*` list means "allow all", so the implementation does not
use empty lists for denied permissions. `allow_network=True` deliberately uses
that global network grant.

Belgie is an embedded language sandbox, not a container or virtual machine. The
Deno permission boundary is appropriate for constrained local code execution;
use an OS- or cloud-isolated sandbox when the threat model requires a separate
kernel, filesystem, or network namespace.

## Opting into packages, network, and rendering

The three higher-risk features are independent:

```python
BelgieSandbox(
    allow_package_imports=True,  # npm:, jsr:, and URL module resolution
    allow_network=True,          # runtime fetch/WebSocket access
    enable_rendering=True,       # install @belgie/render for TSX
)
```

`allow_package_imports=True` lets model-authored module specifiers download and
execute third-party code. It does not enable runtime `fetch`.

`allow_network=True` grants unrestricted runtime network access. It does not
expose host files, environment variables, or subprocesses.

`enable_rendering=True` installs `@belgie/render`, enables package resolution,
and grants the temporary `node_modules` and native-loader reads/FFI/system probes
needed by its Vite build. A script can then return self-contained HTML:

```tsx
import { render } from "@belgie/render";

function Widget() {
  return <main>Hello from Belgie</main>;
}

export default function run() {
  return render({ widget: <Widget />, plugins: [] });
}
```

The HTML is an ordinary tool return. Pydantic AI does not automatically mount it
as an application UI. Rendered documents often exceed the default output limit;
raise `max_output_bytes` deliberately when the consumer can handle that payload.

## Lifecycle and reusable sessions

An owned runtime is scoped to one agent run. Calls in that run share the same
Deno worker, so `globalThis` state and runtime caches can persist between calls.
Separate or concurrent runs receive separate workers and temporary workspaces.
The capability closes its worker and removes the workspace when the run exits,
including cancellation and error paths.

For explicit cross-run reuse, create and enter a session yourself:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.belgie_sandbox import (
    BelgieSandbox,
    BelgieSandboxSession,
)

async with BelgieSandboxSession(allow_package_imports=True) as session:
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[BelgieSandbox(session=session)],
    )
    await agent.run('Run a TypeScript transform.')
    await agent.run('Run another transform in the same Deno worker.')
```

The capability requires an injected session to be open and never enters or
closes it. One session is not safe for overlapping runs because they share
runtime-global state.

`BelgieSandboxSession(runtime=...)` is the advanced escape hatch for a
caller-configured `belgie.Runtime`. Runtime options on that object define its
permissions and environment; the session enters and exits it but does not
modify it.

## Timeouts, output limits, and errors

`timeout` bounds one script. On timeout, the script task is cancelled and
drained before a retry is sent to the model. Cancellation of the parent run is
preserved, and owned runtime cleanup is shielded so teardown can complete.

The result is serialized as compact JSON and measured in UTF-8 bytes before it
reaches model history. Results over `max_output_bytes` are rejected with a retry
prompt asking for a smaller value or summary; they are not silently truncated.

Script syntax, module, permission, JavaScript, timeout, and invalid JSON failures
become `ModelRetry`, because the model can revise its module. Missing Belgie,
runtime startup failures, unopened sessions, and lifecycle misuse raise typed
`BelgieSandboxError` subclasses instead of retrying model code against an
unusable runtime.

The public errors are:

- `BelgieSandboxError`
- `BelgieSandboxExecutionError`
- `BelgieSandboxTimeoutError`
- `BelgieSandboxUnavailableError`

## Configuration

```python
from pydantic_ai_harness.belgie_sandbox import BelgieSandbox

BelgieSandbox(
    allow_package_imports=False,
    allow_network=False,
    enable_rendering=False,
    max_old_generation_size_mb=128,
    timeout=30.0,
    max_output_bytes=50 * 1024,
    max_retries=3,
    session=None,
    instructions=None,
)
```

Set `max_old_generation_size_mb=None` to leave the V8 old-generation limit
unset. Set `instructions=''` to suppress capability instructions, or pass a
string to replace them. The detailed `run_typescript` tool description remains
available.

Settings that create or restrict an owned runtime cannot be combined with an
injected `session`; configure the session instead. Tool-level `timeout`,
`max_output_bytes`, and `max_retries` still apply to an injected session.

## Agent specs

Register the capability type when loading YAML:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - BelgieSandbox:
      timeout: 20
      max_output_bytes: 102400
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.belgie_sandbox import BelgieSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[BelgieSandbox])
```

Caller-owned sessions and custom `belgie.Runtime` objects are Python-only
configuration and are not represented in agent specs.

## Limitations

- The capability requires asyncio. Belgie's async Python bindings do not run
  under Trio.
- Durable execution capabilities are rejected at agent construction. A live,
  process-local Deno session cannot cross Temporal activity, Prefect task, DBOS
  workflow, or replay boundaries safely.
- Output is returned when execution completes; streaming logs and incremental
  results are not exposed.
- Relative host-file imports and direct filesystem tools are not part of this
  capability.
- Native npm add-ons need permissions beyond the default package-import profile;
  use a caller-configured runtime only after reviewing that package's access.

## Related

- [Belgie](https://pypi.org/project/belgie/)
- [Code Mode](../code_mode/)
- [Modal Sandbox](../modal_sandbox/)
- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/belgie_sandbox/)

The API may change between releases while Pydantic AI Harness is on 0.x
versions.
