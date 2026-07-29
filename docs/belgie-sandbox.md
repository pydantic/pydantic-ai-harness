---
title: Belgie Sandbox
description: Give a Pydantic AI agent a restricted embedded Deno runtime for JavaScript, TypeScript, and TSX.
---

# Belgie Sandbox

`BelgieSandbox` gives a Pydantic AI agent a restricted embedded Deno runtime for
JavaScript, TypeScript, and TSX. It contributes one `run_typescript` tool and
leaves every other agent tool visible.

[Belgie](https://pypi.org/project/belgie/) bundles Deno inside its Python
package. The capability is useful when a task needs the JavaScript ecosystem or
browser-style language APIs without executing model-authored code directly in
the application process.

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

The model writes a complete module whose default export, or named `run` export,
is called without arguments:

```typescript
export default function run(): Record<string, string[]> {
  const words = ["ant", "ape", "bear"];
  return Object.groupBy(words, (word) => word[0]);
}
```

The exported function must return a JSON-serializable value. `console.log`
output is not captured.

## Tool and composition model

The capability is additive: an agent with `search`, `save`, and
`BelgieSandbox()` keeps those two tools and gains `run_typescript`. JavaScript
inside the sandbox cannot call the other agent tools.

`run_typescript` uses a distinct name and remains a peer of `CodeMode`'s
`run_code`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.belgie_sandbox import BelgieSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[CodeMode(), BelgieSandbox()],
)
```

The tool includes `code_arg_language=typescript` metadata for instrumentation
and renderers. `CodeMode` therefore leaves it native instead of wrapping it into
the Python tool.

Deferred loading uses a stable capability ID:

```python
BelgieSandbox(defer_loading=True)
```

With no explicit `id`, this registers as `belgie_sandbox`. The tool and
instructions reach the model after the capability is loaded.

## Default isolation

Each agent run gets a separate temporary Belgie `Environment` and Deno
`Runtime`. The runtime starts lazily on the first tool call, so an unused
capability does not start a worker.

The default profile:

- disables npm, JSR, URL, and relative imports;
- denies runtime network access, including `fetch`;
- denies host environment variables, filesystem paths, subprocesses, writes, FFI, and
  system information;
- permits reads only from the run's temporary workspace;
- applies a 30-second execution deadline and a 50 KiB JSON result limit;
- limits V8's old-generation heap to 128 MiB.

Deno interprets an empty permission allow-list as "allow all". The
implementation omits denied permissions instead of passing empty lists.
`allow_network=True` deliberately uses the global network grant.

Belgie provides an embedded language sandbox, not a container or virtual
machine. Use an OS- or cloud-isolated sandbox when untrusted code must have a
separate kernel, filesystem, or network namespace.

## Opting into packages and network

Enable the two higher-risk features independently:

```python
BelgieSandbox(
    allow_package_imports=True,
    allow_network=True,
)
```

`allow_package_imports=True` allows npm, JSR, and URL module resolution. A
model-selected import can download and execute third-party code, but this option
does not grant runtime `fetch`.

`allow_network=True` grants unrestricted Deno runtime network access. Host
files, environment variables, subprocesses, and FFI remain denied.

## Lifecycle and reusable sessions

An owned environment and runtime live for one agent run. Multiple calls in that
run share a Deno worker, so `globalThis` state and runtime caches can persist.
Separate or concurrent agent runs receive separate workers. The capability
closes the runtime and removes its workspace on success, failure, or
cancellation.

Use a caller-owned session for explicit reuse across runs:

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness.belgie_sandbox import (
    BelgieSandbox,
    BelgieSandboxSession,
)

async def main() -> None:
    async with BelgieSandboxSession(allow_package_imports=True) as session:
        agent = Agent(
            'anthropic:claude-sonnet-4-6',
            capabilities=[BelgieSandbox(session=session)],
        )
        await agent.run('Run a TypeScript transform.')
        await agent.run('Run another transform in the same Deno worker.')

asyncio.run(main())
```

An injected session must already be open. The capability does not enter or
close it. Do not use one session for overlapping runs because they share
runtime-global state.

For a custom environment or permission profile, construct a
`belgie.Runtime` and pass it to `BelgieSandboxSession(runtime=...)`. The session
enters and exits that runtime without modifying its configuration.

## Timeouts, output limits, and errors

`timeout` bounds each module execution. A timeout cancels and drains the script
task before returning a retry prompt. Cancellation of the parent agent run is
preserved, and owned cleanup is shielded from cancellation.

Successful return values are serialized as compact JSON and measured in UTF-8
bytes. A result larger than `max_output_bytes` becomes a retry asking the model
for a smaller value or summary; results are not silently truncated.

Script syntax, module loading, permission, JavaScript, timeout, and invalid JSON
failures become `ModelRetry`. Missing Belgie, runtime startup failures, unopened
sessions, and lifecycle misuse raise typed errors because changing the
model-authored module cannot repair those failures:

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
    max_old_generation_size_mb=128,
    timeout=30.0,
    max_output_bytes=50 * 1024,
    max_retries=3,
    session=None,
    instructions=None,
)
```

Set `max_old_generation_size_mb=None` to leave the V8 old-generation limit
unset. Set `instructions=''` to suppress default instructions, or pass custom
text to replace them. The detailed `run_typescript` description remains.

Owned-runtime settings cannot be combined with an injected session; configure
the session instead. `timeout`, `max_output_bytes`, and `max_retries` still
apply at the tool level.

## Agent specs

Register `BelgieSandbox` as a custom capability type:

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

Caller-owned sessions and custom runtime objects are Python-only configuration.

## Limitations

- The capability requires asyncio; Belgie's async Python bindings do not run
  under Trio.
- Pydantic AI durable execution capabilities are rejected at agent
  construction. A live Deno worker cannot cross Temporal activity, Prefect
  task, DBOS workflow, or replay boundaries.
- Tool output is returned after execution. Streaming logs and incremental
  results are not exposed.
- Relative host-file imports and direct filesystem tools are outside this
  capability's contract.
- Native npm add-ons may need permissions beyond the package-import profile.
  Use a caller-configured runtime only after reviewing the package's access.

## API reference

- [Belgie](https://pypi.org/project/belgie/)
- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Code Mode](code-mode.md)
- [Modal Sandbox](modal-sandbox.md)
- [Belgie Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/belgie_sandbox/)
- [Pydantic AI Harness version policy](index.md#version-policy)

The API may change between releases while Pydantic AI Harness is on 0.x
versions.

::: pydantic_ai_harness.belgie_sandbox.BelgieSandbox

::: pydantic_ai_harness.belgie_sandbox.BelgieSandboxSession

::: pydantic_ai_harness.belgie_sandbox.BelgieSandboxError

::: pydantic_ai_harness.belgie_sandbox.BelgieSandboxExecutionError

::: pydantic_ai_harness.belgie_sandbox.BelgieSandboxTimeoutError

::: pydantic_ai_harness.belgie_sandbox.BelgieSandboxUnavailableError
