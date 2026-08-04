---
title: E2B Sandbox
description: Give a Pydantic AI agent an E2B cloud sandbox with bounded command and file tools.
---

# E2B Sandbox

`E2BSandbox` gives a Pydantic AI agent an isolated cloud computer for running
model-generated commands and working with files. Use it for coding, tests, data
processing, and other workloads that should not execute on the application host.

Every agent run gets a fresh [E2B sandbox](https://e2b.dev/docs) by default. The
capability also supports attaching to an existing sandbox or injecting a session
that the application keeps open across several runs.

## Quick start

Install Harness with its E2B extra, then set an E2B API key:

```bash
uv add "pydantic-ai-harness[e2b]"
export E2B_API_KEY=...
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[E2BSandbox(template='base')],
)

result = agent.run_sync('Create a Python script that prints the first ten primes and run it.')
print(result.output)
```

The capability contributes four tools:

| Tool | Purpose |
| --- | --- |
| `run_command` | Run a Bash command and return labelled, bounded stdout and stderr. |
| `read_file` | Stream a UTF-8 text file with bounded memory and line paging. |
| `write_file` | Write UTF-8 text to a file. |
| `list_directory` | List directory entries, marking directories with `/`. |

A non-zero command exit is reported to the model instead of raised. Recoverable
service and filesystem errors become `ModelRetry`; an expired or missing
sandbox raises `E2BSandboxUnavailableError`, and rejected credentials raise
`E2BSandboxAuthError`.

## Logfire

E2B lifecycle and tool operations use the active Pydantic AI OpenTelemetry
tracer. With Pydantic AI instrumentation enabled, Logfire shows:

- `e2b.sandbox.create`, `e2b.sandbox.connect`, and `e2b.sandbox.kill`
- `e2b.sandbox.run_command`, `read_file`, `write_file`, and `list_directory`
- sandbox ID, template, lifecycle mode, outcome, exit code, timeout, truncation,
  file size, and directory entry count

Harness does not add commands, paths, file contents, stdout, or stderr to its
`e2b.*` span attributes, and disables automatic exception events on those spans
so provider error messages do not reintroduce that content. Pydantic AI's own
tool spans can include tool arguments and results, so set `include_content=False`
when those values must stay out of telemetry:

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

logfire.configure()
logfire.instrument_pydantic_ai(include_content=False)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[E2BSandbox()],
)
```

Without Pydantic AI instrumentation, capability-managed sessions use the run's
no-op tracer and the added operation spans have negligible overhead.

## Lifecycle

The default mode is **owned**. A new sandbox is created as the run enters the
toolset and killed when the run exits, even if the model never calls a sandbox
tool. `sandbox_timeout` is the E2B-side lifetime backstop.

Attach to a sandbox managed elsewhere by ID:

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

capability = E2BSandbox(sandbox_id='sbx_existing')
```

The capability connects for each run and leaves the sandbox running afterward.
Creation settings such as `template`, `env`, and `sandbox_timeout` cannot be
combined with `sandbox_id`.

Inject a session to reuse one sandbox across runs while controlling its lifetime:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxSession

async with E2BSandboxSession(template='base', sandbox_timeout=1800) as session:
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[E2BSandbox(session=session, max_command_timeout=600)],
    )
    await agent.run('Install the project dependencies.')
    await agent.run('Run the test suite in the same sandbox.')
```

The injected session must already be open. The capability uses it but never
opens or kills it. Reused sandboxes share one filesystem and process space, so
do not use the same session for overlapping runs that require isolation.

## Command output and file-read bounds

The E2B Python SDK buffers process output in the command handle. Harness
redirects its capture wrapper's own streams away from the SDK, captures command
stdout and stderr inside the sandbox, and reads every capture artifact through a
bounded stream. Each command stream retains at most `max_output_bytes`, then the
model-facing result applies `max_output_bytes` and `max_output_lines` again.
Truncation is marked and stream labels are preserved.

E2B starts commands through `/bin/bash -l -c` before Harness's command-level
redirection runs. Output emitted by Bash login startup files can therefore reach
the SDK outside this bound. A command can modify its user login files for later
calls, so `max_output_bytes` is not a hard SDK-memory ceiling in a reused
sandbox. Keep login profiles silent in E2B templates. A hard bound for startup
output requires a public E2B raw-exec or no-buffer command API.

A completed command returns the bounded tail. The tail pipeline flushes only
when the streams close, so a timed-out command instead returns a bounded prefix
(at most `max_output_bytes`) captured incrementally and marked as truncated to
the first bytes. The truncation flag is exact: the prefix capture retains one
byte past the limit to tell "exactly full" from "cut off".

This capture wrapper requires Bash, `setsid`, `mkfifo`, `cat`, `dd`, `wc`,
`tee`, and GNU `tail`. E2B's standard `base` template supplies them. Custom
templates must retain those programs.

`read_file` checks metadata and then streams at most `max_read_bytes + 1` bytes,
so a file that grows after the metadata check still cannot create an unbounded
client buffer. Large files are refused with a hint to slice them using
`run_command`. `list_directory` materializes E2B's complete listing before it
truncates the displayed result.

## Timeouts and cancellation

`default_command_timeout` applies when the model omits `timeout_seconds`.
`max_command_timeout` caps model-supplied values. In owned mode a command cannot
outlive `sandbox_timeout`; attached or injected modes use a 300-second ceiling
unless `max_command_timeout` is set explicitly.

When E2B reports a timeout or the client wait is cancelled, Harness makes
best-effort attempts to kill both the command's process group and its E2B command
handle. Exiting an owned session also kills the whole sandbox; a sandbox that is
already gone counts as cleaned up. If the kill request itself fails, the run
keeps its result: Harness emits a `RuntimeWarning` naming the sandbox id and the
`sandbox_timeout` backstop, and the session keeps the sandbox reference so
`close()` can be retried. The E2B async SDK uses asyncio internally, so real E2B
runs require an asyncio event loop.

## Durable execution

`E2BSandbox` cannot currently be combined with Pydantic AI's Temporal, DBOS, or
Prefect durability capabilities. Its run-scoped E2B client and sandbox lifecycle
cannot safely cross or replay activity, step, or task boundaries. Harness rejects
that combination when the agent is constructed and when durability capabilities
are supplied for a single run, before a sandbox or tool call is started.

## Configuration

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

E2BSandbox(
    template=None,
    sandbox_id=None,
    session=None,
    sandbox_timeout=300,
    workdir='/home/user',
    env=None,
    metadata=None,
    allow_internet_access=True,
    default_command_timeout=60.0,
    max_command_timeout=None,
    max_output_bytes=50 * 1024,
    max_output_lines=2000,
    max_read_bytes=5 * 1024 * 1024,
    instructions=None,
)
```

`template`, `sandbox_timeout`, `env`, `metadata`, and
`allow_internet_access` cannot be combined with `sandbox_id`. An injected
`session` additionally rejects `sandbox_id` and a non-default `workdir`, because
the caller-owned session already controls those settings. These conflicts fail
during capability construction.

## Composition

Do not combine `E2BSandbox` with another unprefixed capability that provides
`run_command`, `read_file`, `write_file`, or `list_directory`. Pydantic AI
rejects duplicate tool names. Use `PrefixTools` and custom instructions when an
agent needs both:

```python
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

sandbox = PrefixTools(
    wrapped=E2BSandbox(
        instructions='Use the e2b_-prefixed tools for work in the E2B sandbox.',
    ),
    prefix='e2b',
)
```

## Agent specs

Register `E2BSandbox` as a custom capability type when loading an agent spec:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - E2BSandbox:
      template: base
      sandbox_timeout: 600
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.e2b_sandbox import E2BSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[E2BSandbox])
```

## Lower-level session

`E2BSandboxSession` is public for applications that need explicit lifecycle and
byte-oriented access:

```python
from pydantic_ai_harness.e2b_sandbox import E2BSandboxSession

async with E2BSandboxSession(template='base') as session:
    result = await session.exec('echo hello', timeout=30, max_output_bytes=50 * 1024)
    print(result.stdout, result.returncode)
```

## API reference

- [E2B Python SDK](https://github.com/e2b-dev/E2B/tree/main/packages/python-sdk)
- [Pydantic AI Logfire integration](/ai/logfire/)
- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [E2B Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/e2b_sandbox/)
- [Pydantic AI Harness version policy](index.md#version-policy)

The API may change between releases while Pydantic AI Harness is on 0.x
versions.

::: pydantic_ai_harness.e2b_sandbox.E2BSandbox

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxSession

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxExecResult

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxError

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxTerminalError

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxAuthError

::: pydantic_ai_harness.e2b_sandbox.E2BSandboxUnavailableError
