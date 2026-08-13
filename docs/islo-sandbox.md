---
title: Islo Sandbox
description: Give a Pydantic AI agent per-run Islo command and file isolation.
---

# Islo Sandbox

`IsloSandbox` gives an agent an isolated cloud environment for running commands
and working with files, keeping model-generated code away from the application
host.

This is a sandbox capability, not a model provider. Islo supplies execution and
filesystem isolation; the model configured on your Pydantic AI agent still
performs inference.

## Quick start

Install the extra and set an Islo API key:

```bash
uv add "pydantic-ai-harness[islo]"
export ISLO_API_KEY=...
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.islo_sandbox import IsloSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[IsloSandbox()],
)

result = agent.run_sync('Create a Python script and run its tests.')
print(result.output)
```

The capability adds four tools:

| Tool | Purpose |
| --- | --- |
| `run_command` | Run a shell command through `sh -c`. |
| `read_file` | Read bounded UTF-8 text with line paging. |
| `write_file` | Upload UTF-8 text to a file. |
| `list_directory` | List entries, marking directories with `/`. |

Command output labels stdout and stderr, reports non-zero exits, and keeps the
tail when truncating. File reads keep the head and return a continuation offset.

## Lifecycle

Each run creates a fresh sandbox by default. On exit, the capability requests
deletion and Islo's `delete_after` lifecycle policy remains the server-side
cleanup backstop. Expect a sandbox cold start for every run, including runs where
the model does not call a sandbox tool.

Attach to a sandbox managed elsewhere by name:

```python
from pydantic_ai_harness.islo_sandbox import IsloSandbox

IsloSandbox(sandbox_name='existing-sandbox')
```

Attached sandboxes are left running. To reuse an owned sandbox across runs,
enter a session yourself:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.islo_sandbox import IsloSandbox, IsloSandboxSession

async with IsloSandboxSession(sandbox_timeout=1800) as session:
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[IsloSandbox(session=session, max_command_timeout=600)],
    )
    await agent.run('Install dependencies.')
    await agent.run('Run the tests in the same sandbox.')
```

The caller must enter an injected session and owns its lifetime. Reused
sandboxes share process and filesystem state, so use separate sandboxes for
overlapping runs that need isolation.

## Timeouts, cancellation, and limits

`default_command_timeout` supplies the client wait limit and
`max_command_timeout` caps model-supplied values. Values round up to whole
seconds. Without `max_command_timeout`, the ceiling is `sandbox_timeout` (900
seconds by default) in every lifecycle mode. Raise `max_command_timeout`
explicitly for a reused sandbox. An owned sandbox rejects a command ceiling
above its own lifetime.

As of 2026-08-13, Islo's `timeout_secs` command field is a compatibility hint,
not a documented server-enforced deadline, and Islo does not document an exec
cancellation endpoint. If the client deadline expires, the result explicitly
warns that the remote command may still be running. It can continue until it
exits or the sandbox is deleted. This limitation is why the integration uses a
finite sandbox lifetime and never represents a client timeout as a confirmed
remote kill.

`max_output_bytes` and `max_output_lines` bound each stdout and stderr payload.
Islo's provider-side `truncated` signal is retained in the model-facing marker.
Labels and status annotations add a small amount beyond the payload limits.

`read_file` streams at most `max_read_bytes + 1` bytes and refuses an oversized
file. It supports UTF-8 text only. Use a bounded shell command for large or
binary content.

`list_directory` uses a POSIX shell fallback because the SDK has no directory
listing endpoint. The image must provide `sh`; filenames containing newline
characters are not supported.

Islo's Python SDK currently requires an asyncio event loop.

## Errors and cleanup

Recoverable control-plane, command, and file failures become model retry prompts.
A missing or terminal sandbox raises `IsloSandboxUnavailableError`; rejected
credentials raise `IsloSandboxAuthError`. Both inherit from
`IsloSandboxTerminalError`, so the run stops instead of asking the model to retry
against a sandbox or credential that cannot recover.

If owned deletion fails, direct `IsloSandboxSession.close()` retains the
control-plane handle so callers can retry cleanup. The lifecycle `delete_after`
policy remains the final server-side backstop.

The tool names overlap with Shell and FileSystem. Pydantic AI rejects duplicate
tool names. Use `PrefixTools` and custom `instructions` when composing these
capabilities because prefixing does not rewrite the default instructions.

## Configuration

```python
from pydantic_ai_harness.islo_sandbox import IsloSandbox

IsloSandbox(
    image='ghcr.io/islo-labs/islo-runner:latest',
    sandbox_name=None,
    session=None,
    sandbox_timeout=900,
    workdir='/workspace',
    env=None,
    vcpus=None,
    memory_mb=None,
    disk_gb=None,
    internet_enabled=None,
    gateway_profile=None,
    base_url=None,
    compute_url=None,
    default_command_timeout=60.0,
    max_command_timeout=None,
    max_output_bytes=50 * 1024,
    max_output_lines=2000,
    max_read_bytes=5 * 1024 * 1024,
    poll_interval=0.5,
    instructions=None,
)
```

Creation-only settings cannot be combined with `sandbox_name` or an injected
`session`. `base_url` and `compute_url` apply to a capability-created client,
including attach mode, and must be absolute HTTPS URLs. Configure them on an
injected session instead. Set
`instructions=''` to disable default model instructions.

## Agent specs

Register `IsloSandbox` as a custom capability type:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - IsloSandbox:
      sandbox_timeout: 900
      internet_enabled: false
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.islo_sandbox import IsloSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[IsloSandbox])
```

## API reference

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Pydantic AI toolsets](/ai/tools-toolsets/toolsets/)
- [Islo documentation](https://docs.islo.dev/)
- [Islo Python SDK](https://github.com/islo-labs/python-sdk)
- [Islo Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/islo_sandbox/)
- [Pydantic AI Harness version policy](index.md#version-policy)

The API may change between releases while Pydantic AI Harness is on 0.x
versions.

::: pydantic_ai_harness.islo_sandbox.IsloSandbox

::: pydantic_ai_harness.islo_sandbox.IsloSandboxSession

::: pydantic_ai_harness.islo_sandbox.IsloSandboxExecResult

::: pydantic_ai_harness.islo_sandbox.IsloSandboxError

::: pydantic_ai_harness.islo_sandbox.IsloSandboxTerminalError

::: pydantic_ai_harness.islo_sandbox.IsloSandboxAuthError

::: pydantic_ai_harness.islo_sandbox.IsloSandboxUnavailableError
