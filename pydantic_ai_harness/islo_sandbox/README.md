# Islo Sandbox

`IsloSandbox` gives a Pydantic AI agent an isolated Islo cloud environment for
running shell commands and working with files. Use it when model-generated code
must not execute on the application host.

This is a sandbox capability, not a model provider. Islo supplies execution and
filesystem isolation; your existing Pydantic AI model still performs inference.

## Quick start

Install the optional dependency and configure an Islo API key:

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

result = agent.run_sync('Create a Python script and run it.')
print(result.output)
```

Every run gets a fresh sandbox by default. The capability requests deletion when
the run exits and configures Islo's `delete_after` lifecycle policy as a cleanup
backstop.

## Tools

| Tool | Purpose |
| --- | --- |
| `run_command` | Run a command through `sh -c`. |
| `read_file` | Read bounded UTF-8 text with line paging. |
| `write_file` | Upload UTF-8 text to a file. |
| `list_directory` | List entries, marking directories with `/`. |

Command output keeps stdout and stderr separate, reports non-zero exit codes,
and retains the tail when it must truncate. File reads retain the head and
return the next line offset. A recoverable API failure becomes a model retry. A
missing sandbox or rejected credential raises a typed terminal error so the
model cannot loop against an unusable environment.

## Lifecycle modes

The default mode creates and owns one sandbox per agent run:

```python
IsloSandbox(
    image='ghcr.io/islo-labs/islo-runner:latest',
    sandbox_timeout=900,
)
```

Attach to a sandbox managed elsewhere by name:

```python
IsloSandbox(sandbox_name='existing-sandbox')
```

Attached sandboxes are left running. Creation-only options such as `image`,
`env`, and resource sizing cannot be combined with `sandbox_name`.

To reuse an owned sandbox across several runs, enter a session yourself:

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

The caller must enter an injected session and owns its lifetime. Do not reuse one
sandbox concurrently for workloads that need filesystem or process isolation.

## Timeout semantics

`default_command_timeout` and the optional per-tool `timeout_seconds` control how
long the client waits. Values round up to whole seconds before they are sent to
Islo and are capped by `max_command_timeout`, or by `sandbox_timeout` for an
owned sandbox.

As of 2026-08-13, Islo documents `timeout_secs` as a compatibility hint and does
not expose an exec-cancellation endpoint. A client wait timeout is therefore
reported as `remote command may still be running`. The command can continue
until it exits or the sandbox is deleted. Keep command deadlines and the sandbox
lifetime conservative for untrusted work.

## Output and file limits

`max_output_bytes` and `max_output_lines` bound each command stream before it is
returned to the model. Islo can also report provider-side truncation; the tool
preserves that signal in its truncation marker. Labels and status annotations
add a small amount beyond the configured payload limits.

`read_file` streams no more than `max_read_bytes + 1` bytes into the client and
rejects an oversized file. It accepts UTF-8 text only. Use a bounded shell
command such as `head`, `tail`, `sed`, or `grep` for large or binary files.

`list_directory` uses a POSIX shell fallback because the Islo SDK does not expose
a directory-list endpoint. It requires a runner image with `sh`, materializes a
bounded listing, and does not support filenames containing newline characters.

## Configuration

```python
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

Use `base_url` and `compute_url` for an Islo-compatible deployment. They apply
to a capability-created client, including attach mode, but not to an injected
session. Set `instructions=''` to disable the default model instructions.

## Tool-name composition

The tool names overlap with the Shell and FileSystem capabilities. Pydantic AI
rejects duplicate names. Prefix this capability when composing them:

```python
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai_harness.islo_sandbox import IsloSandbox

sandbox = PrefixTools(
    wrapped=IsloSandbox(
        instructions='Use the islo_-prefixed tools for work in the Islo sandbox.',
    ),
    prefix='islo',
)
```

Prefixing does not rewrite default instructions, so provide instructions that
name the prefixed tools.

## Lower-level API

`IsloSandboxSession` exposes explicit lifecycle, command, and byte-file access:

```python
from pydantic_ai_harness.islo_sandbox import IsloSandboxSession

async with IsloSandboxSession() as session:
    result = await session.exec(['sh', '-c', 'echo hello'], timeout=30)
    print(result.stdout, result.returncode)
```

Public errors are `IsloSandboxError`, `IsloSandboxTerminalError`,
`IsloSandboxAuthError`, and `IsloSandboxUnavailableError`. The toolset class is
an implementation detail.

## Agent specs

Register the class as a custom capability type:

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

## References

- [Islo documentation](https://docs.islo.dev/)
- [Islo Python SDK](https://github.com/islo-labs/python-sdk)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Unified Islo Sandbox docs](https://pydantic.dev/docs/ai/harness/islo-sandbox/)
- [Islo Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/islo_sandbox/)
