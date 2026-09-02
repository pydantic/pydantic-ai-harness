# Daytona Sandbox

`DaytonaSandbox` gives an agent an isolated cloud computer for running commands
and working with files. Use it when model-generated code should not run on the
application host.

Each agent run gets a fresh [Daytona sandbox](https://www.daytona.io/docs/en/)
that is deleted when the run ends. You can instead attach a sandbox you manage.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Quick start

Install the extra and set a Daytona API key:

```bash
uv add "pydantic-ai-harness[daytona]"
export DAYTONA_API_KEY=...
```

Add the capability:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import DaytonaSandbox

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[DaytonaSandbox()],
)

result = agent.run_sync('Create a Python script and run its tests.')
print(result.output)
```

## Tools

| Tool | Purpose |
| --- | --- |
| `run_command` | Run a shell command with a bounded timeout and output. |
| `read_file` | Read UTF-8 text with line paging. |
| `write_file` | Write UTF-8 text and create parent directories. |
| `list_directory` | List entries, marking directories with `/`. |

Non-zero command exits are returned to the model. File errors become retryable
tool errors. Missing sandboxes and rejected credentials raise
`DaytonaSandboxUnavailableError` and `DaytonaSandboxAuthError`.

## Lifecycle

The default creates one sandbox per run and deletes it on exit. Daytona also
stops it after `auto_stop_minutes` of inactivity and deletes it immediately
after stopping, which bounds an orphan if teardown cannot reach the control
plane.

Set `sandbox_id` to attach an existing sandbox. Attached sandboxes are not
deleted. `snapshot` selects the snapshot for a fresh sandbox and cannot be used
with `sandbox_id`.

```python
DaytonaSandbox(sandbox_id='sandbox-id')
```

`workdir` applies to commands and relative file paths. `env` is passed when the
sandbox is created and on every command. Set `network_block_all=True` on a fresh
sandbox or session to block outbound traffic:

```python
DaytonaSandbox(network_block_all=True)
```

To reuse one sandbox across sequential runs while controlling its lifetime,
open a `DaytonaSandboxSession` and pass it to the capability:

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness import DaytonaSandbox
from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxSession


async def main() -> None:
    async with DaytonaSandboxSession() as session:
        agent = Agent(
            'openai:gpt-5.6-sol',
            capabilities=[DaytonaSandbox(session=session)],
        )
        await agent.run('Install the project dependencies.')
        await agent.run('Run the tests in the same sandbox.')


asyncio.run(main())
```

The caller opens and closes an injected session. The capability does neither.
An attached session (`DaytonaSandboxSession(sandbox_id=...)`) is also left
running when the session closes. Do not share one session between overlapping
runs that need isolated files or processes.

`session=` cannot be combined with `sandbox_id`, `snapshot`, a non-default
`auto_stop_minutes`, `workdir`, `env`, or `network_block_all` on the capability.
Configure those on `DaytonaSandboxSession`, which owns the sandbox. Attached
sandboxes retain their existing network settings.

`DaytonaSandboxSession.exec` returns a `DaytonaSandboxExecResult` for
applications that need command access outside an agent run. It streams command
output from Daytona and retains only the last `max_output_bytes`. The caller
must provide a positive whole-second timeout.

Use `DaytonaSandboxSession.process` when an application needs to exchange input
with a long-running command or handle stdout and stderr separately:

```python
async with DaytonaSandboxSession(network_block_all=True) as session:
    async with session.process(
        'worker',
        'python worker.py',
        on_stdout=print,
        on_stderr=print,
        max_input_bytes=64 * 1024,
    ) as process:
        await process.send('{"task":"run"}\n')
        returncode = await process.wait(timeout=60)
```

The caller supplies the process identity, input bound, and every wait. Starting,
input, output streaming, and deletion use Daytona's process-session API. Context
exit terminates the remote process session, including after a timeout or
cancellation. Input echo is disabled so protocol input cannot appear on stdout.

## Limits

Commands default to `default_command_timeout=60` seconds. Model-requested
timeouts are capped by `max_command_timeout=300`. Tool output is bounded by
`max_output_bytes` and `max_output_lines`; command output keeps the tail and
directory output keeps the head. `read_file` refuses files larger than
`max_read_bytes` before decoding them.

The Daytona SDK is currently constrained to `>=0.198.0,<0.199.0`. Later SDKs
require `typing-extensions>=4.16.0`, while another Harness extra currently pins
4.15.0. Raise the ceiling after those extras resolve together.

## Composition

The tools use common names. Prefix them when another capability contributes the
same names:

```python
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai_harness import DaytonaSandbox

PrefixTools(wrapped=DaytonaSandbox(instructions=''), prefix='daytona')
```

Daytona sandboxes isolate processes and files from the application host. Network
access and credentials inside the sandbox remain separate trust boundaries. Use
`network_block_all=True` when tools do not require outbound access.
The Daytona SDK is asyncio-native, so this capability does not support trio.
Durable execution is rejected because a live sandbox session cannot survive
activity replay or worker restart.

## Source

- [Daytona Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/daytona_sandbox/)
