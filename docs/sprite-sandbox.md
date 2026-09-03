---
title: Fly.io Sprites Sandbox
description: Give a Pydantic AI agent a persistent Fly.io Sprite with command and file tools.
---

# Fly.io Sprites Sandbox

`SpriteSandbox` gives an agent a persistent [Fly.io Sprite](https://docs.sprites.dev/):
a Linux cloud computer for running commands and working with files. Use it for
coding, data processing, and long-running tasks that should not execute
model-generated commands on the application host. By default, each agent run
gets a fresh Sprite that is destroyed when the run ends. You can also attach to
an existing Sprite or reuse one across several runs.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Quick start

Install the `sprites` extra and set a Fly.io Sprites API token:

```bash
uv add "pydantic-ai-harness[sprites,anthropic]"
export SPRITE_TOKEN=...
export ANTHROPIC_API_KEY=...
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SpriteSandbox

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[SpriteSandbox()],
)

result = agent.run_sync('Create a Python program and run it.')
print(result.output)
```

The capability adds four tools:

| Tool | Purpose |
| --- | --- |
| `run_command` | Run a Bash command with a bounded timeout and combined output. |
| `read_file` | Read a UTF-8 file with bounded output and line paging. |
| `write_file` | Write a UTF-8 file and create parent directories. |
| `list_directory` | List entries, marking directories with `/`. |

## Lifecycle

The default mode creates a uniquely named Sprite with an ownership label per
agent run. On exit, the session fetches the Sprite and verifies that label before
destroying it. If the label changed, cleanup raises
`SpriteSandboxOwnershipError` instead of risking deletion of another Sprite.
The label is a stale-handle guard, not an authorization boundary: the current
Fly.io Sprites API does not support label-conditional deletion, so other actors
with write access to the same organization must remain trusted.

Attach to a Sprite you manage by name. It is left running when the run ends:

```python
from pydantic_ai_harness import SpriteSandbox

SpriteSandbox(sprite_name='my-existing-sprite')
```

Reuse one Sprite across multiple runs with a caller-owned session:

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness import SpriteSandbox
from pydantic_ai_harness.sprites import SpriteSandboxSession


async def main():
    async with SpriteSandboxSession() as session:
        agent = Agent(
            'anthropic:claude-fable-5',
            capabilities=[SpriteSandbox(session=session)],
        )
        await agent.run('Install the project dependencies.')
        await agent.run('Run the tests in the same Sprite.')


asyncio.run(main())
```

An injected session must already be open. The capability never opens, closes,
or destroys it. Attached and injected Sprites can retain files and processes, so
do not share one between overlapping runs that need isolation.

## Timeouts and output limits

Every command gets a finite deadline. `default_command_timeout` supplies the
normal limit and `max_command_timeout` caps model-supplied values. The command
runs in a process group inside the Sprite, so a timeout terminates the shell and
its child processes. A successful command can deliberately detach a background
process (for example, `server &`) before the foreground shell exits. That process
keeps running until it is explicitly stopped or the Sprite is destroyed; this is
part of the persistent-computer behavior, not a timeout escape.

The in-Sprite byte cut preserves the beginning and end of combined stdout and
stderr before the SDK returns them. The tool layer then applies
`max_output_bytes` and `max_output_lines` to the retained payload, keeping the
tail where diagnostics commonly appear. All cuts are marked where the configured
limit has enough room for the marker. After timeout or exit annotations are
added, the final command result is clamped again to both configured caps.

`read_file` uses a size-limited read inside the Sprite, and `list_directory`
applies its entry and byte limits there before returning data. All command-helper
responses also flow through bounded SDK streaming sinks. Those host-side sinks
abort the response if a helper executable inside the mutable Sprite is replaced
and tries to bypass its in-Sprite limit.

The Fly.io Sprites Python SDK is synchronous. The capability runs its calls in worker
threads, so SDK requests do not block the agent event loop.

## Errors and composition

Recoverable command and filesystem failures become model retry prompts. A
missing Sprite raises `SpriteSandboxUnavailableError` and rejected credentials
raise `SpriteSandboxAuthError`; both are terminal because repeating a tool call
cannot fix them.

Do not combine this capability with another unprefixed capability that registers
`run_command`, `read_file`, `write_file`, or `list_directory`. Use Pydantic AI's
`PrefixTools` and replace the default instructions when an agent needs both.

## Configuration

```python
from pydantic_ai_harness import SpriteSandbox

SpriteSandbox(
    token=None,
    sprite_name=None,
    session=None,
    base_url='https://api.sprites.dev',
    api_timeout=30.0,
    runtime=None,
    workdir=None,
    default_command_timeout=60.0,
    max_command_timeout=300.0,
    max_output_bytes=50 * 1024,
    max_output_lines=2000,
    max_read_bytes=5 * 1024 * 1024,
    instructions=None,
)
```

Set `instructions=''` to disable the default instructions, or pass custom text.
`runtime` only applies to a newly created Sprite. Connection and lifecycle
settings cannot be combined with an injected `session` because it already owns
them.

## API reference

- [Fly.io Sprites documentation](https://docs.sprites.dev/)
- [Fly.io Sprites Python SDK](https://github.com/superfly/sprites-py)
- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Fly.io Sprites Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/sprites/)

::: pydantic_ai_harness.sprites.SpriteSandbox

::: pydantic_ai_harness.sprites.SpriteSandboxSession

::: pydantic_ai_harness.sprites.SpriteSandboxExecResult

::: pydantic_ai_harness.sprites.SpriteSandboxError

::: pydantic_ai_harness.sprites.SpriteSandboxTerminalError

::: pydantic_ai_harness.sprites.SpriteSandboxAuthError

::: pydantic_ai_harness.sprites.SpriteSandboxUnavailableError

::: pydantic_ai_harness.sprites.SpriteSandboxOwnershipError
