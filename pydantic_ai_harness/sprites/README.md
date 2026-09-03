# Fly.io Sprites Sandbox

`SpriteSandbox` gives a Pydantic AI agent a persistent
[Fly.io Sprite](https://docs.sprites.dev/): a Linux cloud computer for running
commands and working with files. By default, every agent run creates a fresh
Sprite and destroys it at the end. You can instead attach to an existing Sprite
or reuse a caller-owned session across runs.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Quick start

Install the optional dependency and provide a Fly.io Sprites API token:

```bash
uv add "pydantic-ai-harness[sprites]"
export SPRITE_TOKEN=...
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

The capability contributes four tools:

| Tool | Purpose |
|---|---|
| `run_command` | Run a Bash command with a bounded timeout and combined output. |
| `read_file` | Read a UTF-8 file with bounded output and line paging. |
| `write_file` | Write a UTF-8 file and create parent directories. |
| `list_directory` | List entries, marking directories with `/`. |

## Lifecycle

The default mode creates a uniquely named Sprite with an ownership label for
each agent run. On exit, the session fetches the Sprite and verifies that label
before destroying it. If the label changed, cleanup stops with
`SpriteSandboxOwnershipError` rather than risk deleting a different Sprite.
The label is a stale-handle guard, not an authorization boundary: the current
Fly.io Sprites API does not support label-conditional deletion, so other actors
with write access to the same organization must remain trusted.

Attach to a Sprite you manage by name. It is left running:

```python
from pydantic_ai_harness import SpriteSandbox

SpriteSandbox(sprite_name='my-existing-sprite')
```

To reuse one Sprite across multiple runs, own its session explicitly:

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
or destroys it. Do not share one session between overlapping runs that need
filesystem or process isolation.

## Limits and errors

Every model-facing command gets a finite deadline. `default_command_timeout`
sets the normal limit and `max_command_timeout` caps model-supplied values. The
command runs in a process group inside the Sprite, so a timeout terminates the
shell and its child processes. The in-Sprite byte cut preserves the beginning
and end of combined stdout and stderr before the SDK returns it. The tool layer
then applies byte and line limits to the retained payload, keeping the tail where
diagnostics commonly appear. The final command result, including truncation and
status annotations, is clamped to both configured caps.

File reads reject files larger than `max_read_bytes` before decoding them and
enforce that limit again while reading inside the Sprite. Directory listings
apply entry and byte limits inside the Sprite before returning data to the SDK.

Recoverable command and file failures become model retry prompts. A missing
Sprite raises `SpriteSandboxUnavailableError`; rejected credentials raise
`SpriteSandboxAuthError`. Both are terminal because retrying the same tool call
cannot repair them.

The Fly.io Sprites Python SDK is synchronous. This integration runs SDK calls in worker
threads so it does not block the agent event loop.

## Configuration

```python
from pydantic_ai_harness import SpriteSandbox

SpriteSandbox(
    token=None,                    # defaults to SPRITE_TOKEN at run start
    sprite_name=None,              # attach instead of creating per run
    session=None,                  # reuse an open SpriteSandboxSession
    base_url='https://api.sprites.dev',
    api_timeout=30.0,
    runtime=None,                  # None, 'default', or 'dev' for owned Sprites
    workdir=None,                  # defaults to the Sprite's current directory
    default_command_timeout=60.0,
    max_command_timeout=300.0,
    max_output_bytes=50 * 1024,
    max_output_lines=2000,
    max_read_bytes=5 * 1024 * 1024,
    instructions=None,
)
```

Set `instructions=''` to add no default instructions, or pass custom text to
replace them. Connection and lifecycle settings cannot be combined with an
injected `session` because that session already owns them.

Do not combine this capability with another unprefixed capability that registers
the same tool names. Use Pydantic AI's `PrefixTools` when an agent needs both.

## Further reading

- [Fly.io Sprites documentation](https://docs.sprites.dev/)
- [Fly.io Sprites Python SDK](https://github.com/superfly/sprites-py)
- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Fly.io Sprites Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/sprites/)
