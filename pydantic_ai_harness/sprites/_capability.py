"""Fly.io Sprites capability that gives agents a persistent cloud computer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness._sandbox_output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from pydantic_ai_harness.sprites._session import (
    DEFAULT_API_TIMEOUT as _DEFAULT_API_TIMEOUT,
)
from pydantic_ai_harness.sprites._session import (
    DEFAULT_BASE_URL as _DEFAULT_BASE_URL,
)
from pydantic_ai_harness.sprites._session import (
    DEFAULT_MAX_COMMAND_TIMEOUT as _DEFAULT_MAX_COMMAND_TIMEOUT,
)
from pydantic_ai_harness.sprites._session import SpriteSandboxSession
from pydantic_ai_harness.sprites._toolset import SpriteSandboxToolset

_DEFAULT_MAX_READ_BYTES = 5 * 1024 * 1024

_OWNED_INSTRUCTIONS = (
    'You have a Fly.io Sprite: an isolated, persistent Linux computer created for this run. Use `run_command` '
    'to run shell commands, and `read_file` / `write_file` / `list_directory` to manage files. Commands '
    'run through Bash, so pipes and redirection work. A command times out after {default_timeout}s unless '
    'you pass `timeout_seconds` (up to {max_timeout}s). This Sprite is destroyed when the run ends.'
)

_REUSED_INSTRUCTIONS = (
    'You have a Fly.io Sprite: an isolated, persistent Linux computer. Use `run_command` to run shell commands, '
    'and `read_file` / `write_file` / `list_directory` to manage files. Commands run through Bash, so pipes '
    'and redirection work. A command times out after {default_timeout}s unless you pass `timeout_seconds` '
    '(up to {max_timeout}s). This Sprite is reused across runs, so earlier files can still be present.'
)


@dataclass(kw_only=True)
class SpriteSandbox(AbstractCapability[AgentDepsT]):
    """Access to a persistent cloud computer powered by [Fly.io Sprites](https://sprites.dev).

    By default, each agent run creates a fresh Sprite and destroys it when the run
    ends. Set `sprite_name` to attach to a Sprite you manage, or pass an already-open
    `SpriteSandboxSession` to reuse one across runs. Attached and injected Sprites are
    left running.

    Requires the `sprites` extra (`uv add "pydantic-ai-harness[sprites]"`) and a
    Fly.io Sprites API token in `SPRITE_TOKEN`, or passed via `token`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.sprites import SpriteSandbox

    agent = Agent('anthropic:claude-fable-5', capabilities=[SpriteSandbox()])
    result = agent.run_sync('Write and run a Python program that prints the first 10 primes.')
    print(result.output)
    ```
    """

    token: str | None = None
    """Fly.io Sprites API token. When omitted, `SPRITE_TOKEN` is read when a run starts."""

    sprite_name: str | None = None
    """Attach to an existing Sprite by name instead of creating one per run.

    An attached Sprite is not destroyed when the run ends. Cannot be combined with
    `runtime`, which applies only to Sprite creation, or with `session`.
    """

    session: SpriteSandboxSession | None = None
    """Use an already-open, caller-owned session across runs.

    The capability never opens or closes an injected session. Connection and Sprite
    settings cannot be combined with it because the session already owns them.
    """

    base_url: str = _DEFAULT_BASE_URL
    """Fly.io Sprites API base URL."""

    api_timeout: float = _DEFAULT_API_TIMEOUT
    """Timeout in seconds for Fly.io Sprites API operations, excluding Sprite creation."""

    runtime: str | None = None
    """Runtime channel for a newly created Sprite: `default`, `dev`, or None."""

    workdir: str | None = None
    """Working directory for commands and relative filesystem paths."""

    default_command_timeout: float = 60.0
    """Default timeout in seconds for one `run_command`."""

    max_command_timeout: float = _DEFAULT_MAX_COMMAND_TIMEOUT
    """Hard ceiling in seconds for a command, including a model-supplied timeout."""

    max_output_bytes: int = DEFAULT_MAX_BYTES
    """Maximum command or file payload retained in UTF-8 bytes."""

    max_output_lines: int = DEFAULT_MAX_LINES
    """Maximum command or file payload lines retained alongside the byte limit."""

    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES
    """Largest file `read_file` reads in full before returning a window."""

    instructions: str | None = None
    """Instructions added to the system prompt. Set `''` to disable them."""

    def __post_init__(self) -> None:
        """Validate limits and reject settings the selected lifecycle mode would ignore."""
        for name, value in (
            ('max_output_bytes', self.max_output_bytes),
            ('max_output_lines', self.max_output_lines),
            ('max_read_bytes', self.max_read_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.')
        for name, value in (
            ('api_timeout', self.api_timeout),
            ('default_command_timeout', self.default_command_timeout),
            ('max_command_timeout', self.max_command_timeout),
        ):
            if type(value) is bool or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number, got {value!r}.')
        if self.default_command_timeout > self.max_command_timeout:
            raise ValueError('default_command_timeout cannot exceed max_command_timeout.')
        if not self.base_url:
            raise ValueError('base_url must not be empty.')
        if self.runtime not in (None, 'default', 'dev'):
            raise ValueError(f"runtime must be 'default', 'dev', or None, got {self.runtime!r}.")
        if self.instructions is not None and type(self.instructions) is not str:
            raise ValueError(f'instructions must be a string or None, got {self.instructions!r}.')

        if self.session is not None:
            conflicts = [
                name
                for name, value, default in (
                    ('token', self.token, None),
                    ('sprite_name', self.sprite_name, None),
                    ('base_url', self.base_url, _DEFAULT_BASE_URL),
                    ('api_timeout', self.api_timeout, _DEFAULT_API_TIMEOUT),
                    ('runtime', self.runtime, None),
                    ('workdir', self.workdir, None),
                )
                if value != default
            ]
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} cannot be combined with `session`, which already owns '
                    'the Sprite and its connection settings.'
                )
        elif self.sprite_name is not None and self.runtime is not None:
            raise ValueError('runtime only applies when creating a Sprite; remove it when `sprite_name` is set.')

    def get_instructions(self) -> str | None:
        """Explain the Sprite tools and the configured lifecycle to the model."""
        if self.instructions is not None:
            return self.instructions or None
        template = (
            _REUSED_INSTRUCTIONS if self.sprite_name is not None or self.session is not None else _OWNED_INSTRUCTIONS
        )
        return template.format(
            default_timeout=self.default_command_timeout,
            max_timeout=self.max_command_timeout,
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build and return the Sprite toolset."""
        return SpriteSandboxToolset[AgentDepsT](
            token=self.token,
            sprite_name=self.sprite_name,
            base_url=self.base_url,
            api_timeout=self.api_timeout,
            runtime=self.runtime,
            workdir=self.workdir,
            default_command_timeout=self.default_command_timeout,
            max_command_timeout=self.max_command_timeout,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
            max_read_bytes=self.max_read_bytes,
            session=self.session,
        )
