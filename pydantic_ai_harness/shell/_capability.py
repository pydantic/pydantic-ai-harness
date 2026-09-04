"""Shell capability that provides command execution for agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.shell._toolset import ShellToolset

_DEFAULT_DENIED_COMMANDS: tuple[str, ...] = (
    'rm',
    'rmdir',
    'mkfs',
    'dd',
    'format',
    'shutdown',
    'reboot',
    'halt',
    'poweroff',
    'init',
)


LLM_API_KEY_ENV_PATTERNS: tuple[str, ...] = (
    'ANTHROPIC_*',
    'GATEWAY_*',
    'GEMINI_*',
    'GOOGLE_*',
    'OPENAI_*',
    'OPENROUTER_*',
    'PYDANTIC_AI_GATEWAY_API_KEY',
)
"""Glob patterns for common LLM provider credentials, for `denied_env_patterns`.

Pass these to keep provider credentials in an explicit `env` out of commands.
With `LocalSandbox`, commands also receive its fixed `PATH`, `HOME`, `LANG` and
`TMPDIR`. This is not an isolation boundary. Covers provider prefixes only --
not other secrets, and the prefixes are coarse (`GOOGLE_*` also strips
`GOOGLE_APPLICATION_CREDENTIALS`), so treat it as a starting point.
"""


@dataclass
class Shell(AbstractCapability[AgentDepsT]):
    """Shell command execution for agents.

    Commands execute inside the run's sandbox, starting in `cwd`. Use
    `allowed_commands` or `denied_commands` to control what the agent can invoke.
    """

    cwd: str | Path = '.'
    """Working directory for command execution: a sandbox path, absolute or relative to the sandbox working directory."""

    allowed_commands: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only these command names may be executed (allowlist)."""

    denied_commands: Sequence[str] = _DEFAULT_DENIED_COMMANDS
    """These command names are always rejected (denylist).

    Defaults to blocking destructive commands (rm, dd, shutdown, etc.).
    Set to an empty list to disable.
    """

    denied_operators: Sequence[str] = field(default_factory=list[str])
    """Shell operators that are blocked (e.g. '>', '>>', '|' for restrictive mode)."""

    default_timeout: float = 30.0
    """Default timeout in seconds for command execution."""

    max_timeout: float = 600.0
    """Longest a single foreground command may run, in seconds."""

    max_output_chars: int = 50_000
    """Maximum characters of output returned to the model. Must be positive."""

    allow_interactive: bool = False
    """If True, allow interactive commands (vi, nano, ssh, etc.). Blocked by default."""

    env: Mapping[str, str] | None = None
    """Explicit environment for commands.

    Passes these variables explicitly to the sandbox. With `LocalSandbox`, they
    are added to its fixed `PATH`, `HOME`, `LANG`, and `TMPDIR` environment. This
    is not a security boundary: use OS-level isolation for untrusted commands.
    """

    denied_env_patterns: Sequence[str] = field(default_factory=list[str])
    """Glob patterns for environment variable names to strip from `env`.

    Follows the `denied_*` naming convention but matches by glob (`fnmatch`,
    e.g. `OPENAI_*`), since env secrets cluster by prefix -- unlike
    `denied_commands`, which matches executable names exactly. Only an explicit
    `env` is filtered; the sandbox's own environment is not visible to the
    toolset. If `env` is `None`, the sandbox backend decides what environment commands receive.
    See `LLM_API_KEY_ENV_PATTERNS` for a ready-made provider-credential denylist.
    """

    def __post_init__(self) -> None:
        """Resolve the built-in denylist according to the selected policy."""
        if self.denied_env_patterns and self.env is None:
            raise ValueError('denied_env_patterns requires an explicit env mapping.')
        if self.denied_commands is _DEFAULT_DENIED_COMMANDS:
            self.denied_commands = [] if self.allowed_commands else list(_DEFAULT_DENIED_COMMANDS)

    def get_toolset(self) -> ShellToolset[AgentDepsT]:
        """Build and return the shell toolset."""
        return ShellToolset[AgentDepsT](
            cwd=Path(self.cwd),
            allowed_commands=self.allowed_commands,
            denied_commands=self.denied_commands,
            denied_operators=self.denied_operators,
            default_timeout=self.default_timeout,
            max_timeout=self.max_timeout,
            max_output_chars=self.max_output_chars,
            allow_interactive=self.allow_interactive,
            env=self.env,
            denied_env_patterns=self.denied_env_patterns,
        )
