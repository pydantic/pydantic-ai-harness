"""Daytona sandbox capability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._base import BaseDurabilityCapability  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.daytona_sandbox._session import DEFAULT_AUTO_STOP_MINUTES, DaytonaSandboxSession
from pydantic_ai_harness.daytona_sandbox._toolset import DaytonaSandboxToolset
from pydantic_ai_harness.modal_sandbox._tool_output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent

_DEFAULT_ID = 'daytona_sandbox'
_DEFAULT_MAX_READ_BYTES = 5 * 1024 * 1024


@dataclass(kw_only=True)
class DaytonaSandbox(AbstractCapability[AgentDepsT]):
    """Commands and files in an isolated Daytona cloud sandbox.

    Each run creates a fresh sandbox and deletes it on exit. Set `sandbox_id` to
    attach to a sandbox you manage instead; attached sandboxes are left running.

    Requires the `daytona` extra and `DAYTONA_API_KEY`.
    """

    id: str | None = _DEFAULT_ID
    """Stable capability and toolset ID."""

    sandbox_id: str | None = None
    """Existing sandbox ID or name to attach to instead of creating one."""

    session: DaytonaSandboxSession | None = None
    """An open caller-owned session to reuse across runs.

    The capability uses the session but does not open or close it. Cannot be
    combined with `sandbox_id` or fresh-sandbox provisioning settings.
    """

    snapshot: str | None = None
    """Daytona snapshot used for a fresh sandbox. None uses Daytona's default."""

    auto_stop_minutes: int = DEFAULT_AUTO_STOP_MINUTES
    """Idle minutes before Daytona stops a fresh sandbox; stopped sandboxes auto-delete."""

    workdir: str | None = None
    """Working directory used by commands and relative file paths."""

    env: Mapping[str, str] | None = None
    """Environment variables for a fresh sandbox and its commands."""

    network_block_all: bool = False
    """Block outbound network traffic from a fresh sandbox."""

    default_command_timeout: int = 60
    """Default command timeout in seconds."""

    max_command_timeout: int = 300
    """Maximum command timeout the model may request, in seconds."""

    max_output_bytes: int = DEFAULT_MAX_BYTES
    """Maximum UTF-8 bytes returned by each tool."""

    max_output_lines: int = DEFAULT_MAX_LINES
    """Maximum lines returned by each tool."""

    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES
    """Largest file `read_file` will load."""

    instructions: str | None = None
    """Override the default sandbox instructions. Set `''` to disable them."""

    def __post_init__(self) -> None:
        for name, value in (
            ('auto_stop_minutes', self.auto_stop_minutes),
            ('default_command_timeout', self.default_command_timeout),
            ('max_command_timeout', self.max_command_timeout),
            ('max_output_bytes', self.max_output_bytes),
            ('max_output_lines', self.max_output_lines),
            ('max_read_bytes', self.max_read_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.')
        if self.default_command_timeout > self.max_command_timeout:
            raise ValueError('default_command_timeout cannot exceed max_command_timeout.')
        if self.sandbox_id is not None and self.snapshot is not None:
            raise ValueError('snapshot cannot be combined with sandbox_id.')
        if type(self.network_block_all) is not bool:
            raise ValueError(f'network_block_all must be a boolean, got {self.network_block_all!r}.')
        if self.sandbox_id is not None and self.network_block_all:
            raise ValueError('network_block_all cannot configure an attached sandbox.')
        if self.session is not None:
            conflicts = [
                name
                for name, value, default in (
                    ('sandbox_id', self.sandbox_id, None),
                    ('snapshot', self.snapshot, None),
                    ('auto_stop_minutes', self.auto_stop_minutes, DEFAULT_AUTO_STOP_MINUTES),
                    ('workdir', self.workdir, None),
                    ('env', self.env, None),
                    ('network_block_all', self.network_block_all, False),
                )
                if value != default
            ]
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} cannot be combined with `session`, which already owns '
                    'the sandbox and its configuration.'
                )
        if self.instructions is not None and type(self.instructions) is not str:
            raise ValueError(f'instructions must be a string or None, got {self.instructions!r}.')
        if self.env is not None:
            self.env = dict(self.env)

    def get_instructions(self) -> str | None:
        if self.instructions is not None:
            return self.instructions or None
        reused = self.sandbox_id is not None or self.session is not None
        lifetime = 'persists after this run' if reused else 'is deleted after this run'
        return (
            'You have an isolated Daytona cloud sandbox. Use `run_command` for shell commands and '
            '`read_file`, `write_file`, and `list_directory` for files. '
            f'The sandbox {lifetime}. Commands default to {self.default_command_timeout}s and are capped at '
            f'{self.max_command_timeout}s.'
        )

    def for_agent(self, agent: AbstractAgent[AgentDepsT, object]) -> AbstractCapability[AgentDepsT]:
        """Reject durable execution, which cannot replay a live sandbox session."""
        siblings: list[AbstractCapability[AgentDepsT]] = []
        agent.root_capability.apply(siblings.append)
        if any(isinstance(sibling, BaseDurabilityCapability) for sibling in siblings):
            raise UserError(
                'DaytonaSandbox does not support durable execution: a live sandbox session cannot survive '
                'activity replay or worker restart.'
            )
        return self

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return DaytonaSandboxToolset[AgentDepsT](
            id=self.id or _DEFAULT_ID,
            sandbox_id=self.sandbox_id,
            session=self.session,
            snapshot=self.snapshot,
            auto_stop_minutes=self.auto_stop_minutes,
            workdir=self.workdir,
            env=self.env,
            network_block_all=self.network_block_all,
            default_command_timeout=self.default_command_timeout,
            max_command_timeout=self.max_command_timeout,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
            max_read_bytes=self.max_read_bytes,
        )
