"""Capability configuration for Islo-backed sandbox tools."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness._sandbox_tool_output import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from pydantic_ai_harness.islo_sandbox._session import (
    DEFAULT_IMAGE as _DEFAULT_IMAGE,
)
from pydantic_ai_harness.islo_sandbox._session import (
    DEFAULT_SANDBOX_TIMEOUT as _DEFAULT_SANDBOX_TIMEOUT,
)
from pydantic_ai_harness.islo_sandbox._session import (
    DEFAULT_WORKDIR as _DEFAULT_WORKDIR,
)
from pydantic_ai_harness.islo_sandbox._session import IsloSandboxSession
from pydantic_ai_harness.islo_sandbox._toolset import IsloSandboxToolset

_DEFAULT_MAX_READ_BYTES = 5 * 1024 * 1024

_OWNED_INSTRUCTIONS = (
    'You have an Islo sandbox: an isolated, ephemeral cloud development environment. Use `run_command` '
    'to run shell commands and `read_file` / `write_file` / `list_directory` to manage files. Commands '
    'run through `sh`. A command waits for up to {default_timeout}s unless you pass `timeout_seconds` '
    '(up to {max_timeout}s). The sandbox is reset between runs.'
)

_REUSED_INSTRUCTIONS = (
    'You have an Islo sandbox: an isolated cloud development environment. Use `run_command` to run '
    'shell commands and `read_file` / `write_file` / `list_directory` to manage files. Commands run '
    'through `sh`. A command waits for up to {default_timeout}s unless you pass `timeout_seconds` '
    '(up to {max_timeout}s). This sandbox persists across runs, so earlier files may still be present.'
)


@dataclass(kw_only=True)
class IsloSandbox(AbstractCapability[AgentDepsT]):
    """Give an agent isolated command execution and file access through Islo.

    Each run gets an owned sandbox by default. Set `sandbox_name` to attach to a
    sandbox managed elsewhere, or pass an open `session` to reuse one across
    runs. The capability provides bounded command, read, write, and listing
    tools; creation settings cover the image, workdir, environment, resources,
    network policy, and Islo endpoints.

    Requires the `islo` extra (`uv add "pydantic-ai-harness[islo]"`) and an Islo API
    key in `ISLO_API_KEY`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import IsloSandbox

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[IsloSandbox()])
    result = agent.run_sync('Write a Python script that prints the first 10 primes and run it.')
    print(result.output)
    ```
    """

    image: str = _DEFAULT_IMAGE
    """Container image for owned sandboxes, as a registry tag. Applies only when creating one."""

    sandbox_name: str | None = None
    """Attach to an existing sandbox by name instead of creating one. Attached sandboxes are not deleted.

    Use this to reuse a sandbox managed elsewhere. The settings that only apply when creating a
    sandbox (`image`, `sandbox_timeout`, `workdir`, `env`, `vcpus`, `memory_mb`, `disk_gb`,
    `internet_enabled`, `gateway_profile`) cannot be combined with `sandbox_name`.
    """

    session: IsloSandboxSession | None = None
    """Use a sandbox session you own and keep open across runs, instead of a per-run one.

    Pass an already-entered `IsloSandboxSession` to reuse one sandbox across runs while
    controlling its lifetime yourself: the capability uses it but never opens or closes it.
    Cannot be combined with `sandbox_name`, the creation settings, `base_url`, `compute_url`,
    or `poll_interval`, which the session already owns. A shared session is not
    concurrency-safe across overlapping runs.
    """

    sandbox_timeout: int = _DEFAULT_SANDBOX_TIMEOUT
    """Lifetime ceiling in seconds for an owned sandbox, applied as Islo's `delete_after` policy.

    This bounds the whole sandbox; `default_command_timeout` bounds a single command. Positive
    integer.
    """

    workdir: str | None = _DEFAULT_WORKDIR
    """Working directory for commands, and the base that relative tool paths resolve against."""

    env: Mapping[str, str] | None = None
    """Environment variables for the created sandbox. Copied at construction."""

    vcpus: int | None = None
    """vCPUs for the created sandbox. Positive integer, or `None` for the Islo default."""

    memory_mb: int | None = None
    """Memory in MiB for the created sandbox. Positive integer, or `None` for the Islo default."""

    disk_gb: int | None = None
    """Disk in GiB for the created sandbox. Positive integer, or `None` for the Islo default."""

    internet_enabled: bool | None = None
    """Whether the created sandbox reaches the internet. `None` leaves Islo's default policy."""

    gateway_profile: str | None = None
    """Islo gateway profile governing egress for the created sandbox."""

    base_url: str | None = None
    """Control-plane URL for an Islo-compatible deployment. Must be an absolute HTTPS URL.

    Applies to a client the capability creates, including in attach mode. Passing it alongside
    `session` raises: configure the endpoint on the session instead.
    """

    compute_url: str | None = None
    """Compute-plane URL for an Islo-compatible deployment. Must be an absolute HTTPS URL.

    Same rules as `base_url`.
    """

    default_command_timeout: float = 60.0
    """Seconds a command may run when the model supplies no `timeout_seconds`. Positive and finite."""

    max_command_timeout: int | None = None
    """Ceiling in seconds on a model-supplied `timeout_seconds`; requests above it are clamped.

    Defaults to `sandbox_timeout` when unset, and may not exceed it for an owned sandbox.
    """

    max_output_bytes: int = DEFAULT_MAX_BYTES
    """Byte cap on rendered command output; longer output is tail-truncated. Positive integer."""

    max_output_lines: int = DEFAULT_MAX_LINES
    """Line cap on rendered command output; longer output is tail-truncated. Positive integer."""

    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES
    """Byte cap on a single `read_file`. Reading past it asks the model to slice the file instead."""

    poll_interval: float = 0.5
    """Seconds between polls while waiting for a sandbox to start or a command to finish.

    Positive and finite. Passing it alongside `session` raises: set it on the session instead.
    """

    instructions: str | None = None
    """Replace the default model instructions. `''` supplies none; `None` keeps the defaults."""

    def __post_init__(self) -> None:
        """Validate limits and reject settings ignored by the selected lifecycle."""
        self._validate_configuration()
        if self.env is not None:
            self.env = dict(self.env)

        if self.session is not None:
            conflicts = self._non_default_session_settings()
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} cannot be combined with `session`; configure them on the session instead.'
                )
        elif self.sandbox_name is not None:
            conflicts = self._non_default_creation_settings()
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} only apply when creating a sandbox, but `sandbox_name` attaches to an '
                    'existing one. Remove them, or drop `sandbox_name` to create a sandbox.'
                )

        ceiling = self.max_command_timeout
        if (
            self.sandbox_name is None
            and self.session is None
            and ceiling is not None
            and ceiling > self.sandbox_timeout
        ):
            raise ValueError('max_command_timeout cannot exceed sandbox_timeout for an owned sandbox.')

    def _validate_configuration(self) -> None:
        for name, value in (
            ('sandbox_timeout', self.sandbox_timeout),
            ('max_output_bytes', self.max_output_bytes),
            ('max_output_lines', self.max_output_lines),
            ('max_read_bytes', self.max_read_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.')
        for name, value in (('vcpus', self.vcpus), ('memory_mb', self.memory_mb), ('disk_gb', self.disk_gb)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f'{name} must be a positive integer or None, got {value!r}.')
        if type(self.default_command_timeout) not in {int, float} or not math.isfinite(self.default_command_timeout):
            raise ValueError(
                f'default_command_timeout must be a positive finite number, got {self.default_command_timeout!r}.'
            )
        if self.default_command_timeout <= 0:
            raise ValueError(
                f'default_command_timeout must be a positive finite number, got {self.default_command_timeout!r}.'
            )
        if self.max_command_timeout is not None and (
            type(self.max_command_timeout) is not int or self.max_command_timeout <= 0
        ):
            raise ValueError(
                f'max_command_timeout must be a positive integer or None, got {self.max_command_timeout!r}.'
            )
        if (
            type(self.poll_interval) not in {int, float}
            or not math.isfinite(self.poll_interval)
            or self.poll_interval <= 0
        ):
            raise ValueError(f'poll_interval must be a positive finite number, got {self.poll_interval!r}.')
        if self.instructions is not None and type(self.instructions) is not str:
            raise ValueError(f'instructions must be a string or None, got {self.instructions!r}.')

    def _non_default_creation_settings(self) -> list[str]:
        return [
            name
            for name, value, default in (
                ('image', self.image, _DEFAULT_IMAGE),
                ('sandbox_timeout', self.sandbox_timeout, _DEFAULT_SANDBOX_TIMEOUT),
                ('workdir', self.workdir, _DEFAULT_WORKDIR),
                ('env', self.env, None),
                ('vcpus', self.vcpus, None),
                ('memory_mb', self.memory_mb, None),
                ('disk_gb', self.disk_gb, None),
                ('internet_enabled', self.internet_enabled, None),
                ('gateway_profile', self.gateway_profile, None),
            )
            if value != default
        ]

    def _non_default_session_settings(self) -> list[str]:
        values = self._non_default_creation_settings()
        if self.sandbox_name is not None:
            values.append('sandbox_name')
        if self.base_url is not None:
            values.append('base_url')
        if self.compute_url is not None:
            values.append('compute_url')
        if self.poll_interval != 0.5:
            values.append('poll_interval')
        return values

    def get_instructions(self) -> str | None:
        """Return default or caller-supplied model instructions."""
        if self.instructions is not None:
            return self.instructions or None
        ceiling = self.max_command_timeout if self.max_command_timeout is not None else self.sandbox_timeout
        default_timeout = min(max(1, math.ceil(self.default_command_timeout)), ceiling)
        template = (
            _REUSED_INSTRUCTIONS if self.sandbox_name is not None or self.session is not None else _OWNED_INSTRUCTIONS
        )
        return template.format(default_timeout=default_timeout, max_timeout=ceiling)

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build the Islo-backed run-scoped toolset."""
        return IsloSandboxToolset[AgentDepsT](
            image=self.image,
            sandbox_name=self.sandbox_name,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            env=self.env,
            vcpus=self.vcpus,
            memory_mb=self.memory_mb,
            disk_gb=self.disk_gb,
            internet_enabled=self.internet_enabled,
            gateway_profile=self.gateway_profile,
            base_url=self.base_url,
            compute_url=self.compute_url,
            default_command_timeout=self.default_command_timeout,
            max_command_timeout=self.max_command_timeout,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
            max_read_bytes=self.max_read_bytes,
            poll_interval=self.poll_interval,
            session=self.session,
        )
