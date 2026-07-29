"""Belgie Sandbox capability configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset
from typing_extensions import Self

from pydantic_ai_harness.belgie_sandbox._session import (
    DEFAULT_MAX_OLD_GENERATION_SIZE_MB,
    DEFAULT_TIMEOUT,
    BelgieSandboxSession,
)
from pydantic_ai_harness.belgie_sandbox._toolset import DEFAULT_MAX_OUTPUT_BYTES, BelgieSandboxToolset

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

_AgentOutputT = TypeVar('_AgentOutputT')

_DEFAULT_CAPABILITY_ID = 'belgie_sandbox'
_DEFAULT_CAPABILITY_DESCRIPTION = (
    'Run JavaScript, TypeScript, or TSX modules in a restricted embedded Deno sandbox via `run_typescript`.'
)


@dataclass(kw_only=True)
class BelgieSandbox(AbstractCapability[AgentDepsT]):
    """Execute JavaScript, TypeScript, or TSX in an embedded Belgie sandbox.

    This is an additive capability: `run_typescript` is exposed alongside the
    agent's existing tools. Default sessions use a temporary workspace and deny
    host access, dependency downloads, and runtime network access.
    """

    allow_package_imports: bool = False
    """Allow model-authored npm, JSR, and URL imports to resolve and download packages."""

    allow_network: bool = False
    """Allow unrestricted runtime network access, including `fetch`."""

    max_old_generation_size_mb: int | None = DEFAULT_MAX_OLD_GENERATION_SIZE_MB
    """V8 old-generation heap limit in MiB, or None to leave it unbounded."""

    timeout: float = DEFAULT_TIMEOUT
    """Maximum seconds for one `run_typescript` execution."""

    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    """Maximum UTF-8 JSON bytes returned by one script."""

    max_retries: int = 3
    """Maximum model retries for `run_typescript`."""

    session: BelgieSandboxSession | None = None
    """An already-open caller-owned session to reuse instead of creating one per run."""

    instructions: str | None = None
    """Custom model instructions; an empty string disables capability instructions."""

    def __post_init__(self) -> None:
        """Validate config and fill deferred-loading routing defaults."""
        for name, value in (
            ('allow_package_imports', self.allow_package_imports),
            ('allow_network', self.allow_network),
        ):
            if type(value) is not bool:
                raise ValueError(f'{name} must be a bool, got {value!r}.')
        if self.max_old_generation_size_mb is not None and (
            type(self.max_old_generation_size_mb) is not int or self.max_old_generation_size_mb <= 0
        ):
            raise ValueError(
                'max_old_generation_size_mb must be a positive integer or None, '
                f'got {self.max_old_generation_size_mb!r}.'
            )
        if type(self.timeout) is bool or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError(f'timeout must be a positive finite number, got {self.timeout!r}.')
        if type(self.max_output_bytes) is not int or self.max_output_bytes <= 0:
            raise ValueError(f'max_output_bytes must be a positive integer, got {self.max_output_bytes!r}.')
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError(f'max_retries must be a non-negative integer, got {self.max_retries!r}.')
        if self.instructions is not None and type(self.instructions) is not str:
            raise ValueError(f'instructions must be a string or None, got {self.instructions!r}.')

        if self.session is not None:
            conflicts = [
                name
                for name, value, default in (
                    ('allow_package_imports', self.allow_package_imports, False),
                    ('allow_network', self.allow_network, False),
                    (
                        'max_old_generation_size_mb',
                        self.max_old_generation_size_mb,
                        DEFAULT_MAX_OLD_GENERATION_SIZE_MB,
                    ),
                )
                if value != default
            ]
            if conflicts:
                raise ValueError(
                    f'{", ".join(conflicts)} cannot be combined with `session`, which already owns '
                    'the Belgie runtime configuration.'
                )

        if self.defer_loading and self.id is None:
            self.id = _DEFAULT_CAPABILITY_ID
        if self.defer_loading and self.description is None:
            self.description = _DEFAULT_CAPABILITY_DESCRIPTION

    def get_instructions(self) -> str | None:
        """Describe the effective sandbox contract to the model."""
        if self.instructions is not None:
            return self.instructions or None
        if self.session is not None:
            return (
                'Use `run_typescript` to execute complete JavaScript, TypeScript, or TSX modules in a '
                'caller-managed Belgie runtime. Export a default function or named `run` function and '
                'return JSON-serializable data. Runtime access and state lifetime depend on the supplied session.'
            )

        package_text = (
            'npm, JSR, and URL imports are enabled'
            if self.allow_package_imports
            else 'npm, JSR, URL, and relative imports are disabled'
        )
        network_text = 'runtime network access is enabled' if self.allow_network else 'runtime `fetch` is disabled'
        return (
            'Use `run_typescript` to execute a complete JavaScript, TypeScript, or TSX module in a '
            'temporary Belgie Deno sandbox. Export a default function or named `run` function and return '
            f'JSON-serializable data. {package_text}; {network_text}. Host files, environment '
            f'variables, subprocesses, and writes are unavailable. Each call has a {self.timeout:g}s deadline '
            f'and a {self.max_output_bytes}-byte JSON output limit. The runtime is reset between agent runs.'
        )

    def for_agent(self, agent: AbstractAgent[AgentDepsT, _AgentOutputT]) -> Self:
        """Reject durable engines that cannot carry the run-scoped Deno session."""
        from pydantic_ai.durable_exec._base import BaseDurabilityCapability

        durability: list[str] = []

        def collect(capability: AbstractCapability[AgentDepsT]) -> None:
            if isinstance(capability, BaseDurabilityCapability):
                durability.append(type(capability).__name__)

        agent.root_capability.apply(collect)
        if durability:
            names = ', '.join(sorted(durability))
            raise UserError(
                f'BelgieSandbox does not support durable execution capabilities ({names}): '
                'its Deno runtime is live, process-local state that cannot cross activity, task, or replay boundaries.'
            )
        return self

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build the additive Belgie Sandbox toolset."""
        return BelgieSandboxToolset[AgentDepsT](
            allow_package_imports=self.allow_package_imports,
            allow_network=self.allow_network,
            max_old_generation_size_mb=self.max_old_generation_size_mb,
            timeout=float(self.timeout),
            max_output_bytes=self.max_output_bytes,
            max_retries=self.max_retries,
            toolset_id=self.id or _DEFAULT_CAPABILITY_ID,
            session=self.session,
        )
