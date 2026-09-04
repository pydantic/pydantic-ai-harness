"""Capability that supplies a Daytona sandbox to an agent run."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness._sandbox_provider import absolute_path
from pydantic_ai_harness.daytona_sandbox._backend import (
    DEFAULT_AUTO_STOP_MINUTES,
    DaytonaSandboxBackend,
)


def _sandbox_name(identity: str) -> str:
    """Return a Daytona-safe, deterministic name for one conversation."""
    return f'pydantic-ai-{hashlib.sha256(identity.encode()).hexdigest()[:32]}'


@dataclass(kw_only=True)
class DaytonaSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated Daytona sandbox through `ctx.sandbox`.

    One named sandbox is created or reused per conversation, so a follow-up run continues in
    the workspace the previous one left behind. The name is derived from the conversation, which
    also makes creation safe to retry across durable workers: a retry attaches to the sandbox the
    first attempt made rather than provisioning a second one.

    Nothing here deletes a sandbox. A conversation can span many runs, so the end of a run is not
    the end of the workspace; Daytona stops an idle sandbox after `auto_stop_minutes` and deletes
    it immediately after that. Raise `auto_stop_minutes` for longer work, or delete one yourself
    with `DaytonaSandboxBackend.delete_by_id`.

    Set `sandbox_id` to attach to a sandbox managed elsewhere instead.
    """

    sandbox_id: str | None = None
    """Existing Daytona sandbox ID or name to attach to without deleting it."""

    snapshot: str | None = None
    """Daytona snapshot used for an owned sandbox."""

    auto_stop_minutes: int = DEFAULT_AUTO_STOP_MINUTES
    """Idle minutes before Daytona stops an owned sandbox."""

    workdir: str | None = None
    """Absolute working directory for commands."""

    env: Mapping[str, str] | None = None
    """Environment variables configured on an owned sandbox."""

    network_block_all: bool = False
    """Whether to block outbound traffic from an owned sandbox."""

    def __post_init__(self) -> None:
        if self.auto_stop_minutes <= 0:
            raise ValueError(f'auto_stop_minutes must be a positive integer, got {self.auto_stop_minutes!r}.')
        self.workdir = absolute_path('workdir', self.workdir)
        if self.env is not None:
            self.env = dict(self.env)
        if self.sandbox_id is None:
            return
        conflicts = [
            name
            for name, value, default in (
                ('snapshot', self.snapshot, None),
                ('auto_stop_minutes', self.auto_stop_minutes, DEFAULT_AUTO_STOP_MINUTES),
                ('env', self.env, None),
                ('network_block_all', self.network_block_all, False),
            )
            if value != default
        ]
        if conflicts:
            raise ValueError(
                f'{", ".join(conflicts)} only apply when creating a sandbox, but `sandbox_id` '
                'attaches to an existing one.'
            )

    def get_sandbox(self, ctx: RunContext[AgentDepsT], *, ref: SandboxRef | None) -> SandboxBackend:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None:
            # An environment the run was pointed at explicitly, or one an earlier run left behind.
            return DaytonaSandboxBackend(ref=ref, working_dir=self.workdir)
        if self.sandbox_id is not None:
            return DaytonaSandboxBackend(ref=SandboxRef(sandbox_id=self.sandbox_id), working_dir=self.workdir)
        return DaytonaSandboxBackend(
            name=_sandbox_name(ctx.conversation_id or ctx.run_id or ''),
            snapshot=self.snapshot,
            auto_stop_minutes=self.auto_stop_minutes,
            working_dir=self.workdir,
            env=self.env,
            network_block_all=self.network_block_all,
        )
