"""Capability that supplies a Daytona workspace to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from pydantic_ai_harness.daytona_workspace._backend import DEFAULT_AUTO_STOP_MINUTES, DaytonaWorkspaceBackend

if TYPE_CHECKING:
    from daytona import AsyncDaytona


@dataclass(kw_only=True)
class DaytonaWorkspace(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Daytona](https://www.daytona.io) workspace through `ctx.workspace`.

    A run with no reference creates a fresh workspace. Pass a `WorkspaceRef` supplied by the
    application to attach to an environment managed elsewhere.

    This capability supplies execution only. Compose it with tools or
    capabilities that consume
    [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace].
    """

    client: AsyncDaytona | None = None
    """A caller-owned `daytona.AsyncDaytona` client. When omitted, the backend creates one while
    acquiring the workspace and closes it again on release; supply one to keep the client open
    across runs and own its lifecycle with `async with AsyncDaytona() as client:`."""

    snapshot: str | None = None
    """Daytona snapshot used for a newly created workspace."""

    auto_stop_minutes: int = DEFAULT_AUTO_STOP_MINUTES
    """Idle minutes before Daytona stops a newly created workspace."""

    workdir: str | None = None
    """Absolute working directory for commands and relative filesystem paths."""

    env: Mapping[str, str] | None = None
    """Environment variables configured on a newly created workspace."""

    network_block_all: bool = False
    """Whether to block outbound traffic from a newly created workspace."""

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        del ctx
        if ref is not None and ref.provider != 'daytona':
            return None
        return DaytonaWorkspaceBackend(
            client=self.client,
            ref=ref,
            snapshot=self.snapshot,
            auto_stop_minutes=self.auto_stop_minutes,
            working_dir=self.workdir,
            env=self.env,
            network_block_all=self.network_block_all,
        )
