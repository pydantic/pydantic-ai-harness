"""Capability that supplies an E2B sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from pydantic_ai_harness.e2b_workspace._backend import (
    DEFAULT_SANDBOX_TIMEOUT,
    E2BWorkspaceBackend,
)


@dataclass(kw_only=True)
class E2BWorkspace(AbstractCapability[AgentDepsT]):
    """Supply an isolated [E2B](https://e2b.dev) sandbox through `ctx.workspace`.

    A run with no reference creates a fresh sandbox. Pass a `WorkspaceRef` supplied by the
    application to attach to an environment managed elsewhere.

    This capability supplies execution only. Compose it with tools or
    capabilities that consume
    [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace].
    """

    template: str | None = None
    """E2B template name or ID for a newly created workspace."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Server-side lifetime backstop for a newly created workspace, in seconds."""

    workdir: str | None = None
    """Absolute working directory for commands and relative filesystem paths."""

    env: Mapping[str, str] | None = None
    """Environment variables configured on a newly created workspace."""

    metadata: Mapping[str, str] | None = None
    """Metadata added to a newly created workspace."""

    allow_internet_access: bool = True
    """Whether a newly created workspace may reach the internet."""

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        del ctx
        if ref is not None and ref.provider != 'e2b':
            return None
        return E2BWorkspaceBackend(
            ref=ref,
            template=self.template,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            env=self.env,
            metadata=self.metadata,
            allow_internet_access=self.allow_internet_access,
        )
