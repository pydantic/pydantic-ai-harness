"""Capability that supplies a Modal workspace to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from pydantic_ai_harness.modal_workspace._backend import (
    DEFAULT_APP_NAME,
    DEFAULT_IMAGE,
    DEFAULT_SANDBOX_TIMEOUT,
    ModalWorkspaceBackend,
)


@dataclass(kw_only=True)
class ModalWorkspace(AbstractCapability[AgentDepsT]):
    """Supply a Modal workspace through `ctx.workspace`.

    A run with an explicit `WorkspaceRef` attaches to that workspace. Without a reference, the
    first workspace operation creates a fresh Modal sandbox. The application owns persistence of
    the returned reference and the lifecycle of the native Modal handle.
    """

    image: str = DEFAULT_IMAGE
    """Registry image used when creating a workspace."""

    name: str | None = None
    """Optional Modal name used only when creating a workspace."""

    app_name: str = DEFAULT_APP_NAME
    """Modal app used when creating a workspace."""

    create_app_if_missing: bool = True
    """Whether Modal may create the app."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Server-side lifetime for a newly created workspace, in seconds."""

    workdir: str | None = None
    """Absolute working directory for a newly created workspace."""

    env: Mapping[str, str] | None = None
    """Environment variables configured on a newly created workspace."""

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build a backend without performing Modal I/O."""
        del ctx
        if ref is not None and ref.provider != 'modal':
            return None
        return ModalWorkspaceBackend(
            ref=ref,
            name=self.name,
            image=self.image,
            app_name=self.app_name,
            create_app_if_missing=self.create_app_if_missing,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            env=self.env,
        )
