"""Capability that supplies a Fly.io Sprite workspace to an agent run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from pydantic_ai_harness.sprites._backend import SpriteWorkspaceBackend

if TYPE_CHECKING:
    from sprites import SpritesClient


@dataclass(kw_only=True)
class SpriteWorkspace(AbstractCapability[AgentDepsT]):
    """Supply a persistent [Fly.io Sprite](https://sprites.dev) workspace through `ctx.workspace`.

    A run with no reference creates a fresh Sprite. Pass a `WorkspaceRef` supplied by the
    application to attach to a Sprite managed elsewhere. Acquisition is lazy, and ending a run
    does not disconnect or destroy the Sprite.

    This capability supplies execution only. Compose it with tools or
    capabilities that consume
    [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace].
    """

    client: SpritesClient | None = None
    """A caller-owned `sprites.SpritesClient`. When omitted, the backend creates one on first use
    from `token` (or `SPRITE_TOKEN`) and closes it again on `disconnect`; supply one to own its
    lifecycle, and the backend never closes it."""

    token: str | None = None
    """API token for a backend-owned client; defaults to `SPRITE_TOKEN` on first use."""

    base_url: str = 'https://api.sprites.dev'
    """Sprites API endpoint for a backend-owned client."""

    api_timeout: float = 30.0
    """HTTP timeout in seconds for a backend-owned client; creation uses the SDK's own timeout."""

    runtime: str | None = None
    """Runtime for a newly created Sprite."""

    workdir: str | None = None
    """Absolute working directory for commands and relative filesystem paths."""

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        del ctx
        if ref is not None and ref.provider != 'sprites':
            return None
        return SpriteWorkspaceBackend(
            client=self.client,
            ref=ref,
            token=self.token,
            base_url=self.base_url,
            api_timeout=self.api_timeout,
            runtime=self.runtime,
            working_dir=self.workdir,
        )
