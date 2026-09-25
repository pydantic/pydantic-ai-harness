"""Capability that supplies a Fly.io Sprite sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_working_dir
from pydantic_ai_harness.sprites_sandbox._backend import SpritesSandboxBackend

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult
    from sprites import AsyncSpritesClient


class _SuppliedBackend(SpritesSandboxBackend):
    """The backend `SpritesSandbox` built for one run; the capability closes its client when the run ends."""


@dataclass(kw_only=True)
class SpritesSandbox(AbstractCapability[AgentDepsT]):
    """Supply a persistent [Fly.io Sprite](https://sprites.dev) workspace through `ctx.workspace`.

    A run with no reference creates a fresh Sprite. Pass a `WorkspaceRef` supplied by the
    application to attach to a Sprite managed elsewhere. Acquisition is lazy, and ending a run
    does not delete the Sprite. Commands run under `sh -c` in the Sprite's own shell environment.

    This capability supplies execution only. Compose it with tools or
    capabilities that consume
    [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace], such as `Coder`, `Shell`,
    and `FileSystem`.
    """

    client: AsyncSpritesClient | None = None
    """A caller-owned `sprites.AsyncSpritesClient`, which is never closed for you. When omitted,
    each run's backend creates one on first use from `SPRITE_TOKEN` and closes it when the run
    ends. Supply one to share its connections across runs, on one event loop."""

    runtime: str | None = None
    """Runtime for a newly created Sprite."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the Sprite's default."""

    env: Mapping[str, str] | None = None
    """Environment variables every command gets, on top of the Sprite's own; a command's `env` is layered on top."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                'defer_loading must be False for SpritesSandbox: a deferred capability never supplies the workspace.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        del ctx
        if ref is not None and ref.provider != 'sprites':
            return None
        return _SuppliedBackend(
            client=self.client,
            ref=ref,
            runtime=self.runtime,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        try:
            return await handler()
        finally:
            backend = innermost_backend(ctx.workspace)
            if isinstance(backend, _SuppliedBackend):
                await backend.aclose()
