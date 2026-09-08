"""Supply a Fly.io Sprite through the core sandbox capability hook."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT, RunContext
from sprites import SpritesClient

from pydantic_ai_harness._sandbox_provider import absolute_path
from pydantic_ai_harness.sprites._backend import SpriteSandboxBackend


@dataclass(kw_only=True)
class SpriteSandbox(AbstractCapability[AgentDepsT]):
    """Supply a persistent Fly.io Sprite as `ctx.sandbox`.

    Defaults to one named Sprite per conversation. An explicit `sprite_name` or
    run reference attaches to an existing Sprite and raises if it is missing.
    Acquisition is lazy; ending a run does not destroy or disconnect the Sprite.
    Lifecycle methods are explicit and callers finish in-flight commands first.
    """

    token: str | None = None
    """API token; defaults to SPRITE_TOKEN on first use."""
    sprite_name: str | None = None
    """Name of an existing Sprite to attach to."""
    base_url: str = 'https://api.sprites.dev'
    """Sprites API endpoint."""
    api_timeout: float = 30.0
    """SDK HTTP timeout in seconds; creation uses the SDK's 120-second timeout."""
    runtime: str | None = None
    """Runtime for newly created Sprites."""
    workdir: str | None = None
    """Absolute working directory inside the Sprite."""
    client: SpritesClient | None = None
    """Optional caller-owned client, which this capability never closes."""

    def __post_init__(self) -> None:
        self.workdir = absolute_path('workdir', self.workdir)
        if self.sprite_name is not None and self.runtime is not None:
            raise ValueError('runtime applies only to creation, not sprite_name attachment.')

    def get_sandbox(self, ctx: RunContext[AgentDepsT], *, ref: SandboxRef | None) -> SandboxBackend:
        """Return a configured backend without acquiring a Sprite."""
        existing = ref or (SandboxRef(sandbox_id=self.sprite_name) if self.sprite_name is not None else None)
        identity = ctx.conversation_id or ctx.run_id or ''
        return SpriteSandboxBackend(
            token=self.token,
            ref=existing,
            name=f'pydantic-ai-{hashlib.sha256(identity.encode()).hexdigest()[:32]}',
            base_url=self.base_url,
            api_timeout=self.api_timeout,
            runtime=self.runtime if existing is None else None,
            working_dir=self.workdir,
            client=self.client,
        )
