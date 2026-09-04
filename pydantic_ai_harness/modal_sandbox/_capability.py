"""Capability that supplies a Modal sandbox to an agent run."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness._sandbox_provider import absolute_path
from pydantic_ai_harness.modal_sandbox._backend import (
    DEFAULT_APP_NAME,
    DEFAULT_IMAGE,
    DEFAULT_SANDBOX_TIMEOUT,
    ModalSandboxBackend,
)


def _sandbox_name(identity: str) -> str:
    """Return a Modal-safe, deterministic name for one conversation."""
    digest = hashlib.sha256(identity.encode()).hexdigest()[:32]
    return f'pydantic-ai-{digest}'


@dataclass(kw_only=True)
class ModalSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Modal](https://modal.com) sandbox through `ctx.sandbox`.

    One named sandbox is created or reused per conversation, so a follow-up run continues in
    the workspace the previous one left behind. The name is derived from the conversation, which
    also makes creation safe to retry across durable workers: a retry attaches to the sandbox the
    first attempt made rather than provisioning a second one.

    Nothing here terminates a sandbox. A conversation can span many runs, so the end of a run is
    not the end of the workspace; Modal reaps an idle sandbox at `sandbox_timeout`. Raise that
    for longer work, or terminate one yourself through `result.sandbox`.

    Set `sandbox_id` to attach to an existing sandbox instead.

    This capability supplies execution only. Compose it with tools or capabilities
    that consume [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
    """

    image: str = DEFAULT_IMAGE
    """Registry image used for an owned sandbox."""

    sandbox_id: str | None = None
    """Existing Modal sandbox ID to attach to without terminating it."""

    app_name: str = DEFAULT_APP_NAME
    """Deployed Modal app that owns named sandboxes."""

    create_app_if_missing: bool = True
    """Whether Modal may create the app when it does not exist."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Server-side lifetime backstop for an owned sandbox, in seconds."""

    workdir: str | None = None
    """Absolute working directory for an owned sandbox, or Modal's default."""

    env: Mapping[str, str] | None = None
    """Environment variables configured on an owned sandbox."""

    def __post_init__(self) -> None:
        if self.sandbox_timeout <= 0:
            raise ValueError(f'sandbox_timeout must be a positive integer, got {self.sandbox_timeout!r}.')
        self.workdir = absolute_path('workdir', self.workdir)
        if self.env is not None:
            self.env = dict(self.env)
        if self.sandbox_id is None:
            return
        conflicts = [
            name
            for name, value, default in (
                ('image', self.image, DEFAULT_IMAGE),
                ('app_name', self.app_name, DEFAULT_APP_NAME),
                ('create_app_if_missing', self.create_app_if_missing, True),
                ('sandbox_timeout', self.sandbox_timeout, DEFAULT_SANDBOX_TIMEOUT),
                ('workdir', self.workdir, None),
                ('env', self.env, None),
            )
            if value != default
        ]
        if conflicts:
            raise ValueError(
                f'{", ".join(conflicts)} only apply when creating a sandbox, but `sandbox_id` '
                'attaches to an existing one. Remove them, or drop `sandbox_id` to create a sandbox.'
            )

    def get_sandbox(self, ctx: RunContext[AgentDepsT], *, ref: SandboxRef | None) -> SandboxBackend:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None:
            # An environment the run was pointed at explicitly, or one an earlier run left behind.
            return ModalSandboxBackend(ref=ref)
        if self.sandbox_id is not None:
            return ModalSandboxBackend(ref=SandboxRef(sandbox_id=self.sandbox_id))
        return ModalSandboxBackend(
            name=_sandbox_name(ctx.conversation_id or ctx.run_id or ''),
            image=self.image,
            app_name=self.app_name,
            create_app_if_missing=self.create_app_if_missing,
            sandbox_timeout=self.sandbox_timeout,
            workdir=self.workdir,
            env=self.env,
        )
