"""Capability that supplies an E2B sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness._sandbox_provider import absolute_path
from pydantic_ai_harness.e2b_sandbox._backend import (
    DEFAULT_SANDBOX_TIMEOUT,
    E2BSandboxBackend,
)

_CONVERSATION_METADATA_KEY = 'pydantic-ai-conversation-id'


@dataclass(kw_only=True)
class E2BSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [E2B](https://e2b.dev) sandbox through `ctx.sandbox`.

    One sandbox is created or reused per conversation, marked with the conversation id in E2B
    metadata, so a follow-up run continues in the workspace the previous one left behind. Reusing
    an existing match also makes creation safe to retry across durable workers: a retry attaches
    to the sandbox the first attempt made rather than provisioning a second one.

    Nothing here kills a sandbox. A conversation can span many runs, so the end of a run is not
    the end of the workspace; E2B reaps an idle sandbox at `sandbox_timeout`. Raise that for
    longer work, or kill one yourself through `result.sandbox`.

    Set `sandbox_id` to attach to an environment managed elsewhere instead.

    This capability supplies execution only. Compose it with tools or
    capabilities that consume
    [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
    """

    template: str | None = None
    """E2B template name or ID for an owned sandbox."""

    sandbox_id: str | None = None
    """Existing E2B sandbox ID to attach to without killing it."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Server-side lifetime backstop for an owned sandbox, in seconds."""

    workdir: str | None = None
    """Absolute working directory for commands and relative filesystem paths."""

    env: Mapping[str, str] | None = None
    """Environment variables configured on an owned sandbox."""

    metadata: Mapping[str, str] | None = None
    """Metadata added to an owned sandbox."""

    allow_internet_access: bool = True
    """Whether an owned sandbox may reach the internet."""

    def __post_init__(self) -> None:
        if self.sandbox_timeout <= 0:
            raise ValueError(f'sandbox_timeout must be a positive integer, got {self.sandbox_timeout!r}.')
        self.workdir = absolute_path('workdir', self.workdir)
        if self.env is not None:
            self.env = dict(self.env)
        if self.metadata is not None:
            self.metadata = dict(self.metadata)
            if _CONVERSATION_METADATA_KEY in self.metadata:
                raise ValueError(
                    f'metadata key {_CONVERSATION_METADATA_KEY!r} is reserved: it is how a follow-up '
                    "run finds the conversation's sandbox again."
                )
        if self.sandbox_id is None:
            return
        conflicts = [
            name
            for name, value, default in (
                ('template', self.template, None),
                ('sandbox_timeout', self.sandbox_timeout, DEFAULT_SANDBOX_TIMEOUT),
                ('env', self.env, None),
                ('metadata', self.metadata, None),
                ('allow_internet_access', self.allow_internet_access, True),
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
            return E2BSandboxBackend(ref=ref, working_dir=self.workdir)
        if self.sandbox_id is not None:
            return E2BSandboxBackend(ref=SandboxRef(sandbox_id=self.sandbox_id), working_dir=self.workdir)
        identity = {_CONVERSATION_METADATA_KEY: ctx.conversation_id or ctx.run_id or ''}
        return E2BSandboxBackend(
            identity=identity,
            template=self.template,
            sandbox_timeout=self.sandbox_timeout,
            working_dir=self.workdir,
            env=self.env,
            metadata={**(self.metadata or {}), **identity},
            allow_internet_access=self.allow_internet_access,
        )
