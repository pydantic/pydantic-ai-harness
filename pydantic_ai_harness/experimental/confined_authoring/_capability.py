"""Confined authoring capability: let an agent author its own sandboxed tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset, CombinedToolset

from pydantic_ai_harness.experimental.confined_authoring._slots import (
    InjectedFunction,
    render_function_catalog,
)
from pydantic_ai_harness.experimental.confined_authoring._store import SlotStore
from pydantic_ai_harness.experimental.confined_authoring._toolset import authoring_toolsets

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_monty import ResourceLimits


_GUIDANCE_PREFIX = (
    'You can author your own tools with `author_tool_slot(name, description, code, parameters, uses, returns)`. '
    'A tool slot is a Monty script (a subset of Python): it reads each declared parameter as a bound variable, '
    'calls the injected functions it lists in `uses` (each is async -- `await` it and use its result), and its '
    'final expression becomes the tool return value. Only the functions you list in `uses` are reachable; there '
    'is no import, filesystem, environment, clock, subprocess, or network access inside the sandbox. A slot is '
    'validated when authored and, on success, becomes callable on your next run, not the current one. Use '
    '`list_tool_slots` and `disable_tool_slot` to manage them.\n\n'
    'Injected functions available to authored slots:\n\n'
)


@dataclass
class ConfinedAuthoring(AbstractCapability[AgentDepsT]):
    """Let an agent author, validate, and persist its own sandboxed tools.

    Exposes `author_tool_slot`, `list_tool_slots`, and `disable_tool_slot`, and
    serves the slots the agent has authored as real tools. Each authored tool is
    a Monty script that runs in a sandbox and can call only the injected
    functions its slot declares -- default-deny, with no ambient import,
    filesystem, environment, clock, subprocess, or network access. Slots are
    validated before they are served (typed arguments, a static type-check
    against the declared functions and parameters, a missing-`await` check, and a
    return-type check), persisted to a manifest, and reloaded on the next run.

    Unlike
    [`RuntimeAuthoring`][pydantic_ai_harness.experimental.authoring.RuntimeAuthoring],
    which imports arbitrary Python capabilities into the host process, an authored
    slot never runs host Python: its only reach into the host is the injected
    functions the host provided and the slot declared. That makes it usable when
    the authoring model is untrusted, when slots must be isolated per tenant, or
    when a host deliberately removes broad tools (shell, file writes) and lets the
    injected-function allowlist be the only escape hatch.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent, RunContext
    from pydantic_ai_harness.experimental.confined_authoring import ConfinedAuthoring, InjectedFunction


    async def http_get(ctx: RunContext[None], kwargs: dict[str, object]) -> object:
        return {'status': 200, 'url': kwargs['url']}


    authoring = ConfinedAuthoring[None](
        directory=Path('.slots'),
        functions=[
            InjectedFunction(
                name='http_get',
                call=http_get,
                parameters={'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
                returns={'type': 'object'},
                description='Fetch a URL and return the response.',
            )
        ],
    )
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[authoring])
    ```

    Because the capability serves its own slots, an authored tool is live on the
    next `agent.run(...)` with nothing to thread through -- the toolset reloads
    the manifest at the start of each run.
    """

    directory: Path
    """Directory holding the `slots.json` manifest of authored slots."""

    functions: Sequence[InjectedFunction[AgentDepsT]] = ()
    """The capability-scoped pool of injected functions authored slots may call. A slot reaches the host only
    through the subset of this pool it declares; nothing else is available inside a slot's sandbox."""

    guidance: str | None = None
    """System-prompt guidance on authoring, with the available-functions catalog appended. Cache-stable. Leave
    `None` for the default, or set `''` to omit guidance entirely."""

    max_retries: int = 3
    """Maximum retries for a served slot tool when its sandbox execution raises."""

    resource_limits: ResourceLimits | None = None
    """Sandbox limits for slot execution. `None` uses a memory/allocation backstop plus a default
    30s cap on in-sandbox compute. A partial mapping merges onto that backstop, overriding only the
    caps it names."""

    @property
    def store(self) -> SlotStore[AgentDepsT]:
        """The disk-backed slot store over `directory` and this capability's function pool."""
        return SlotStore[AgentDepsT](self.directory, self.functions)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance plus the catalog of injected functions slots may call."""
        if self.guidance is not None:
            return self.guidance or None
        return _GUIDANCE_PREFIX + render_function_catalog(self.functions)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """The authoring tools plus the served slots, over this capability's store."""
        toolsets = authoring_toolsets(self.store, max_retries=self.max_retries, resource_limits=self.resource_limits)
        return CombinedToolset[AgentDepsT](toolsets)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: the capability holds live injected functions and a disk-backed store."""
        return None
