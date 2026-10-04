"""Deprecated pre-rename name for `PydanticAIDocs`, kept for agent-spec compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.pydantic_ai_docs._capability import PydanticAIDocs


# `init=False` inherits `PydanticAIDocs.__init__`, whose annotations resolve against that module's
# globals when the agent-spec schema is built; a regenerated `__init__` would resolve them here (#552).
@dataclass(init=False)
class PyaiDocs(PydanticAIDocs[AgentDepsT]):
    """Deprecated name for `PydanticAIDocs`.

    Keeps the pre-rename `'PyaiDocs'` agent-spec block name, so specs saved before the
    rename still load when this class is passed via `custom_capability_types`. Specs
    saved through this class keep the old block name; migrate to `PydanticAIDocs` to
    save under the new one.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'PyaiDocs'
