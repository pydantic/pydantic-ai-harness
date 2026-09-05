"""Notion capability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.notion._toolset import MCPToolsetClient, NotionToolset

_DESCRIPTION = "Search and read the authenticated user's Notion workspace, with explicitly selected write tools."


@dataclass
class Notion(AbstractCapability[AgentDepsT]):
    """Search and read Notion through its official hosted MCP server.

    The default toolset exposes a conservative discovery/read surface. Add mutation
    tool names explicitly with `mutations`. Authentication and MCP session state stay
    in Pydantic AI/FastMCP, including when a caller supplies a prebuilt client.
    """

    _: KW_ONLY

    client: MCPToolsetClient = field(repr=False)
    """Caller-owned OAuth client or in-process server for Notion's hosted MCP contract."""

    mutations: str | Sequence[str] = ()
    """Exact Notion MCP mutation tool names to expose, such as `notion-update-page`."""

    include_instructions: bool = True
    """Inject Notion identity, search-routing, and mutation guidance."""

    expected_identity: tuple[str, str] | None = None
    """Expected `(workspace_id, user_id)` for a restored connection or deferred mutation."""

    id: str | None = None
    """Optional stable capability and toolset ID.

    Notion tools have fixed names and one MCP client is one authenticated identity, so
    anonymous instances collide instead of merging their access policies. Set an ID for
    deferred loading or durable execution. Build a separate agent or dynamic toolset for
    each connected user rather than sharing an instance across users.
    """

    description: str | None = _DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    def __post_init__(self) -> None:
        self.mutations = NotionToolset.normalize_mutations(self.mutations)

    def get_toolset(self) -> NotionToolset[AgentDepsT]:
        """Build the Notion MCP toolset."""
        return NotionToolset[AgentDepsT](
            client=self.client,
            mutations=self.mutations,
            include_instructions=self.include_instructions,
            expected_identity=self.expected_identity,
            id=self.id,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: the capability holds a live authenticated MCP client."""
        return None
