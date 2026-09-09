"""Linear hosted MCP capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import is_read_only

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Linear support with: uv add "pydantic-ai-harness[linear]"') from exc


@dataclass(kw_only=True)
class Linear(AbstractCapability[AgentDepsT]):
    """Use Linear's hosted tools with the permissions of the connected user."""

    description: str | None = 'Use Linear issues, projects, and teams.'
    auth: Auth | str | None = field(default=None, repr=False)
    """API key, OAuth token, `'oauth'`, or HTTP authentication. Defaults to `LINEAR_ACCESS_TOKEN`, then OAuth."""
    read_only: bool = False
    """Use Linear's read-only endpoint. A custom client is filtered by `readOnlyHint` instead."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Linear connection and optional read-only selection."""
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or 'linear', include_instructions=self.include_instructions
            )
        else:
            toolset = MCPToolset(
                'https://mcp.linear.app/mcp/readonly' if self.read_only else 'https://mcp.linear.app/mcp',
                id=self.id or 'linear',
                auth=self.auth if self.auth is not None else environ.get('LINEAR_ACCESS_TOKEN', 'oauth'),
                headers=None,
                include_instructions=self.include_instructions,
            )
        if self.read_only and self.client is not None:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset
