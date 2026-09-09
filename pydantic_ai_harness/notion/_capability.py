"""Notion hosted MCP capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import is_read_only

try:
    from fastmcp.client.auth import OAuth
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Notion support with: uv add "pydantic-ai-harness[notion]"') from exc


@dataclass(kw_only=True)
class Notion(AbstractCapability[AgentDepsT]):
    """Use Notion's hosted tools with the permissions of the connected user."""

    description: str | None = 'Search and change Notion workspace content.'
    auth: Auth | str | None = field(default=None, repr=False)
    """OAuth access token, `'oauth'`, or HTTP authentication. Defaults to `NOTION_ACCESS_TOKEN`, then OAuth."""
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Notion connection and optional read-only selection."""
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or 'notion', include_instructions=self.include_instructions
            )
        else:
            auth = self.auth if self.auth is not None else environ.get('NOTION_ACCESS_TOKEN', 'oauth')
            if auth == 'oauth':
                auth = OAuth(additional_client_metadata={'token_endpoint_auth_method': 'none'})
            toolset = MCPToolset(
                'https://mcp.notion.com/mcp',
                id=self.id or 'notion',
                auth=auth,
                headers=None,
                include_instructions=self.include_instructions,
            )
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset
