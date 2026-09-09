"""GitHub hosted MCP capability."""

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
    raise ImportError('Install GitHub support with: uv add "pydantic-ai-harness[github]"') from exc


GITHUB_MCP_URL = 'https://api.githubcopilot.com/mcp/'


@dataclass(kw_only=True)
class GitHub(AbstractCapability[AgentDepsT]):
    """Use GitHub's hosted tools with the permissions of the connected credential."""

    description: str | None = 'Read and change GitHub resources.'
    auth: Auth | str | None = field(default=None, repr=False)
    """PAT or HTTP authentication. Defaults to `GITHUB_TOKEN`."""
    read_only: bool = False
    """Use the server's native read-only mode. A custom client is filtered by `readOnlyHint` instead."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """
    url: str = GITHUB_MCP_URL
    """Hosted endpoint, including GitHub Enterprise Cloud data-residency endpoints."""
    toolsets: list[str] | None = None
    """Native GitHub toolsets. `None` keeps the server defaults."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the GitHub connection and optional read-only selection."""
        headers = {'X-MCP-Readonly': 'true'} if self.read_only else {}
        if self.toolsets is not None:
            headers['X-MCP-Toolsets'] = ','.join(self.toolsets)
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or 'github', include_instructions=self.include_instructions
            )
        else:
            toolset = MCPToolset(
                self.url,
                id=self.id or 'github',
                auth=self.auth if self.auth is not None else environ.get('GITHUB_TOKEN'),
                headers=headers,
                include_instructions=self.include_instructions,
            )
        if self.read_only and self.client is not None:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset
