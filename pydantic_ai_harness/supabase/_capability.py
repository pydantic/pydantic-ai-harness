"""Supabase hosted MCP capability."""

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
    raise ImportError('Install Supabase support with: uv add "pydantic-ai-harness[supabase]"') from exc

from urllib.parse import urlencode


@dataclass(kw_only=True)
class Supabase(AbstractCapability[AgentDepsT]):
    """Connect to Supabase using its native project, feature, and read-only settings."""

    description: str | None = 'Use Supabase project and account tools.'
    auth: Auth | str | None = field(default=None, repr=False)
    """PAT, `'oauth'`, or HTTP authentication. Defaults to `SUPABASE_ACCESS_TOKEN`, then OAuth."""
    read_only: bool = False
    """Use the server's native read-only mode. A custom client is filtered by `readOnlyHint` instead."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """
    project_ref: str | None = None
    """Native project selection. Omit to retain account-level tools."""
    features: list[str] | None = None
    """Native feature groups. `None` keeps the server defaults."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Supabase connection and optional read-only selection."""
        query: dict[str, str] = {}
        if self.project_ref is not None:
            query['project_ref'] = self.project_ref
        if self.features is not None:
            query['features'] = ','.join(self.features)
        if self.read_only:
            query['read_only'] = 'true'
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or 'supabase', include_instructions=self.include_instructions
            )
        else:
            toolset = MCPToolset(
                'https://mcp.supabase.com/mcp' + ('?' + urlencode(query) if query else ''),
                id=self.id or 'supabase',
                auth=self.auth if self.auth is not None else environ.get('SUPABASE_ACCESS_TOKEN', 'oauth'),
                headers=None,
                include_instructions=self.include_instructions,
            )
        if self.read_only and self.client is not None:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset
