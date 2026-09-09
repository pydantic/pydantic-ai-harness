"""Connect Composio sessions over MCP.

Composio documents `session.mcp.url` and `session.mcp.headers` as the connection
contract (verified 2026-09-08). Re-check session setup before changing transport:
https://docs.composio.dev/docs/sessions-via-mcp
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Composio support with: uv add "pydantic-ai-harness[composio]"') from exc


@dataclass(kw_only=True)
class Composio(AbstractCapability[AgentDepsT]):
    """Use the hosted tools of a caller-configured Composio session.

    Create or restore a session with Composio's SDK, then pass its MCP URL and
    headers. Composio owns connected accounts, tool selection, and session state.
    """

    url: str | None = None
    """The URL returned by `session.mcp.url`. Required unless `client` is supplied."""
    headers: Mapping[str, str | None] | None = field(default=None, repr=False)
    """The headers returned by `session.mcp.headers`; unset values are omitted."""
    description: str | None = 'Discover and use connected applications through Composio.'
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a configured MCP client or transport.

    The supplied client owns its URL and authentication; `url` and `headers`
    are not applied to it.
    """

    def get_toolset(self) -> MCPToolset[AgentDepsT]:
        """Build the session connection using Pydantic AI's MCP lifecycle."""
        if self.client is not None:
            return MCPToolset(self.client, id=self.id or 'composio', include_instructions=self.include_instructions)
        if self.url is None:
            raise ValueError('Provide the Composio session URL or a configured client.')
        return MCPToolset(
            self.url,
            headers={key: value for key, value in (self.headers or {}).items() if value is not None},
            id=self.id or 'composio',
            include_instructions=self.include_instructions,
        )
