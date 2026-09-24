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
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import MCPClientFunc, per_run

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Composio support with: uv add "pydantic-ai-harness[composio]"') from exc


@dataclass(kw_only=True)
class Composio(AbstractCapability[AgentDepsT]):
    """Give an agent the tools of a Composio session.

    Create or restore the session with Composio's SDK, then pass its MCP URL and headers.
    """

    url: str | None = None
    """The URL returned by `session.mcp.url`. Required unless `client` is supplied."""
    headers: Mapping[str, str | None] | None = field(default=None, repr=False)
    """The headers returned by `session.mcp.headers`; unset values are omitted."""
    description: str | None = 'Discover and use connected applications through Composio.'
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | MCPClientFunc[AgentDepsT] | None = field(default=None, repr=False)
    """Your own MCP client or transport, or a function that returns one for each run.

    `url` and `headers` are ignored when it is set. A function can pick the current user's session
    from `ctx.deps`; returning `None` gives that run no Composio tools.
    """

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the session connection using Pydantic AI's MCP lifecycle."""
        if self.client is not None:
            return per_run(self.client, self._from_client, id=self.id or 'composio')
        if self.url is None:
            raise ValueError('Provide the Composio session URL or a configured client.')
        return MCPToolset(
            self.url,
            headers={key: value for key, value in (self.headers or {}).items() if value is not None},
            id=self.id or 'composio',
            include_instructions=self.include_instructions,
        )

    def _from_client(self, client: MCPToolsetClient) -> MCPToolset[AgentDepsT]:
        return MCPToolset(client, id=self.id or 'composio', include_instructions=self.include_instructions)
