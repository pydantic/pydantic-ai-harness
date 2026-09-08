"""LogfireMCP hosted MCP capability."""

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
    raise ImportError('Install LogfireMCP support with: uv add "pydantic-ai-harness[logfire-mcp]"') from exc

from datetime import datetime, timezone

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.tools import RunContext

LOGFIRE_US_MCP_URL = 'https://logfire-us.pydantic.dev/mcp'
LOGFIRE_EU_MCP_URL = 'https://logfire-eu.pydantic.dev/mcp'

_INSTRUCTIONS = (
    'Timestamps in tool schemas and examples, and project creation timestamps, are examples or metadata rather than '
    'the current time. Query transport bounds apply in addition to SQL time predicates and default to a short '
    'window, so widen them explicitly when needed. Create a Logfire link only when the user asks for one.'
)


@dataclass(kw_only=True)
class LogfireMCP(AbstractCapability[AgentDepsT]):
    """Query Logfire telemetry and manage observability resources through its hosted tools."""

    description: str | None = 'Query Logfire telemetry and manage observability resources.'
    auth: Auth | str | None = field(default=None, repr=False)
    """API key, `'oauth'`, or HTTP authentication. Defaults to `LOGFIRE_API_KEY`, then OAuth."""
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Include server instructions, query guidance, and the current UTC time."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """
    url: str = LOGFIRE_US_MCP_URL
    """Hosted US, hosted EU, or self-hosted MCP endpoint."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the LogfireMCP connection and optional read-only selection."""
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or 'logfire-mcp', include_instructions=self.include_instructions
            )
        else:
            toolset = MCPToolset(
                self.url,
                id=self.id or 'logfire-mcp',
                auth=self.auth if self.auth is not None else environ.get('LOGFIRE_API_KEY', 'oauth'),
                headers=None,
                include_instructions=self.include_instructions,
            )
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return query guidance and the current UTC time."""
        if not self.include_instructions:
            return None
        return [_INSTRUCTIONS, self._current_utc]

    def _current_utc(self, ctx: RunContext[AgentDepsT]) -> str:
        current_utc = datetime.now(timezone.utc).isoformat(timespec='seconds')
        return f'Current UTC time is `{current_utc}`.'
