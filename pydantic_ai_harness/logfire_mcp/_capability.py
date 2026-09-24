"""LogfireMCP hosted MCP capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import MCPAuth, MCPAuthFunc, MCPClientFunc, is_read_only, per_run_auth, per_run_client

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install LogfireMCP support with: uv add "pydantic-ai-harness[logfire-mcp]"') from exc

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.messages import ModelRequest
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
    auth: MCPAuth | MCPAuthFunc[AgentDepsT] | None = field(default=None, repr=False)
    """A Logfire API key, an `httpx.Auth`, or a function of the run context that returns one.

    Unset, it uses `LOGFIRE_API_KEY`. If the function returns `None`, that run has no Logfire tools.
    """
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Include server instructions, query guidance, and the current UTC time."""
    client: MCPToolsetClient | MCPClientFunc[AgentDepsT] | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport, or a callable that returns one for each run.

    The supplied client owns its URL, authentication, and server configuration.
    """
    url: str = LOGFIRE_US_MCP_URL
    """Hosted US, hosted EU, or self-hosted MCP endpoint."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the LogfireMCP connection and optional read-only selection."""
        id = self.id or 'logfire-mcp'
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = per_run_client(self.client, self._from_client, id=id)
        else:
            toolset = per_run_auth(self.auth, self._connect, id=id)
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset

    def _from_client(self, client: MCPToolsetClient) -> MCPToolset[AgentDepsT]:
        return MCPToolset(client, id=self.id or 'logfire-mcp', include_instructions=self.include_instructions)

    def _connect(self, auth: MCPAuth | None) -> MCPToolset[AgentDepsT]:
        if auth is None:
            auth = environ.get('LOGFIRE_API_KEY')
        if auth is None:
            raise UserError('Set `LOGFIRE_API_KEY` or pass `auth` to connect to Logfire.')
        return MCPToolset(
            self.url,
            id=self.id or 'logfire-mcp',
            auth=auth,
            headers=None,
            include_instructions=self.include_instructions,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return query guidance and the current UTC time."""
        if not self.include_instructions:
            return None
        return [_INSTRUCTIONS, self._current_utc]

    def _current_utc(self, ctx: RunContext[AgentDepsT]) -> str | None:
        # The run stamps each request as it is made, so this needs no clock read of its own, which
        # Temporal's workflow sandbox would reject.
        stamps = [
            message.timestamp for message in ctx.messages if isinstance(message, ModelRequest) and message.timestamp
        ]
        if not stamps:
            return None
        return f'Current UTC time is `{max(stamps).isoformat(timespec="seconds")}`.'
