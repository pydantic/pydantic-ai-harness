"""Logfire hosted MCP capability.

Provider contract, verified 2026-09-07:

- `https://logfire-us.pydantic.dev/mcp` and `https://logfire-eu.pydantic.dev/mcp` are the hosted
  Streamable HTTP endpoints.
- OAuth and API-key bearer tokens are both accepted. API keys carry scopes such as `project:read`,
  and Logfire checks them on every request.
- `project_list` returns the projects the credential can reach; project tools take a `project`
  argument in `organization/project` form.

Source: https://pydantic.dev/docs/logfire/guides/mcp-server/. Re-check the endpoint and
authentication sections before changing connection behavior.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Logfire MCP capability. '
        'Install it with: uv add "pydantic-ai-harness[logfire-mcp]"'
    ) from _import_error

if TYPE_CHECKING:
    from fastmcp import Client as FastMCPClient
    from fastmcp import FastMCP
    from fastmcp.client.transports import ClientTransport
    from mcp.server.fastmcp import FastMCP as FastMCP1Server

LOGFIRE_US_MCP_URL = 'https://logfire-us.pydantic.dev/mcp'
"""Logfire's hosted MCP endpoint for the US data region."""

LOGFIRE_EU_MCP_URL = 'https://logfire-eu.pydantic.dev/mcp'
"""Logfire's hosted MCP endpoint for the EU data region."""

_DEFAULT_DESCRIPTION = 'Query Logfire telemetry and manage dashboards, alerts, and issues.'


@dataclass
class LogfireMCP(AbstractCapability[AgentDepsT]):
    """Query Logfire telemetry and manage observability resources through Logfire's hosted MCP server.

    Logfire enforces access: an API key's project and scopes decide what the agent can read
    or change. The server's own tool descriptions and instructions guide the model; pass
    `allowed_tools` to narrow what it sees.
    """

    _: KW_ONLY

    id: str | None = None
    """Capability ID. Leave unset so two Logfire configurations do not merge."""

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    url: str = LOGFIRE_US_MCP_URL
    """MCP endpoint. Use `LOGFIRE_EU_MCP_URL` for EU data, or a self-hosted `/mcp` URL."""

    auth: Auth | Literal['oauth'] | str | None = field(default='oauth', repr=False)
    """`'oauth'` for browser login, a Logfire API key for headless use, a custom `httpx.Auth`, or `None`."""

    allowed_tools: list[str] | None = None
    """Exact MCP tool names to expose. `None` exposes every tool the server returns."""

    include_instructions: bool = True
    """Add the instructions the Logfire server sends on connect to the agent's instructions."""

    client: FastMCPClient[Any] | ClientTransport | FastMCP | FastMCP1Server | None = field(default=None, repr=False)
    """Prebuilt FastMCP client or transport, or an in-process server, used instead of `url`.

    It carries its own authentication, so `auth` is ignored.
    """

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Logfire MCP toolset, with an exact-name filter when configured."""
        client = self.client if self.client is not None else self.url
        auth = self.auth if self.client is None else None
        toolset: AbstractToolset[AgentDepsT] = MCPToolset(
            client, id=self.id or 'logfire-mcp', auth=auth, include_instructions=self.include_instructions
        )
        if self.allowed_tools is None:
            return toolset
        allowed_tools = frozenset(self.allowed_tools)
        return toolset.filtered(lambda _ctx, tool: tool.name in allowed_tools)

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
        url: str = LOGFIRE_US_MCP_URL,
        auth: Literal['oauth'] | str | None = 'oauth',
        allowed_tools: list[str] | None = None,
        include_instructions: bool = True,
    ) -> LogfireMCP[AgentDepsT]:
        """Construct from serializable options, excluding runtime client injection."""
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            url=url,
            auth=auth,
            allowed_tools=allowed_tools,
            include_instructions=include_instructions,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the agent-spec capability name."""
        return 'LogfireMCP'
