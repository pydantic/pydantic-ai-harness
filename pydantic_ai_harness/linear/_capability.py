"""Linear hosted MCP capability.

Provider contract, verified 2026-09-04:

- `https://mcp.linear.app/mcp` is the primary Streamable HTTP endpoint.
- `https://mcp.linear.app/mcp/readonly` exposes read tools only.
- Both OAuth and bearer tokens are supported.

Source: https://linear.app/docs/mcp. Re-check the endpoint and authentication
sections before changing connection or access behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from httpx import Auth
from pydantic import AnyUrl
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Linear capability. Install it with: uv add "pydantic-ai-harness[linear]"'
    ) from _import_error

LINEAR_MCP_URL = 'https://mcp.linear.app/mcp'
"""Linear's read-write Streamable HTTP MCP endpoint."""

LINEAR_READ_ONLY_MCP_URL = 'https://mcp.linear.app/mcp/readonly'
"""Linear's Streamable HTTP endpoint that exposes read tools only."""

_DEFAULT_DESCRIPTION = 'Use Linear issues, projects, and teams.'
_READ_INSTRUCTIONS = (
    'Use the Linear tools for issues, projects, and teams. Prefer specific queries. Use identifiers returned by '
    'Linear or supplied by the user instead of guessing from names.'
)
_WRITE_INSTRUCTIONS = (
    ' Before changing Linear data, verify the requested action and target. Before creating, search for an existing '
    'match when a read tool is available.'
)


@dataclass
class Linear(AbstractCapability[AgentDepsT]):
    """Read or update Linear work through Linear's hosted MCP server.

    The default connection uses Linear's documented read-only endpoint. Pass
    `auth='oauth'` or a bearer token to authenticate. Set `read_only=False` to
    select the read-write endpoint, or inject a client or `MCPToolset` when the
    caller owns connection setup.
    """

    _: KW_ONLY

    id: str | None = None
    """Capability ID. Leave unset so duplicate Linear configurations do not merge."""

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    read_only: bool = True
    """Use Linear's server-enforced read-only endpoint for the default connection."""

    auth: Auth | Literal['oauth'] | str | None = field(default=None, repr=False)
    """OAuth, bearer-token, or custom HTTP auth for URL connections."""

    allowed_tools: Sequence[str] | None = None
    """Exact MCP tool names to expose. `None` exposes every tool returned by the endpoint."""

    include_instructions: bool = True
    """Add short Linear usage guidance to the model instructions."""

    client: MCPToolsetClient | MCPToolset[AgentDepsT] | None = field(default=None, repr=False)
    """Injected MCP connection or toolset.

    A prebuilt `MCPToolset` or FastMCP client keeps its own endpoint and
    authentication. Other URL-shaped clients receive `auth`. `allowed_tools`
    still applies to injected connections.
    """

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Linear MCP toolset, with an exact-name filter when configured."""
        if isinstance(self.client, MCPToolset):
            if self.auth is not None:
                raise UserError('`auth` cannot be used with a prebuilt `MCPToolset`; configure auth on its client.')
            toolset: AbstractToolset[AgentDepsT] = self.client
        else:
            client = (
                self.client
                if self.client is not None
                else (LINEAR_READ_ONLY_MCP_URL if self.read_only else LINEAR_MCP_URL)
            )
            if isinstance(client, (str, AnyUrl)):
                parsed_url = urlsplit(str(client))
                url_scheme = parsed_url.scheme.lower()
            else:
                parsed_url = None
                url_scheme = None
            is_http_url = url_scheme in ('http', 'https')
            if (
                is_http_url
                and parsed_url is not None
                and (parsed_url.username is not None or parsed_url.password is not None)
            ):
                raise UserError('Linear MCP client URLs must not contain credentials; pass `auth` separately.')
            if self.auth is not None and url_scheme == 'http':
                raise UserError('Authenticated Linear MCP URLs must use HTTPS.')
            if self.auth is not None and not is_http_url:
                raise UserError('`auth` cannot be used with a prebuilt client; configure auth on that client.')
            if is_http_url and parsed_url is not None:
                client = urlunsplit(
                    (url_scheme, parsed_url.netloc, parsed_url.path, parsed_url.query, parsed_url.fragment)
                )
            toolset = MCPToolset(
                client,
                id=self.id or 'linear',
                auth=self.auth if is_http_url else None,
            )

        if self.allowed_tools is None:
            return toolset
        allowed_tools = frozenset((self.allowed_tools,) if isinstance(self.allowed_tools, str) else self.allowed_tools)
        return toolset.filtered(lambda _ctx, tool: tool.name in allowed_tools)

    def get_instructions(self) -> str | None:
        """Return concise provider guidance."""
        if not self.include_instructions:
            return None
        connection_may_write = self.client is not None or not self.read_only
        return _READ_INSTRUCTIONS + (_WRITE_INSTRUCTIONS if connection_may_write else '')

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
        read_only: bool = True,
        auth: Literal['oauth'] | str | None = None,
        allowed_tools: Sequence[str] | None = None,
        include_instructions: bool = True,
    ) -> Linear[AgentDepsT]:
        """Construct from serializable options, excluding runtime client injection."""
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            read_only=read_only,
            auth=auth,
            allowed_tools=allowed_tools,
            include_instructions=include_instructions,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the agent-spec capability name."""
        return 'Linear'
