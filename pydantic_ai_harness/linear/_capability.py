"""Linear MCP integration.

Wire contract, verified 2026-09-04:

- `https://mcp.linear.app/mcp` is Linear's Streamable HTTP endpoint.
- `https://mcp.linear.app/mcp/readonly` only exposes read tools.
- Both OAuth and API or OAuth access tokens are supported. Tokens use bearer authentication.

Source: https://linear.app/docs/mcp. Re-check the Setup and FAQ sections when changing
endpoint or authentication behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import AnyUrl
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Linear capability. Install it with: uv add "pydantic-ai-slim[mcp]"'
    ) from _import_error

LINEAR_MCP_URL = 'https://mcp.linear.app/mcp'
"""Linear's read-write Streamable HTTP MCP endpoint."""

LINEAR_READ_ONLY_MCP_URL = 'https://mcp.linear.app/mcp/readonly'
"""Linear's Streamable HTTP endpoint that only exposes read tools."""

_DEFAULT_DESCRIPTION = 'Find and manage work in Linear.'
_INSTRUCTIONS = 'Use Linear tools to find and manage work. Read before changing anything.'


@dataclass
class Linear(AbstractCapability[AgentDepsT]):
    """Linear issues, projects, comments, and related workspace data over Linear's hosted MCP server.

    Uses Pydantic AI's `MCPToolset` for transport, authentication, and lifecycle.
    """

    read_only: bool = True
    """Use Linear's endpoint that only exposes read tools. Set to `False` to enable mutations."""

    allowed_tools: Sequence[str] | None = None
    """Exact tool names to expose. `None` exposes every tool returned by the selected endpoint."""

    auth: httpx.Auth | Literal['oauth'] | str | None = field(default=None, repr=False)
    """OAuth, a bearer token, or custom `httpx.Auth`. `None` starts Linear's OAuth flow."""

    include_instructions: bool = True
    """Add short Linear usage guidance to the agent instructions."""

    client: MCPToolsetClient | MCPToolset[AgentDepsT] | None = field(default=None, repr=False)
    """`MCPToolset` client input or a prebuilt toolset. Prebuilt values own their authentication."""

    description: str | None = _DEFAULT_DESCRIPTION

    def __post_init__(self) -> None:
        if self.client is not None and self.auth is not None:
            is_url = isinstance(self.client, AnyUrl) or (
                isinstance(self.client, str) and urlsplit(self.client).scheme.lower() in ('http', 'https')
            )
            if isinstance(self.client, MCPToolset) or not is_url:
                raise UserError('`auth` cannot be combined with a prebuilt MCP client or toolset.')

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Linear MCP toolset, optionally wrapped by core's exact-name filter."""
        if isinstance(self.client, MCPToolset):
            toolset = self.client
        elif self.client is not None:
            is_url = isinstance(self.client, AnyUrl) or (
                isinstance(self.client, str) and urlsplit(self.client).scheme.lower() in ('http', 'https')
            )
            if is_url and urlsplit(str(self.client)).scheme.lower() != 'https':
                raise UserError('`client` URL must use HTTPS so Linear credentials are encrypted in transit.')
            auth = ('oauth' if self.auth is None else self.auth) if is_url else None
            toolset = MCPToolset[AgentDepsT](self.client, id=self.id, auth=auth)
        else:
            endpoint = LINEAR_READ_ONLY_MCP_URL if self.read_only else LINEAR_MCP_URL
            auth = self.auth
            if auth is None:
                auth = 'oauth'
            toolset = MCPToolset[AgentDepsT](endpoint, id=self.id, auth=auth)

        if self.allowed_tools is None:
            return toolset
        allowed = frozenset(self.allowed_tools)
        return toolset.filtered(lambda _ctx, tool_def: tool_def.name in allowed)

    def get_instructions(self) -> str | None:
        """Return concise guidance for using Linear tools."""
        return _INSTRUCTIONS if self.include_instructions else None

    @classmethod
    def from_spec(
        cls,
        *,
        read_only: bool = True,
        allowed_tools: list[str] | None = None,
        auth: Literal['oauth'] | str | None = None,
        include_instructions: bool = True,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
    ) -> Linear[AgentDepsT]:
        """Construct from serializable options, excluding the runtime-only `client` input."""
        return cls(
            read_only=read_only,
            allowed_tools=allowed_tools,
            auth=auth,
            include_instructions=include_instructions,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )
