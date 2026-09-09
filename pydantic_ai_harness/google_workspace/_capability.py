"""Google Workspace hosted MCP capability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from os import environ
from typing import Literal

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset

from pydantic_ai_harness._mcp import is_read_only

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Google Workspace capability. '
        'Install it with: uv add "pydantic-ai-harness[google-workspace]"'
    ) from _import_error

GoogleWorkspaceService = Literal['gmail', 'drive', 'docs', 'sheets', 'slides', 'calendar', 'chat', 'people']
"""A Google Workspace product Google serves over MCP."""

_MCP_URLS: dict[str, str] = {
    'gmail': 'https://gmailmcp.googleapis.com/mcp/v1',
    'drive': 'https://drivemcp.googleapis.com/mcp/v1',
    'docs': 'https://docsmcp.googleapis.com/mcp/v1',
    'sheets': 'https://sheetsmcp.googleapis.com/mcp/v1',
    'slides': 'https://slidesmcp.googleapis.com/mcp/v1',
    'calendar': 'https://calendarmcp.googleapis.com/mcp/v1',
    'chat': 'https://chatmcp.googleapis.com/mcp/v1',
    'people': 'https://people.googleapis.com/mcp/v1',
}

_DEFAULT_DESCRIPTION = 'Use Gmail, Calendar, Drive, and the other Google Workspace products.'


@dataclass
class GoogleWorkspace(AbstractCapability[AgentDepsT]):
    """Connect an agent to Google's hosted Workspace MCP servers.

    The default serves every tool Google publishes for the selected products, including the ones that
    send, change, and delete; the token's scopes decide what the agent can read or change.
    """

    services: GoogleWorkspaceService | Sequence[GoogleWorkspaceService]
    """Workspace products to expose, such as `'gmail'` or `['gmail', 'calendar']`."""

    _: KW_ONLY

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: Auth | str | None = field(default=None, repr=False)
    """A Google OAuth bearer token, or a custom `httpx.Auth`. Defaults to `$GOOGLE_ACCESS_TOKEN`."""

    read_only: bool = False
    """Expose only the tools Google marks read-only."""

    include_instructions: bool = True
    """Forward the server instructions to the agent."""

    def __post_init__(self) -> None:
        """Normalize `services` to a tuple of products that have an endpoint."""
        self.services = (self.services,) if isinstance(self.services, str) else tuple(dict.fromkeys(self.services))
        if not self.services:
            raise UserError('Google Workspace needs at least one service.')
        for service in self.services:
            if service not in _MCP_URLS:
                raise UserError(f'Unknown Google Workspace service {service!r}; expected one of {sorted(_MCP_URLS)}.')

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build one product-prefixed MCP connection per selected service."""
        auth = self.auth if self.auth is not None else environ.get('GOOGLE_ACCESS_TOKEN')
        if auth is None or auth == '':
            raise UserError('Google Workspace needs a token: pass auth= or set GOOGLE_ACCESS_TOKEN.')
        prefix = self.id or 'google-workspace'
        toolset: AbstractToolset[AgentDepsT] = CombinedToolset(
            [
                MCPToolset[AgentDepsT](
                    _MCP_URLS[service],
                    id=f'{prefix}-{service}',
                    auth=auth,
                    include_instructions=self.include_instructions,
                ).prefixed(service)
                for service in self.services
            ]
        )
        return toolset.filtered(lambda _ctx, tool_def: is_read_only(tool_def)) if self.read_only else toolset
