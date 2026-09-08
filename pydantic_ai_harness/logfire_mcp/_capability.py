"""Logfire hosted MCP: `https://logfire-us.pydantic.dev/mcp`, and `logfire-eu` for EU data.

Both endpoints verified 2026-09-08 against https://pydantic.dev/docs/logfire/guides/mcp-server/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ
from typing import Literal

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import is_read_only

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Logfire MCP capability. '
        'Install it with: uv add "pydantic-ai-harness[logfire-mcp]"'
    ) from _import_error

_LOGFIRE_MCP_URL = 'https://logfire-us.pydantic.dev/mcp'
_DEFAULT_DESCRIPTION = 'Query Logfire telemetry and manage dashboards, alerts, and issues.'


@dataclass(kw_only=True)
class LogfireMCP(AbstractCapability[AgentDepsT]):
    """Connect an agent to Logfire's hosted MCP server.

    The default exposes the write tools too; the credential's project and scopes decide what Logfire
    lets the agent read or change.
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    url: str = _LOGFIRE_MCP_URL
    """MCP endpoint.

    Use `https://logfire-eu.pydantic.dev/mcp` for EU data, or your own `/mcp` URL for a self-hosted
    deployment.
    """

    auth: Auth | Literal['oauth'] | str | None = field(default=None, repr=False)
    """A Logfire API key for headless use, `'oauth'` for browser login, or a custom `httpx.Auth`.

    Defaults to `$LOGFIRE_MCP_TOKEN`, then to `'oauth'`.
    """

    read_only: bool = False
    """Expose only the tools Logfire marks read-only."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Logfire MCP connection."""
        auth = self.auth or environ.get('LOGFIRE_MCP_TOKEN') or 'oauth'
        toolset: AbstractToolset[AgentDepsT] = MCPToolset(
            self.url, id=self.id or 'logfire-mcp', auth=auth, include_instructions=True
        )
        return toolset.filtered(lambda _ctx, tool_def: is_read_only(tool_def)) if self.read_only else toolset

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
        url: str = _LOGFIRE_MCP_URL,
        read_only: bool = False,
    ) -> LogfireMCP[AgentDepsT]:
        """Construct a Logfire MCP capability from serializable options.

        `auth` is absent by design, so a spec file cannot carry a Logfire token: the credential comes
        from `$LOGFIRE_MCP_TOKEN`.
        """
        return cls(id=id, description=description, defer_loading=defer_loading, url=url, read_only=read_only)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the agent-spec capability name."""
        return 'LogfireMCP'
