"""Logfire hosted MCP capability.

Provider contract, verified 2026-09-07:

- `https://logfire-us.pydantic.dev/mcp` and `https://logfire-eu.pydantic.dev/mcp` are the hosted
  Streamable HTTP endpoints.
- OAuth and API-key bearer tokens are both accepted. API keys carry scopes such as `project:read`,
  and Logfire checks them on every request.
- `project_list` returns the projects the credential can reach; pass the returned project identifier
  unchanged to project tools.
- Every tool carries MCP `readOnlyHint` and `destructiveHint` annotations, set by `_tool_scope` in
  `logfire_mcp_capabilities.catalog` (pydantic/platform). `read_only=True` filters on `readOnlyHint`.

Source: https://pydantic.dev/docs/logfire/guides/mcp-server/. Re-check the endpoint and
authentication sections before changing connection behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Any, Literal

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Logfire MCP capability. '
        'Install it with: uv add "pydantic-ai-harness[logfire-mcp]"'
    ) from _import_error

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AgentInstructions

LOGFIRE_US_MCP_URL = 'https://logfire-us.pydantic.dev/mcp'
"""Logfire's hosted MCP endpoint for the US data region."""

LOGFIRE_EU_MCP_URL = 'https://logfire-eu.pydantic.dev/mcp'
"""Logfire's hosted MCP endpoint for the EU data region."""

_DEFAULT_DESCRIPTION = 'Query Logfire telemetry and manage dashboards, alerts, and issues.'

_INSTRUCTIONS = (
    'Timestamps in tool schemas and examples, and project creation timestamps, are examples or metadata rather than '
    'the current time. Query transport bounds apply in addition to SQL time predicates and default to a short '
    'window, so widen them explicitly when needed. Create a Logfire link only when the user asks for one.'
)


def _is_read_only(tool_def: ToolDefinition) -> bool:
    """Whether Logfire marked the tool `readOnlyHint`; an unannotated tool counts as a write."""
    metadata: dict[str, Any] = tool_def.metadata or {}
    annotations = metadata.get('annotations')
    if not isinstance(annotations, Mapping):
        return False
    return annotations.get('readOnlyHint') is True  # pyright: ignore[reportUnknownMemberType]


@dataclass
class LogfireMCP(AbstractCapability[AgentDepsT]):
    """Query Logfire telemetry and manage observability resources through Logfire's hosted MCP server.

    Every tool the credential can reach is exposed, and tools Logfire does not mark read-only
    require approval before they run. Pass `read_only=True` to expose only the read-only tools.
    Logfire enforces access: an API key's project and scopes decide what the agent can read or
    change. The server's own tool descriptions and instructions guide the model; pass
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

    read_only: bool = False
    """Expose only the tools Logfire marks read-only, dropping the ones that change dashboards, alerts, issues,
    and variables. By default every tool is exposed and the ones not marked read-only require approval, handled
    with `DeferredToolRequests` or `HandleDeferredToolCalls`.
    """

    allowed_tools: list[str] | None = None
    """Exact MCP tool names to expose. `None` exposes every tool `read_only` allows."""

    include_instructions: bool = True
    """Add both the Logfire server instructions and this capability's query guidance to the agent."""

    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Prebuilt FastMCP client or transport, or an in-process server, used instead of `url`.

    It carries its own authentication, so `auth` is ignored.
    """

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Logfire MCP toolset and apply the read-only filter, exact-name filter, and approval policy."""
        client = self.client if self.client is not None else self.url
        auth = self.auth if self.client is None else None
        toolset: AbstractToolset[AgentDepsT] = MCPToolset(
            client, id=self.id or 'logfire-mcp', auth=auth, include_instructions=self.include_instructions
        )
        if self.read_only:
            toolset = toolset.filtered(lambda _ctx, tool_def: _is_read_only(tool_def))
        if self.allowed_tools is not None:
            allowed_tools = frozenset(self.allowed_tools)
            toolset = toolset.filtered(lambda _ctx, tool_def: tool_def.name in allowed_tools)
        if not self.read_only:
            toolset = toolset.approval_required(lambda _ctx, tool_def, _args: not _is_read_only(tool_def))
        return toolset

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return cache-stable query guidance and the current run's UTC time."""
        if not self.include_instructions:
            return None
        return [_INSTRUCTIONS, self._current_utc]

    def _current_utc(self, ctx: RunContext[AgentDepsT]) -> str | None:
        for message in reversed(ctx.messages):
            if isinstance(message, ModelRequest):
                for part in reversed(message.parts):
                    if isinstance(part, UserPromptPart):
                        current_utc = part.timestamp.astimezone(timezone.utc).isoformat(timespec='seconds')
                        return f'Current UTC time for this run is `{current_utc}`.'
        return None  # pragma: no cover

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
        url: str = LOGFIRE_US_MCP_URL,
        auth: Literal['oauth'] | str | None = 'oauth',
        read_only: bool = False,
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
            read_only=read_only,
            allowed_tools=allowed_tools,
            include_instructions=include_instructions,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the agent-spec capability name."""
        return 'LogfireMCP'
