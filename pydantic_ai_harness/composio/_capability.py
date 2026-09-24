"""Connect Composio sessions over MCP.

Composio documents `session.mcp.url` and `session.mcp.headers` as the connection
contract (verified 2026-09-08). Re-check session setup before changing transport:
https://docs.composio.dev/docs/sessions-via-mcp
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import one_connection

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Composio support with: uv add "pydantic-ai-harness[composio]"') from exc


_ID = 'composio'


@dataclass(kw_only=True)
class Composio(AbstractCapability[AgentDepsT]):
    """Give an agent the tools of a Composio session.

    Create or restore the session with Composio's SDK, then pass its MCP URL and headers.
    """

    id: str | None = _ID
    """Names this capability in a run, so `defer_loading=True` needs no `id`. Give each `Composio` on one agent its own."""
    url: str | None = None
    """The URL returned by `session.mcp.url`. Required unless `client` is supplied."""
    headers: Mapping[str, str | None] | None = field(default=None, repr=False)
    """The headers returned by `session.mcp.headers`; unset values are omitted."""
    description: str | None = 'Discover and use connected applications through Composio.'
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, for full control of the connection. It cannot be combined with `url` or `headers`."""

    def __post_init__(self) -> None:
        if self.client is not None and (self.url is not None or self.headers is not None):
            raise UserError('`client` owns the connection, so it cannot be combined with `url` or `headers`.')
        if self.client is None and self.url is None:
            raise UserError('Pass `url` from `session.mcp.url`, or your own `client`, to connect to Composio.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two `Composio`s under one `id` are the same session stated twice, or an error if they differ."""
        return one_connection(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the session connection using Pydantic AI's MCP lifecycle."""
        if self.client is not None:
            return MCPToolset(self.client, id=self.id or _ID, include_instructions=self.include_instructions)
        assert self.url is not None  # checked in `__post_init__`
        return MCPToolset(
            self.url,
            headers={key: value for key, value in (self.headers or {}).items() if value is not None},
            id=self.id or _ID,
            include_instructions=self.include_instructions,
        )
