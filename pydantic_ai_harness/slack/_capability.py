"""Slack hosted MCP capability."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import MCPAuth, MCPAuthFunc, credential, is_read_only, per_run

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install Slack support with: uv add "pydantic-ai-harness[slack]"') from exc


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent Slack's hosted tools, acting as the connected user."""

    description: str | None = 'Use Slack messages, channels, and canvases.'
    auth: MCPAuth | MCPAuthFunc[AgentDepsT] | None = field(default=None, repr=False)
    """A Slack user token, an `httpx.Auth`, or a function of the run context that returns one.

    Unset, it uses `SLACK_USER_TOKEN`. If the function returns `None`, that run has no Slack tools.
    """
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Your own MCP client or transport, which then owns the URL and authentication."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Slack connection and optional read-only selection."""
        id = self.id or 'slack'
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=id, include_instructions=self.include_instructions
            )
        else:
            toolset = per_run(self.auth, self._connect, id=id)
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset

    def _connect(self, auth: MCPAuth | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            'https://mcp.slack.com/mcp',
            id=self.id or 'slack',
            auth=credential(auth, env='SLACK_USER_TOKEN', service='Slack'),
            headers=None,
            include_instructions=self.include_instructions,
        )
