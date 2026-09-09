"""AWS hosted MCP capability."""

from __future__ import annotations

from dataclasses import dataclass, field

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import is_read_only

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as exc:  # pragma: no cover
    raise ImportError('Install AWS support with: uv add "pydantic-ai-harness[aws]"') from exc

from typing import Literal


@dataclass(kw_only=True)
class AWS(AbstractCapability[AgentDepsT]):
    """Use AWS's managed MCP server with IAM-controlled access."""

    description: str | None = 'Use AWS knowledge and account tools.'
    auth: Auth | str | None = field(default=None, repr=False)
    """`'oauth'` for browser sign-in, or caller-supplied HTTP authentication. Defaults to OAuth."""
    read_only: bool = False
    """Expose only tools the server marks read-only; unmarked tools are omitted."""
    include_instructions: bool = True
    """Forward the server's instructions to the agent."""
    client: MCPToolsetClient | None = field(default=None, repr=False)
    """Override the connection with a caller-configured MCP client or transport.

    The supplied client owns its URL, authentication, and server configuration.
    """
    region: Literal['us-east-1', 'eu-central-1'] = 'us-east-1'
    """Region hosting the MCP endpoint, independent of the regions your tools operate on."""

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the AWS connection and optional read-only selection."""
        if self.client is not None:
            toolset: AbstractToolset[AgentDepsT] = MCPToolset(
                self.client, id=self.id or 'aws', include_instructions=self.include_instructions
            )
        else:
            toolset = MCPToolset(
                f'https://aws-mcp.{self.region}.api.aws/mcp',
                id=self.id or 'aws',
                auth=self.auth if self.auth is not None else 'oauth',
                headers=None,
                include_instructions=self.include_instructions,
            )
        if self.read_only:
            return toolset.filtered(lambda _ctx, tool: is_read_only(tool))
        return toolset
