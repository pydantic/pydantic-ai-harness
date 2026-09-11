"""Ordinal hosted MCP capability.

Provider contract, verified 2026-09-11:

- `https://app.tryordinal.com/mcp` is the Streamable HTTP endpoint.
- Authentication is OAuth. The previous API-key server at
  `https://app.tryordinal.com/api/mcp` is deprecated and is not used here.

Source: https://docs.tryordinal.com/mcp/introduction
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the Ordinal capability. Install it with: uv add "pydantic-ai-harness[ordinal]"'
    ) from _import_error

_ORDINAL_MCP_URL = 'https://app.tryordinal.com/mcp'
_DEFAULT_DESCRIPTION = 'Work inside an Ordinal workspace: draft, schedule, and analyze social posts.'


@dataclass
class Ordinal(AbstractCapability[AgentDepsT]):
    """Connect an agent to Ordinal's hosted MCP server.

    The first tool call opens a browser for OAuth. After sign-in, the agent
    can reach every workspace the user belongs to.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Ordinal

    agent = Agent('openai:gpt-5', capabilities=[Ordinal()])
    ```
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    def get_toolset(self) -> MCPToolset[AgentDepsT]:
        """Build the Ordinal MCP connection.

        This capability does not emit its own spans. It only constructs the
        hosted connection; Pydantic AI's MCP toolset traces tool calls.
        """
        return MCPToolset(
            _ORDINAL_MCP_URL,
            id=self.id if self.id is not None else 'ordinal',
            auth='oauth',
            include_instructions=True,
        )
