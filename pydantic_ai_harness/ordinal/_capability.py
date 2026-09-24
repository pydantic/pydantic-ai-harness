"""Ordinal hosted MCP capability.

Provider contract, verified 2026-09-11:

- `https://app.tryordinal.com/mcp` is the Streamable HTTP endpoint.
- Authentication is OAuth. The previous API-key server at
  `https://app.tryordinal.com/api/mcp` is deprecated and is not used here.

Source: https://docs.tryordinal.com/mcp/introduction
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness._mcp import MCPAuth, MCPAuthFunc, per_run_auth

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
    """Let an agent draft, schedule, and analyze social posts in Ordinal.

    Without `auth`, the agent opens a browser for sign-in the first time it connects. It can then
    reach every workspace the user belongs to.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Ordinal

    agent = Agent('openai:gpt-5', capabilities=[Ordinal()])
    ```
    """

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    auth: MCPAuth | MCPAuthFunc[AgentDepsT] | None = field(default=None, repr=False)
    """An Ordinal access token, `'oauth'`, an `httpx.Auth`, or a function that returns one for each run.

    Unset, it means `'oauth'`, which opens a browser on the machine running the agent. A function can
    return the current user's token from `ctx.deps`; returning `None` gives that run no Ordinal tools.
    """

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the Ordinal MCP connection.

        This capability does not emit its own spans. It only constructs the
        hosted connection; Pydantic AI's MCP toolset traces tool calls.
        """
        return per_run_auth(self.auth, self._connect, id=self.id if self.id is not None else 'ordinal')

    def _connect(self, auth: MCPAuth | None) -> MCPToolset[AgentDepsT]:
        return MCPToolset(
            _ORDINAL_MCP_URL,
            id=self.id if self.id is not None else 'ordinal',
            auth=auth if auth is not None else 'oauth',
            include_instructions=True,
        )
