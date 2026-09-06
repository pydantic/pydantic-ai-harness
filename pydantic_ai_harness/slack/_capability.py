"""Slack capability backed by Slack's hosted MCP server."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

_INSTRUCTIONS = """\
When prior discussion or another participant's messages are needed, retrieve visible conversation context through
native Slack MCP. Resolve ambiguity before acting, distinguish channel-wide results from thread replies, and follow
provider pagination until the result is complete. If visible context is unavailable, ask for clarification instead of
guessing.
"""

# Externally owned endpoint verified 2026-09-06 against Slack's official MCP overview
# (https://docs.slack.dev/ai/slack-mcp-server/); recheck when integration changes.
_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent native Slack MCP tools using one explicit OAuth token."""

    token: str = field(repr=False)
    id: str | None = 'slack'

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError('Slack token must be non-blank.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge equal-token configurations and reject ambiguous credentials."""
        first = capabilities[0]
        assert isinstance(first, cls)
        for capability in capabilities[1:]:
            assert isinstance(capability, cls)
            if capability.token != first.token:
                raise UserError('Multiple Slack capabilities with different credentials cannot be combined.')
        return super().combine(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return a native Slack MCP toolset configured with this capability's token."""
        return MCPToolset(
            _SLACK_MCP_URL,
            id=f'{self.id or "slack"}-mcp',
            headers={'Authorization': f'Bearer {self.token}'},
            include_instructions=True,  # Core defaults this to false; Slack MCP supplies required instructions.
        )

    def get_instructions(self) -> str:
        """Return static guidance for native Slack MCP tools."""
        return _INSTRUCTIONS
