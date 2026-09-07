"""Give an agent Slack's hosted MCP tools."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

# Slack's hosted MCP server. It decides whether a token is acceptable; a bot token or anything else it
# does not recognise gets a 401 at run time. Verified 2026-09-07 against
# https://docs.slack.dev/ai/slack-mcp-server/; recheck when the integration changes.
_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent Slack's hosted MCP tools, acting as the user whose token is used.

    Slack's MCP server accepts user tokens (`xoxp-`) only. A bot token cannot be used here.
    """

    token: str | None = field(default=None, repr=False)
    """The Slack user token. Defaults to `SLACK_USER_TOKEN`."""
    id: str | None = 'slack'

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get('SLACK_USER_TOKEN')
        if not self.token:
            raise UserError('Slack tools need a user token. Pass Slack(token=...) or set SLACK_USER_TOKEN.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge equal-token configurations and reject different credentials."""
        first = capabilities[0]
        assert isinstance(first, cls)
        for capability in capabilities[1:]:
            assert isinstance(capability, cls)
            if capability.token != first.token:
                raise UserError('Multiple Slack capabilities with different credentials cannot be combined.')
        return super().combine(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Connect to Slack's hosted MCP server as the token's user."""
        return MCPToolset(
            _SLACK_MCP_URL,
            id=f'{self.id or "slack"}-mcp',
            headers={'Authorization': f'Bearer {self.token}'},
            include_instructions=True,  # Core defaults to False; forward the guidance Slack's server sends.
        )
