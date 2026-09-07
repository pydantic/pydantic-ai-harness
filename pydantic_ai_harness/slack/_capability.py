"""Give an agent Slack's hosted MCP tools."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset

_INSTRUCTIONS = """\
Slack tools act with the identity of the connected token.
Before answering about discussion elsewhere in Slack, read the relevant thread or channel with these tools rather than guessing.
Distinguish channel-wide results from thread replies.
If you cannot find the needed context, say so instead of making it up.
"""

# Slack's hosted MCP server. It accepts user tokens only and answers a bot token with `invalid_token_type`.
# Verified 2026-09-07 against https://docs.slack.dev/ai/slack-mcp-server/; recheck when the integration changes.
_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent Slack's hosted MCP tools, acting as the user whose token is used.

    Slack's MCP server accepts user tokens (`xoxp-`) only. A bot token cannot be used here.
    """

    token: str | None = field(default=None, repr=False)
    """The Slack user token. Defaults to `SLACK_USER_TOKEN`."""
    id: str | None = 'slack'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Return a copy with the token resolved for this run."""
        token = self.token or os.environ.get('SLACK_USER_TOKEN')
        if not token:
            raise UserError('Slack tools need a user token. Pass Slack(token=...) or set SLACK_USER_TOKEN.')
        if token.startswith('xoxb-'):
            raise UserError(
                "Slack's MCP server accepts user tokens only, not bot tokens. "
                'Add user token scopes to the app and use its `xoxp-` token; see https://docs.slack.dev/ai/slack-mcp-server/.'
            )
        return replace(self, token=token)

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

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Return the toolset for the resolved token, or `None` before a run resolves one."""
        if self.token is None:
            return None
        return MCPToolset(
            _SLACK_MCP_URL,
            id=f'{self.id or "slack"}-mcp',
            headers={'Authorization': f'Bearer {self.token}'},
            include_instructions=True,  # Core defaults this to false; Slack MCP supplies required instructions.
        )

    def get_instructions(self) -> str:
        """Return guidance for using Slack tools."""
        return _INSTRUCTIONS
