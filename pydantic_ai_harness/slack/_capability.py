"""Give an agent Slack's hosted MCP tools."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from httpx import Auth
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent Slack's hosted MCP tools, acting as the user whose token is used.

    Slack's MCP server accepts user tokens (`xoxp-`) only. A bot token cannot be used here.
    """

    auth: str | Auth | None = field(default=None, repr=False)
    """The Slack user token or an `httpx.Auth` that supplies it. Defaults to `SLACK_USER_TOKEN`."""
    read_only: bool = False
    """Expose only the tools Slack marks read-only, dropping the ones that post, react, or edit as the token's user."""
    id: str | None = 'slack'

    def __post_init__(self) -> None:
        if self.auth is None:
            self.auth = os.environ.get('SLACK_USER_TOKEN')
        if not self.auth:
            raise UserError('Slack tools need a user token. Pass Slack(auth=...) or set SLACK_USER_TOKEN.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge equal-token configurations and reject different credentials."""
        first = capabilities[0]
        assert isinstance(first, cls)
        for capability in capabilities[1:]:
            assert isinstance(capability, cls)
            if capability.auth != first.auth:
                raise UserError('Multiple Slack capabilities with different credentials cannot be combined.')
        return super().combine(capabilities)

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Connect to Slack's hosted MCP server as the token's user."""
        toolset: AbstractToolset[AgentDepsT] = MCPToolset(
            _SLACK_MCP_URL,
            id=f'{self.id or "slack"}-mcp',
            auth=self.auth,
            include_instructions=True,
        )
        if self.read_only:
            toolset = toolset.filtered(_slack_marks_read_only)
        return toolset


def _slack_marks_read_only(ctx: RunContext[AgentDepsT], tool: ToolDefinition) -> bool:
    """Slack annotates every tool with the MCP `readOnlyHint`; core copies the annotations into tool metadata."""
    annotations: object = (tool.metadata or {}).get('annotations')
    if not isinstance(annotations, dict):
        return False
    typed: dict[object, object] = annotations  # pyright: ignore[reportUnknownVariableType]
    return typed.get('readOnlyHint') is True
