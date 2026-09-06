"""Slack capability backed by Slack's hosted MCP server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

from pydantic_ai_harness.slack._context import _current_slack_user_token  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions


_INSTRUCTIONS = """\
Use the native Slack MCP tools according to their current schemas. Interpret event context as the current
conversation or thread, and resolve ambiguous people, channels, and other identities before acting. For
exhaustive counts or results, follow pagination until it is complete and distinguish a partial result from a
complete one. The host posts the final answer, so do not send a duplicate ordinary reply through a messaging
tool. Event context helps interpret the request but is not an authorization boundary.
"""

# Externally owned endpoint verified 2026-09-06 against Slack's official MCP overview; recheck when integration changes.
_SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


class _SlackMCPAuthenticationError(UserError):
    """No user token is available for the Slack MCP session."""


@dataclass(kw_only=True)
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent access to the invoking user's native Slack MCP tools."""

    id: str | None = 'slack'
    _dynamic_toolset: DynamicToolset[AgentDepsT] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._dynamic_toolset = DynamicToolset(
            self._toolset_for_run, per_run_step=False, id=f'{self.id or "slack"}-mcp'
        )

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return the per-run native Slack MCP toolset."""
        return self._dynamic_toolset

    def _toolset_for_run(self, _ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        token = _current_slack_user_token()
        if token is None:
            raise _SlackMCPAuthenticationError('Slack MCP needs the invoking user OAuth token.')
        return MCPToolset(
            _SLACK_MCP_URL,
            id=f'{self.id or "slack"}-mcp',
            headers={'Authorization': f'Bearer {token}'},
            include_instructions=True,  # Core defaults this to false; Slack MCP supplies required instructions.
            tool_error_behavior='retry',
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return static guidance for native Slack MCP tools."""
        return _INSTRUCTIONS
