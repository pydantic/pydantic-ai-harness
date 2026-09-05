"""Slack MCP tool selection and per-run toolset construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

try:
    from pydantic_ai.mcp import MCPToolset
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Slack workspace tools require the Slack extra. Install with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error

SLACK_MCP_URL = 'https://mcp.slack.com/mcp'


class SlackTool(str, Enum):
    """Slack-hosted MCP tools supported by the typed selection API."""

    SEARCH_PUBLIC = 'slack_search_public'
    SEARCH_PUBLIC_AND_PRIVATE = 'slack_search_public_and_private'
    SEARCH_CHANNELS = 'slack_search_channels'
    SEARCH_USERS = 'slack_search_users'
    READ_CHANNEL = 'slack_read_channel'
    READ_THREAD = 'slack_read_thread'
    READ_FILE = 'slack_read_file'
    READ_USER_PROFILE = 'slack_read_user_profile'
    LIST_CHANNEL_MEMBERS = 'slack_list_channel_members'
    SEND_MESSAGE = 'slack_send_message'
    SCHEDULE_MESSAGE = 'slack_schedule_message'
    ADD_REACTION = 'slack_add_reaction'


READ_SLACK_TOOLS = frozenset(
    {
        SlackTool.SEARCH_PUBLIC,
        SlackTool.SEARCH_PUBLIC_AND_PRIVATE,
        SlackTool.SEARCH_CHANNELS,
        SlackTool.SEARCH_USERS,
        SlackTool.READ_CHANNEL,
        SlackTool.READ_THREAD,
        SlackTool.READ_FILE,
        SlackTool.READ_USER_PROFILE,
        SlackTool.LIST_CHANNEL_MEMBERS,
    }
)
"""Every supported Slack search and read tool."""

WORKSPACE_READ_SLACK_TOOLS = frozenset(
    {
        SlackTool.SEARCH_PUBLIC,
        SlackTool.SEARCH_PUBLIC_AND_PRIVATE,
        SlackTool.SEARCH_CHANNELS,
        SlackTool.SEARCH_USERS,
        SlackTool.READ_CHANNEL,
        SlackTool.READ_THREAD,
    }
)
"""Smallest useful set for questions about workspace conversations."""


def _require_slack_tool(value: object) -> SlackTool:
    if not isinstance(value, SlackTool):
        raise TypeError('SlackTools.of() accepts SlackTool values, not raw tool-name strings')
    return value


@dataclass(frozen=True, slots=True, init=False)
class SlackTools:
    """An immutable, typed selection of Slack MCP tools."""

    selected: frozenset[SlackTool | str]

    @classmethod
    def _from_selection(cls, selected: frozenset[SlackTool | str]) -> SlackTools:
        instance = object.__new__(cls)
        object.__setattr__(instance, 'selected', selected)
        return instance

    @classmethod
    def workspace_read(cls) -> SlackTools:
        """Search people and conversations, then read channels and threads."""
        return cls._from_selection(WORKSPACE_READ_SLACK_TOOLS)

    @classmethod
    def read_only(cls) -> SlackTools:
        """Expose every supported search and read tool, including files and profiles."""
        return cls._from_selection(READ_SLACK_TOOLS)

    @classmethod
    def of(cls, *tools: SlackTool) -> SlackTools:
        """Select exact Slack MCP tools."""
        return cls._from_selection(frozenset(_require_slack_tool(tool) for tool in tools))

    @classmethod
    def named(cls, *tool_names: str) -> SlackTools:
        """Select exact provider tool names not yet represented by `SlackTool`.

        Prefer `of`. This compatibility escape hatch still verifies every name
        against Slack's discovered catalog, and `approval='writes'` treats each
        unclassified tool as requiring approval.
        """
        if not tool_names or any(
            not name.startswith('slack_') or not name.removeprefix('slack_') for name in tool_names
        ):
            raise ValueError("SlackTools.named() requires non-empty provider names beginning with 'slack_'")
        return cls._from_selection(frozenset(tool_names))

    def __or__(self, other: SlackTools) -> SlackTools:
        return SlackTools._from_selection(self.selected | other.selected)


@dataclass
class SelectedSlackToolset(WrapperToolset[AgentDepsT], Generic[AgentDepsT]):
    """Expose an exact tool selection and reject server catalog drift."""

    selected: frozenset[SlackTool | str]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        selected_names = {tool.value if isinstance(tool, SlackTool) else tool for tool in self.selected}
        missing = selected_names - tools.keys()
        if missing:
            listed = ', '.join(sorted(missing))
            raise UserError(f'The Slack MCP server did not provide the selected tools: {listed}')
        return {name: tool for name, tool in tools.items() if name in selected_names}


def slack_mcp_toolset(
    *,
    token: str,
    tools: SlackTools,
    approval: Literal['writes', 'all', 'none'],
) -> AbstractToolset[AgentDepsT]:
    """Build one user-authenticated Slack MCP toolset for one agent run."""
    toolset: MCPToolset[AgentDepsT] = MCPToolset(
        SLACK_MCP_URL,
        id='slack-mcp',
        headers={'Authorization': f'Bearer {token}'},
    )
    selected: AbstractToolset[AgentDepsT] = SelectedSlackToolset(toolset, tools.selected)
    if approval == 'none':
        return selected
    if approval == 'all':
        return selected.approval_required()
    read_names = {tool.value for tool in READ_SLACK_TOOLS}
    return selected.approval_required(lambda _ctx, tool_def, _args: tool_def.name not in read_names)
