"""Slack MCP tool selection and per-run toolset construction."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from pydantic_ai_harness.slack._context import SlackContext, SlackMessageContext

try:
    import pydantic_ai.mcp as pydantic_ai_mcp
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Slack workspace tools require the Slack extra. Install with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error

SLACK_MCP_URL = 'https://mcp.slack.com/mcp'
"""Slack's Streamable HTTP endpoint.

Verified 2026-09-05 against https://docs.slack.dev/ai/slack-mcp-server/.
Re-check that page before changing the endpoint, tool catalog, or OAuth scope table.
"""


class SlackTool(str, Enum):
    """Slack-hosted MCP tools supported by the typed selection API.

    Names and scope families were verified 2026-09-05 against Slack's hosted
    MCP documentation. Runtime catalog validation reports provider drift.
    """

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

CURRENT_CONVERSATION_SLACK_TOOLS = frozenset(
    {
        SlackTool.SEARCH_USERS,
        SlackTool.READ_CHANNEL,
        SlackTool.READ_THREAD,
        SlackTool.READ_FILE,
    }
)
"""Read the invoking conversation and resolve the people mentioned in it."""

WORKSPACE_READ_SLACK_TOOLS = frozenset(
    {
        SlackTool.SEARCH_PUBLIC,
        SlackTool.SEARCH_USERS,
        SlackTool.READ_CHANNEL,
        SlackTool.READ_THREAD,
        SlackTool.READ_FILE,
    }
)
"""Search public workspace messages and read conversations by ID."""

_TOOL_USER_SCOPES: dict[SlackTool, frozenset[str]] = {
    SlackTool.SEARCH_PUBLIC: frozenset({'search:read.public'}),
    SlackTool.SEARCH_PUBLIC_AND_PRIVATE: frozenset(
        {'search:read.public', 'search:read.private', 'search:read.mpim', 'search:read.im'}
    ),
    SlackTool.SEARCH_CHANNELS: frozenset(
        {'search:read.public', 'search:read.private', 'search:read.mpim', 'search:read.im'}
    ),
    SlackTool.SEARCH_USERS: frozenset({'search:read.users'}),
    SlackTool.READ_CHANNEL: frozenset({'channels:history', 'groups:history', 'mpim:history', 'im:history'}),
    SlackTool.READ_THREAD: frozenset({'channels:history', 'groups:history', 'mpim:history', 'im:history'}),
    SlackTool.READ_FILE: frozenset({'files:read'}),
    SlackTool.READ_USER_PROFILE: frozenset({'users:read', 'users:read.email'}),
    SlackTool.LIST_CHANNEL_MEMBERS: frozenset({'channels:read', 'groups:read', 'mpim:read'}),
    SlackTool.SEND_MESSAGE: frozenset({'chat:write'}),
    SlackTool.SCHEDULE_MESSAGE: frozenset({'chat:write'}),
    SlackTool.ADD_REACTION: frozenset({'reactions:write'}),
}


def _require_slack_tool(value: object) -> SlackTool:
    if not isinstance(value, SlackTool):
        raise TypeError('SlackTools.of() accepts SlackTool values, not raw tool-name strings')
    return value


def _require_custom_tool(value: object) -> SlackCustomTool:
    if not isinstance(value, SlackCustomTool):
        raise TypeError('SlackTools.custom() accepts SlackCustomTool values')
    return value


@dataclass(frozen=True, slots=True, init=False)
class SlackCustomTool:
    """A provider tool not yet represented by `SlackTool`, with its OAuth scopes."""

    name: str
    user_scopes: frozenset[str]

    def __init__(
        self,
        name: str,
        *,
        user_scopes: Collection[str],
    ) -> None:
        if not name.startswith('slack_') or not name.removeprefix('slack_'):
            raise ValueError("SlackCustomTool.name must begin with 'slack_'")
        if isinstance(user_scopes, str):
            raise TypeError('SlackCustomTool.user_scopes must be a collection of scope strings, not one string')
        scopes = frozenset(scope.strip() for scope in user_scopes)
        if not scopes or any(not scope for scope in scopes):
            raise ValueError('SlackCustomTool.user_scopes must contain non-empty OAuth scope names')
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'user_scopes', scopes)


def _tool_name(tool: SlackTool | SlackCustomTool) -> str:
    return tool.value if isinstance(tool, SlackTool) else tool.name


@dataclass(frozen=True, slots=True, init=False)
class SlackTools:
    """An immutable, typed selection of Slack MCP tools."""

    selected: frozenset[SlackTool | SlackCustomTool]
    _current_conversation_only: frozenset[SlackTool]

    @classmethod
    def _from_selection(
        cls,
        selected: frozenset[SlackTool | SlackCustomTool],
        *,
        current_only: frozenset[SlackTool] = frozenset(),
    ) -> SlackTools:
        instance = object.__new__(cls)
        object.__setattr__(instance, 'selected', selected)
        object.__setattr__(instance, '_current_conversation_only', current_only)
        return instance

    @classmethod
    def none(cls) -> SlackTools:
        """Expose no Slack MCP tools while retaining Slack hosting and context."""
        return cls._from_selection(frozenset())

    @classmethod
    def current_conversation(cls) -> SlackTools:
        """Resolve users and read the channel or thread that invoked the agent."""
        return cls._from_selection(
            CURRENT_CONVERSATION_SLACK_TOOLS,
            current_only=frozenset({SlackTool.READ_CHANNEL, SlackTool.READ_THREAD, SlackTool.READ_FILE}),
        )

    @classmethod
    def workspace_read(cls) -> SlackTools:
        """Search public workspace messages, resolve users, and read conversations."""
        return cls._from_selection(WORKSPACE_READ_SLACK_TOOLS)

    @classmethod
    def read_only(cls) -> SlackTools:
        """Expose every supported search and read tool, including files and profiles."""
        return cls._from_selection(READ_SLACK_TOOLS)

    @classmethod
    def of(cls, *tools: SlackTool) -> SlackTools:
        """Select exact Slack MCP tools."""
        if not tools:
            raise ValueError('SlackTools.of() needs at least one SlackTool; use SlackTools.none() for no tools')
        return cls._from_selection(frozenset(_require_slack_tool(tool) for tool in tools))

    @classmethod
    def custom(cls, *tools: SlackCustomTool) -> SlackTools:
        """Select provider tools not yet represented by `SlackTool`.

        Custom tools remain approval-gated by `approval='writes'` because
        Harness has not classified their behavior.
        """
        if not tools:
            raise TypeError('SlackTools.custom() requires at least one SlackCustomTool')
        return cls._from_selection(frozenset(_require_custom_tool(tool) for tool in tools))

    def restrict_to_current_conversation(self) -> SlackTools:
        """Confine selected channel, thread, and file reads to the invoking Slack context."""
        restricted: set[SlackTool] = set()
        for tool in self.selected:
            if isinstance(tool, SlackTool) and tool in {
                SlackTool.READ_CHANNEL,
                SlackTool.READ_THREAD,
                SlackTool.READ_FILE,
            }:
                restricted.add(tool)
        return SlackTools._from_selection(self.selected, current_only=frozenset(restricted))

    def __or__(self, other: SlackTools) -> SlackTools:
        return SlackTools._from_selection(
            self.selected | other.selected,
            current_only=self._current_conversation_only | other._current_conversation_only,
        )

    @property
    def required_user_scopes(self) -> frozenset[str]:
        """Slack user OAuth scopes required by this typed selection.

        Custom tools contribute the explicit scopes in their `SlackCustomTool` descriptor.
        """
        scopes: set[str] = set()
        for tool in self.selected:
            scopes.update(_TOOL_USER_SCOPES[tool] if isinstance(tool, SlackTool) else tool.user_scopes)
        return frozenset(scopes)


@dataclass
class SelectedSlackToolset(WrapperToolset[AgentDepsT], Generic[AgentDepsT]):
    """Expose an exact tool selection and reject server catalog drift."""

    selected: frozenset[SlackTool | SlackCustomTool]
    current_only: frozenset[SlackTool]
    slack_context: SlackContext | None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        selected_names = {_tool_name(tool) for tool in self.selected}
        missing = selected_names - tools.keys()
        if missing:
            listed = ', '.join(sorted(missing))
            raise UserError(f'The Slack MCP server did not provide the selected tools: {listed}')
        return {name: tool for name, tool in tools.items() if name in selected_names}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if name in {selected.value for selected in self.current_only}:
            self._validate_current_conversation(name, tool_args)
        return await super().call_tool(name, tool_args, ctx, tool)

    def _validate_current_conversation(self, name: str, tool_args: dict[str, Any]) -> None:
        context = self.slack_context
        if context is None:
            raise UserError(
                f'{name} is restricted to the invoking Slack conversation, but this run has no SlackContext. '
                'Use SlackApp, or explicitly select SlackTools.workspace_read() for an unbound run.'
            )
        channel_ids = {context.channel_id}
        thread_coordinates = {(context.channel_id, context.thread_ts)}
        for entity in context.active_entities:
            if entity.entity_type == 'slack#/types/channel_id' and isinstance(entity.value, str):
                channel_ids.add(entity.value)
            elif isinstance(entity.value, SlackMessageContext):
                channel_ids.add(entity.value.channel_id)
                thread_coordinates.add((entity.value.channel_id, entity.value.message_ts))
        if tool_args.get('channel_id') not in channel_ids:
            if name != SlackTool.READ_FILE.value:
                raise UserError(f'{name} is restricted to the invoking Slack conversation and active view.')
        if (
            name == SlackTool.READ_THREAD.value
            and (
                tool_args.get('channel_id'),
                tool_args.get('message_ts'),
            )
            not in thread_coordinates
        ):
            raise UserError(f'{name} is restricted to the invoking Slack thread and active view.')
        if name == SlackTool.READ_FILE.value and tool_args.get('file_id') not in {
            file.file_id for file in context.files
        }:
            raise UserError(f'{name} is restricted to files attached to the invoking Slack message.')


def slack_mcp_toolset(
    *,
    token: str,
    tools: SlackTools,
    approval: Literal['writes', 'all', 'none'],
    slack_context: SlackContext | None = None,
) -> AbstractToolset[AgentDepsT]:
    """Build one user-authenticated Slack MCP toolset for one agent run."""
    toolset: pydantic_ai_mcp.MCPToolset[AgentDepsT] = pydantic_ai_mcp.MCPToolset(
        SLACK_MCP_URL,
        id='slack-mcp',
        headers={'Authorization': f'Bearer {token}'},
    )
    selected: AbstractToolset[AgentDepsT] = SelectedSlackToolset(
        toolset,
        tools.selected,
        tools._current_conversation_only,  # pyright: ignore[reportPrivateUsage] - same-module construction
        slack_context,
    )
    if approval == 'none':
        return selected
    if approval == 'all':
        return selected.approval_required()
    read_names = {tool.value for tool in READ_SLACK_TOOLS}
    return selected.approval_required(lambda _ctx, tool_def, _args: tool_def.name not in read_names)
