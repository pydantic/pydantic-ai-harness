"""Slack capability backed by Slack's hosted MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

from pydantic_ai_harness.slack._approvals import SlackApprovals
from pydantic_ai_harness.slack._client import SlackClient, default_client
from pydantic_ai_harness.slack._context import (
    current_delivery_client,
    current_slack_context,
    fixed_mcp_fallback_allowed,
)
from pydantic_ai_harness.slack._interactions import SlackInteractions
from pydantic_ai_harness.slack._mcp import SlackTool, SlackTools, slack_mcp_toolset
from pydantic_ai_harness.slack._thread import SlackThread
from pydantic_ai_harness.slack._validate import reject_bare_string

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions


DEFAULT_INSTRUCTIONS = """\
You are participating in Slack. Use the Slack tools when the user asks about
people, messages, channels, or threads in their workspace. Resolve names
to Slack user or channel IDs before searching when needed. Follow every cursor
until no page remains when the user asks for a complete count or exhaustive
result. Return ordinary answers as the final output; do not use a Slack write
tool to deliver the reply. Keep the final reply concise and suitable for a
Slack thread.
"""
"""Default guidance for using Slack MCP tools correctly."""


class SlackMCPAuthenticationError(UserError):
    """No user token is available for the Slack MCP session."""


class _SlackSpec(BaseModel):
    """Runtime validation for values loaded from YAML or JSON."""

    model_config = ConfigDict(extra='forbid')

    tools: list[SlackTool] | None = None
    approval: Literal['writes', 'all', 'none'] = 'writes'
    approver_ids: list[str] | None = None
    instructions: str | None = None


@dataclass
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent user-scoped Slack search, read, and action tools.

    `Slack()` exposes a curated workspace conversation set from Slack's hosted MCP server.
    Select additional tools with `SlackTools.of(...)`. Slack actions selected by
    the caller require approval by default.

    [`SlackApp`][pydantic_ai_harness.slack.SlackApp] supplies the invoking user's
    OAuth token for each run. For runs hosted elsewhere, pass `mcp_token` or set
    `SLACK_MCP_TOKEN`. A fresh MCP session is created for every run, so concurrent
    users never share an authenticated Slack session.
    """

    tools: SlackTools = field(default_factory=SlackTools.workspace_read)
    """Exact Slack MCP tools visible to the model."""

    approval: Literal['writes', 'all', 'none'] = 'writes'
    """Which selected Slack MCP calls require Pydantic AI tool approval."""

    mcp_token: str | None = field(default=None, repr=False)
    """Fixed Slack MCP user token for runs not hosted by `SlackApp`."""

    approver_ids: list[str] | None = None
    """Slack users allowed to approve actions. Defaults to the invoking user."""

    instructions: str | None = None
    """Replace `DEFAULT_INSTRUCTIONS`; use an empty string to add no guidance."""

    delivery_token: str | None = field(default=None, repr=False)
    """Bot token used only for approval prompts outside `SlackApp`."""

    delivery_client: SlackClient | None = field(default=None, repr=False)
    """Slack Web API client used only for approval prompts and deterministic delivery."""

    interactions: SlackInteractions | None = field(default=None, repr=False)
    """Prompt registry used by Slack approval buttons."""

    _app_delivery_token: str | None = field(default=None, init=False, repr=False, compare=False)
    _resolved_client: SlackClient | None = field(default=None, init=False, repr=False, compare=False)
    _resolved_interactions: SlackInteractions | None = field(default=None, init=False, repr=False, compare=False)
    _dynamic_toolset: DynamicToolset[AgentDepsT] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.approval not in ('writes', 'all', 'none'):
            raise ValueError("approval must be 'writes', 'all', or 'none'")
        reject_bare_string(self.approver_ids, 'approver_ids')
        self._dynamic_toolset = DynamicToolset(self._toolset_for_run, per_run_step=False, id='slack-mcp')

    @classmethod
    def from_spec(
        cls,
        *,
        tools: list[str] | None = None,
        approval: Literal['writes', 'all', 'none'] = 'writes',
        approver_ids: list[str] | None = None,
        instructions: str | None = None,
    ) -> Slack[AgentDepsT]:
        """Build the serializable part of the capability from an agent spec."""
        spec = _SlackSpec.model_validate(
            {
                'tools': tools,
                'approval': approval,
                'approver_ids': approver_ids,
                'instructions': instructions,
            }
        )
        selected = SlackTools.workspace_read() if spec.tools is None else SlackTools.of(*spec.tools)
        return cls(
            tools=selected,
            approval=spec.approval,
            approver_ids=spec.approver_ids,
            instructions=spec.instructions,
        )

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return a dynamic toolset that creates one MCP session per run."""
        return self._dynamic_toolset

    def _toolset_for_run(self, _ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        slack_context = current_slack_context()
        token = (slack_context.user_token if slack_context is not None else None) or self.mcp_token
        if token is None and fixed_mcp_fallback_allowed():
            token = os.environ.get('SLACK_MCP_TOKEN')
        if token is None and not self.tools.selected:
            return None
        if token is None:
            raise SlackMCPAuthenticationError(
                'Slack MCP needs the invoking user OAuth token. Configure Bolt OAuth, pass mcp_token, '
                'or set SLACK_MCP_TOKEN.'
            )
        toolset = slack_mcp_toolset(token=token, tools=self.tools, approval=self.approval)
        return toolset

    def set_app_delivery_token(self, token: str) -> None:
        """Set the bot token `SlackApp` uses for approval UI delivery."""
        self._app_delivery_token = token

    def resolve_client(self) -> SlackClient:
        """Resolve the Web API client used for approval prompts."""
        if bound_client := current_delivery_client():
            return bound_client
        if self._resolved_client is None:
            self._resolved_client = (
                self.delivery_client
                if self.delivery_client is not None
                else default_client(self.delivery_token or self._app_delivery_token)
            )
        return self._resolved_client

    def resolve_interactions(self) -> SlackInteractions:
        """Resolve the registry backing Slack approval buttons."""
        if self._resolved_interactions is None:
            self._resolved_interactions = self.interactions if self.interactions is not None else SlackInteractions()
        return self._resolved_interactions

    def resolve_prompt(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Route a Slack approval-button click to the suspended run."""
        return self.resolve_interactions().resolve(block_id=block_id, value=value, user_id=user_id)

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        """Render selected tool approvals in the current Slack thread."""
        slack_context = current_slack_context()
        if slack_context is None:
            return None
        thread = SlackThread(
            channel_id=slack_context.channel_id,
            thread_ts=slack_context.thread_ts,
            user_id=slack_context.user_id,
            team_id=slack_context.team_id,
        )
        approvals = SlackApprovals[AgentDepsT](
            self.resolve_client(),
            self.resolve_interactions(),
            thread=thread,
            allowed_user_ids=self.approver_ids,
        )
        return await approvals(ctx, requests)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Describe Slack tool behavior and the current conversation coordinates."""
        base = self.instructions if self.instructions is not None else DEFAULT_INSTRUCTIONS
        if not base:
            return None

        def current_conversation() -> str:
            context = current_slack_context()
            if context is None:
                return ''
            conversation = (
                f'The current Slack channel ID is `{context.channel_id}` and the current thread timestamp is '
                f'`{context.thread_ts}`. The invoking Slack user ID is `{context.user_id}`.'
            )
            if not context.active_entities:
                return conversation
            active = ', '.join(f'`{entity.entity_type}` = `{entity.value}`' for entity in context.active_entities)
            return f"{conversation} The user's active Slack view, in relevance order, is: {active}."

        return [base, current_conversation]
