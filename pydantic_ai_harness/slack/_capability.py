"""Slack capability backed by Slack's hosted MCP server."""

from __future__ import annotations

import os
from collections.abc import Collection
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset

from pydantic_ai_harness.slack._approvals import SlackApprovals
from pydantic_ai_harness.slack._client import SlackClient
from pydantic_ai_harness.slack._context import (
    SlackMessageContext,
    current_delivery_client,
    current_slack_context,
    current_user_token,
    fixed_mcp_fallback_allowed,
)
from pydantic_ai_harness.slack._interactions import DEFAULT_PROMPT_TIMEOUT_SECONDS, SlackInteractions
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
    read_scope: Literal['current_conversation', 'workspace'] = 'current_conversation'
    approval: Literal['writes', 'all', 'none'] = 'writes'
    approver_ids: list[str] | None = None
    approval_timeout_seconds: float = DEFAULT_PROMPT_TIMEOUT_SECONDS
    instructions: str | None = None


@dataclass
class Slack(AbstractCapability[AgentDepsT]):
    """Give an agent user-scoped Slack search, read, and action tools.

    `Slack()` exposes a curated workspace conversation set from Slack's hosted MCP server.
    Select additional tools with `SlackTools.of(...)`. Slack actions selected by
    the caller require approval by default.

    [`SlackApp`][pydantic_ai_harness.slack.SlackApp] supplies the invoking user's
    OAuth token for each run. For runs hosted elsewhere, explicitly select
    `SlackTools.workspace_read()` or another unrestricted tool selection, then
    pass `mcp_token` or set `SLACK_MCP_TOKEN`. A fresh MCP session is created for
    every run, so concurrent users never share an authenticated Slack session.
    """

    tools: SlackTools = field(default_factory=SlackTools.current_conversation)
    """Exact Slack MCP tools visible to the model."""

    approval: Literal['writes', 'all', 'none'] = 'writes'
    """Which selected Slack MCP calls require Pydantic AI tool approval."""

    mcp_token: str | None = field(default=None, repr=False)
    """Fixed Slack MCP user token for runs not hosted by `SlackApp`."""

    approver_ids: Collection[str] | None = None
    """Slack users allowed to approve actions. Defaults to the invoking user."""

    approval_timeout_seconds: float = DEFAULT_PROMPT_TIMEOUT_SECONDS
    """Seconds to wait for an approval click before denying the action."""

    instructions: str | None = None
    """Replace `DEFAULT_INSTRUCTIONS`; use an empty string to add no guidance."""

    _resolved_interactions: SlackInteractions | None = field(default=None, init=False, repr=False, compare=False)
    _dynamic_toolset: DynamicToolset[AgentDepsT] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.approval not in ('writes', 'all', 'none'):
            raise ValueError("approval must be 'writes', 'all', or 'none'")
        if not isfinite(self.approval_timeout_seconds) or self.approval_timeout_seconds <= 0:
            raise ValueError('approval_timeout_seconds must be finite and positive')
        reject_bare_string(self.approver_ids, 'approver_ids')
        if self.approver_ids is not None:
            normalized = frozenset(user_id.strip() for user_id in self.approver_ids)
            if not normalized or any(not user_id for user_id in normalized):
                raise ValueError('approver_ids must contain non-empty Slack user IDs')
            self.approver_ids = normalized
        self._dynamic_toolset = DynamicToolset(self._toolset_for_run, per_run_step=False, id='slack-mcp')

    @classmethod
    def from_spec(
        cls,
        *,
        tools: list[str] | None = None,
        read_scope: Literal['current_conversation', 'workspace'] = 'current_conversation',
        approval: Literal['writes', 'all', 'none'] = 'writes',
        approver_ids: list[str] | None = None,
        approval_timeout_seconds: float = DEFAULT_PROMPT_TIMEOUT_SECONDS,
        instructions: str | None = None,
    ) -> Slack[AgentDepsT]:
        """Build the serializable part of the capability from an agent spec.

        Exact channel, thread, and file reads remain confined to the invoking
        conversation unless `read_scope='workspace'` is explicit.
        """
        spec = _SlackSpec.model_validate(
            {
                'tools': tools,
                'read_scope': read_scope,
                'approval': approval,
                'approver_ids': approver_ids,
                'approval_timeout_seconds': approval_timeout_seconds,
                'instructions': instructions,
            }
        )
        if spec.tools is None:
            selected = (
                SlackTools.current_conversation()
                if spec.read_scope == 'current_conversation'
                else SlackTools.workspace_read()
            )
        elif spec.tools:
            selected = SlackTools.of(*spec.tools)
            if spec.read_scope == 'current_conversation':
                selected = selected.restrict_to_current_conversation()
        else:
            selected = SlackTools.none()
        return cls(
            tools=selected,
            approval=spec.approval,
            approver_ids=spec.approver_ids,
            approval_timeout_seconds=spec.approval_timeout_seconds,
            instructions=spec.instructions,
        )

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Return a dynamic toolset that creates one MCP session per run."""
        return self._dynamic_toolset

    def _toolset_for_run(self, _ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        if not self.tools.selected:
            return None
        token = current_user_token() or self.mcp_token
        if token is None and fixed_mcp_fallback_allowed():
            token = os.environ.get('SLACK_MCP_TOKEN')
        if token is None:
            raise SlackMCPAuthenticationError(
                'Slack MCP needs the invoking user OAuth token. Configure Bolt OAuth, pass mcp_token, '
                'or set SLACK_MCP_TOKEN.'
            )
        toolset = slack_mcp_toolset(
            token=token,
            tools=self.tools,
            approval=self.approval,
            slack_context=current_slack_context(),
        )
        return toolset

    def _resolve_client(self) -> SlackClient:
        """Resolve the Web API client used for approval prompts."""
        if bound_client := current_delivery_client():
            return bound_client
        raise UserError(  # pragma: no cover - public SlackApp runs always bind Bolt's client
            'Slack approval UI requires a SlackApp-hosted run with a bound Bolt client.'
        )

    def _resolve_interactions(self) -> SlackInteractions:
        """Resolve the registry backing Slack approval buttons."""
        if self._resolved_interactions is None:
            self._resolved_interactions = SlackInteractions(timeout_seconds=self.approval_timeout_seconds)
        return self._resolved_interactions

    def _resolve_prompt(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Route a Slack approval-button click to the suspended run."""
        return self._resolve_interactions().resolve(block_id=block_id, value=value, user_id=user_id)

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        """Render selected tool approvals in private messages to the approvers."""
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
            self._resolve_client(),
            self._resolve_interactions(),
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
            if context.files:
                file_ids = ', '.join(f'`{file.file_id}`' for file in context.files)
                conversation += f' File IDs attached to the invoking message: {file_ids}.'
            if not context.active_entities:
                return conversation

            def render_value(value: str | SlackMessageContext) -> str:
                if isinstance(value, SlackMessageContext):
                    return f'channel `{value.channel_id}`, message `{value.message_ts}`'
                return f'`{value}`'

            active = ', '.join(
                f'`{entity.entity_type}` = {render_value(entity.value)}' for entity in context.active_entities
            )
            return f"{conversation} The user's active Slack view, in relevance order, is: {active}."

        return [base, current_conversation]
