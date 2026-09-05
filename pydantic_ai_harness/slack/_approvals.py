"""Resolve tool-approval requests by asking in Slack."""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Generic

from pydantic_ai import ToolDenied
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    DeferredToolApprovalResult,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
)

from pydantic_ai_harness.slack._client import SlackClient
from pydantic_ai_harness.slack._interactions import APPROVE, DENY, MAX_QUESTION_CHARS, SlackInteractions
from pydantic_ai_harness.slack._thread import SlackDepsT, SlackThread


class SlackApprovals(Generic[SlackDepsT]):
    """Ask in Slack before a tool that requires approval runs.

    `Slack` sets this internal handler up for each Slack-hosted run.

    This is not something the model decides to do. The gate is on the tool, so a
    model cannot skip it.

    A prompt nobody answers is denied, and so is a call whose arguments are too
    long for Slack to show in full. An agent with write access to real systems
    should not act because a question timed out, or because the part of the call
    that mattered scrolled off the end.
    """

    def __init__(
        self,
        client: SlackClient,
        interactions: SlackInteractions,
        *,
        thread: SlackThread,
        allowed_user_ids: Collection[str] | None = None,
    ) -> None:
        """Ask through `interactions`, which must be the one the app resolves clicks against.

        Args:
            client: Slack client the prompts are posted with.
            interactions: Prompt registry that lets `SlackApp` route button
                clicks to the run waiting on them.
            thread: Where to ask.
            allowed_user_ids: Who may approve. Defaults to the person whose
                message started the run. Set it to a reviewer group when the
                requester should not approve their own agent's actions.

        """
        self._client = client
        self._interactions = interactions
        self._thread = thread
        # Copied, not kept: these decide who may approve, so a list the caller
        # edits later should not quietly change who this handler accepts.
        self._allowed_user_ids = list(allowed_user_ids) if allowed_user_ids is not None else None

    async def __call__(self, _ctx: RunContext[SlackDepsT], requests: DeferredToolRequests) -> DeferredToolResults:
        """Ask about every pending approval and build the results.

        Every requested approval receives a Slack decision.
        """
        approvals: dict[str, bool | DeferredToolApprovalResult] = {}
        for call in requests.approvals:
            approvals[call.tool_call_id] = await self._decide(self._thread, call)
        return requests.build_results(approvals=approvals)

    async def _decide(self, thread: SlackThread, call: ToolCallPart) -> bool | DeferredToolApprovalResult:
        question = _question(thread, call, _render(call))
        # Checked here rather than left to `ask`, so the denial says what to do
        # about it. Nothing is truncated to fit: approving half a call is
        # approving something nobody read.
        if len(question) > MAX_QUESTION_CHARS:
            return ToolDenied(
                f'This call was not offered for approval: showing it takes {len(question)} characters, '
                f'and Slack can show at most {MAX_QUESTION_CHARS}, so nobody could review the whole call. '
                'Call the tool with smaller arguments, for instance by writing the long part to a file first.'
            )
        answer = await self._interactions.ask(
            self._client,
            thread,
            question,
            allowed_user_ids=self._allowed_user_ids,
        )
        if answer == APPROVE:
            return True
        if answer == DENY:
            return ToolDenied('A person denied this action in Slack.')
        return ToolDenied('Nobody approved this action in Slack in time, so it was not run.')


def _question(thread: SlackThread, call: ToolCallPart, arguments: str | None) -> str:
    source = f'Workspace: {thread.team_id or "unknown"}\nChannel: {thread.channel_id}'
    if thread.thread_ts is not None:  # pragma: no branch - SlackApp approval runs always have a thread root
        source += f'\nThread: {thread.thread_ts}'
    source += f'\nRequested by: {thread.user_id or "unknown"}'
    detail = f'\n\nArguments:\n{arguments}' if arguments else ''
    return f'{source}\n\nRun {call.tool_name}?{detail}'


def _render(call: ToolCallPart) -> str | None:
    """The call's arguments as they will be shown, or `None` when it takes none."""
    # Arguments the model sent as malformed JSON come back wrapped under an
    # `INVALID_JSON` key rather than raising, and showing that is the point: the
    # person approving should see exactly what was asked for.
    arguments = call.args_as_dict()
    if not arguments:
        return None
    return json.dumps(arguments, indent=2, default=str, ensure_ascii=False)
