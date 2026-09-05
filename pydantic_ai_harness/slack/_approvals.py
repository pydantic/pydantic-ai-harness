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
from pydantic_ai_harness.slack._interactions import MAX_QUESTION_CHARS, SlackInteractions
from pydantic_ai_harness.slack._thread import SlackDepsT, SlackThread, ThreadResolver, resolve_thread
from pydantic_ai_harness.slack._validate import string_sequence

APPROVE = 'Approve'
DENY = 'Deny'


class SlackApprovals(Generic[SlackDepsT]):
    """Ask in Slack before a tool that requires approval runs.

    [`SlackChat(approvals=True)`][pydantic_ai_harness.slack.SlackChat] sets this
    up for you. Build one directly to pass as the handler of
    `HandleDeferredToolCalls`, or to give approvals their own reviewer group.

    Unlike an `ask_user` tool, this is not something the model decides to do. The
    gate is on the tool, so a model that would rather not ask does not get to
    skip it -- which is the point of having it.

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
        thread: SlackThread | ThreadResolver[SlackDepsT] | None = None,
        allowed_user_ids: Collection[str] | None = None,
    ) -> None:
        """Ask through `interactions`, which must be the one the app resolves clicks against.

        Args:
            client: Slack client the prompts are posted with.
            interactions: Prompt registry. `SlackBot` finds this on the agent's
                `SlackChat` so button clicks reach the run waiting on them; pass
                the same instance explicitly when you build the app yourself.
            thread: Where to ask, or a callable working it out from the run
                context. Omit to ask in the thread the run is bound to.
            allowed_user_ids: Who may approve. Defaults to the person whose
                message started the run. Set it to a reviewer group when the
                requester should not approve their own agent's actions.

        Raises:
            ValueError: If `allowed_user_ids` is a string rather than a collection
                of ids.
        """
        self._client = client
        self._interactions = interactions
        self._thread = thread
        self._allowed_user_ids = (
            frozenset(string_sequence(allowed_user_ids, 'allowed_user_ids')) if allowed_user_ids is not None else None
        )

    async def __call__(self, ctx: RunContext[SlackDepsT], requests: DeferredToolRequests) -> DeferredToolResults | None:
        """Ask about every pending approval and build the results.

        Returns `None` when there is no Slack conversation to ask in, leaving the
        calls for another handler rather than approving them unasked.
        """
        thread = resolve_thread(self._thread, ctx)
        if thread is None:
            return None
        approvals: dict[str, bool | DeferredToolApprovalResult] = {}
        for call in requests.approvals:
            approvals[call.tool_call_id] = await self._decide(thread, call)
        return requests.build_results(approvals=approvals)

    async def _decide(self, thread: SlackThread, call: ToolCallPart) -> bool | DeferredToolApprovalResult:
        question = _question(call, _render(call))
        # Checked here rather than left to `ask`, so the denial says what to do
        # about it. Nothing is truncated to fit: approving half a call is
        # approving something nobody read.
        if len(question) > MAX_QUESTION_CHARS:
            return ToolDenied(
                f'This call was not offered for approval: showing it takes {len(question)} characters, '
                f'and Slack can show at most {MAX_QUESTION_CHARS}, so nobody could review the whole call. '
                'Call the tool with smaller arguments, for instance by writing the long part to a file first.'
            )
        try:
            answer = await self._interactions.ask(
                self._client,
                thread,
                question,
                [APPROVE, DENY],
                allowed_user_ids=self._allowed_user_ids,
            )
        except ValueError as error:
            # Nobody is allowed to answer, so nobody can approve. Denying says why
            # in the transcript rather than failing the run at the approval gate.
            return ToolDenied(f'This action could not be approved in Slack: {error}')
        if answer == APPROVE:
            return True
        if answer == DENY:
            return ToolDenied('A person denied this action in Slack.')
        return ToolDenied('Nobody approved this action in Slack in time, so it was not run.')


def _question(call: ToolCallPart, arguments: str | None) -> str:
    detail = f'\n```\n{arguments}\n```' if arguments else ''
    return f'Run `{call.tool_name}`?{detail}'


def _render(call: ToolCallPart) -> str | None:
    """The call's arguments as they will be shown, or `None` when it takes none."""
    # Arguments the model sent as malformed JSON come back wrapped under an
    # `INVALID_JSON` key rather than raising, and showing that is the point: the
    # person approving should see exactly what was asked for.
    arguments = call.args_as_dict()
    if not arguments:
        return None
    return json.dumps(arguments, indent=2, default=str, ensure_ascii=False)
