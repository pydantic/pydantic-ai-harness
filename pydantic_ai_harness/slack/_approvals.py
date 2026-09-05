"""Resolve tool-approval requests by asking in Slack."""

from __future__ import annotations

import json
from collections.abc import Collection

from pydantic_ai import ToolDenied
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolApprovalResult, DeferredToolRequests, DeferredToolResults, RunContext

from pydantic_ai_harness.slack._interactions import SlackInteractions
from pydantic_ai_harness.slack._thread import SlackThread

APPROVE = 'Approve'
DENY = 'Deny'

_MAX_ARGUMENT_CHARS = 2900
"""Longest rendered arguments a prompt shows. Slack caps one section block at 3000
characters, and the arguments are never truncated to fit: approving half a call is
approving something nobody read. A call whose arguments do not fit is denied
instead."""


class SlackApprovals:
    """Ask in Slack before a tool that requires approval runs.

    Pass an instance as the handler of `HandleDeferredToolCalls`. Each pending
    call is posted as a question with Approve and Deny buttons, and the run
    continues as soon as someone answers.

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
        interactions: SlackInteractions,
        *,
        allowed_user_ids: Collection[str] | None = None,
    ) -> None:
        """Ask through `interactions`, which must be the one the app resolves clicks against.

        Args:
            interactions: Shared prompt registry. Give the same instance to
                [`SlackAgent`][pydantic_ai_harness.slack.SlackAgent] so button
                clicks reach the run waiting on them.
            allowed_user_ids: Who may approve. Defaults to the person whose
                message started the run. Set it to a reviewer group when the
                requester should not approve their own agent's actions.

        Raises:
            ValueError: If `allowed_user_ids` is a string rather than a collection
                of ids.
        """
        if isinstance(allowed_user_ids, str):
            raise ValueError('allowed_user_ids must be a collection of user ids, not a string')
        self._interactions = interactions
        self._allowed_user_ids = frozenset(allowed_user_ids) if allowed_user_ids is not None else None

    async def __call__(self, ctx: RunContext[SlackThread], requests: DeferredToolRequests) -> DeferredToolResults:
        """Ask about every pending approval and build the results."""
        approvals: dict[str, bool | DeferredToolApprovalResult] = {}
        for call in requests.approvals:
            approvals[call.tool_call_id] = await self._decide(ctx.deps, call)
        return requests.build_results(approvals=approvals)

    async def _decide(self, thread: SlackThread, call: ToolCallPart) -> bool | DeferredToolApprovalResult:
        arguments = _render(call)
        if arguments is not None and len(arguments) > _MAX_ARGUMENT_CHARS:
            return ToolDenied(
                f'This call was not offered for approval: its arguments are {len(arguments)} characters, '
                f'and Slack can show at most {_MAX_ARGUMENT_CHARS}, so nobody could review the whole call. '
                'Call the tool with smaller arguments, for instance by writing the long part to a file first.'
            )
        answer = await self._interactions.ask(
            thread,
            _question(call, arguments),
            [APPROVE, DENY],
            allowed_user_ids=self._allowed_user_ids,
        )
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
