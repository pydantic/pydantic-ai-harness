"""Tools that let an agent talk to the Slack thread it is running in.

The `reportUnknownMemberType` suppressions on the Slack calls here and in the
sibling modules cover one thing only: `slack_sdk` types every Web API method
with a trailing `**kwargs`, so strict Pyright cannot fully resolve the call.
They do not hide a renamed or removed method (`reportAttributeAccessIssue`) or
a changed parameter type (`reportArgumentType`), both of which still fail the
build. A renamed keyword argument is the one drift that gets through, because
the SDK's own `**kwargs` accepts it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.slack._interactions import SlackInteractions
from pydantic_ai_harness.slack._thread import SlackThread

StepStatus = Literal['pending', 'running', 'done', 'failed']
"""State of one plan step, rendered as an icon in the posted checklist."""

_STATUS_ICONS: dict[StepStatus, str] = {
    'pending': ':white_circle:',
    'running': ':hourglass_flowing_sand:',
    'done': ':white_check_mark:',
    'failed': ':x:',
}

_MAX_MESSAGE_CHARS = 3500
"""Below Slack's 4000-character limit, leaving room for the mrkdwn Slack adds."""

_MAX_PLAN_STEPS = 20


class PlanStep(BaseModel):
    """One line of the checklist the agent shows while it works."""

    text: str = Field(description='What this step does, in a few words.')
    status: StepStatus = Field(default='pending', description='Where this step has got to.')


class SlackChatToolset(FunctionToolset[SlackThread]):
    """Tools for reporting progress, asking questions, and sending files in Slack.

    The agent's `deps` must be a [`SlackThread`][pydantic_ai_harness.slack.SlackThread]
    naming where the run is talking. Every tool posts into that thread, so one
    `Agent` serves every conversation.

    What the model chooses to say belongs here. Mechanical liveness -- a typing
    indicator, an error notice when a run fails -- belongs in the application,
    which knows those things without asking the model.

    `upload_file` is only registered when `file_root` is set. Without it there is
    no directory to judge a model-supplied path against, and uploading an
    arbitrary path off the host is not a sensible default.
    """

    def __init__(
        self,
        *,
        interactions: SlackInteractions | None = None,
        file_root: Path | str | None = None,
        max_message_chars: int = _MAX_MESSAGE_CHARS,
    ) -> None:
        """Configure which tools the agent gets.

        Args:
            interactions: Shared prompt registry backing `ask_user`. Omit to leave
                `ask_user` unregistered, which is right for an agent that should
                never block waiting for a person.
            file_root: Directory that `upload_file` may read from. Paths outside it
                are refused. Omit to leave `upload_file` unregistered.
            max_message_chars: Longest single message `post_message` will send.
                Longer text is refused with a retry telling the model to split it,
                rather than being truncated where a reader cannot see the loss.
        """
        if max_message_chars <= 0:
            raise ValueError('max_message_chars must be positive')
        self._interactions = interactions
        self._file_root = Path(file_root).expanduser().resolve() if file_root is not None else None
        self._max_message_chars = max_message_chars
        self._plans: dict[str, str] = {}

        super().__init__()
        self.add_function(self.post_message)
        self.add_function(self.post_plan)
        self.add_function(self.set_status)
        if interactions is not None:
            self.add_function(self.ask_user)
        if self._file_root is not None:
            self.add_function(self.upload_file)

    async def post_message(self, ctx: RunContext[SlackThread], text: str) -> str:
        """Post a message into the Slack thread now, without ending your turn.

        Use this to report something worth knowing while you keep working: what
        you found, what you are about to do, why you changed approach. Your final
        answer is delivered separately, so do not repeat it here.
        """
        if len(text) > self._max_message_chars:
            raise ModelRetry(
                f'That message is {len(text)} characters and Slack takes at most {self._max_message_chars}. '
                'Send it as several shorter messages, or put the long form in a file.'
            )
        if not text.strip():
            raise ModelRetry('Message text cannot be empty.')
        thread = ctx.deps
        await thread.client.chat_postMessage(channel=thread.channel_id, thread_ts=thread.thread_ts, text=text)  # pyright: ignore[reportUnknownMemberType]
        return 'Posted.'

    async def post_plan(self, ctx: RunContext[SlackThread], steps: list[PlanStep]) -> str:
        """Show or update the checklist of what you are doing.

        Call it once with the whole plan before starting multi-step work, then
        call it again with the same steps and updated statuses as you go. Each
        call replaces the previous checklist rather than posting a new one, so
        send the complete list every time.
        """
        if not steps:
            raise ModelRetry('A plan needs at least one step.')
        if len(steps) > _MAX_PLAN_STEPS:
            raise ModelRetry(f'Keep the plan to {_MAX_PLAN_STEPS} steps or fewer; group the smaller ones.')
        thread = ctx.deps
        text = '\n'.join(f'{_STATUS_ICONS[step.status]} {step.text}' for step in steps)
        existing = self._plans.get(thread.key)
        if existing is None:
            response = await thread.client.chat_postMessage(  # pyright: ignore[reportUnknownMemberType]
                channel=thread.channel_id, thread_ts=thread.thread_ts, text=text
            )
            timestamp = response.get('ts')
            if isinstance(timestamp, str):
                self._plans[thread.key] = timestamp
            return 'Plan posted.'
        await thread.client.chat_update(channel=thread.channel_id, ts=existing, text=text)  # pyright: ignore[reportUnknownMemberType]
        return 'Plan updated.'

    async def set_status(self, ctx: RunContext[SlackThread], status: str) -> str:
        """Set the short working-state line shown next to your name.

        For example `reading the deploy logs`. It is replaced by your next status
        and is not part of the conversation, so use it for the current activity
        only.
        """
        thread = ctx.deps
        try:
            await thread.client.assistant_threads_setStatus(  # pyright: ignore[reportUnknownMemberType]
                channel_id=thread.channel_id, thread_ts=thread.thread_ts, status=status
            )
        except Exception:
            # The status line only exists in agent and assistant threads. In a
            # plain channel thread Slack rejects the call, and that is not a
            # reason to fail the turn or make the model retry.
            return 'Status is not available in this conversation; carry on without it.'
        return 'Status set.'

    async def ask_user(self, ctx: RunContext[SlackThread], question: str, options: list[str]) -> str:
        """Ask the person a multiple-choice question and wait for them to answer.

        Use it when you genuinely cannot proceed without a decision that is theirs
        to make. It blocks your turn until someone clicks, so do not use it to
        confirm things you can reasonably decide yourself.
        """
        if self._interactions is None:  # pragma: no cover - not registered without interactions
            raise ModelRetry('Asking the user is not enabled for this agent.')
        answer = await self._interactions.ask(ctx.deps, question, options)
        if answer is None:
            return 'Nobody answered in time. Choose a reasonable default, say which you chose, and carry on.'
        return answer

    async def upload_file(
        self,
        ctx: RunContext[SlackThread],
        path: str,
        title: str | None = None,
        comment: str | None = None,
    ) -> str:
        """Send a file you produced into the thread.

        Prefer this over pasting long output into a message: a spreadsheet, a
        report, a diff, or an image reads far better as a file. `path` must be
        inside the directory this agent was given.
        """
        if self._file_root is None:  # pragma: no cover - not registered without a root
            raise ModelRetry('Uploading files is not enabled for this agent.')
        resolved = (self._file_root / path).resolve()
        if not resolved.is_relative_to(self._file_root):
            raise ModelRetry(f'{path} is outside the directory you may send files from.')
        if not resolved.is_file():
            raise ModelRetry(f'There is no file at {path}. Write it first, then send it.')
        thread = ctx.deps
        await thread.client.files_upload_v2(  # pyright: ignore[reportUnknownMemberType]
            channel=thread.channel_id,
            thread_ts=thread.thread_ts,
            file=str(resolved),
            title=title,
            initial_comment=comment,
        )
        return f'Sent {resolved.name}.'
