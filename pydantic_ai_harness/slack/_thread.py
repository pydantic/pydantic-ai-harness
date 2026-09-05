"""Where a Slack message goes, and how a run says which thread it is in."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

from pydantic_ai.tools import RunContext

SlackDepsT = TypeVar('SlackDepsT')
"""The deps type a thread resolver reads. Its own variable because pydantic-ai's
`AgentDepsT` is contravariant, which a resolver's parameter position cannot use."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackThread:
    """A place in Slack: a channel, and optionally a thread inside it.

    Holds addressing only -- no client, no credentials -- so it is cheap to
    build, safe to log, and can be rebuilt from stored state.

    `thread_ts` is the thread root, not the triggering message: a mention that
    starts a thread uses its own `ts`, and a reply inside a thread uses the
    thread's existing `thread_ts`. Leave it unset to post to the channel itself.
    """

    channel_id: str
    """Channel, group, or DM to post in. A name like `#alerts` works too."""

    thread_ts: str | None = None
    """Thread root replies belong to. Unset posts to the channel, not a thread."""

    user_id: str | None = None
    """Slack user who asked. Approval prompts default to this person."""

    team_id: str | None = None
    """Workspace the message came from. Included in the conversation key when set,
    which keeps history separate across workspaces for a multi-workspace install."""

    @property
    def key(self) -> str:
        """Stable identifier for this conversation's history.

        Pass it as `conversation_id` to `Agent.run()` and use it as the
        [`ConversationStore`][pydantic_ai_harness.slack.ConversationStore] key.
        """
        prefix = f'{self.team_id}:' if self.team_id else ''
        return f'{prefix}{self.channel_id}:{self.thread_ts}' if self.thread_ts else f'{prefix}{self.channel_id}'


_CURRENT_THREAD: ContextVar[SlackThread | None] = ContextVar('pydantic_ai_harness.slack.thread', default=None)


@contextmanager
def bind_thread(thread: SlackThread) -> Generator[None]:
    """Say that everything inside this block is talking to `thread`.

    `SlackApp` uses this internal compatibility context for direct
    `SlackApprovals` handlers. New integrations should use `SlackContext`.
    """
    token = _CURRENT_THREAD.set(thread)
    try:
        yield
    finally:
        _CURRENT_THREAD.reset(token)


def current_thread() -> SlackThread | None:
    """The Slack thread bound around the current run, if any."""
    return _CURRENT_THREAD.get()


ThreadResolver = Callable[[RunContext[SlackDepsT]], 'SlackThread | None']
"""Works out which Slack thread a run is talking to, from its run context."""


def resolve_thread(
    thread: SlackThread | ThreadResolver[SlackDepsT] | None, ctx: RunContext[SlackDepsT]
) -> SlackThread | None:
    """Which thread a configured `thread` means for this run, if any.

    A fixed thread is itself, a resolver is asked, and an unset one falls back to
    the thread bound around the current run.
    """
    if callable(thread):
        thread = thread(ctx)
    return thread if thread is not None else current_thread()
