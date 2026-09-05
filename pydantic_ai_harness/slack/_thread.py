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
        return conversation_key(channel_id=self.channel_id, thread_ts=self.thread_ts, team_id=self.team_id)


def conversation_key(*, channel_id: str, thread_ts: str | None = None, team_id: str | None = None) -> str:
    """Build the history key for one Slack thread.

    Kept separate from [`SlackThread`][pydantic_ai_harness.slack.SlackThread] so an
    application can compute the key straight from a raw Slack event -- for
    instance to look up history before deciding whether to start a run.
    """
    prefix = f'{team_id}:' if team_id else ''
    return f'{prefix}{channel_id}:{thread_ts}' if thread_ts else f'{prefix}{channel_id}'


_CURRENT_THREAD: ContextVar[SlackThread | None] = ContextVar('pydantic_ai_harness.slack.thread', default=None)


@contextmanager
def bind_thread(thread: SlackThread) -> Generator[None]:
    """Say that everything inside this block is talking to `thread`.

    [`SlackChat`][pydantic_ai_harness.slack.SlackChat]'s tools post to the bound
    thread, so an agent answering a Slack message replies where it was asked
    without the thread going anywhere near its `deps`.
    [`SlackBot`][pydantic_ai_harness.slack.SlackBot] binds it for you; bind it
    yourself when you drive the agent from your own Slack listeners.

    The binding follows into tasks started inside the block, so tools running
    concurrently in one turn all see the same thread. It does not cross a process
    boundary: under durable execution, configure `SlackChat(thread=...)` or
    `SlackChat(channels=[...])` instead, which a worker can rebuild from the run
    context.
    """
    token = _CURRENT_THREAD.set(thread)
    try:
        yield
    finally:
        _CURRENT_THREAD.reset(token)


def current_thread() -> SlackThread | None:
    """The thread bound by [`bind_thread`][pydantic_ai_harness.slack.bind_thread], if any."""
    return _CURRENT_THREAD.get()


ThreadResolver = Callable[[RunContext[SlackDepsT]], 'SlackThread | None']
"""Works out which Slack thread a run is talking to, from its run context."""


def resolve_thread(
    thread: SlackThread | ThreadResolver[SlackDepsT] | None, ctx: RunContext[SlackDepsT]
) -> SlackThread | None:
    """Which thread a configured `thread` means for this run, if any.

    A fixed thread is itself, a resolver is asked, and an unset one falls back to
    whatever [`bind_thread`][pydantic_ai_harness.slack.bind_thread] set. Every
    piece that posts settles it this way, so an agent's tools and its approval
    prompts always land in the same conversation.
    """
    if callable(thread):
        thread = thread(ctx)
    return thread if thread is not None else current_thread()
