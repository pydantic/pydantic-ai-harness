"""Typed addressing for a Slack conversation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

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
