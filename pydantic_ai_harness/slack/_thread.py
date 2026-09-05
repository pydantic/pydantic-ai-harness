"""Per-message Slack context passed to an agent run as `deps`."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'slack-sdk is required for the Slack package. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error


@dataclass(frozen=True, slots=True, kw_only=True)
class SlackThread:
    """Where one agent run is talking, and to whom.

    Build one per inbound Slack message and pass it as the run's `deps`. The
    tools in [`SlackChatToolset`][pydantic_ai_harness.slack.SlackChatToolset]
    read the destination from here, so a single `Agent` serves every thread.

    `thread_ts` is the thread root, not the triggering message: a mention that
    starts a thread uses its own `ts`, and a reply inside a thread uses the
    thread's existing `thread_ts`. [`conversation_key`][pydantic_ai_harness.slack.conversation_key]
    derives the history key from the same value.
    """

    client: AsyncWebClient
    """Authenticated Slack Web API client used for every call in this run."""

    channel_id: str
    """Channel, group, or DM the message arrived in."""

    thread_ts: str
    """Timestamp of the thread root that replies belong to."""

    user_id: str
    """Slack user who sent the message. Approval prompts default to this person."""

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


def conversation_key(*, channel_id: str, thread_ts: str, team_id: str | None = None) -> str:
    """Build the history key for one Slack thread.

    Kept separate from [`SlackThread`][pydantic_ai_harness.slack.SlackThread] so an
    application can compute the key straight from a raw Slack event -- for
    instance to look up history before deciding whether to start a run.
    """
    prefix = f'{team_id}:' if team_id else ''
    return f'{prefix}{channel_id}:{thread_ts}'
