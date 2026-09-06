"""Typed subset of the Slack Web API used by the host."""

from __future__ import annotations

from typing import Protocol


class SlackResponse(Protocol):
    """Typed subset of a Slack Web API response used at runtime boundaries."""

    def get(self, key: str) -> object: ...  # pragma: no cover


class SlackClient(Protocol):
    """The Slack Web API methods used by the host."""

    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str | None = None,
        markdown_text: str | None = None,
        thread_ts: str | None = None,
        mrkdwn: bool | None = None,
    ) -> SlackResponse:
        """Post a message, optionally into a thread."""
        ...  # pragma: no cover

    async def agents_sessions_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str | None = None,
        status: str,
    ) -> SlackResponse:
        """Set the lifecycle status for an agent session."""
        ...  # pragma: no cover
