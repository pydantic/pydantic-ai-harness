"""The Slack Web API surface the Slack primitives call.

Typed as a `Protocol` so the package works against `slack_sdk`'s
`AsyncWebClient` without importing it, and so tests can substitute a fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class SlackResponse(Protocol):
    """The part of a `slack_sdk` API response these primitives read."""

    def get(self, key: str, default: object = None) -> object:
        """Return the value for `key`, or `default` when the response omits it."""
        ...  # pragma: no cover


class SlackClient(Protocol):
    """The subset of `slack_sdk.web.async_client.AsyncWebClient` used here.

    A real `AsyncWebClient` satisfies this as-is: the methods below name only the
    keyword arguments these primitives pass, and an implementation may accept
    more. Pass any object with these methods to substitute a different transport
    or a fake.
    """

    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str | None = None,
        blocks: Sequence[object] | None = None,
        thread_ts: str | None = None,
    ) -> SlackResponse:
        """Post a message, optionally into a thread."""
        ...  # pragma: no cover

    async def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        text: str | None = None,
        blocks: Sequence[object] | None = None,
    ) -> SlackResponse:
        """Replace the text and blocks of a message already posted."""
        ...  # pragma: no cover

    async def files_upload_v2(
        self,
        *,
        channel: str,
        file: str,
        title: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> SlackResponse:
        """Upload one file and share it into a channel or thread."""
        ...  # pragma: no cover

    async def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str,
    ) -> SlackResponse:
        """Set the working-state line shown while the agent runs."""
        ...  # pragma: no cover
