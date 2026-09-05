"""The part of the Slack Web API this package calls.

`slack_sdk` ships `py.typed` but leaves generics unparameterised in its
signatures -- bare `Dict`, `List`, `PathLike`, and an unannotated `**kwargs` --
and one unknown anywhere in a signature makes the whole method partially
unknown, so strict Pyright reports every call through `AsyncWebClient`.

Naming the methods used here fixes that without a suppression at each call,
and it checks more than suppressions did: the static conformance check below
assigns a real `AsyncWebClient` to `SlackClient`, so a method the SDK renames or
retypes fails. Keyword names are checked too, which a suppressed call is not --
the SDK's own `**kwargs` accepts anything.

The signatures mirror `AsyncWebClient`'s, so a real client satisfies this as-is
and nothing has to be wrapped or adapted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

try:
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'slack-sdk is required for the Slack package. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error


class SlackResponse(Protocol):
    """Typed subset of a Slack Web API response used at runtime boundaries."""

    def get(self, key: str) -> object: ...  # pragma: no cover


class SlackClient(Protocol):
    """The Slack Web API methods this package calls.

    `slack_sdk`'s `AsyncWebClient` satisfies this as it stands. Implement it to
    route calls elsewhere, or to stand in for Slack in tests.
    """

    async def conversations_open(self, *, users: str | Sequence[str]) -> SlackResponse:
        """Open or resume a private conversation with one or more users."""
        ...  # pragma: no cover

    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str | None = None,
        markdown_text: str | None = None,
        blocks: Sequence[dict[str, object]] | None = None,
        thread_ts: str | None = None,
        mrkdwn: bool | None = None,
    ) -> SlackResponse:
        """Post a message, optionally into a thread."""
        ...  # pragma: no cover

    async def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        text: str | None = None,
        blocks: Sequence[dict[str, object]] | None = None,
        mrkdwn: bool | None = None,
    ) -> SlackResponse:
        """Replace the text and blocks of a message already posted."""
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


if TYPE_CHECKING:
    _sdk_conformance_check: SlackClient = AsyncWebClient(token='xoxb-type-check')
