"""The part of the Slack Web API this package calls.

`slack_sdk` ships `py.typed` but leaves generics unparameterised in its
signatures -- bare `Dict`, `List`, `PathLike`, and an unannotated `**kwargs` --
and one unknown anywhere in a signature makes the whole method partially
unknown, so strict Pyright reports every call through `AsyncWebClient`.

Naming the four methods used here fixes that without a suppression at each call,
and it checks more than suppressions did: a real `AsyncWebClient` is verified
against this protocol wherever one is assigned to
[`SlackThread.client`][pydantic_ai_harness.slack.SlackThread], so a method the
SDK renames or retypes fails there. Keyword names are checked too, which a
suppressed call is not -- the SDK's own `**kwargs` accepts anything.

The signatures mirror `AsyncWebClient`'s, so a real client satisfies this as-is
and nothing has to be wrapped or adapted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

try:
    from slack_sdk.web.async_slack_response import AsyncSlackResponse
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'slack-sdk is required for the Slack package. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error


class SlackClient(Protocol):
    """The Slack Web API methods this package calls.

    `slack_sdk`'s `AsyncWebClient` satisfies this as it stands. Implement it to
    route calls elsewhere, or to stand in for Slack in tests.
    """

    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str | None = None,
        blocks: Sequence[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
    ) -> AsyncSlackResponse:
        """Post a message, optionally into a thread."""
        ...  # pragma: no cover

    async def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        text: str | None = None,
        blocks: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncSlackResponse:
        """Replace the text and blocks of a message already posted."""
        ...  # pragma: no cover

    async def files_upload_v2(
        self,
        *,
        channel: str | None = None,
        file: str | None = None,
        title: str | None = None,
        initial_comment: str | None = None,
        thread_ts: str | None = None,
    ) -> AsyncSlackResponse:
        """Upload a file by path and share it into a channel or thread.

        The SDK also takes bytes and open file objects. Only the path form is
        named here, because that is all this package sends.
        """
        ...  # pragma: no cover

    async def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str,
    ) -> AsyncSlackResponse:
        """Set the working-state line shown in an agent or assistant thread."""
        ...  # pragma: no cover
