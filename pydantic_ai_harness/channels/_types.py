"""Provider-neutral channel types and conversation storage."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

from pydantic_ai.messages import ModelMessage
from typing_extensions import Self


class ChannelError(RuntimeError):
    """A channel transport or provider request failed."""


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """A text message normalized by a channel adapter.

    `conversation_id` and `sender_id` are opaque, adapter-scoped identifiers.
    `message_id` identifies the provider message for logging and deduplication.
    """

    conversation_id: str
    sender_id: str
    message_id: str
    text: str


class ChannelAdapter(Protocol):
    """Connect one messaging provider to a [`ChannelHost`][pydantic_ai_harness.channels.ChannelHost]."""

    async def __aenter__(self) -> Self:
        """Open adapter resources; providers may validate credentials here."""
        ...  # pragma: no cover

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close resources owned by the adapter."""
        ...  # pragma: no cover

    def messages(self) -> AsyncIterator[InboundMessage]:
        """Yield inbound messages until the adapter closes or the task is cancelled."""
        ...  # pragma: no cover

    async def send_text(self, conversation_id: str, text: str) -> None:
        """Send text once, applying provider-specific limits.

        Raise [`ChannelError`][pydantic_ai_harness.channels.ChannelError] when
        delivery does not complete. The caller must not assume a raised error
        means the provider did not receive the message.
        """
        ...  # pragma: no cover


class ConversationStore(Protocol):
    """Store Pydantic AI message history for one channel host."""

    async def load(self, conversation_id: str) -> Sequence[ModelMessage]:
        """Load the current history, or an empty sequence for a new conversation."""
        ...  # pragma: no cover

    async def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        """Replace the conversation history with `messages`."""
        ...  # pragma: no cover

    async def delete(self, conversation_id: str) -> None:
        """Delete a conversation's history if it exists."""
        ...  # pragma: no cover


class InMemoryConversationStore:
    """Keep conversation history in memory for the lifetime of one process."""

    def __init__(self) -> None:
        self._messages: dict[str, list[ModelMessage]] = {}

    async def load(self, conversation_id: str) -> Sequence[ModelMessage]:
        """Load a copy of the conversation history."""
        return list(self._messages.get(conversation_id, ()))

    async def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        """Replace the conversation history."""
        self._messages[conversation_id] = list(messages)

    async def delete(self, conversation_id: str) -> None:
        """Delete the conversation history if it exists."""
        self._messages.pop(conversation_id, None)
