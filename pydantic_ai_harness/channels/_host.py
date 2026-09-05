"""Provider-neutral channel event handling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol
from weakref import WeakValueDictionary

import anyio
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelEvent:
    """A text event normalized by a provider adapter.

    `reply_to_id` identifies the provider object that should receive the response. For a threaded
    provider this is the thread root; for a reply-based provider it is the source message.
    `delivery_id` identifies the provider destination when it differs from `conversation_id`.
    """

    event_id: str
    conversation_id: str
    sender_id: str
    text: str
    reply_to_id: str | None = None
    delivery_id: str | None = None


class ChannelAdapter(Protocol):
    """The outbound half of a messaging provider adapter."""

    async def reply(self, event: ChannelEvent, text: str) -> None:
        """Send `text` in response to `event`."""
        ...  # pragma: no cover - structural protocol


class ConversationStore(Protocol):
    """Storage for Pydantic AI message history keyed by normalized conversation ID."""

    async def load(self, conversation_id: str) -> Sequence[ModelMessage]:
        """Load the conversation's messages, or an empty sequence for a new conversation."""
        ...  # pragma: no cover - structural protocol

    async def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        """Replace the conversation's stored messages."""
        ...  # pragma: no cover - structural protocol


class InMemoryConversationStore:
    """Keep conversation history in this Python process."""

    def __init__(self) -> None:
        self._messages: dict[str, list[ModelMessage]] = {}

    async def load(self, conversation_id: str) -> Sequence[ModelMessage]:
        """Load a copy of the current history."""
        return list(self._messages.get(conversation_id, ()))

    async def save(self, conversation_id: str, messages: Sequence[ModelMessage]) -> None:
        """Store a copy of `messages`."""
        self._messages[conversation_id] = list(messages)


class ChannelHost(Generic[AgentDepsT]):
    """Run an agent for inbound channel events and send its text response.

    Events for one conversation run serially. Events for different conversations may overlap.
    Serialization is local to this host instance; a multi-process receiver should partition its
    queue by `conversation_id` or provide equivalent coordination.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, str],
        adapter: ChannelAdapter,
        *,
        store: ConversationStore | None = None,
    ) -> None:
        self._agent = agent
        self._adapter = adapter
        self._store = store if store is not None else InMemoryConversationStore()
        self._conversation_locks: WeakValueDictionary[str, anyio.Lock] = WeakValueDictionary()

    async def handle(self, event: ChannelEvent, *, deps: AgentDepsT = None) -> AgentRunResult[str]:
        """Run the agent for `event`, reply, then commit the resulting message history.

        The agent must produce text. Failures from the agent, adapter, or store are surfaced
        without host-level retries.
        """
        lock = self._conversation_locks.setdefault(event.conversation_id, anyio.Lock())
        async with lock:
            history = await self._store.load(event.conversation_id)
            result = await self._agent.run(
                event.text,
                message_history=history,
                conversation_id=event.conversation_id,
                deps=deps,
            )
            await self._adapter.reply(event, result.output)
            await self._store.save(event.conversation_id, result.all_messages())
            return result
