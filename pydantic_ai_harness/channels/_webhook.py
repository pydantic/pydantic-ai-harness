"""Private queue bridge for webhook channel adapters."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import AsyncGenerator

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from pydantic_ai_harness.channels._types import InboundMessage


class WebhookInbox:
    def __init__(self, max_queued_messages: int) -> None:
        self._max_queued_messages = max_queued_messages
        self._send_stream: MemoryObjectSendStream[InboundMessage] | None = None
        self._receive_stream: MemoryObjectReceiveStream[InboundMessage] | None = None

    def open(self) -> None:
        self._send_stream, self._receive_stream = anyio.create_memory_object_stream[InboundMessage](
            self._max_queued_messages
        )

    def put(self, message: InboundMessage) -> bool:
        send_stream = self._send_stream
        if send_stream is None:  # pragma: no cover
            return False
        try:
            send_stream.send_nowait(message)
        except (anyio.WouldBlock, anyio.BrokenResourceError, anyio.ClosedResourceError):
            return False
        return True

    def close(self) -> None:
        send_stream = self._send_stream
        receive_stream = self._receive_stream
        self._send_stream = None
        self._receive_stream = None
        assert send_stream is not None and receive_stream is not None
        send_stream.close()
        receive_stream.close()

    async def messages(self) -> AsyncGenerator[InboundMessage, None]:
        receive_stream = self._receive_stream
        if receive_stream is None:  # pragma: no cover
            return
        try:
            async with receive_stream:
                async for message in receive_stream:
                    yield message
        except anyio.ClosedResourceError:
            return


class RecentMessageIds:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._ids: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, message_id: str) -> bool:
        return message_id in self._ids

    def add(self, message_id: str) -> None:
        self._ids[message_id] = None
        if len(self._ids) > self._max_size:
            self._ids.popitem(last=False)

    def clear(self) -> None:
        self._ids.clear()
