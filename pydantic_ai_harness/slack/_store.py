"""Conversation history keyed by Slack thread."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import anyio
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter


class ConversationStore(Protocol):
    """Where a Slack agent keeps one thread's message history.

    Implement this to put history somewhere that outlives the process. Keys come
    from [`conversation_key`][pydantic_ai_harness.slack.conversation_key], so they
    are safe to use verbatim as row ids.
    """

    async def load(self, key: str) -> Sequence[ModelMessage]:
        """Return the stored history, or an empty sequence for a new conversation."""
        ...  # pragma: no cover

    async def save(self, key: str, messages: Sequence[ModelMessage]) -> None:
        """Replace the stored history."""
        ...  # pragma: no cover

    async def delete(self, key: str) -> None:
        """Drop the history if it exists. Used by a reset command."""
        ...  # pragma: no cover


class InMemoryConversationStore:
    """Keep history in this process only.

    History is lost on restart, which is usually the wrong trade for an agent
    people talk to over days. Use it for tests and local experiments, and
    [`FileConversationStore`][pydantic_ai_harness.slack.FileConversationStore] or
    your own database-backed store otherwise.
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[ModelMessage]] = {}

    async def load(self, key: str) -> Sequence[ModelMessage]:
        """Return a copy of the stored history."""
        return list(self._messages.get(key, ()))

    async def save(self, key: str, messages: Sequence[ModelMessage]) -> None:
        """Replace the stored history with a copy of `messages`."""
        self._messages[key] = list(messages)

    async def delete(self, key: str) -> None:
        """Drop the history if it exists."""
        self._messages.pop(key, None)


class FileConversationStore:
    """Keep each conversation's history in its own JSON file.

    Enough for a single-process bot that should survive a restart. Writes go to a
    temporary file that is then renamed, so a crash mid-write leaves the previous
    history intact rather than a truncated file. Concurrent writers in separate
    processes are not coordinated; use a database-backed store for that.
    """

    def __init__(self, directory: Path | str) -> None:
        """Store conversations under `directory`, creating it on first write."""
        self._directory = Path(directory).expanduser()

    def _path(self, key: str) -> Path:
        # Keys carry Slack ids and separators, so hash rather than sanitize: the
        # digest is a valid filename on every platform and cannot collide with a
        # sibling key through character replacement.
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._directory / f'{digest}.json'

    async def load(self, key: str) -> Sequence[ModelMessage]:
        """Return the stored history, or an empty sequence when the file is absent."""
        path = anyio.Path(self._path(key))
        try:
            raw = await path.read_bytes()
        except FileNotFoundError:
            return []
        return ModelMessagesTypeAdapter.validate_json(raw)

    async def save(self, key: str, messages: Sequence[ModelMessage]) -> None:
        """Write the history, replacing any previous file for this key."""
        directory = anyio.Path(self._directory)
        await directory.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        temporary = anyio.Path(path.with_suffix('.json.tmp'))
        await temporary.write_bytes(ModelMessagesTypeAdapter.dump_json(list(messages)))
        await temporary.rename(path)

    async def delete(self, key: str) -> None:
        """Drop the history file if it exists."""
        await anyio.Path(self._path(key)).unlink(missing_ok=True)
