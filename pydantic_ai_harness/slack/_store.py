"""Conversation history keyed by Slack thread."""

from __future__ import annotations

import hashlib
import secrets
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

    Enough for a single-process bot that should survive a restart. Each write goes
    to its own temporary file that then replaces the old one, so a crash mid-write
    leaves the previous history intact rather than a truncated file, and two saves
    for the same conversation cannot overwrite each other's partial file. Which of
    two concurrent saves wins is still whichever finishes last; use a
    database-backed store when that matters.
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
        # These files hold whole conversations, so keep them to the owner rather
        # than whatever a umask of 022 would leave readable.
        await directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self._path(key)
        # A name unique to this write. A shared one lets two saves for the same key
        # clobber each other's half-written file and rename the wrong bytes in.
        temporary = anyio.Path(path.with_suffix(f'.{secrets.token_hex(8)}.tmp'))
        try:
            await temporary.touch(mode=0o600)
            await temporary.write_bytes(ModelMessagesTypeAdapter.dump_json(list(messages)))
            # `replace`, not `rename`: on Windows renaming onto an existing file raises.
            await temporary.replace(path)
        finally:
            # A failed write would otherwise leave the transcript sitting in the
            # temporary file. After a successful replace there is nothing to remove.
            await temporary.unlink(missing_ok=True)

    async def delete(self, key: str) -> None:
        """Drop the history file if it exists."""
        await anyio.Path(self._path(key)).unlink(missing_ok=True)
