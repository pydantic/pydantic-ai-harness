"""A proposed change to the workspace: the diff a listener sees, and the request that announces it."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.filesystem._events import (
    MAX_EVENT_DIFF_CHARS,
    FileChangeRequestEvent,
    FileEditedEvent,
    FileOperation,
)

_REFUSALS: dict[FileOperation, str] = {
    'write': 'was not written',
    'edit': 'was not edited',
    'create_directory': 'was not created',
}


def unified_diff(old: str, new: str, *, path: str) -> tuple[str, bool]:
    """Unified diff from `old` to `new`, cut at `MAX_EVENT_DIFF_CHARS`.

    Returns the diff and whether it was cut. Two equal texts diff to an
    empty string, so a `create_directory` proposes no diff at all.
    """
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile=f'a/{path}', tofile=f'b/{path}', lineterm=''
    )
    diff = '\n'.join(lines)
    if len(diff) <= MAX_EVENT_DIFF_CHARS:
        return diff, False
    return diff[:MAX_EVENT_DIFF_CHARS], True


@dataclass(kw_only=True)
class Change:
    """A change to announce before it is applied and to report after."""

    path: str
    root_dir: str
    operation: FileOperation
    diff: str
    truncated: bool

    @classmethod
    def propose(cls, *, path: str, root_dir: str, operation: FileOperation, old: str = '', new: str = '') -> Change:
        diff, truncated = unified_diff(old, new, path=path)
        return cls(path=path, root_dir=root_dir, operation=operation, diff=diff, truncated=truncated)

    async def request(self, ctx: RunContext[AgentDepsT] | None) -> str | None:
        """Announce the change to the run's listeners.

        Returns the tool result for the model when a listener cancelled the
        change, or `None` when it may proceed. Outside a run there is nobody
        to ask, so a direct call always proceeds.
        """
        if ctx is None:
            return None
        event = FileChangeRequestEvent(
            path=self.path, root_dir=self.root_dir, operation=self.operation, diff=self.diff, truncated=self.truncated
        )
        await ctx.emit(event)
        if not event.cancelled:
            return None
        reason = event.cancel_reason or 'cancelled by a listener'
        return f'[{self.path!r} {_REFUSALS[self.operation]}: {reason}]'

    def edited(self, *, content_hash: str) -> FileEditedEvent:
        """The notification for this change once `edit_file` has applied it."""
        return FileEditedEvent(
            path=self.path, root_dir=self.root_dir, content_hash=content_hash, diff=self.diff, truncated=self.truncated
        )
