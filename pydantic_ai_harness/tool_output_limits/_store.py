"""Storage backends for spilled tool outputs.

`WorkspaceStore` is the default: it writes each payload to a file inside the run's workspace
(`ctx.workspace`), so with a remote sandbox the spilled files live in the sandbox, next to the
files the agent's other tools see. `OverflowStore` is the narrow protocol for any other backend
(a blob store, a durable engine's storage): persist a payload under a key, read it back by
handle. `LocalFileStore` implements it on the host filesystem for runs with no workspace.
"""

from __future__ import annotations

import posixpath
import re
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic_ai.workspaces import Workspace, WorkspaceError

from pydantic_ai_harness._workspace import supports_commands

_WORKSPACE_SUBDIR = 'pydantic-ai-harness/tool-output'
"""Where spills go below the workspace's `$TMPDIR` (or `/tmp`); Shell keeps its jobs beside it."""

_FILESYSTEM_ONLY_SUBDIR = '.pydantic-ai-harness/tool-output'
"""Where spills go below the working directory of a workspace that cannot run commands."""

_DIRECTORY_TIMEOUT = 30.0
"""Deadline for the command that locates the spill directory; it returns at once in a healthy workspace."""


@runtime_checkable
class OverflowStore(Protocol):
    """Persist and retrieve spilled tool-output payloads outside the run's workspace.

    `write` takes a caller-chosen `key` and returns a `handle`. The handle is the only
    thing a later `read` needs, so it must be self-contained (a backend can encode the
    run, the call, and the retry into it). Implementations may return the key unchanged.
    """

    async def write(self, key: str, data: bytes) -> str:
        """Persist `data` under `key` and return a handle that `read` accepts."""
        ...  # pragma: no cover

    async def read(self, handle: str) -> bytes:
        """Return the payload previously stored for `handle`.

        Raise `FileNotFoundError` (or another `OSError`) when the handle is unknown.
        """
        ...  # pragma: no cover


_UNSAFE_SEGMENT = re.compile(r'[^A-Za-z0-9._-]+')


def _safe_segment(segment: str) -> str:
    """Make one path segment filesystem-safe without collapsing distinct keys.

    Empty or dot-only segments are replaced so a handle can never escape the root via
    `.`/`..`; the resolve-within-root check in `read` is the second line of defense.
    """
    cleaned = _UNSAFE_SEGMENT.sub('_', segment)
    if cleaned in ('', '.', '..'):
        return '_'
    return cleaned


def _segments(key: str) -> list[str]:
    return [_safe_segment(part) for part in key.split('/') if part] or ['_']


@dataclass
class WorkspaceStore:
    """The default store: each spilled payload is a file inside the run's workspace.

    The handle is the file's absolute workspace path, so the model can also open it with any
    workspace file tool (such as `FileSystem`'s `read_file`). Files live under
    `$TMPDIR/pydantic-ai-harness/tool-output` in the workspace (`/tmp` when `TMPDIR` is unset),
    created owner-only; a workspace that cannot run commands uses
    `.pydantic-ai-harness/tool-output` below its working directory instead.

    Files are not pruned: they go away with the sandbox, or with the host's temp-dir cleanup
    for a local workspace. Set `directory` to place them elsewhere.
    """

    directory: str | None = None
    """Workspace directory for spilled files, absolute or relative to the working directory."""

    async def write(self, workspace: Workspace, key: str, data: bytes) -> str:
        """Write `data` into `workspace` and return its absolute workspace path as the handle."""
        path = posixpath.join(await self._directory(workspace), *_segments(key))
        await workspace.write_bytes(path, data)
        return path

    async def read(self, workspace: Workspace, handle: str) -> bytes:
        """Read a payload back; a handle outside the store directory raises `PermissionError`."""
        directory = await self._directory(workspace)
        path = posixpath.normpath(posixpath.join(directory, handle))
        if not path.startswith(directory.rstrip('/') + '/'):
            raise PermissionError(f'Handle {handle!r} is outside the store directory.')
        return await workspace.read_bytes(path)

    async def _directory(self, workspace: Workspace) -> str:
        if self.directory is not None:
            return await workspace.resolve(self.directory)
        if not supports_commands(workspace):
            return await workspace.resolve(_FILESYSTEM_ONLY_SUBDIR)
        result = await workspace.run(
            f'd="${{TMPDIR:-/tmp}}/{_WORKSPACE_SUBDIR}" && (umask 077 && mkdir -p "$d") && printf %s "$d"',
            shell=True,
            timeout=_DIRECTORY_TIMEOUT,
        )
        directory = posixpath.normpath(result.stdout)
        if result.exit_code != 0 or not posixpath.isabs(directory):
            raise WorkspaceError(result.stderr.strip() or 'The workspace has no writable temporary directory.')
        return directory


@dataclass
class LocalFileStore:
    """`OverflowStore` that writes each payload to a file on the host running the agent.

    Use it for runs with no workspace attached; with a workspace, the default `WorkspaceStore`
    keeps spills where the agent's other tools can see them.

    The handle equals the key: a relative `run_id/tool_call_id.retry` path under
    `base_dir`. The root is stable and shareable on purpose -- a later agent or run can
    read a spill a previous run produced, so the store is not isolated per instance.

    Security comes from two mechanisms, not isolation: the root is created with `0700`
    perms (owner-only), and `read` resolves the target (following symlinks) and rejects
    anything that escapes the root via symlink, `..`, or an absolute path. Handle segments
    are also sanitized by `_safe_segment`.

    Files are kept after the run by default (a later `read_tool_result` may need them).
    Set `cleanup_after` to opt into age-based pruning; see that field.
    """

    base_dir: Path | None = None
    """Root directory for spilled files. Defaults to a stable temp subdirectory."""

    cleanup_after: timedelta | None = None
    """Opt-in TTL for spilled files. `None` (default) keeps files forever.

    When set, a `write` schedules a background prune (a daemon thread, off the hot path)
    that deletes files whose modification time is older than `cleanup_after`. Pruning is
    best-effort: any failure is caught and surfaced via `warnings.warn`, never propagated
    into the agent run. Modification time (`st_mtime`) is the age signal; last-read time
    (`st_atime`) is unreliable on `noatime`/`relatime` mounts and is not used.
    """

    _root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._root = (
            self.base_dir if self.base_dir is not None else Path(tempfile.gettempdir()) / 'pyai_harness_overflow'
        )

    def _path(self, key: str) -> Path:
        return self._root.joinpath(*_segments(key))

    def _ensure_root(self) -> None:
        """Create the root directory owned by the current user with `0700` perms."""
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            self._root.chmod(0o700)
        except OSError:  # pragma: no cover - best effort on a root we do not own
            pass

    async def write(self, key: str, data: bytes) -> str:
        self._ensure_root()
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._schedule_cleanup()
        return key

    async def read(self, handle: str) -> bytes:
        target = self._path(handle).resolve()
        root = self._root.resolve()
        if not target.is_relative_to(root):
            raise PermissionError(f'Handle {handle!r} resolves outside the store root.')
        return target.read_bytes()

    # --- opt-in TTL pruning (non-blocking, non-erroring) ---

    def _schedule_cleanup(self) -> threading.Thread | None:
        """Fire a background prune when `cleanup_after` is set. Never blocks `write`."""
        if self.cleanup_after is None:
            return None
        thread = threading.Thread(target=self._run_prune, name='overflow-prune', daemon=True)
        thread.start()
        return thread

    def _run_prune(self) -> None:
        try:
            self._prune_sync()
        except Exception as exc:  # never let cleanup fail a run or block the hot path
            warnings.warn(f'LocalFileStore cleanup failed: {exc}', stacklevel=2)

    def _prune_sync(self) -> None:
        """Delete files older than `cleanup_after` (by `st_mtime`)."""
        assert self.cleanup_after is not None
        cutoff = time.time() - self.cleanup_after.total_seconds()
        for path in self._root.rglob('*'):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:  # pragma: no cover - file vanished mid-prune
                continue
