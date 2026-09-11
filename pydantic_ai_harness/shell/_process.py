"""Process plumbing for the shell toolset.

Everything here is about one spawned process: classifying a spawn failure,
reading its output as lines, killing its whole group, and the bookkeeping
for a background process. Policy (allow and deny lists, environment) stays in
`_toolset.py`.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import os
import re
import signal
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar

import anyio
import anyio.abc
from pydantic_ai.exceptions import ModelRetry

from pydantic_ai_harness.shell._events import MAX_EVENT_LINE_CHARS

_IO_DRAIN_TIMEOUT: float = 2.0
_KILL_GRACE_PERIOD: float = 2.0

_P = ParamSpec('_P')
_SelfT = TypeVar('_SelfT')

# Spawning a command fails with a bare `OSError` for causes that have no
# dedicated subclass, and with `FileNotFoundError`/`NotADirectoryError` for
# causes that do. The errno says whose fault it is: these are the model's, and
# it can act on them. Every other errno (EMFILE, ENOMEM) is the host's, and must
# keep aborting the run rather than sending the model into a retry loop it
# can't win.
#
# ENOENT and ENOTDIR reach here only from the working directory, since the
# command string is handed to a shell that always exists -- a command whose own
# executable is missing is reported by that shell on stderr, not by the spawn.
#
# Keyed by `OSError.errno`, which the stdlib types as `int | None`.
_RECOVERABLE_ERRNOS: dict[int | None, str] = {
    errno.ENOENT: 'The working directory no longer exists.',
    errno.ENOTDIR: 'The working directory is no longer a directory.',
}


def recoverable(
    fn: Callable[Concatenate[_SelfT, _P], Awaitable[str]],
) -> Callable[Concatenate[_SelfT, _P], Awaitable[str]]:
    """Convert model-correctable errors into `ModelRetry`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run. A denied command, a command
    the OS refuses to spawn, and a working directory the model's own earlier
    command destroyed are all things the model can recover from, so surface them
    as a retry instead of crashing the agent.
    """

    @functools.wraps(fn)
    async def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except PermissionError as e:
            raise ModelRetry(str(e)) from e
        except OSError as e:
            reason = _RECOVERABLE_ERRNOS.get(e.errno)
            if reason is None:
                raise
            # `str(e)` embeds the absolute host path; the reason alone doesn't.
            raise ModelRetry(reason) from e

    return wrapper


def is_interactive_command(command: str) -> bool:
    """Detect commands that typically require interactive input."""
    interactive_patterns = [
        r'^(vi|vim|nano|emacs|less|more|top|htop|man)\b',
        r'^sudo\s',
        r'^passwd\b',
        r'^ssh\b',
        r'^telnet\b',
        r'^ftp\b',
    ]
    return any(re.match(p, command.strip()) for p in interactive_patterns)


class BackgroundProcess:
    """State for a background command using temp files for output."""

    __slots__ = (
        'command',
        'command_id',
        'proc',
        'stdout_path',
        'stderr_path',
        'started_at',
        'finished',
        'exit_code',
        'stop_lock',
    )

    def __init__(
        self,
        *,
        command: str,
        command_id: str,
        proc: anyio.abc.Process,
        stdout_path: str,
        stderr_path: str,
    ) -> None:
        self.command = command
        self.command_id = command_id
        self.proc = proc
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.started_at = time.monotonic()
        self.finished = False
        self.exit_code: int | None = None
        # Serializes duplicate stops: the second waits for the first to
        # finish tearing down instead of racing its cleanup.
        self.stop_lock = anyio.Lock()


def read_bg_output(bg: BackgroundProcess) -> tuple[str, str]:
    """Read current output from background process temp files."""
    try:
        stdout = Path(bg.stdout_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        stdout = ''
    try:
        stderr = Path(bg.stderr_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        stderr = ''
    return stdout, stderr


def cleanup_bg_files(bg: BackgroundProcess) -> None:
    """Remove temp files for a background process."""
    try:
        os.unlink(bg.stdout_path)
    except OSError:
        pass
    try:
        os.unlink(bg.stderr_path)
    except OSError:
        pass


async def kill_process_group(proc: anyio.abc.Process) -> None:
    """SIGTERM the process group, then SIGKILL whatever is left of it.

    The child was spawned with `start_new_session=True`, so its pid is the
    group id for the group's whole life. It stays addressable through
    `os.killpg` while any member lives, even after the leader exited and was
    reaped; resolving the group with `os.getpgid` at kill time would miss
    exactly that window and leave surviving members unkillable.

    Waiting only for the group leader is not enough: a child the shell forked
    while the SIGTERM was in flight misses it, and a leader that traps the
    signal never exits. So the group is swept with SIGKILL once the leader is
    gone or the grace period is up, and the sweep runs in `finally` because a
    cancelled caller (a native `Task.cancel()`, which no anyio shield stops)
    must still leave nothing behind. On an already empty group the sweep is
    a no-op. The final SIGKILL is not reaped: a caller that keeps the process
    pairs this with `proc.wait()` and `proc.aclose()`.
    """
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        return

    try:
        with anyio.CancelScope(shield=True), anyio.move_on_after(_KILL_GRACE_PERIOD):
            await proc.wait()
    finally:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


LineSink = Callable[[str, bool], Awaitable[None]]
"""Receives each completed output line and whether it was cut at `MAX_EVENT_LINE_CHARS`."""

# Bytes of one unterminated line kept for the sink; a UTF-8 character is at
# most four bytes, so this always covers the characters the sink will keep.
_LINE_HEAD_BYTES = MAX_EVENT_LINE_CHARS * 4


class _LineBuffer:
    """Split a byte stream into lines for a sink without buffering whole lines.

    The raw bytes still go to the model in full; this only feeds the per-line
    sink, so a single multi-megabyte line costs the head plus a flag, not a
    growing copy per chunk.
    """

    def __init__(self) -> None:
        self._head = b''
        self._overflowed = False

    def feed(self, chunk: bytes) -> list[tuple[str, bool]]:
        """Return the lines completed by `chunk`."""
        lines: list[tuple[str, bool]] = []
        while True:
            newline = chunk.find(b'\n')
            if newline < 0:
                self._extend(chunk)
                return lines
            self._extend(chunk[:newline])
            lines.append(self._take())
            chunk = chunk[newline + 1 :]

    def flush(self) -> tuple[str, bool] | None:
        """Return the unterminated final line, if any."""
        if not self._head and not self._overflowed:
            return None
        return self._take()

    def _extend(self, piece: bytes) -> None:
        room = _LINE_HEAD_BYTES - len(self._head)
        if len(piece) > room:
            self._overflowed = True
        self._head += piece[:room]

    def _take(self) -> tuple[str, bool]:
        text = self._head.decode('utf-8', errors='replace').rstrip('\r')
        truncated = self._overflowed or len(text) > MAX_EVENT_LINE_CHARS
        self._head = b''
        self._overflowed = False
        return text[:MAX_EVENT_LINE_CHARS], truncated


class OutputReader:
    """Collect one pipe into bytes, handing each completed line to a sink.

    Reading resumes where a cancelled `read` stopped: after a timeout kills
    the command, `drain_with_timeout` picks up the same line buffer, so a
    tail written before the deadline still reaches the sink as one line.
    """

    def __init__(self, stream: anyio.abc.ByteReceiveStream, *, on_line: LineSink | None) -> None:
        self._stream = stream
        self._on_line = on_line
        self._lines = _LineBuffer()
        self._pending: list[tuple[str, bool]] = []
        self.chunks: list[bytes] = []

    @property
    def text(self) -> str:
        return b''.join(self.chunks).decode('utf-8', errors='replace')

    async def read(self) -> None:
        """Read to end of file, buffering lines for later delivery."""
        async for chunk in self._stream:
            self.chunks.append(chunk)
            if self._on_line is not None:
                self._pending.extend(self._lines.feed(chunk))

    async def deliver(self) -> None:
        """Invoke callbacks for buffered lines and any unterminated final line."""
        if self._on_line is None:
            return
        for line, truncated in self._pending:
            await self._on_line(line, truncated)
        self._pending.clear()
        if (last := self._lines.flush()) is not None:
            await self._on_line(*last)

    async def flush(self) -> None:  # pragma: no cover
        """Hand the sink the unterminated last line, if any."""
        if self._on_line is not None and (last := self._lines.flush()) is not None:
            await self._on_line(*last)

    async def drain(self) -> None:
        """Read to end of file, treating a pipe closed under us as the end."""
        try:
            await self.read()
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            pass


async def drain_with_timeout(*readers: OutputReader) -> None:
    """Finish reading after a kill, for as long as a grandchild may hold the pipe."""
    with anyio.move_on_after(_IO_DRAIN_TIMEOUT):
        async with anyio.create_task_group() as tg:
            for reader in readers:
                tg.start_soon(reader.drain)


async def run_to_exit(proc: anyio.abc.Process, *readers: OutputReader, timeout: float) -> tuple[int, bool]:
    """Read the pipes to end of file and reap the process; return its exit code and whether the deadline hit.

    The whole group is killed on timeout and on cancellation alike:
    `proc.aclose()` kills only the shell, and its children would outlive
    the run.

    Callback latency from `on_line` sinks does not consume the timeout: lines
    are buffered during the timed IO phase and delivered afterward, so a slow
    listener cannot turn a completed command into a spurious timeout.
    """
    try:
        with anyio.fail_after(timeout):
            async with anyio.create_task_group() as tg:
                for reader in readers:
                    tg.start_soon(reader.read)
            exit_code = await proc.wait()
        for reader in readers:
            await reader.deliver()
        return exit_code, False
    except TimeoutError:
        await kill_process_group(proc)
        exit_code = await proc.wait()
        with anyio.CancelScope(shield=True):
            await drain_with_timeout(*readers)
            for reader in readers:
                await reader.deliver()
        return exit_code, True
    except BaseException:
        await kill_process_group(proc)
        raise
    finally:
        await proc.aclose()
