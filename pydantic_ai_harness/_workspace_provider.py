"""Private helpers shared by workspace provider backends."""

from __future__ import annotations

import posixpath
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import anyio
from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import WorkspaceCommand, WorkspaceTimeoutError


async def stop_shielded(stop: Callable[[], Awaitable[object]], *, grace: float = 2.0) -> None:
    """Attempt to stop a command under cancellation without cancelling its shared sandbox."""

    async def best_effort_stop() -> None:
        try:
            await stop()
        except Exception:
            # Preserve the command timeout/cancellation if a provider's stop request fails.
            pass

    # A child task owns the stop so repeated cancellation of the caller cannot interrupt cleanup.
    with anyio.move_on_after(grace, shield=True):
        async with anyio.create_task_group() as group:
            group.start_soon(best_effort_stop)


@asynccontextmanager
async def command_deadline(
    timeout: float | None,
    *,
    stop: Callable[[], Awaitable[object]],
    output: Callable[[], tuple[str, str]] = lambda: ('', ''),
) -> AsyncGenerator[None, None]:
    """Bound only the command phase, after sandbox acquisition; stop on timeout or cancellation."""
    stopped = False
    with anyio.move_on_after(timeout) as scope:
        try:
            yield
        except BaseException:
            await stop_shielded(stop)
            stopped = True
            raise
    if scope.cancelled_caught:
        if not stopped:
            await stop_shielded(stop)
        stdout, stderr = output()
        raise WorkspaceTimeoutError(f'Command timed out after {timeout}s', stdout=stdout, stderr=stderr)


def absolute_path(name: str, value: str | None) -> str | None:
    """Validate an absolute POSIX path without changing symlink traversal."""
    if value is None:
        return None
    if not posixpath.isabs(value):
        raise ValueError(f'{name} must be an absolute workspace path or None, got {value!r}.')
    return value


def command_argv(command: WorkspaceCommand, shell: bool) -> list[str]:
    """The argv that runs `command`, with the same `shell` rules as core's local backend.

    A shell string runs under `/bin/sh -c`; an argv sequence runs as given. An empty argv is
    refused: a provider that quotes it with `shlex.join` would get `''`, which a shell runs as a
    successful no-op.
    """
    if isinstance(command, str):
        if not shell:
            raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
        return ['/bin/sh', '-c', command]
    if isinstance(command, bytes):
        raise TypeError('a bytes command is not supported; pass a string or argv sequence')
    if shell:
        raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
    if not command:
        raise TypeError('an argv sequence needs at least the program to run')
    for argument in command:
        if type(argument) is not str:
            raise TypeError('argv elements must be strings')
        if '\x00' in argument:
            raise ValueError('argv elements must not contain NUL bytes')
    return list(command)


def check_working_dir(value: str | None) -> None:
    """Raise `UserError` unless a provider's `working_dir` is an absolute POSIX path or `None`."""
    if value is not None and not posixpath.isabs(value):
        raise UserError(f'working_dir must be an absolute POSIX path or None, got {value!r}.')


def check_integer(name: str, value: int | None, *, minimum: int = 1, optional: bool = False) -> None:
    """Raise `UserError` unless `value` is an integer of at least `minimum`, or `None` when `optional`.

    `bool` is rejected: it is an `int` subclass, but `True` is never a meant count.
    """
    if (type(value) is int and value >= minimum) or (value is None and optional):
        return
    raise UserError(f'{name} must be an integer of at least {minimum}{" or None" if optional else ""}, got {value!r}.')
