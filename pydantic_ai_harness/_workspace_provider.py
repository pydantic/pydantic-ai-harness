"""Private helpers shared by workspace provider backends."""

from __future__ import annotations

import posixpath

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import WorkspaceCommand


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
    if shell:
        raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
    if not command:
        raise TypeError('an argv sequence needs at least the program to run')
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
