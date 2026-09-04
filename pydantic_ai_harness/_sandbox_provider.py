"""Private helpers shared by the sandbox provider backends."""

from __future__ import annotations

import posixpath
from collections.abc import Awaitable, Callable

import anyio
from anyio.lowlevel import checkpoint
from typing_extensions import Never


def absolute_path(name: str, value: str | None) -> str | None:
    """Normalize an absolute POSIX path, preserving `None`."""
    if value is None:
        return None
    if not posixpath.isabs(value):
        raise ValueError(f'{name} must be an absolute sandbox path or None, got {value!r}.')
    return posixpath.normpath(value)


async def cleanup_call(call: Callable[[], Awaitable[object]], *, timeout: float) -> Exception | None:
    """Run one teardown RPC shielded from cancellation and bounded by `timeout`.

    Returns the failure instead of raising so the caller owns translation; a bare
    `TimeoutError` means the bound expired. Shielded because teardown must still go out while
    a run is being cancelled; bounded so a wedged control plane cannot hang teardown.
    """
    error: Exception | None = None
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(timeout) as scope:
            try:
                await call()
            except Exception as exc:
                error = exc
        if scope.cancel_called:
            return TimeoutError()
    return error


async def raise_after_cleanup(error: Exception) -> Never:
    """Deliver pending cancellation before raising a cleanup error."""
    await checkpoint()
    raise error
