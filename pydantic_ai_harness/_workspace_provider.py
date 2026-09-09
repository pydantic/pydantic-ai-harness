"""Private helpers shared by workspace provider backends."""

from __future__ import annotations

import posixpath


def absolute_path(name: str, value: str | None) -> str | None:
    """Validate an absolute POSIX path without changing symlink traversal."""
    if value is None:
        return None
    if not posixpath.isabs(value):
        raise ValueError(f'{name} must be an absolute workspace path or None, got {value!r}.')
    return value
