"""The DB-API surface the SQLite-backed stores use, so any compatible driver can be passed in.

The stores open stdlib `sqlite3` themselves for a `database=` path, and that stays the default.
A caller-owned `connection=`, though, only ever has to be a
[DB-API 2.0](https://peps.python.org/pep-0249/) connection speaking SQLite, which is what lets
`turso.connect(...)` (or `turso.sync.connect(..., remote_url=...)`, the embedded-replica form) be
handed to the same store.

Two things make that work. The cursor-returning `execute` family is a DB-API optional extension
that both drivers implement, so the stores' `conn.execute(...).fetchone()` style needs no cursor
juggling. And the exception classes are exposed as attributes of the connection — also a DB-API
extension — so a store catches `conn.DatabaseError` rather than `sqlite3.DatabaseError`, which a
non-stdlib driver's errors do not inherit from.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol


class SqliteCursor(Protocol):
    """The cursor surface the stores use."""

    @property
    def lastrowid(self) -> int | None: ...  # pragma: no cover

    @property
    def rowcount(self) -> int: ...  # pragma: no cover

    def fetchone(self) -> Any: ...  # pragma: no cover

    def fetchall(self) -> list[Any]: ...  # pragma: no cover

    def __iter__(self) -> Iterator[Any]: ...  # pragma: no cover


class SqliteConnection(Protocol):
    """The connection surface the stores use, satisfied by `sqlite3` and by `turso`."""

    @property
    def DatabaseError(self) -> type[Exception]:
        """DB-API base for errors the database raises; `IntegrityError` and friends subclass it."""
        ...  # pragma: no cover

    def execute(self, sql: str, parameters: Any = ..., /) -> SqliteCursor: ...  # pragma: no cover

    def executescript(self, sql_script: str, /) -> SqliteCursor: ...  # pragma: no cover

    def commit(self) -> None: ...  # pragma: no cover

    def rollback(self) -> None: ...  # pragma: no cover

    def close(self) -> None: ...  # pragma: no cover

    @property
    def in_transaction(self) -> bool: ...  # pragma: no cover
