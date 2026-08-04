"""Storage backends for scheduled agent runs."""

from __future__ import annotations

import re
import sqlite3
import threading
from typing import Protocol, runtime_checkable

import anyio.to_thread

from pydantic_ai_harness.scheduling._types import Schedule

_VALID_TABLE_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,62}')


class ScheduleConflictError(Exception):
    """Raised when a schedule save uses a stale version."""


@runtime_checkable
class ScheduleStore(Protocol):
    """Async whole-record storage for schedules.

    Use one runner per store. Writers must re-read and retry when `save` raises
    `ScheduleConflictError` because another writer changed the record.
    """

    async def add(self, schedule: Schedule) -> Schedule:
        """Store a new schedule, raising `ValueError` for a duplicate id."""
        ...  # pragma: no cover

    async def get(self, schedule_id: str) -> Schedule | None:
        """Return one schedule by id, or `None`."""
        ...  # pragma: no cover

    async def list(self) -> list[Schedule]:
        """Return schedules in insertion order."""
        ...  # pragma: no cover

    async def save(self, schedule: Schedule) -> None:
        """Replace a matching version with `version + 1`, or raise on conflict or unknown id."""
        ...  # pragma: no cover

    async def remove(self, schedule_id: str) -> bool:
        """Remove a schedule and return whether it existed."""
        ...  # pragma: no cover


class InMemoryScheduleStore:
    """Insertion-ordered process-local schedule storage."""

    def __init__(self) -> None:
        self._schedules: dict[str, Schedule] = {}

    async def add(self, schedule: Schedule) -> Schedule:
        """Store a deep copy of a new schedule."""
        if schedule.id in self._schedules:
            raise ValueError(f'A schedule with id {schedule.id!r} already exists.')
        stored = schedule.model_copy(deep=True)
        self._schedules[schedule.id] = stored
        return stored.model_copy(deep=True)

    async def get(self, schedule_id: str) -> Schedule | None:
        """Return a deep copy of one schedule, or `None`."""
        schedule = self._schedules.get(schedule_id)
        return None if schedule is None else schedule.model_copy(deep=True)

    async def list(self) -> list[Schedule]:
        """Return deep copies of schedules in insertion order."""
        return [schedule.model_copy(deep=True) for schedule in self._schedules.values()]

    async def save(self, schedule: Schedule) -> None:
        """Replace a matching version with a bumped deep copy."""
        stored = self._schedules.get(schedule.id)
        if stored is None:
            raise ValueError(f'Unknown schedule id {schedule.id!r}.')
        if schedule.version != stored.version:
            raise ScheduleConflictError(f'Schedule {schedule.id!r} changed since it was read.')
        self._schedules[schedule.id] = schedule.model_copy(update={'version': schedule.version + 1}, deep=True)

    async def remove(self, schedule_id: str) -> bool:
        """Remove a schedule and return whether it existed."""
        return self._schedules.pop(schedule_id, None) is not None


class SqliteScheduleStore:
    """File-backed SQLite schedule storage.

    Empty and `:memory:` databases are unsupported because each operation uses a fresh connection.
    Blocking SQLite work runs in a worker thread. The version check runs inside the `UPDATE`
    statement itself, so it holds across store instances and processes sharing one database file.
    """

    def __init__(self, database: str = '.agent-schedules.db', *, table: str = 'schedules') -> None:
        if not _VALID_TABLE_RE.fullmatch(table):
            raise ValueError(f'invalid table name: {table!r}')
        if database in ('', ':memory:'):
            raise ValueError(
                "SqliteScheduleStore does not support empty or ':memory:' databases; "
                'use InMemoryScheduleStore or a file-backed database.'
            )
        self._database = database
        self._table = table
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database)
        if not self._ready:
            try:
                connection.execute(
                    f'CREATE TABLE IF NOT EXISTS "{self._table}" '
                    '(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, '
                    'data TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 0)'
                )
                connection.commit()
            except BaseException:
                connection.close()
                raise
            self._ready = True
        return connection

    def _add_sync(self, schedule: Schedule) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    f'INSERT INTO "{self._table}" (id, data, version) VALUES (?, ?, ?)',
                    (schedule.id, schedule.model_dump_json(), schedule.version),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f'A schedule with id {schedule.id!r} already exists.') from exc
            finally:
                connection.close()

    def _get_sync(self, schedule_id: str) -> Schedule | None:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(f'SELECT data FROM "{self._table}" WHERE id = ?', (schedule_id,)).fetchone()
            finally:
                connection.close()
        return None if row is None else Schedule.model_validate_json(str(row[0]))

    def _list_sync(self) -> list[Schedule]:
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(f'SELECT data FROM "{self._table}" ORDER BY seq').fetchall()
            finally:
                connection.close()
        return [Schedule.model_validate_json(str(row[0])) for row in rows]

    def _save_sync(self, schedule: Schedule) -> None:
        with self._lock:
            connection = self._connect()
            try:
                bumped = schedule.model_copy(update={'version': schedule.version + 1}, deep=True)
                cursor = connection.execute(
                    f'UPDATE "{self._table}" SET data = ?, version = ? WHERE id = ? AND version = ?',
                    (bumped.model_dump_json(), bumped.version, schedule.id, schedule.version),
                )
                if cursor.rowcount == 0:
                    row = connection.execute(f'SELECT 1 FROM "{self._table}" WHERE id = ?', (schedule.id,)).fetchone()
                    if row is None:
                        raise ValueError(f'Unknown schedule id {schedule.id!r}.')
                    raise ScheduleConflictError(f'Schedule {schedule.id!r} changed since it was read.')
                connection.commit()
            finally:
                connection.close()

    def _remove_sync(self, schedule_id: str) -> bool:
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute(f'DELETE FROM "{self._table}" WHERE id = ?', (schedule_id,))
                connection.commit()
                return cursor.rowcount > 0
            finally:
                connection.close()

    async def add(self, schedule: Schedule) -> Schedule:
        """Persist a new schedule and return a detached copy."""
        await anyio.to_thread.run_sync(self._add_sync, schedule)
        return schedule.model_copy(deep=True)

    async def get(self, schedule_id: str) -> Schedule | None:
        """Return one persisted schedule by id, or `None`."""
        return await anyio.to_thread.run_sync(self._get_sync, schedule_id)

    async def list(self) -> list[Schedule]:
        """Return persisted schedules in insertion order."""
        return await anyio.to_thread.run_sync(self._list_sync)

    async def save(self, schedule: Schedule) -> None:
        """Replace a matching persisted version with a bumped version."""
        await anyio.to_thread.run_sync(self._save_sync, schedule)

    async def remove(self, schedule_id: str) -> bool:
        """Remove a persisted schedule and return whether it existed."""
        return await anyio.to_thread.run_sync(self._remove_sync, schedule_id)
