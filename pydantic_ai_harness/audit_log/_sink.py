"""Storage backends for audit records.

`AuditSink` is the pluggable interface; a backend adapter (a warehouse, an
object store, a knowledge graph) implements it. Three stdlib reference sinks
ship here: `InMemoryAuditSink`, `JsonlAuditSink`, and `SqliteAuditSink`.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio.to_thread
from pydantic import TypeAdapter
from typing_extensions import TypeVar

from pydantic_ai_harness.audit_log._types import RunAuditRecord, ToolCallRecord

_RecordT = TypeVar('_RecordT')

_TOOL_ADAPTER: TypeAdapter[ToolCallRecord] = TypeAdapter(ToolCallRecord)
_RUN_ADAPTER: TypeAdapter[RunAuditRecord] = TypeAdapter(RunAuditRecord)


@runtime_checkable
class AuditSink(Protocol):
    """Async protocol for audit-record backends.

    Every method is async so one protocol covers in-memory sinks and
    file/database sinks that would otherwise block the event loop.
    `record_tool_call` and `record_run` are append/write; `list_tool_calls`
    returns a run's tool calls in the order they were recorded, and `get_run`
    returns the run's latest record, or `None` when the run is absent.
    """

    async def record_tool_call(self, record: ToolCallRecord) -> None: ...  # pragma: no cover

    async def record_run(self, record: RunAuditRecord) -> None: ...  # pragma: no cover

    async def list_tool_calls(self, *, run_id: str) -> list[ToolCallRecord]: ...  # pragma: no cover

    async def get_run(self, *, run_id: str) -> RunAuditRecord | None: ...  # pragma: no cover


class InMemoryAuditSink:
    """Process-local sink, suitable for tests and single-process runs."""

    def __init__(self) -> None:
        self._tool_calls: dict[str, list[ToolCallRecord]] = defaultdict(list)
        self._runs: dict[str, RunAuditRecord] = {}

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        self._tool_calls[record.run_id].append(record)

    async def record_run(self, record: RunAuditRecord) -> None:
        self._runs[record.run_id] = record

    async def list_tool_calls(self, *, run_id: str) -> list[ToolCallRecord]:
        return list(self._tool_calls.get(run_id, ()))

    async def get_run(self, *, run_id: str) -> RunAuditRecord | None:
        return self._runs.get(run_id)


_KIND_TOOL = 'tool_call'
_KIND_RUN = 'run'


class JsonlAuditSink:
    """Append-only JSON Lines sink: one file, one JSON object per record.

    Each line is `{"kind": "tool_call"|"run", "record": {...}}`. Writes append,
    so a run recorded more than once keeps every line and `get_run` returns the
    last. File I/O runs on a worker thread so the event loop is not blocked.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _append(self, kind: str, payload: bytes) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({'kind': kind, 'record': json.loads(payload)}).encode('utf-8')
        with self._path.open('ab') as handle:
            handle.write(line + b'\n')

    def _read(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        envelopes: list[dict[str, object]] = []
        for raw in self._path.read_text(encoding='utf-8').splitlines():
            if raw.strip():
                envelopes.append(json.loads(raw))
        return envelopes

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        await anyio.to_thread.run_sync(self._append, _KIND_TOOL, _TOOL_ADAPTER.dump_json(record))

    async def record_run(self, record: RunAuditRecord) -> None:
        await anyio.to_thread.run_sync(self._append, _KIND_RUN, _RUN_ADAPTER.dump_json(record))

    async def list_tool_calls(self, *, run_id: str) -> list[ToolCallRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_tool_calls, run_id)

    def _sync_list_tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        records: list[ToolCallRecord] = []
        for envelope in self._read():
            if envelope['kind'] == _KIND_TOOL:
                record = _TOOL_ADAPTER.validate_python(envelope['record'])
                if record.run_id == run_id:
                    records.append(record)
        return records

    async def get_run(self, *, run_id: str) -> RunAuditRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_run, run_id)

    def _sync_get_run(self, run_id: str) -> RunAuditRecord | None:
        latest: RunAuditRecord | None = None
        for envelope in self._read():
            if envelope['kind'] == _KIND_RUN:
                record = _RUN_ADAPTER.validate_python(envelope['record'])
                if record.run_id == run_id:
                    latest = record
        return latest


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    conversation_id TEXT,
    parent_run_id TEXT,
    agent_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id, seq);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    outcome TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    error TEXT,
    conversation_id TEXT,
    parent_run_id TEXT,
    agent_name TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL
);
"""

_TOOL_COLUMNS = (
    'run_id',
    'tool_call_id',
    'tool_name',
    'arguments',
    'result',
    'error',
    'started_at',
    'ended_at',
    'conversation_id',
    'parent_run_id',
    'agent_name',
)
_RUN_COLUMNS = (
    'run_id',
    'outcome',
    'input_tokens',
    'output_tokens',
    'total_tokens',
    'error',
    'conversation_id',
    'parent_run_id',
    'agent_name',
    'started_at',
    'ended_at',
)


def _to_json_row(record: _RecordT, adapter: TypeAdapter[_RecordT], columns: tuple[str, ...]) -> tuple[object, ...]:
    data = adapter.dump_python(record, mode='json')
    return tuple(data[column] for column in columns)


class SqliteAuditSink:
    """SQLite-backed sink. One file holds the `tool_calls` and `runs` tables.

    Pass either `database=` (a path; short-lived connections opened per call
    with WAL enabled) or `connection=` (a caller-owned `sqlite3.Connection`).
    A caller-owned connection must be created with `check_same_thread=False`,
    since methods dispatch SQL onto worker threads via `anyio.to_thread`. Runs
    upsert on their `run_id`; tool calls append.
    """

    def __init__(
        self,
        *,
        database: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if (database is None) == (connection is None):
            raise ValueError('provide exactly one of `database=` or `connection=`')
        self._database = Path(database) if database is not None else None
        self._connection = connection
        self._schema_ready = False

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        assert self._database is not None
        self._database.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._database, isolation_level=None, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        if self._connection is None:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        conn.executescript(_SQLITE_SCHEMA)
        conn.commit()
        self._schema_ready = True

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_record_tool_call, record)

    def _sync_record_tool_call(self, record: ToolCallRecord) -> None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            placeholders = ', '.join('?' * len(_TOOL_COLUMNS))
            conn.execute(
                f'INSERT INTO tool_calls ({", ".join(_TOOL_COLUMNS)}) VALUES ({placeholders})',
                _to_json_row(record, _TOOL_ADAPTER, _TOOL_COLUMNS),
            )
        finally:
            self._maybe_close(conn)

    async def record_run(self, record: RunAuditRecord) -> None:
        await anyio.to_thread.run_sync(self._sync_record_run, record)

    def _sync_record_run(self, record: RunAuditRecord) -> None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            placeholders = ', '.join('?' * len(_RUN_COLUMNS))
            conn.execute(
                f'INSERT OR REPLACE INTO runs ({", ".join(_RUN_COLUMNS)}) VALUES ({placeholders})',
                _to_json_row(record, _RUN_ADAPTER, _RUN_COLUMNS),
            )
        finally:
            self._maybe_close(conn)

    async def list_tool_calls(self, *, run_id: str) -> list[ToolCallRecord]:
        return await anyio.to_thread.run_sync(self._sync_list_tool_calls, run_id)

    def _sync_list_tool_calls(self, run_id: str) -> list[ToolCallRecord]:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            rows = conn.execute(
                f'SELECT {", ".join(_TOOL_COLUMNS)} FROM tool_calls WHERE run_id = ? ORDER BY seq',
                (run_id,),
            ).fetchall()
        finally:
            self._maybe_close(conn)
        return [_TOOL_ADAPTER.validate_python(dict(zip(_TOOL_COLUMNS, row))) for row in rows]

    async def get_run(self, *, run_id: str) -> RunAuditRecord | None:
        return await anyio.to_thread.run_sync(self._sync_get_run, run_id)

    def _sync_get_run(self, run_id: str) -> RunAuditRecord | None:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(
                f'SELECT {", ".join(_RUN_COLUMNS)} FROM runs WHERE run_id = ?',
                (run_id,),
            ).fetchone()
        finally:
            self._maybe_close(conn)
        if row is None:
            return None
        return _RUN_ADAPTER.validate_python(dict(zip(_RUN_COLUMNS, row)))
