"""Storage backends for audit records.

`AuditSink` is the pluggable interface, and it is the extension point of this
capability: this module ships no persistent or production backend, by
design. Implement the four methods against SQLite, Postgres, Mongo, a
warehouse, an object store, or a knowledge graph, and pass the instance as
`sink=`; that choice, and its transaction, durability, and concurrency
semantics, belongs to the consumer. Two trivial, dependency-free reference
sinks ship here for tests and simple single-process use: `InMemoryAuditSink`
and `JsonlAuditSink`.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio.to_thread
from pydantic import TypeAdapter

from pydantic_ai_harness.audit_log._types import RunAuditRecord, ToolCallRecord

logger = logging.getLogger(__name__)

_TOOL_ADAPTER: TypeAdapter[ToolCallRecord] = TypeAdapter(ToolCallRecord)
_RUN_ADAPTER: TypeAdapter[RunAuditRecord] = TypeAdapter(RunAuditRecord)


@runtime_checkable
class AuditSink(Protocol):
    """Async protocol for audit-record backends -- implement this for your own backend.

    Every method is async so one protocol covers in-memory sinks and
    file/database sinks that would otherwise block the event loop.
    `record_tool_call` and `record_run` are append/write; `list_tool_calls`
    returns a run's tool calls in the order they were recorded, and `get_run`
    returns the run's latest record, or `None` when the run is absent. This
    library takes no position on which backend a real deployment should use;
    it defines the shape and ships trivial references, not a recommendation.
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
    last. File I/O runs on a worker thread so the event loop is not blocked. A
    line that fails to parse as JSON (e.g. a partial append torn by a crash) is
    skipped and logged rather than raised, so one bad line does not hide every
    valid record recorded before it.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _append(self, kind: str, payload: bytes) -> None:
        """Append one record as its own line, first isolating any torn trailing line.

        If a prior crash left the file's last byte not a newline (a write cut
        short mid-record), naively appending this record plus a terminating
        newline would concatenate it onto that fragment, producing one
        combined line that `_read` cannot parse -- hiding the new record
        along with the torn one, not just the torn one. Opened `a+b` (append
        *and* read) rather than `ab` so the trailing byte can be inspected:
        when the file is non-empty and does not already end in a newline, a
        separating newline is written first, leaving the torn fragment as its
        own (skippable) line and this record starting cleanly on the next.
        Reading the trailing byte does not race the append: `a+b` writes
        always land at end-of-file regardless of the read seek, the same
        O_APPEND guarantee `ab` relies on.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({'kind': kind, 'record': json.loads(payload)}).encode('utf-8')
        with self._path.open('a+b') as handle:
            if handle.tell() > 0:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b'\n':
                    handle.write(b'\n')
            handle.write(line + b'\n')

    def _read(self) -> list[dict[str, object]]:
        """Read every envelope in the file, skipping a line that fails to parse as JSON.

        A torn trailing record -- a partial append cut short by a crash, or any
        other single malformed line -- must not make every read raise and hide
        every prior valid record; the unparseable line is logged and skipped.
        """
        if not self._path.exists():
            return []
        envelopes: list[dict[str, object]] = []
        for raw in self._path.read_text(encoding='utf-8').splitlines():
            if not raw.strip():
                continue
            try:
                envelopes.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning('Skipping unparseable line in audit sink %s.', self._path)
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
