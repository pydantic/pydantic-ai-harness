"""Conformance and backend-specific tests for the audit sinks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from pydantic_ai_harness.audit_log import (
    AuditSink,
    InMemoryAuditSink,
    JsonlAuditSink,
    RunAuditRecord,
    RunOutcome,
    SqliteAuditSink,
    ToolCallRecord,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _tool(run_id: str, tool_call_id: str, *, tool_name: str = 'lookup', arguments: str = '{"q": 1}') -> ToolCallRecord:
    return ToolCallRecord(run_id=run_id, tool_call_id=tool_call_id, tool_name=tool_name, arguments=arguments)


def _run(run_id: str, *, outcome: RunOutcome = 'completed') -> RunAuditRecord:
    return RunAuditRecord(run_id=run_id, outcome=outcome, input_tokens=3, output_tokens=4, total_tokens=7)


@pytest.fixture(params=['memory', 'jsonl', 'sqlite-db', 'sqlite-conn'])
def sink(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[AuditSink]:
    connection: sqlite3.Connection | None = None
    if request.param == 'sqlite-conn':
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        made: AuditSink = SqliteAuditSink(connection=connection)
    elif request.param == 'sqlite-db':
        made = SqliteAuditSink(database=tmp_path / 'audit.db')
    elif request.param == 'jsonl':
        made = JsonlAuditSink(tmp_path / 'audit.jsonl')
    else:
        made = InMemoryAuditSink()
    yield made
    if connection is not None:
        connection.close()


class TestSinkConformance:
    """Every reference sink honors the same read/write contract."""

    async def test_records_and_reads_back_in_order(self, sink: AuditSink):
        await sink.record_tool_call(_tool('r1', 'c1'))
        await sink.record_tool_call(_tool('r1', 'c2'))
        await sink.record_tool_call(_tool('r2', 'c3'))
        await sink.record_run(_run('r1'))

        r1_calls = await sink.list_tool_calls(run_id='r1')
        assert [c.tool_call_id for c in r1_calls] == ['c1', 'c2']
        assert [c.run_id for c in r1_calls] == ['r1', 'r1']
        assert [c.tool_call_id for c in await sink.list_tool_calls(run_id='r2')] == ['c3']

        run = await sink.get_run(run_id='r1')
        assert run is not None
        assert run.outcome == 'completed'
        assert (run.input_tokens, run.output_tokens, run.total_tokens) == (3, 4, 7)

    async def test_missing_run_reads_empty(self, sink: AuditSink):
        assert await sink.list_tool_calls(run_id='absent') == []
        assert await sink.get_run(run_id='absent') is None

    async def test_record_fields_round_trip(self, sink: AuditSink):
        record = ToolCallRecord(
            run_id='r1',
            tool_call_id='c1',
            tool_name='lookup',
            arguments='{"q": 1}',
            result='ok',
            error=None,
            conversation_id='conv',
            parent_run_id='parent',
            agent_name='librarian',
        )
        await sink.record_tool_call(record)
        (read_back,) = await sink.list_tool_calls(run_id='r1')
        assert read_back.result == 'ok'
        assert read_back.conversation_id == 'conv'
        assert read_back.parent_run_id == 'parent'
        assert read_back.agent_name == 'librarian'
        assert read_back.started_at == record.started_at


class TestInMemoryAuditSink:
    async def test_runs_are_keyed_by_run_id(self):
        sink = InMemoryAuditSink()
        await sink.record_run(_run('r1', outcome='completed'))
        await sink.record_run(_run('r2', outcome='failed'))
        assert (await sink.get_run(run_id='r2')).outcome == 'failed'  # type: ignore[union-attr]


class TestJsonlAuditSink:
    async def test_creates_nested_parent_dirs(self, tmp_path: Path):
        sink = JsonlAuditSink(tmp_path / 'nested' / 'deep' / 'audit.jsonl')
        await sink.record_tool_call(_tool('r1', 'c1'))
        assert [c.tool_call_id for c in await sink.list_tool_calls(run_id='r1')] == ['c1']

    async def test_get_run_returns_latest_append(self, tmp_path: Path):
        sink = JsonlAuditSink(tmp_path / 'audit.jsonl')
        await sink.record_run(_run('r1', outcome='completed'))
        await sink.record_run(_run('r1', outcome='failed'))
        run = await sink.get_run(run_id='r1')
        assert run is not None and run.outcome == 'failed'

    async def test_blank_lines_are_skipped(self, tmp_path: Path):
        path = tmp_path / 'audit.jsonl'
        sink = JsonlAuditSink(path)
        await sink.record_tool_call(_tool('r1', 'c1'))
        # A stray blank line in the log must not break reads.
        with path.open('a', encoding='utf-8') as handle:
            handle.write('\n')
        await sink.record_tool_call(_tool('r1', 'c2'))
        assert [c.tool_call_id for c in await sink.list_tool_calls(run_id='r1')] == ['c1', 'c2']

    async def test_list_ignores_run_records_and_other_runs(self, tmp_path: Path):
        sink = JsonlAuditSink(tmp_path / 'audit.jsonl')
        await sink.record_tool_call(_tool('r1', 'c1'))
        await sink.record_tool_call(_tool('r2', 'c2'))
        await sink.record_run(_run('r1'))
        assert [c.tool_call_id for c in await sink.list_tool_calls(run_id='r1')] == ['c1']
        assert await sink.get_run(run_id='r2') is None


class TestSqliteAuditSink:
    def test_requires_exactly_one_of_database_or_connection(self):
        with pytest.raises(ValueError, match='exactly one'):
            SqliteAuditSink()
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        try:
            with pytest.raises(ValueError, match='exactly one'):
                SqliteAuditSink(database='x.db', connection=connection)
        finally:
            connection.close()

    async def test_schema_is_created_once_and_reused(self, tmp_path: Path):
        sink = SqliteAuditSink(database=tmp_path / 'audit.db')
        # The first write creates the schema; the second reuses it (idempotent).
        await sink.record_tool_call(_tool('r1', 'c1'))
        await sink.record_run(_run('r1'))
        assert len(await sink.list_tool_calls(run_id='r1')) == 1
        assert (await sink.get_run(run_id='r1')) is not None

    async def test_run_upserts_on_run_id(self, tmp_path: Path):
        sink = SqliteAuditSink(database=tmp_path / 'audit.db')
        await sink.record_run(_run('r1', outcome='completed'))
        await sink.record_run(_run('r1', outcome='failed'))
        run = await sink.get_run(run_id='r1')
        assert run is not None and run.outcome == 'failed'

    async def test_get_run_on_fresh_db_creates_schema(self, tmp_path: Path):
        # `get_run` as the very first operation must create the schema, not fail on a missing table.
        sink = SqliteAuditSink(database=tmp_path / 'fresh.db')
        assert await sink.get_run(run_id='absent') is None

    async def test_list_on_fresh_db_creates_schema(self, tmp_path: Path):
        sink = SqliteAuditSink(database=tmp_path / 'fresh.db')
        assert await sink.list_tool_calls(run_id='absent') == []

    async def test_creates_nested_parent_dirs(self, tmp_path: Path):
        sink = SqliteAuditSink(database=tmp_path / 'nested' / 'deep' / 'audit.db')
        await sink.record_run(_run('r1'))
        assert (await sink.get_run(run_id='r1')) is not None

    async def test_caller_owned_deferred_connection_writes_survive_close(self, tmp_path: Path):
        # A caller-owned connection defaults to sqlite3's deferred isolation
        # mode (unlike the internally-managed `database=` path, which forces
        # autocommit). Writes must be committed regardless, or closing the
        # caller's connection rolls them back and a fresh connection to the
        # same file finds nothing.
        db_path = tmp_path / 'caller-owned.db'
        owned = sqlite3.connect(db_path, check_same_thread=False)
        assert owned.isolation_level is not None  # deferred mode, not autocommit
        try:
            sink = SqliteAuditSink(connection=owned)
            await sink.record_tool_call(_tool('r1', 'c1'))
            await sink.record_run(_run('r1'))
        finally:
            owned.close()  # a commit-less close on a deferred connection rolls back

        reader = sqlite3.connect(db_path, check_same_thread=False)
        try:
            read_sink = SqliteAuditSink(connection=reader)
            assert len(await read_sink.list_tool_calls(run_id='r1')) == 1
            run = await read_sink.get_run(run_id='r1')
            assert run is not None and run.outcome == 'completed'
        finally:
            reader.close()

    async def test_memory_database_persists_across_calls(self):
        # `database=':memory:'` must behave like a working sink across
        # multiple calls, not silently discard everything after the first
        # write (SQLite hands out a brand-new, empty `:memory:` database to
        # every `connect()` call, so opening and closing a connection per
        # operation -- the normal `database=` lifecycle -- destroys the data).
        sink = SqliteAuditSink(database=':memory:')
        await sink.record_tool_call(_tool('r1', 'c1'))
        await sink.record_run(_run('r1'))
        assert len(await sink.list_tool_calls(run_id='r1')) == 1
        run = await sink.get_run(run_id='r1')
        assert run is not None and run.outcome == 'completed'
