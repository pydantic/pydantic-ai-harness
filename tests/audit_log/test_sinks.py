"""Conformance and backend-specific tests for the audit sinks.

Only the two dependency-free reference sinks (`InMemoryAuditSink`,
`JsonlAuditSink`) are covered here. A persistent backend (SQLite, Postgres,
Mongo, a warehouse) is the consumer's own `AuditSink` implementation and is
not this library's concern -- see `docs/audit-log.md` "Sinks".
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from pydantic_ai_harness.audit_log import (
    AuditSink,
    InMemoryAuditSink,
    JsonlAuditSink,
    RunAuditRecord,
    RunOutcome,
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


@pytest.fixture(params=['memory', 'jsonl'])
def sink(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[AuditSink]:
    made: AuditSink = JsonlAuditSink(tmp_path / 'audit.jsonl') if request.param == 'jsonl' else InMemoryAuditSink()
    yield made


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

    async def test_torn_trailing_line_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        # A partial append (e.g. a process crash mid-write) leaves a trailing
        # line that is not valid JSON and has no terminating newline. That one
        # torn record must not make every read raise and hide the valid
        # records recorded before it.
        path = tmp_path / 'audit.jsonl'
        sink = JsonlAuditSink(path)
        await sink.record_tool_call(_tool('r1', 'c1'))
        await sink.record_run(_run('r1'))
        with path.open('a', encoding='utf-8') as handle:
            handle.write('{"kind": "tool_call", "record": {"run_id": "r1"')

        with caplog.at_level(logging.WARNING):
            calls = await sink.list_tool_calls(run_id='r1')
            run = await sink.get_run(run_id='r1')

        assert [c.tool_call_id for c in calls] == ['c1']
        assert run is not None and run.outcome == 'completed'
        assert 'unparseable' in caplog.text.lower()  # the skip is logged, not silently dropped

    async def test_append_after_a_pre_existing_torn_line_does_not_lose_the_new_record(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        # A prior crash can leave the file's last line torn -- no terminating
        # newline -- *before* the process restarts and appends again. Naively
        # opening `ab` and writing `line + b"\n"` would concatenate the new
        # record onto that fragment into one combined, unparseable line, so
        # `_read`'s skip-on-malformed-line behavior would drop the newly
        # written record too, not just the torn one. `_append` must insert a
        # separating newline first so the torn fragment stays its own
        # (skipped) line and the newly appended record is readable on its own.
        path = tmp_path / 'audit.jsonl'
        path.write_bytes(b'{"kind": "tool_call", "record": {"run_id": "torn", "tool_call_id": "lost"')  # no \n
        sink = JsonlAuditSink(path)

        await sink.record_tool_call(_tool('r1', 'c1'))
        await sink.record_run(_run('r1'))

        with caplog.at_level(logging.WARNING):
            calls = await sink.list_tool_calls(run_id='r1')
            run = await sink.get_run(run_id='r1')

        assert [c.tool_call_id for c in calls] == ['c1']  # the new record is readable, not swallowed
        assert run is not None and run.outcome == 'completed'
        assert await sink.list_tool_calls(run_id='torn') == []  # the torn fragment never merges into a real record
        assert 'unparseable' in caplog.text.lower()  # the torn fragment is skipped, not silently merged

    async def test_trailing_line_torn_mid_multibyte_utf8_char_does_not_fail_the_whole_read(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        # A crash can tear an append in the middle of a multi-byte UTF-8
        # character, not just mid-JSON-structure. Decoding the whole file as
        # UTF-8 in one call would raise `UnicodeDecodeError` for the entire
        # read and hide every valid record recorded before the torn one --
        # this one bad line must be skipped, exactly like a torn JSON line.
        path = tmp_path / 'audit.jsonl'
        sink = JsonlAuditSink(path)
        await sink.record_tool_call(_tool('r1', 'c1'))
        await sink.record_run(_run('r1'))
        # A line ending in a multi-byte UTF-8 character ('世', 3 bytes: \xe4\xb8\x96),
        # truncated by one byte so the file ends mid-character, with no
        # terminating newline -- what a write cut short by a crash leaves behind.
        torn = '{"kind": "tool_call", "record": {"run_id": "r1", "tool_name": "世'.encode()[:-1]
        with path.open('ab') as handle:
            handle.write(torn)

        with caplog.at_level(logging.WARNING):
            calls = await sink.list_tool_calls(run_id='r1')
            run = await sink.get_run(run_id='r1')

        assert [c.tool_call_id for c in calls] == ['c1']  # records before the torn line are still readable
        assert run is not None and run.outcome == 'completed'
        assert 'unparseable' in caplog.text.lower()  # the torn line is skipped, not fatal to the whole read

    @pytest.mark.parametrize(
        'malformed_line',
        [
            pytest.param('null', id='json-null'),
            pytest.param('{}', id='empty-object'),
            pytest.param('[]', id='json-array'),
            pytest.param('42', id='bare-number'),
            pytest.param('{"kind": "tool_call", "record": {}}', id='tool-call-missing-required-fields'),
            pytest.param('{"kind": "run", "record": {}}', id='run-missing-required-fields'),
            pytest.param('{"kind": "mystery", "record": {"foo": "bar"}}', id='unrecognized-kind'),
            pytest.param('{"kind": "tool_call", "record": {"run_id": "r1"', id='unterminated-json'),
            pytest.param(
                ('[' * 200_000) + '1' + (']' * 200_000),
                id='recursion-error-deeply-nested-json',
            ),
        ],
    )
    async def test_read_skips_any_malformed_or_foreign_line(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, malformed_line: str
    ):
        # Each of these decodes as UTF-8 and, except for the unterminated-json
        # and deeply-nested-json cases, parses as valid JSON -- but none is a
        # well-formed `tool_call` or `run` envelope. Before the fix, `_read`
        # caught only `UnicodeDecodeError`/`json.JSONDecodeError`, so a line
        # like this propagated a `TypeError` (not subscriptable), `KeyError`
        # (missing `kind`/`record`), or pydantic `ValidationError` (a `record`
        # missing its required fields) straight out of `list_tool_calls`/
        # `get_run`, crashing the read and hiding every valid record around
        # it. The deeply-nested-json case is different again: nesting an
        # array past the interpreter's recursion limit makes `json.loads`
        # itself raise `RecursionError`, not a shape or validation failure at
        # all -- which is why `_read` now catches `Exception` rather than an
        # enumerated list of the specific types the other cases raise; the
        # next pathological line would only need a different exception type
        # to escape again. Placing the malformed line between two valid
        # tool-call records proves it is skipped in place, not just
        # tolerated at the end of the file.
        path = tmp_path / 'audit.jsonl'
        sink = JsonlAuditSink(path)
        await sink.record_tool_call(_tool('r1', 'before'))
        await sink.record_run(_run('r1'))
        with path.open('a', encoding='utf-8') as handle:
            handle.write(malformed_line + '\n')
        await sink.record_tool_call(_tool('r1', 'after'))

        with caplog.at_level(logging.WARNING):
            calls = await sink.list_tool_calls(run_id='r1')
            run = await sink.get_run(run_id='r1')

        assert [c.tool_call_id for c in calls] == ['before', 'after']
        assert run is not None and run.outcome == 'completed'
        assert 'unparseable' in caplog.text.lower()  # the skip is logged, not silent

    async def test_read_skips_every_malformed_variant_coexisting_in_one_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        # The variants from test_read_skips_any_malformed_or_foreign_line, all
        # interleaved with valid records in one file, plus a genuinely torn
        # (no trailing newline) fragment at the end -- the crash-mid-write
        # shape the older torn-line tests cover. Different failure modes
        # coexisting in one read must not interact: every bad line is
        # independently skipped and logged, and every valid record survives.
        path = tmp_path / 'audit.jsonl'
        sink = JsonlAuditSink(path)
        malformed_lines = [
            'null',
            '{}',
            '[]',
            '42',
            '{"kind": "tool_call", "record": {}}',
            '{"kind": "run", "record": {}}',
            '{"kind": "mystery", "record": {"foo": "bar"}}',
        ]
        await sink.record_tool_call(_tool('r1', 'c0'))
        await sink.record_run(_run('r1'))
        for index, line in enumerate(malformed_lines):
            with path.open('a', encoding='utf-8') as handle:
                handle.write(line + '\n')
            await sink.record_tool_call(_tool('r1', f'c{index + 1}'))
        with path.open('a', encoding='utf-8') as handle:
            handle.write('{"kind": "tool_call", "record": {"run_id": "r1", "tool_call_id": "torn"')  # no \n

        bad_line_count = len(malformed_lines) + 1  # + the trailing torn fragment

        # Each accessor does its own full-file read, so count the warnings from
        # one read at a time rather than across both -- that call-count is an
        # unrelated implementation detail, not what this test is about.
        with caplog.at_level(logging.WARNING):
            calls = await sink.list_tool_calls(run_id='r1')
        assert caplog.text.lower().count('unparseable') == bad_line_count  # every bad line logged once, none silent

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            run = await sink.get_run(run_id='r1')
        assert caplog.text.lower().count('unparseable') == bad_line_count  # the second read skips them all too

        assert [c.tool_call_id for c in calls] == [f'c{i}' for i in range(len(malformed_lines) + 1)]
        assert run is not None and run.outcome == 'completed'
