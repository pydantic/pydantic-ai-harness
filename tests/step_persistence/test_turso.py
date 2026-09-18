"""`SqliteStepStore` against a real Turso connection.

Turso is a SQLite fork whose Python client speaks DB-API 2.0, so it is passed to the existing
store as a caller-owned `connection=` rather than needing a backend of its own. These tests run
the real driver -- the point is that the store's SQL and its error handling hold against a
non-stdlib connection, which no fake would prove.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import turso
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    SqliteStepStore,
    StepEvent,
    ToolEffectRecord,
)

pytestmark = pytest.mark.anyio

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def turso_store(tmp_path: Path) -> SqliteStepStore:
    return SqliteStepStore(connection=turso.connect(str(tmp_path / 'runs.db')), media_store=None)


async def test_every_record_type_round_trips(turso_store: SqliteStepStore) -> None:
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hello')])]

    await turso_store.register_run(RunRecord(run_id='r1', conversation_id='c1', started_at=TS))
    await turso_store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, timestamp=TS))
    await turso_store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=messages, timestamp=TS))
    await turso_store.record_tool_effect(
        ToolEffectRecord(run_id='r1', tool_call_id='t1', tool_name='get_weather', status='started', started_at=TS)
    )

    run = await turso_store.get_run(run_id='r1')
    assert run is not None
    assert run.conversation_id == 'c1'
    assert [event.kind for event in await turso_store.list_events(run_id='r1')] == ['run_started']
    snapshot = await turso_store.latest_snapshot(run_id='r1')
    assert snapshot is not None
    assert snapshot.messages == messages
    effect = await turso_store.get_tool_effect(run_id='r1', tool_call_id='t1')
    assert effect is not None
    assert effect.status == 'started'
    assert [record.run_id for record in await turso_store.list_runs()] == ['r1']
    assert [record.tool_call_id for record in await turso_store.list_unresolved_tool_effects(run_id='r1')] == ['t1']


async def test_reused_run_id_raises_the_drivers_integrity_error(turso_store: SqliteStepStore) -> None:
    """The single-shot `run_id` contract rests on the primary key, not on a `sqlite3` exception class."""
    await turso_store.register_run(RunRecord(run_id='r1'))

    with pytest.raises(turso.IntegrityError):
        await turso_store.register_run(RunRecord(run_id='r1'))


async def test_legacy_database_without_state_column_migrates(tmp_path: Path) -> None:
    """The `ALTER TABLE` migrations are attempt-and-catch, and Turso raises a different class.

    `turso.DatabaseError` does not inherit from `sqlite3.OperationalError`, so catching the
    driver's own `DatabaseError` off the connection is what keeps this path working. The
    "column is already there" re-check that makes the broad catch safe is unchanged.
    """
    db = tmp_path / 'runs.db'
    setup = turso.connect(str(db))
    setup.executescript(
        'CREATE TABLE snapshots ('
        'seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, step_index INTEGER NOT NULL, '
        'conversation_id TEXT, parent_run_id TEXT, agent_name TEXT, timestamp TEXT NOT NULL, '
        'messages TEXT NOT NULL);'
    )
    setup.execute(
        'INSERT INTO snapshots (run_id, step_index, conversation_id, parent_run_id, agent_name, '
        "timestamp, messages) VALUES ('r1', 3, NULL, NULL, NULL, '2026-01-01T00:00:00+00:00', '[]')"
    )
    setup.commit()

    store = SqliteStepStore(connection=turso.connect(str(db)), media_store=None)
    migrated = await store.latest_snapshot(run_id='r1')

    assert migrated is not None
    assert migrated.state == 'complete'
    assert migrated.step_index == 3
