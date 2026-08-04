"""Tests for public schedule-store backends."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.scheduling import (
    InMemoryScheduleStore,
    IntervalTrigger,
    Schedule,
    ScheduleConflictError,
    ScheduleStore,
    SqliteScheduleStore,
)

pytestmark = pytest.mark.anyio

StoreFactory = Callable[[], ScheduleStore]


@pytest.fixture(params=['memory', 'sqlite'])
def store_factory(request: pytest.FixtureRequest, tmp_path: Path) -> StoreFactory:
    """Build each store backend behind the public protocol."""
    if request.param == 'memory':
        return InMemoryScheduleStore
    return lambda: SqliteScheduleStore(str(tmp_path / 'schedules.db'))


def _schedule(schedule_id: str, name: str = 'job') -> Schedule:
    return Schedule(
        id=schedule_id,
        name=name,
        prompt=f'prompt:{name}',
        trigger=IntervalTrigger(every=timedelta(hours=1)),
        next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestScheduleStoreCrud:
    async def test_empty_and_unknown(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        assert await store.list() == []
        assert await store.get('missing') is None
        assert await store.remove('missing') is False
        with pytest.raises(ValueError, match='Unknown schedule id'):
            await store.save(_schedule('missing'))

    async def test_add_get_save_remove_and_order(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        first = await store.add(_schedule('first', 'First'))
        await store.add(_schedule('second', 'Second'))
        assert [item.id for item in await store.list()] == ['first', 'second']
        first.name = 'Changed'
        assert (await store.get('first')).name == 'First'  # type: ignore[union-attr]
        await store.save(first)
        persisted = await store.get('first')
        assert persisted is not None
        assert persisted.name == 'Changed'
        assert persisted.version == 1
        assert (await store.list())[0].version == 1
        with pytest.raises(ScheduleConflictError, match='changed since it was read'):
            await store.save(first)
        assert await store.remove('first') is True
        assert await store.get('first') is None

    async def test_duplicate_id_rejected(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        await store.add(_schedule('same'))
        with pytest.raises(ValueError, match='already exists'):
            await store.add(_schedule('same'))


class TestInMemoryScheduleStoreCopies:
    async def test_list_and_save_do_not_leak_aliases(self) -> None:
        store = InMemoryScheduleStore()
        original = _schedule('copy')
        await store.add(original)
        original.prompt = 'mutated outside'
        listed = await store.list()
        listed[0].prompt = 'mutated listing'
        stored = await store.get('copy')
        assert stored is not None
        assert stored.prompt == 'prompt:job'


class TestSqliteScheduleStore:
    def test_rejects_memory_and_bad_table(self, tmp_path: Path) -> None:
        for database in ('', ':memory:'):
            with pytest.raises(ValueError, match='does not support'):
                SqliteScheduleStore(database)
        with pytest.raises(ValueError, match='invalid table name'):
            SqliteScheduleStore(str(tmp_path / 'db.sqlite'), table='bad name; drop')

    async def test_roundtrip_across_instances(self, tmp_path: Path) -> None:
        database = str(tmp_path / 'persistent.db')
        expected = _schedule('kept')
        expected.usage_limits = UsageLimits(request_limit=3, total_tokens_limit=10_000)
        await SqliteScheduleStore(database).add(expected)
        loaded = await SqliteScheduleStore(database).get('kept')
        assert loaded is not None
        assert loaded.model_dump() == expected.model_dump()

    async def test_version_check_holds_across_store_instances(self, tmp_path: Path) -> None:
        database = str(tmp_path / 'shared.db')
        store_a = SqliteScheduleStore(database)
        store_b = SqliteScheduleStore(database)
        await store_a.add(_schedule('shared'))

        from_a = await store_a.get('shared')
        from_b = await store_b.get('shared')
        assert from_a is not None
        assert from_b is not None
        from_a.name = 'from a'
        from_b.name = 'from b'
        await store_a.save(from_a)

        with pytest.raises(ScheduleConflictError, match='changed since it was read'):
            await store_b.save(from_b)
        stored = await store_b.get('shared')
        assert stored is not None
        assert stored.name == 'from a'
        assert stored.version == 1

    async def test_schema_failure_closes_the_connection(self, tmp_path: Path) -> None:
        database = tmp_path / 'corrupt.db'
        database.write_bytes(b'not a sqlite database')
        store = SqliteScheduleStore(str(database))
        with pytest.raises(sqlite3.DatabaseError):
            await store.add(_schedule('any'))

    async def test_reserved_word_table_name(self, tmp_path: Path) -> None:
        store = SqliteScheduleStore(str(tmp_path / 'reserved.db'), table='select')
        added = await store.add(_schedule('quoted'))
        added.name = 'updated'
        await store.save(added)
        loaded = await store.get('quoted')
        assert loaded is not None
        assert loaded.name == 'updated'
        assert [schedule.id for schedule in await store.list()] == ['quoted']
        assert await store.remove('quoted') is True
