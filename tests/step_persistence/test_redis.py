"""Tests for `RedisStepStore` against a fake implementing the `RedisClient` protocol.

The store speaks only the commands on that protocol, so a dict-backed fake
reaches every path without a running server. What the fake cannot answer -- that
the real server accepts these commands, and that `EXPIRE` actually removes a key
-- lives in `integration_tests/redis/test_live_step_store.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.media import DiskMediaStore
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RedisClient,
    RedisStepStore,
    RunRecord,
    StepEvent,
    StepPersistence,
    StepStore,
    ToolEffectRecord,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Restrict async tests to asyncio (Agent.run uses `asyncio.create_task`)."""
    return 'asyncio'


class FakeRedis:
    """The commands `RedisStepStore` uses, over plain dicts.

    Replies are `bytes` by default, as `redis.asyncio.Redis` returns them unless
    it was built with `decode_responses=True`; `decode=True` covers the other
    configuration. Expiry is recorded, not enforced -- tests that need a key to
    be gone remove it directly, which is what an elapsed TTL looks like from the
    store's side.
    """

    def __init__(self, *, decode: bool = False) -> None:
        self.strings: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.expiries: dict[str, int] = {}
        self._decode = decode

    def _out(self, value: str) -> str | bytes:
        return value if self._decode else value.encode()

    async def get(self, name: str) -> object:
        value = self.strings.get(name)
        return None if value is None else self._out(value)

    async def set(self, name: str, value: str, *, ex: int | None = None, nx: bool = False) -> object:
        if nx and name in self.strings:
            return None
        self.strings[name] = value
        if ex is not None:
            self.expiries[name] = ex
        return True

    async def delete(self, *names: str) -> object:
        for name in names:
            self.strings.pop(name, None)
            self.expiries.pop(name, None)
        return len(names)

    async def expire(self, name: str, seconds: int) -> object:
        self.expiries[name] = seconds
        return True

    async def incr(self, name: str) -> int:
        value = int(self.strings.get(name, '0')) + 1
        self.strings[name] = str(value)
        return value

    async def rpush(self, name: str, *values: str) -> object:
        self.lists.setdefault(name, []).extend(values)
        return len(self.lists[name])

    async def lrange(self, name: str, start: int, end: int) -> list[str | bytes]:
        stop = None if end == -1 else end + 1
        return [self._out(item) for item in self.lists.get(name, [])[start:stop]]

    async def sadd(self, name: str, *values: str) -> object:
        self.sets.setdefault(name, set()).update(values)
        return len(values)

    async def srem(self, name: str, *values: str) -> object:
        self.sets.setdefault(name, set()).difference_update(values)
        return len(values)

    async def smembers(self, name: str) -> list[str | bytes]:
        return [self._out(member) for member in sorted(self.sets.get(name, set()))]

    async def zadd(self, name: str, mapping: Mapping[str, float]) -> object:
        self.zsets.setdefault(name, {}).update(mapping)
        return len(mapping)

    async def zrange(self, name: str, start: int, end: int) -> list[str | bytes]:
        ordered = sorted(self.zsets.get(name, {}).items(), key=lambda entry: (entry[1], entry[0]))
        stop = None if end == -1 else end + 1
        return [self._out(member) for member, _ in ordered[start:stop]]

    async def zrem(self, name: str, *values: str) -> object:
        members = self.zsets.setdefault(name, {})
        for value in values:
            members.pop(value, None)
        return len(values)

    async def hset(self, name: str, key: str, value: str) -> object:
        self.hashes.setdefault(name, {})[key] = value
        return 1

    async def hget(self, name: str, key: str) -> object:
        value = self.hashes.get(name, {}).get(key)
        return None if value is None else self._out(value)

    async def hgetall(self, name: str) -> Mapping[str | bytes, str | bytes]:
        return {self._out(key): self._out(value) for key, value in self.hashes.get(name, {}).items()}


class UntypedReplyRedis(FakeRedis):
    """A client whose `GET` hands back something that is neither `str` nor `bytes`."""

    async def get(self, name: str) -> object:
        return 1


def _messages(content: str = 'hello') -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=content)])]


def _at(minute: int) -> datetime:
    return datetime(2026, 1, 1, 12, minute, tzinfo=timezone.utc)


class TestRedisStepStoreConstruction:
    def test_fake_satisfies_the_client_protocol(self) -> None:
        assert isinstance(FakeRedis(), RedisClient)

    def test_store_satisfies_the_step_store_protocol(self) -> None:
        assert isinstance(RedisStepStore(FakeRedis()), StepStore)

    def test_media_is_inline_by_default(self) -> None:
        store = RedisStepStore(FakeRedis())
        assert store._media_store is None  # pyright: ignore[reportPrivateUsage]

    def test_rejects_invalid_max_snapshots_per_run(self) -> None:
        with pytest.raises(ValueError, match='max_snapshots_per_run must be'):
            RedisStepStore(FakeRedis(), max_snapshots_per_run=0)

    @pytest.mark.parametrize('value', [0, -1, 1.5])
    def test_rejects_invalid_expire_seconds(self, value: object) -> None:
        with pytest.raises(ValueError, match='expire_seconds must be an int >= 1 or None'):
            RedisStepStore(FakeRedis(), expire_seconds=value)  # pyright: ignore[reportArgumentType]

    async def test_prefix_namespaces_every_key(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, prefix='tenant-a')
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )

        touched = {*client.strings, *client.lists, *client.sets, *client.hashes, *client.zsets}
        assert touched == {
            'tenant-a:run:{r1}',
            'tenant-a:runs',
            'tenant-a:events:{r1}',
            'tenant-a:snapshots:seq:{r1}',
            'tenant-a:snapshots:{r1}',
            'tenant-a:snapshot:{r1}:1',
            'tenant-a:tool_effects:{r1}',
        }


class TestRedisStepStoreProtocol:
    async def test_register_and_get_run(self) -> None:
        store = RedisStepStore(FakeRedis())
        record = RunRecord(run_id='r1', conversation_id='c1', agent_name='a', metadata={'k': 'v'})
        await store.register_run(record)

        loaded = await store.get_run(run_id='r1')
        assert loaded is not None
        assert (loaded.run_id, loaded.conversation_id, loaded.agent_name, loaded.metadata) == (
            'r1',
            'c1',
            'a',
            {'k': 'v'},
        )

    async def test_register_duplicate_run_raises(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='r1'))
        with pytest.raises(ValueError, match="run_id 'r1' is already registered"):
            await store.register_run(RunRecord(run_id='r1'))

    async def test_missing_lookups_return_none_or_empty(self) -> None:
        store = RedisStepStore(FakeRedis())
        assert await store.get_run(run_id='nope') is None
        assert await store.list_events(run_id='nope') == []
        assert await store.latest_snapshot(run_id='nope') is None
        assert await store.list_snapshots(run_id='nope') == []
        assert await store.get_tool_effect(run_id='nope', tool_call_id='t1') is None
        assert await store.list_unresolved_tool_effects(run_id='nope') == []
        assert await store.list_runs() == []

    async def test_list_runs_chronological(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='b', started_at=_at(2)))
        await store.register_run(RunRecord(run_id='a', started_at=_at(1)))
        await store.register_run(RunRecord(run_id='c', started_at=_at(3)))

        assert [r.run_id for r in await store.list_runs()] == ['a', 'b', 'c']

    async def test_list_runs_filters(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', parent_run_id='p1', started_at=_at(1)))
        await store.register_run(RunRecord(run_id='r2', conversation_id='c2', parent_run_id='p1', started_at=_at(2)))
        await store.register_run(RunRecord(run_id='r3', conversation_id='c1', started_at=_at(3)))

        assert [r.run_id for r in await store.list_runs(parent_run_id='p1')] == ['r1', 'r2']
        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['r1', 'r3']
        both = await store.list_runs(parent_run_id='p1', conversation_id='c2')
        assert [r.run_id for r in both] == ['r2']

    async def test_list_runs_sorts_by_instant_not_iso_string(self) -> None:
        """`13:00+02:00` precedes `12:00Z` as an instant but follows it lexicographically."""
        store = RedisStepStore(FakeRedis())
        earlier = datetime(2026, 1, 1, 13, 0, tzinfo=timezone(timedelta(hours=2)))
        later = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        await store.register_run(RunRecord(run_id='later', started_at=later))
        await store.register_run(RunRecord(run_id='earlier', started_at=earlier))

        assert [r.run_id for r in await store.list_runs()] == ['earlier', 'later']

    async def test_list_runs_heals_an_index_pointing_at_an_expired_run(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=60)
        await store.register_run(RunRecord(run_id='alive', conversation_id='c1', started_at=_at(1)))
        await store.register_run(RunRecord(run_id='gone', conversation_id='c1', started_at=_at(2)))
        del client.strings['pydantic-ai-harness:step:run:{gone}']

        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['alive']
        assert client.sets['pydantic-ai-harness:step:runs:conversation:c1'] == {'alive'}

    async def test_append_and_list_events(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(
            StepEvent(run_id='r1', kind='tool_call_started', step_index=1, tool_name='search', tool_call_id='t1')
        )
        await store.append_event(StepEvent(run_id='r2', kind='run_started', step_index=0))

        events = await store.list_events(run_id='r1')
        assert [(e.kind, e.step_index, e.tool_name) for e in events] == [
            ('run_started', 0, None),
            ('tool_call_started', 1, 'search'),
        ]

    async def test_save_and_load_snapshot(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=3, messages=_messages('a'), conversation_id='c1')
        )

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert (snap.step_index, snap.conversation_id, snap.state) == (3, 'c1', 'complete')
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert prompt.content == 'a'

    async def test_latest_snapshot_default_skips_interrupted(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages(), state='interrupted')
        )

        default = await store.latest_snapshot(run_id='r1')
        assert default is not None and default.step_index == 0
        opted = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert opted is not None and (opted.step_index, opted.state) == (1, 'interrupted')

    async def test_latest_snapshot_skips_a_vanished_payload(self) -> None:
        """A payload that expired ahead of its index member must not hide the one below it."""
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=60)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        del client.strings['pydantic-ai-harness:step:snapshot:{r1}:2']

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None and snap.step_index == 0

    async def test_snapshot_seq_monotonic_across_reset_step_index(self) -> None:
        """A reused `run_id` whose `step_index` restarts must not clobber the earlier snapshot."""
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=7, messages=_messages('first')))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('second')))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [7, 0]
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None and snap.step_index == 0

    async def test_index_members_that_are_not_snapshots_are_skipped(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        client.zsets['pydantic-ai-harness:step:snapshots:{r1}'].update({'nonsense': 2.0, 'x:complete': 3.0})

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0]

    async def test_tool_effect_upsert_and_run_scope(self) -> None:
        store = RedisStepStore(FakeRedis())
        started = ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        await store.record_tool_effect(started)
        await store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id='t1',
                tool_name='ship',
                run_id='r1',
                status='completed',
                ended_at=_at(5),
                effect_summary='shipped',
            )
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r2', status='started')
        )

        record = await store.get_tool_effect(run_id='r1', tool_call_id='t1')
        assert record is not None
        assert (record.status, record.effect_summary, record.ended_at) == ('completed', 'shipped', _at(5))
        other = await store.get_tool_effect(run_id='r2', tool_call_id='t1')
        assert other is not None and other.status == 'started'

    async def test_list_unresolved_tool_effects(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='a', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t2', tool_name='b', run_id='r1', status='completed')
        )
        await store.record_tool_effect(ToolEffectRecord(tool_call_id='t3', tool_name='c', run_id='r1', status='failed'))

        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['t1']

    async def test_decode_responses_client(self) -> None:
        """A client built with `decode_responses=True` returns `str`, and reads still work."""
        store = RedisStepStore(FakeRedis(decode=True))
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )

        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['r1']
        assert [e.kind for e in await store.list_events(run_id='r1')] == ['run_started']
        assert await store.latest_snapshot(run_id='r1') is not None
        assert [r.tool_call_id for r in await store.list_unresolved_tool_effects(run_id='r1')] == ['t1']

    async def test_a_reply_that_is_neither_str_nor_bytes_raises(self) -> None:
        store = RedisStepStore(UntypedReplyRedis())
        with pytest.raises(ValueError, match='expected a string reply, got int'):
            await store.get_run(run_id='r1')

    async def test_corrupted_snapshot_payload_raises(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        key = 'pydantic-ai-harness:step:snapshot:{r1}:1'
        client.strings[key] = client.strings[key].replace('"timestamp"', '"step_index": "x", "timestamp"', 1)

        with pytest.raises(ValueError, match='snapshot payload has wrong types'):
            await store.latest_snapshot(run_id='r1')

    async def test_non_ascii_metadata_round_trips(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='r1', metadata={'ключ': 'значение'}))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, metadata={'emoji': '🚀'}))

        run = await store.get_run(run_id='r1')
        assert run is not None and run.metadata == {'ключ': 'значение'}
        assert [e.metadata for e in await store.list_events(run_id='r1')] == [{'emoji': '🚀'}]


class TestRedisStepStoreListSnapshots:
    async def test_write_order_and_interrupted_filter(self) -> None:
        store = RedisStepStore(FakeRedis())
        # Descending then ascending `step_index` so write order is neither a
        # `step_index` sort nor its reverse.
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=_messages()))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages(), state='interrupted')
        )
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=9, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [2, 1]
        opted = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in opted] == [2, 0, 1]
        assert [s.state for s in opted] == ['complete', 'interrupted', 'complete']

    async def test_skips_a_vanished_payload(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=60)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        del client.strings['pydantic-ai-harness:step:snapshot:{r1}:1']

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [1]

    async def test_skips_an_unparsable_payload(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        client.strings['pydantic-ai-harness:step:snapshot:{r1}:1'] = '{"not": "a snapshot"}'

        with caplog.at_level(logging.WARNING):
            snapshots = await store.list_snapshots(run_id='r1')
        assert [s.step_index for s in snapshots] == [1]
        assert 'Skipping unparsable snapshot for run r1' in caplog.text

    async def test_store_is_accepted_as_a_search_substrate(self) -> None:
        """`SnapshotHistorySource` rejects stores lacking `list_snapshots` at construction."""
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('remember this')))

        source = SnapshotHistorySource(store)
        assert [type(m).__name__ for m in await source.run_history(run_id='r1')] == ['ModelRequest']


class TestRedisStepStoreRetention:
    async def test_unbounded_keeps_every_snapshot(self) -> None:
        store = RedisStepStore(FakeRedis())
        for step in range(4):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0, 1, 2, 3]

    async def test_prunes_to_the_newest_window(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, max_snapshots_per_run=2)
        for step in range(5):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [3, 4]
        assert set(client.zsets['pydantic-ai-harness:step:snapshots:{r1}']) == {'4:complete', '5:complete'}
        assert 'pydantic-ai-harness:step:snapshot:{r1}:1' not in client.strings

    async def test_keeps_newest_complete_below_the_window(self) -> None:
        store = RedisStepStore(FakeRedis(), max_snapshots_per_run=2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        for step in (1, 2, 3):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages(), state='interrupted')
            )

        retained = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in retained] == [0, 2, 3]
        resumable = await store.latest_snapshot(run_id='r1')
        assert resumable is not None and resumable.step_index == 0

    async def test_one_over_the_bound_can_have_nothing_to_drop(self) -> None:
        """The retain set exceeds the bound here, so the prune finds no candidate."""
        client = FakeRedis()
        store = RedisStepStore(client, max_snapshots_per_run=2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        for step in (1, 2):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages(), state='interrupted')
            )

        assert len(client.zsets['pydantic-ai-harness:step:snapshots:{r1}']) == 3

    async def test_pruning_is_scoped_to_the_written_run(self) -> None:
        store = RedisStepStore(FakeRedis(), max_snapshots_per_run=1)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        for step in range(3):
            await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=step, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0]
        assert [s.step_index for s in await store.list_snapshots(run_id='r2')] == [2]


class TestRedisStepStoreExpiry:
    async def test_no_expiry_by_default(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )

        assert client.expiries == {}

    async def test_every_write_refreshes_the_runs_keys(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=120)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )

        assert client.expiries == {
            'pydantic-ai-harness:step:run:{r1}': 120,
            'pydantic-ai-harness:step:events:{r1}': 120,
            'pydantic-ai-harness:step:snapshots:seq:{r1}': 120,
            'pydantic-ai-harness:step:snapshots:{r1}': 120,
            'pydantic-ai-harness:step:snapshot:{r1}:1': 120,
            'pydantic-ai-harness:step:tool_effects:{r1}': 120,
        }

    async def test_index_sets_never_expire(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=120)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', parent_run_id='p1'))

        assert 'pydantic-ai-harness:step:runs' not in client.expiries
        assert 'pydantic-ai-harness:step:runs:conversation:c1' not in client.expiries
        assert 'pydantic-ai-harness:step:runs:parent:p1' not in client.expiries


class TestRedisStepStoreMedia:
    async def test_large_binary_and_text_externalized_and_restored(self, tmp_path: Path) -> None:
        media = tmp_path / 'media'
        store = RedisStepStore(FakeRedis(), media_store=DiskMediaStore(media), media_threshold_bytes=64 * 1024)
        big_binary = b'\xab' * 100_000
        big_text = 'Z' * 100_000
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[BinaryContent(data=big_binary, media_type='image/png')]),
                    ToolReturnPart(tool_name='scrape', content=big_text, tool_call_id='t1'),
                ]
            ),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        assert len(list(media.iterdir())) == 2

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = prompt.content[0]
        assert isinstance(binary, BinaryContent)
        assert binary.data == big_binary
        tool_return = request.parts[1]
        assert isinstance(tool_return, ToolReturnPart)
        assert tool_return.content == big_text

    async def test_below_threshold_stays_inline(self, tmp_path: Path) -> None:
        media = tmp_path / 'media'
        store = RedisStepStore(FakeRedis(), media_store=DiskMediaStore(media), media_threshold_bytes=64 * 1024)
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='small')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        assert not media.exists() or list(media.iterdir()) == []
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        response = snap.messages[0]
        assert isinstance(response, ModelResponse)
        text = response.parts[0]
        assert isinstance(text, TextPart)
        assert text.content == 'small'


class TestRedisStepStoreThroughAgent:
    async def test_agent_run_round_trips_through_step_persistence(self) -> None:
        store = RedisStepStore(FakeRedis(), expire_seconds=3600)
        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='vision')],
        )
        result = await agent.run('classify this')
        assert isinstance(result.output, str)

        runs = await store.list_runs()
        assert len(runs) == 1
        assert runs[0].agent_name == 'vision'
        kinds = [e.kind for e in await store.list_events(run_id=runs[0].run_id)]
        assert kinds[0] == 'run_started' and kinds[-1] == 'run_completed'
        snap = await store.latest_snapshot(run_id=runs[0].run_id)
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert prompt.content == 'classify this'
