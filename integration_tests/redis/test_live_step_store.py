"""Live Redis conformance tests for `RedisStepStore`.

The unit suite in `tests/step_persistence/test_redis.py` drives the store
through a dict-backed fake. The fake implements the same protocol, so what it
cannot answer is everything that depends on the server itself:

- that a real server accepts this command surface at all, including `SET ... NX`
  as the single-shot `run_id` guard;
- that `ZRANGE` orders the snapshot index by score, so `seq` 10 follows `seq` 2
  rather than preceding it as the member text would sort;
- that `EXPIRE` reaches every key kind the store writes (string, list, set,
  zset, hash), that a later write refreshes the window, and that an elapsed TTL
  really removes the key -- the state the index self-heal exists for;
- that `INCR` allocates `seq` on the server, so two processes sharing a run
  cannot hand out the same snapshot slot;
- that a client left on redis-py's default `bytes` replies reads back what a
  `decode_responses=True` client wrote.

This file covers exactly those. It is not a second copy of the unit suite --
API-shape coverage belongs there, where it runs on every matrix leg.

Run against a local server with `make integration-redis` after starting one, e.g.
`docker run -d -p 6379:6379 redis:8`. Without a reachable server the tests skip,
unless `REDIS_REQUIRE_LIVE` is set (CI does), where an unreachable server fails
instead -- a service container that never came up must not pass as a silent skip.

External assumptions, verified 2026-08-29 against Redis 8 (`redis:8`):

- `SET` with `NX` writes only when the key is absent, replying nil otherwise.
  Source: <https://redis.io/docs/latest/commands/set/>.
- `ZRANGE key 0 -1` returns members ordered by score, low to high. Source:
  <https://redis.io/docs/latest/commands/zrange/>.
- `TTL` replies -1 for a key with no expiry and -2 for a key that is gone.
  Source: <https://redis.io/docs/latest/commands/ttl/>.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from typing import NoReturn
from urllib.parse import urlsplit

import anyio
import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from redis.asyncio import Redis
from redis.exceptions import RedisError

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RedisStepStore,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run live server tests once under asyncio."""
    return 'asyncio'


def _requires_live() -> bool:
    return os.environ.get('REDIS_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}


def _redis_url() -> str:
    return os.environ.get('REDIS_TEST_URL', 'redis://127.0.0.1:6379')


def _redis_target() -> str:
    """Host and port, without the credentials a Redis URL may carry.

    `REDIS_TEST_URL` accepts `redis://user:password@host:6379`, and the only place
    this string is used is a skip or failure message -- which CI keeps. Naming the
    variable rather than echoing an unparsable value keeps that true for a URL
    `urlsplit` cannot make sense of.
    """
    host = urlsplit(_redis_url()).hostname
    if host is None:
        return 'the server named by REDIS_TEST_URL'
    port = urlsplit(_redis_url()).port
    return host if port is None else f'{host}:{port}'


def _unavailable(message: str) -> NoReturn:
    if _requires_live():
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
def prefix() -> str:
    """A key namespace unique to one test, so concurrent runs cannot collide."""
    return f'harness-test:{uuid.uuid4().hex}'


@pytest.fixture
async def redis_client(prefix: str) -> AsyncGenerator[Redis, None]:
    """A connected client whose `{prefix}:*` keys are removed after the test."""
    client = Redis.from_url(_redis_url(), decode_responses=True)
    try:
        await client.ping()
    except (RedisError, OSError) as error:
        await client.aclose()
        _unavailable(f'no reachable Redis at {_redis_target()}: {error}')

    try:
        yield client
    finally:
        # Closing sits in its own `finally` so a server that went away mid-test takes the
        # key cleanup down without also leaking the connection into the rest of the suite.
        try:
            keys = [key async for key in client.scan_iter(match=f'{prefix}:*')]
            if keys:
                await client.delete(*keys)
        finally:
            await client.aclose()


def _messages(content: str = 'hello') -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=content)])]


class TestLiveRedisStepStore:
    """What the fake reproduces rather than executes."""

    async def test_the_command_surface_round_trips(self, redis_client: Redis, prefix: str) -> None:
        """Every command the store issues, accepted by the server and read back."""
        store = RedisStepStore(redis_client, prefix=prefix)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', parent_run_id='p1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('a')))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages('b'), state='interrupted')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )

        assert [r.run_id for r in await store.list_runs(parent_run_id='p1', conversation_id='c1')] == ['r1']
        assert [e.kind for e in await store.list_events(run_id='r1')] == ['run_started']
        settled = await store.latest_snapshot(run_id='r1')
        assert settled is not None and settled.step_index == 0
        frontier = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert frontier is not None and frontier.step_index == 1
        assert [s.step_index for s in await store.list_snapshots(run_id='r1', include_interrupted=True)] == [0, 1]
        assert [r.tool_call_id for r in await store.list_unresolved_tool_effects(run_id='r1')] == ['t1']

    async def test_set_nx_refuses_a_second_registration(self, redis_client: Redis, prefix: str) -> None:
        """The single-shot `run_id` contract, enforced by the server rather than a pre-check."""
        store = RedisStepStore(redis_client, prefix=prefix)
        await store.register_run(RunRecord(run_id='r1', agent_name='first'))

        with pytest.raises(ValueError, match="run_id 'r1' is already registered"):
            await store.register_run(RunRecord(run_id='r1', agent_name='second'))

        kept = await store.get_run(run_id='r1')
        assert kept is not None and kept.agent_name == 'first'

    async def test_snapshot_index_orders_by_score_not_member_text(self, redis_client: Redis, prefix: str) -> None:
        """`10:complete` sorts before `2:complete` as text; the score has to decide."""
        store = RedisStepStore(redis_client, prefix=prefix)
        for step in range(11):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == list(range(11))
        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None and latest.step_index == 10

    async def test_expire_seconds_reaches_every_key_kind(self, redis_client: Redis, prefix: str) -> None:
        """String, list, set, zset, and hash all take the TTL, the shared index sets included."""
        store = RedisStepStore(redis_client, prefix=prefix, expire_seconds=60)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )

        ttls = {key: await redis_client.ttl(key) async for key in redis_client.scan_iter(match=f'{prefix}:*')}
        run_scoped = {key: ttl for key, ttl in ttls.items() if '{r1}' in key}
        shared = {key: ttl for key, ttl in ttls.items() if '{r1}' not in key}
        assert len(run_scoped) == 6
        assert all(0 < ttl <= 60 for ttl in run_scoped.values()), run_scoped
        assert set(shared) == {f'{prefix}:runs:all', f'{prefix}:runs:conversation:c1'}
        assert all(0 < ttl <= 60 for ttl in shared.values()), shared

    async def test_a_later_write_refreshes_the_window(self, redis_client: Redis, prefix: str) -> None:
        """An active run keeps its window; only a run nothing writes to falls out of it."""
        store = RedisStepStore(redis_client, prefix=prefix, expire_seconds=2)
        await store.register_run(RunRecord(run_id='r1'))
        run_key = f'{prefix}:run:{{r1}}'
        await anyio.sleep(1.1)
        assert await redis_client.ttl(run_key) == 1

        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))

        assert await redis_client.ttl(run_key) == 2

    async def test_an_elapsed_ttl_leaves_a_stale_index_the_read_heals(self, redis_client: Redis, prefix: str) -> None:
        """A newer run's registration keeps the set alive past an older run's key; the read drops the stale id."""
        store = RedisStepStore(redis_client, prefix=prefix, expire_seconds=2)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await anyio.sleep(1.2)
        await store.register_run(RunRecord(run_id='r2', conversation_id='c1'))
        await anyio.sleep(1.1)
        assert await redis_client.ttl(f'{prefix}:run:{{r1}}') == -2
        assert await redis_client.smembers(f'{prefix}:runs:conversation:c1') == {'r1', 'r2'}

        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['r2']

        assert await redis_client.smembers(f'{prefix}:runs:conversation:c1') == {'r2'}

    async def test_seq_is_allocated_by_the_server(self, redis_client: Redis, prefix: str) -> None:
        """Two store instances on one run must not hand out the same snapshot slot."""
        one = RedisStepStore(redis_client, prefix=prefix)
        two = RedisStepStore(redis_client, prefix=prefix)
        await one.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('from one')))
        await two.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('from two')))

        assert await redis_client.zrange(f'{prefix}:snapshots:{{r1}}', 0, -1) == ['1:complete', '2:complete']
        assert len(await one.list_snapshots(run_id='r1')) == 2

    async def test_a_bytes_client_reads_what_a_decoded_client_wrote(self, redis_client: Redis, prefix: str) -> None:
        """redis-py replies with `bytes` unless asked otherwise, and both must work."""
        decoded = RedisStepStore(redis_client, prefix=prefix)
        await decoded.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await decoded.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=4, messages=_messages('a')))
        await decoded.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )

        raw_client = Redis.from_url(_redis_url())
        try:
            raw = RedisStepStore(raw_client, prefix=prefix)
            assert [r.run_id for r in await raw.list_runs(conversation_id='c1')] == ['r1']
            snapshot = await raw.latest_snapshot(run_id='r1')
            assert snapshot is not None and snapshot.step_index == 4
            assert [r.tool_call_id for r in await raw.list_unresolved_tool_effects(run_id='r1')] == ['t1']
        finally:
            await raw_client.aclose()
