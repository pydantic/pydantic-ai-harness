"""Redis-backed step-persistence store over a caller-owned client.

Like `RedisSpendStore` and `RedisPlanStore`, this store never imports `redis`:
it depends only on the `RedisClient` protocol below, so installing the harness
pulls in no Redis driver. Pass your own `redis.asyncio.Redis` client.

Keys under the configured `prefix`, with a `{run_id}` hash tag so one run's keys
land on a single Redis Cluster slot:

```text
<prefix>:run:{<run_id>}              string  RunRecord JSON
<prefix>:runs                        set     every run id
<prefix>:runs:conversation:<cid>     set     run ids in one conversation
<prefix>:runs:parent:<pid>           set     run ids spawned by one run
<prefix>:events:{<run_id>}           list    one StepEvent JSON per RPUSH
<prefix>:snapshots:seq:{<run_id>}    string  INCR counter allocating `seq`
<prefix>:snapshots:{<run_id>}        zset    member `<seq>:<state>`, score `seq`
<prefix>:snapshot:{<run_id>}:<seq>   string  ContinuableSnapshot JSON
<prefix>:tool_effects:{<run_id>}     hash    tool_call_id -> ToolEffectRecord JSON
```

Snapshots are keyed by a per-run monotonic `seq` from `INCR`, not by
`step_index`: `ctx.run_step` resets each `Agent.run`, so a reused `run_id` would
clobber an earlier snapshot. Matches `SqliteStepStore`'s `AUTOINCREMENT seq`.
The index member carries the state (`12:complete`) rather than only the payload,
so choosing what to read and what to prune costs one `ZRANGE` and no fetches.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pydantic_ai_harness.media import MediaStore, externalize_media, restore_media
from pydantic_ai_harness.step_persistence._store import (
    _DEFAULT_MEDIA_THRESHOLD_BYTES,  # pyright: ignore[reportPrivateUsage]
    _event_from_dict,  # pyright: ignore[reportPrivateUsage]
    _event_to_dict,  # pyright: ignore[reportPrivateUsage]
    _load_json_object,  # pyright: ignore[reportPrivateUsage]
    _opt_str,  # pyright: ignore[reportPrivateUsage]
    _retained_seqs,  # pyright: ignore[reportPrivateUsage]
    _run_from_dict,  # pyright: ignore[reportPrivateUsage]
    _run_to_dict,  # pyright: ignore[reportPrivateUsage]
    _snapshot_state,  # pyright: ignore[reportPrivateUsage]
    _tool_effect_from_dict,  # pyright: ignore[reportPrivateUsage]
    _tool_effect_to_dict,  # pyright: ignore[reportPrivateUsage]
    _validate_max_snapshots,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    RunRecord,
    SnapshotState,
    StepEvent,
    ToolEffectRecord,
)

_logger = logging.getLogger(__name__)

_DEFAULT_PREFIX = 'pydantic-ai-harness:step'


@runtime_checkable
class RedisClient(Protocol):
    """The part of a Redis client `RedisStepStore` uses.

    `redis.asyncio.Redis` satisfies this, as does any wrapper or fake exposing
    the same commands. Arguments are passed positionally (except `set`'s `ex`
    and `nx`), so a client that names them differently still works.

    Declared as returning `Awaitable` rather than as `async def`, which would
    narrow the requirement to a `Coroutine`: `redis.asyncio.Redis` types its
    commands as `Awaitable`, so an `async def` protocol refuses the one client
    this exists to accept.
    """

    def get(self, name: str, /) -> Awaitable[object]: ...  # pragma: no cover

    def set(
        self, name: str, value: str, /, *, ex: int | None = None, nx: bool = False
    ) -> Awaitable[object]: ...  # pragma: no cover

    def delete(self, *names: str) -> Awaitable[object]: ...  # pragma: no cover

    def expire(self, name: str, seconds: int, /) -> Awaitable[object]: ...  # pragma: no cover

    def incr(self, name: str, /) -> Awaitable[int]: ...  # pragma: no cover

    def rpush(self, name: str, *values: str) -> Awaitable[object]: ...  # pragma: no cover

    def lrange(self, name: str, start: int, end: int, /) -> Awaitable[Sequence[str | bytes]]: ...  # pragma: no cover

    def sadd(self, name: str, *values: str) -> Awaitable[object]: ...  # pragma: no cover

    def srem(self, name: str, *values: str) -> Awaitable[object]: ...  # pragma: no cover

    def smembers(self, name: str, /) -> Awaitable[Iterable[str | bytes]]: ...  # pragma: no cover

    def zadd(self, name: str, mapping: Mapping[str, float], /) -> Awaitable[object]: ...  # pragma: no cover

    def zrange(self, name: str, start: int, end: int, /) -> Awaitable[Sequence[object]]:
        """Members by ascending score.

        `Sequence[object]` rather than of strings because a driver types this
        command for the `withscores=True` shape too. The store never asks for
        scores, and `_as_text` rejects a member that is not a string.
        """
        ...  # pragma: no cover

    def zrem(self, name: str, *values: str) -> Awaitable[object]: ...  # pragma: no cover

    def hset(self, name: str, key: str, value: str, /) -> Awaitable[object]: ...  # pragma: no cover

    def hget(self, name: str, key: str, /) -> Awaitable[object]: ...  # pragma: no cover

    def hgetall(self, name: str, /) -> Awaitable[Mapping[str | bytes, str | bytes]]: ...  # pragma: no cover


def _as_text(value: object) -> str:
    """Decode one reply, whichever encoding the client is configured for.

    `redis.asyncio.Redis` hands back `bytes` unless built with
    `decode_responses=True`, and both configurations have to work.
    """
    if isinstance(value, (bytes, bytearray)):
        return value.decode('utf-8')
    if isinstance(value, str):
        return value
    raise ValueError(f'expected a string reply, got {type(value).__name__}')


def _validate_expire_seconds(value: object) -> None:
    """Reject a TTL Redis would read as an immediate delete.

    `EXPIRE` treats zero or negative as "delete now", the opposite of what a
    caller asking for retention means, and a non-`int` would only fail at the
    first write. Mirrors `_validate_max_snapshots`.
    """
    if value is None:
        return
    if not isinstance(value, int) or value < 1:
        raise ValueError(f'expire_seconds must be an int >= 1 or None, got {value!r}')


def _parse_member(member: str) -> tuple[int, SnapshotState] | None:
    """Split a `<seq>:<state>` index member, or `None` when it is not one.

    A foreign member is skipped rather than raised on, the way `FileStepStore`
    skips a snapshot filename that is not an integer.
    """
    seq_text, separator, state_text = member.rpartition(':')
    if not separator:
        return None
    try:
        return int(seq_text), _snapshot_state(state_text)
    except ValueError:
        return None


class RedisStepStore:
    """Redis-backed store shared by every worker pointed at the same server.

    ```python
    from redis.asyncio import Redis

    from pydantic_ai_harness.step_persistence import RedisStepStore

    store = RedisStepStore(Redis.from_url('redis://localhost'))
    ```

    `prefix` namespaces every key, so one server can hold several deployments.

    `expire_seconds` (default `None`) gives a run's keys a TTL, refreshed on
    every write to that run. Each snapshot payload carries the TTL of its own
    write, so older snapshots expire ahead of the newest. The index sets hold
    other runs and never expire; `list_runs` drops a member whose run key is
    gone, so they self-heal rather than growing forever.

    `media_store` defaults to `None`, unlike the file, sqlite, and Mongo stores:
    payloads stay inline, because moving large binary or text parts into an
    in-memory database is a choice the caller should make deliberately. Pass any
    `MediaStore` to externalize parts at or above `media_threshold_bytes`.

    `max_snapshots_per_run` (default `None`, unbounded) prunes to the retain set
    after each write (see `_prune_snapshots`).

    A write spans several keys, so a crash mid-`save_snapshot` can leave a
    payload no index member points at, and under `expire_seconds` a payload can
    expire ahead of its index member. Reads skip an index member whose payload is
    gone, the tolerance `FileStepStore` has for a snapshot file that vanishes.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        prefix: str = _DEFAULT_PREFIX,
        expire_seconds: int | None = None,
        media_store: MediaStore | None = None,
        media_threshold_bytes: int = _DEFAULT_MEDIA_THRESHOLD_BYTES,
        max_snapshots_per_run: int | None = None,
    ) -> None:
        _validate_max_snapshots(max_snapshots_per_run)
        _validate_expire_seconds(expire_seconds)
        self._client = client
        self._prefix = prefix
        self._expire_seconds = expire_seconds
        self._media_store = media_store
        self._media_threshold_bytes = media_threshold_bytes
        self._max_snapshots_per_run = max_snapshots_per_run

    def _run_key(self, run_id: str) -> str:
        return f'{self._prefix}:run:{{{run_id}}}'

    @property
    def _runs_key(self) -> str:
        return f'{self._prefix}:runs'

    def _conversation_key(self, conversation_id: str) -> str:
        return f'{self._prefix}:runs:conversation:{conversation_id}'

    def _parent_key(self, parent_run_id: str) -> str:
        return f'{self._prefix}:runs:parent:{parent_run_id}'

    def _events_key(self, run_id: str) -> str:
        return f'{self._prefix}:events:{{{run_id}}}'

    def _seq_key(self, run_id: str) -> str:
        return f'{self._prefix}:snapshots:seq:{{{run_id}}}'

    def _snapshot_index_key(self, run_id: str) -> str:
        return f'{self._prefix}:snapshots:{{{run_id}}}'

    def _snapshot_key(self, run_id: str, seq: int) -> str:
        return f'{self._prefix}:snapshot:{{{run_id}}}:{seq}'

    def _tool_effects_key(self, run_id: str) -> str:
        return f'{self._prefix}:tool_effects:{{{run_id}}}'

    async def _touch(self, *names: str) -> None:
        """Refresh the expiry on the keys a write just touched."""
        if self._expire_seconds is None:
            return
        for name in names:
            await self._client.expire(name, self._expire_seconds)

    async def register_run(self, record: RunRecord) -> None:
        """Store lineage for a run, refusing a `run_id` that is already here.

        `SET ... NX` enforces the single-shot `run_id` contract that
        `SqliteStepStore` gets from its primary key. `StepPersistence.before_run`
        pre-checks with `get_run`; this covers a direct caller and the race
        between that check and the write.
        """
        created = await self._client.set(
            self._run_key(record.run_id),
            json.dumps(_run_to_dict(record)),
            ex=self._expire_seconds,
            nx=True,
        )
        if not created:
            raise ValueError(f'run_id {record.run_id!r} is already registered in this store')
        await self._client.sadd(self._runs_key, record.run_id)
        if record.conversation_id is not None:
            await self._client.sadd(self._conversation_key(record.conversation_id), record.run_id)
        if record.parent_run_id is not None:
            await self._client.sadd(self._parent_key(record.parent_run_id), record.run_id)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        raw = await self._client.get(self._run_key(run_id))
        if raw is None:
            return None
        return _run_from_dict(_load_json_object(_as_text(raw)))

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        """Return matching `RunRecord`s sorted by `started_at` ascending.

        Reads the narrowest index set that answers the query, so a filtered
        listing never walks every run; only when both filters are set does one
        still have to be applied in Python. Sorts on the parsed instant, not the
        stored ISO string, which would misorder mixed offsets.
        """
        if parent_run_id is not None:
            index_key = self._parent_key(parent_run_id)
        elif conversation_id is not None:
            index_key = self._conversation_key(conversation_id)
        else:
            index_key = self._runs_key
        records: list[RunRecord] = []
        for member in await self._client.smembers(index_key):
            run_id = _as_text(member)
            raw = await self._client.get(self._run_key(run_id))
            if raw is None:
                # The run key expired under `expire_seconds` while the index set,
                # which has no TTL, kept pointing at it. Drop the member so the
                # index does not accumulate ids that resolve to nothing.
                await self._client.srem(index_key, run_id)
                continue
            records.append(_run_from_dict(_load_json_object(_as_text(raw))))
        if parent_run_id is not None and conversation_id is not None:
            records = [record for record in records if record.conversation_id == conversation_id]
        return sorted(records, key=lambda record: record.started_at)

    async def append_event(self, event: StepEvent) -> None:
        await self._client.rpush(self._events_key(event.run_id), json.dumps(_event_to_dict(event)))
        await self._touch(self._events_key(event.run_id), self._run_key(event.run_id))

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        raw_events = await self._client.lrange(self._events_key(run_id), 0, -1)
        return [_event_from_dict(_load_json_object(_as_text(raw))) for raw in raw_events]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        """Write one snapshot under a freshly allocated `seq`, then prune.

        The payload lands before the index member, so an interrupted write
        leaves an unreferenced payload rather than an index member pointing at
        one that never arrived.
        """
        messages_json: object = json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages).decode('utf-8'))
        if self._media_store is not None:
            messages_json = await externalize_media(
                messages_json,
                media_store=self._media_store,
                threshold_bytes=self._media_threshold_bytes,
            )
        run_id = snapshot.run_id
        seq = await self._client.incr(self._seq_key(run_id))
        payload: dict[str, object] = {
            'run_id': run_id,
            'step_index': snapshot.step_index,
            'conversation_id': snapshot.conversation_id,
            'parent_run_id': snapshot.parent_run_id,
            'agent_name': snapshot.agent_name,
            'timestamp': snapshot.timestamp.isoformat(),
            'state': snapshot.state,
            'messages': messages_json,
        }
        await self._client.set(self._snapshot_key(run_id, seq), json.dumps(payload), ex=self._expire_seconds)
        await self._client.zadd(self._snapshot_index_key(run_id), {f'{seq}:{snapshot.state}': float(seq)})
        await self._touch(self._snapshot_index_key(run_id), self._seq_key(run_id), self._run_key(run_id))
        await self._prune_snapshots(run_id)

    async def _snapshot_entries(self, run_id: str) -> list[tuple[int, SnapshotState]]:
        """The run's `(seq, state)` pairs, ascending, from the index alone."""
        members = await self._client.zrange(self._snapshot_index_key(run_id), 0, -1)
        entries: list[tuple[int, SnapshotState]] = []
        for member in members:
            entry = _parse_member(_as_text(member))
            if entry is not None:
                entries.append(entry)
        return entries

    async def _prune_snapshots(self, run_id: str) -> None:
        """Drop this run's snapshots outside the retain set when bounded.

        `_retained_seqs` keeps the newest overall and the newest `complete` even
        when the newest `keep` are all `interrupted`, which is also why a run one
        over the bound can have nothing to drop. Externalized media is
        content-addressed and shared, so a dropped snapshot never deletes a blob.
        """
        if self._max_snapshots_per_run is None:
            return
        entries = await self._snapshot_entries(run_id)
        if len(entries) <= self._max_snapshots_per_run:
            return
        retained = _retained_seqs(entries, self._max_snapshots_per_run)
        dropped = [(seq, state) for seq, state in entries if seq not in retained]
        if not dropped:
            return
        await self._client.delete(*[self._snapshot_key(run_id, seq) for seq, _ in dropped])
        await self._client.zrem(self._snapshot_index_key(run_id), *[f'{seq}:{state}' for seq, state in dropped])

    async def _snapshot_from_payload(self, run_id: str, text: str) -> ContinuableSnapshot:
        data = _load_json_object(text)
        step_index = data['step_index']
        timestamp_raw = data['timestamp']
        if not (isinstance(step_index, int) and isinstance(timestamp_raw, str)):
            raise ValueError('snapshot payload has wrong types')
        messages_json = data['messages']
        if self._media_store is not None:
            messages_json = await restore_media(messages_json, media_store=self._media_store)
        messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(messages_json)
        return ContinuableSnapshot(
            run_id=run_id,
            step_index=step_index,
            messages=messages,
            conversation_id=_opt_str(data.get('conversation_id')),
            parent_run_id=_opt_str(data.get('parent_run_id')),
            agent_name=_opt_str(data.get('agent_name')),
            timestamp=datetime.fromisoformat(timestamp_raw),
            state=_snapshot_state(data.get('state')),
        )

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        """The newest snapshot passing the state gate, or `None`.

        Walks candidates newest-first, skipping one whose payload is gone (a
        raced prune, or a payload that expired ahead of the index). Retention
        keeps the newest overall and the newest `complete`, so skipping still
        reaches the right snapshot, as in `FileStepStore`.
        """
        entries = await self._snapshot_entries(run_id)
        candidates = sorted(
            (seq for seq, state in entries if include_interrupted or state == 'complete'),
            reverse=True,
        )
        for seq in candidates:
            raw = await self._client.get(self._snapshot_key(run_id, seq))
            if raw is None:
                continue
            return await self._snapshot_from_payload(run_id, _as_text(raw))
        return None

    async def list_snapshots(self, *, run_id: str, include_interrupted: bool = False) -> list[ContinuableSnapshot]:
        """Return retained snapshots for `run_id` in write order.

        Mirrors the `latest_snapshot` gate. A payload that fails to parse is
        skipped and logged, so one damaged value does not hide the rest of the
        run. Not part of the `StepStore` protocol: `conversation_search` consumes
        it through its narrower `SnapshotStore`.
        """
        snapshots: list[ContinuableSnapshot] = []
        for seq, state in await self._snapshot_entries(run_id):
            if state == 'interrupted' and not include_interrupted:
                continue
            raw = await self._client.get(self._snapshot_key(run_id, seq))
            if raw is None:
                continue
            try:
                snapshots.append(await self._snapshot_from_payload(run_id, _as_text(raw)))
            except Exception:
                _logger.warning('Skipping unparsable snapshot for run %s', run_id, exc_info=True)
        return snapshots

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        await self._client.hset(
            self._tool_effects_key(record.run_id),
            record.tool_call_id,
            json.dumps(_tool_effect_to_dict(record)),
        )
        await self._touch(self._tool_effects_key(record.run_id), self._run_key(record.run_id))

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        raw = await self._client.hget(self._tool_effects_key(run_id), tool_call_id)
        if raw is None:
            return None
        return _tool_effect_from_dict(_load_json_object(_as_text(raw)))

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        fields = await self._client.hgetall(self._tool_effects_key(run_id))
        records = [_tool_effect_from_dict(_load_json_object(_as_text(raw))) for raw in fields.values()]
        return [record for record in records if record.status == 'started']
