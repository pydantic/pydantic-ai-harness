"""Tests for scheduling CRUD tools through the public capability surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.scheduling import (
    InMemoryScheduleStore,
    IntervalTrigger,
    OnceTrigger,
    Schedule,
    ScheduleConflictError,
    Scheduling,
)

pytestmark = pytest.mark.anyio


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def _call(capability: Scheduling[None], name: str, arguments: dict[str, object]) -> str:
    toolset = capability.get_toolset()
    ctx = _ctx()
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool(name, arguments, ctx, tools[name])
    assert isinstance(result, str)
    return result


async def _add(capability: Scheduling[None], schedule: Schedule) -> None:
    await capability.resolved_store.add(schedule)


class TestSchedulingTools:
    async def test_create_list_get_update_delete(self) -> None:
        capability = Scheduling[None]()
        assert await _call(capability, 'list_schedules', {}) == 'No schedules.'
        created = await _call(
            capability,
            'create_schedule',
            {'name': 'Review', 'prompt': 'Review changes', 'schedule': 'every 2h', 'deliver_to': 'ops'},
        )
        schedule_id = (await capability.resolved_store.list())[0].id
        assert schedule_id in created
        listing = await _call(capability, 'list_schedules', {})
        assert 'Review' in listing
        assert 'Review changes' not in listing
        details = await _call(capability, 'get_schedule', {'schedule_id': schedule_id})
        assert 'prompt: Review changes' in details
        assert 'version:' not in details
        updated = await _call(
            capability,
            'update_schedule',
            {'schedule_id': schedule_id, 'name': 'Review PR', 'prompt': 'Review it', 'max_runs': 2},
        )
        assert 'Review PR' in updated
        stored = await capability.resolved_store.get(schedule_id)
        assert stored is not None
        assert stored.prompt == 'Review it'
        assert stored.max_runs == 2
        deleted = await _call(capability, 'delete_schedule', {'schedule_id': schedule_id})
        assert 'deleted' in deleted
        assert await capability.resolved_store.list() == []

    async def test_past_once_and_once_max_runs_are_retries(self) -> None:
        capability = Scheduling[None]()
        with pytest.raises(ModelRetry, match='time is in the past'):
            await _call(
                capability,
                'create_schedule',
                {'name': 'Past', 'prompt': 'work', 'schedule': '2020-01-01T00:00:00Z'},
            )
        with pytest.raises(ModelRetry, match='only valid for recurring'):
            await _call(
                capability,
                'create_schedule',
                {'name': 'Once', 'prompt': 'work', 'schedule': 'in 1h', 'max_runs': 2},
            )
        assert await capability.resolved_store.list() == []

    async def test_parse_and_model_validation_are_retries(self) -> None:
        capability = Scheduling[None]()
        with pytest.raises(ModelRetry, match='Invalid schedule'):
            await _call(
                capability,
                'create_schedule',
                {'name': 'Bad', 'prompt': 'work', 'schedule': 'whenever', 'max_runs': 0},
            )
        with pytest.raises(ModelRetry, match='Invalid schedule'):
            await _call(
                capability,
                'create_schedule',
                {'name': 'Bad limit', 'prompt': 'work', 'schedule': 'every 1h', 'max_runs': 0},
            )
        with pytest.raises(ModelRetry, match='Invalid schedule: Duration is too large'):
            await _call(
                capability,
                'create_schedule',
                {'name': 'Huge interval', 'prompt': 'work', 'schedule': 'every 1000000000d'},
            )
        await _add(capability, _recurring('invalid-update'))
        with pytest.raises(ModelRetry, match='Invalid schedule update'):
            await _call(capability, 'update_schedule', {'schedule_id': 'invalid-update', 'max_runs': -1})

    async def test_not_found_lists_known_ids(self) -> None:
        capability = Scheduling[None]()
        await _add(capability, _recurring('known'))
        with pytest.raises(ModelRetry, match='Known ids: known'):
            await _call(capability, 'get_schedule', {'schedule_id': 'missing'})

    async def test_trigger_update_recomputes_and_resets_runs(self) -> None:
        capability = Scheduling[None]()
        schedule = _recurring('change')
        schedule.runs_completed = 4
        await _add(capability, schedule)
        await _call(
            capability,
            'update_schedule',
            {'schedule_id': 'change', 'schedule': 'every 2h'},
        )
        stored = await capability.resolved_store.get('change')
        assert stored is not None
        assert stored.runs_completed == 0
        assert stored.trigger == IntervalTrigger(every=timedelta(hours=2))
        assert stored.next_run_at is not None
        assert stored.next_run_at > datetime.now(timezone.utc)

    async def test_render_variants_and_optional_update_fields(self) -> None:
        capability = Scheduling[None]()
        day = _recurring('day')
        day.trigger = IntervalTrigger(every=timedelta(days=1))
        day.last_status = 'error'
        day.last_error = 'RuntimeError: old failure'
        day.last_delivery_error = 'RuntimeError: delivery failure'
        minute = _recurring('minute')
        minute.trigger = IntervalTrigger(every=timedelta(minutes=30))
        once = Schedule(
            id='once',
            name='once',
            prompt='one shot',
            trigger=OnceTrigger(at=datetime(2030, 1, 1, tzinfo=timezone.utc)),
            next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
            last_output='previous output',
        )
        for item in (day, minute, once):
            await _add(capability, item)
        listing = await _call(capability, 'list_schedules', {})
        assert 'schedule: every 1d' in listing
        assert 'schedule: every 30m' in listing
        assert 'schedule: 2030-01-01T00:00:00+00:00' in listing
        assert 'last status: error' in listing
        assert 'last error: RuntimeError: old failure' in listing
        assert 'last delivery error: RuntimeError: delivery failure' in listing
        details = await _call(capability, 'get_schedule', {'schedule_id': 'once'})
        assert 'last output: previous output' in details
        await _call(
            capability,
            'update_schedule',
            {'schedule_id': 'minute', 'deliver_to': 'alerts'},
        )
        updated = await capability.resolved_store.get('minute')
        assert updated is not None
        assert updated.deliver_to == 'alerts'

    async def test_update_rejects_a_past_once(self) -> None:
        capability = Scheduling[None]()
        await _add(capability, _recurring('past-update'))
        with pytest.raises(ModelRetry, match='time is in the past'):
            await _call(
                capability,
                'update_schedule',
                {'schedule_id': 'past-update', 'schedule': '2020-01-01T00:00:00Z'},
            )

    async def test_same_trigger_does_not_reset_runs(self) -> None:
        capability = Scheduling[None]()
        schedule = _recurring('same-trigger')
        schedule.runs_completed = 3
        await _add(capability, schedule)
        await _call(
            capability,
            'update_schedule',
            {'schedule_id': 'same-trigger', 'schedule': 'every 1h'},
        )
        stored = await capability.resolved_store.get('same-trigger')
        assert stored is not None
        assert stored.runs_completed == 3

    async def test_update_clears_fields_and_drops_inherited_limit_for_once(self) -> None:
        capability = Scheduling[None]()
        schedule = _recurring('limited')
        schedule.max_runs = 3
        schedule.deliver_to = 'alerts'
        await _add(capability, schedule)

        await _call(
            capability,
            'update_schedule',
            {'schedule_id': 'limited', 'deliver_to': '', 'max_runs': 0},
        )
        cleared = await capability.resolved_store.get('limited')
        assert cleared is not None
        assert cleared.deliver_to is None
        assert cleared.max_runs is None

        cleared.max_runs = 3
        await capability.resolved_store.save(cleared)
        await _call(capability, 'update_schedule', {'schedule_id': 'limited', 'schedule': 'in 1h'})
        converted = await capability.resolved_store.get('limited')
        assert converted is not None
        assert isinstance(converted.trigger, OnceTrigger)
        assert converted.max_runs is None

    async def test_update_to_once_rejects_explicit_max_runs(self) -> None:
        capability = Scheduling[None]()
        await _add(capability, _recurring('limited'))
        with pytest.raises(ModelRetry, match='only valid for recurring'):
            await _call(
                capability,
                'update_schedule',
                {'schedule_id': 'limited', 'schedule': 'in 1h', 'max_runs': 2},
            )

    async def test_pause_retries_on_a_concurrent_claim(self) -> None:
        store = _ClaimAfterReadStore()
        capability = Scheduling[None](store=store)
        original = _recurring('racing')
        original.next_run_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        await _add(capability, original)

        await _call(capability, 'pause_schedule', {'schedule_id': 'racing'})

        stored = await store.get('racing')
        assert stored is not None
        assert stored.enabled is False
        assert stored.runs_completed == 1
        assert stored.next_run_at == datetime(2026, 1, 1, 1, tzinfo=timezone.utc)

    async def test_conflict_retry_exhaustion_is_a_model_retry(self) -> None:
        store = _AlwaysConflictStore()
        capability = Scheduling[None](store=store)
        await _add(capability, _recurring('contended'))

        with pytest.raises(ModelRetry, match='changed while this update was being applied'):
            await _call(capability, 'pause_schedule', {'schedule_id': 'contended'})
        assert store.save_attempts == 3

    @pytest.mark.parametrize(
        ('tool_name', 'arguments', 'paused'),
        [
            ('update_schedule', {'name': 'updated'}, False),
            ('resume_schedule', {}, True),
            ('run_schedule_now', {}, False),
        ],
    )
    async def test_mutating_tools_retry_a_conflict(
        self, tool_name: str, arguments: dict[str, object], paused: bool
    ) -> None:
        store = _ConflictOnceStore()
        capability = Scheduling[None](store=store)
        schedule = _recurring(tool_name)
        schedule.enabled = not paused
        await _add(capability, schedule)
        arguments['schedule_id'] = tool_name

        await _call(capability, tool_name, arguments)

        assert store.save_attempts == 2

    async def test_pause_resume_recomputes_and_run_now_queues(self) -> None:
        capability = Scheduling[None]()
        schedule = _recurring('queue')
        schedule.next_run_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        await _add(capability, schedule)
        assert 'paused' in await _call(capability, 'pause_schedule', {'schedule_id': 'queue'})
        with pytest.raises(ModelRetry, match='Resume it'):
            await _call(capability, 'run_schedule_now', {'schedule_id': 'queue'})
        assert 'resumed' in await _call(capability, 'resume_schedule', {'schedule_id': 'queue'})
        resumed = await capability.resolved_store.get('queue')
        assert resumed is not None
        assert resumed.next_run_at is not None
        assert resumed.next_run_at > datetime.now(timezone.utc)
        assert 'next runner tick' in await _call(capability, 'run_schedule_now', {'schedule_id': 'queue'})
        queued = await capability.resolved_store.get('queue')
        assert queued is not None
        assert queued.next_run_at is not None
        assert queued.next_run_at <= datetime.now(timezone.utc)

    async def test_completed_schedules_are_terminal(self) -> None:
        capability = Scheduling[None]()
        completed = _recurring('completed')
        completed.next_run_at = None
        paused_completed = _recurring('paused-completed')
        paused_completed.enabled = False
        paused_completed.next_run_at = None
        await _add(capability, completed)
        await _add(capability, paused_completed)

        with pytest.raises(ModelRetry, match='completed.*Create a new schedule'):
            await _call(capability, 'run_schedule_now', {'schedule_id': 'completed'})
        with pytest.raises(ModelRetry, match='completed.*Create a new schedule'):
            await _call(
                capability,
                'update_schedule',
                {'schedule_id': 'completed', 'schedule': 'every 2h'},
            )
        await _call(capability, 'update_schedule', {'schedule_id': 'completed', 'name': 'renamed'})
        renamed = await capability.resolved_store.get('completed')
        assert renamed is not None
        assert renamed.name == 'renamed'
        assert renamed.next_run_at is None
        with pytest.raises(ModelRetry, match='not paused'):
            await _call(capability, 'resume_schedule', {'schedule_id': 'completed'})

        await _call(capability, 'resume_schedule', {'schedule_id': 'paused-completed'})
        resumed = await capability.resolved_store.get('paused-completed')
        assert resumed is not None
        assert resumed.enabled is True
        assert resumed.next_run_at is None

    async def test_lowering_max_runs_below_attempt_count_completes_schedule(self) -> None:
        capability = Scheduling[None]()
        schedule = _recurring('lower-limit')
        schedule.runs_completed = 3
        await _add(capability, schedule)

        await _call(capability, 'update_schedule', {'schedule_id': 'lower-limit', 'max_runs': 2})
        updated = await capability.resolved_store.get('lower-limit')
        assert updated is not None
        assert updated.max_runs == 2
        assert updated.next_run_at is None

    async def test_resume_of_an_expired_once_is_a_retry(self) -> None:
        capability = Scheduling[None]()
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        expired = Schedule(
            id='expired-once',
            name='expired-once',
            prompt='work',
            trigger=OnceTrigger(at=past),
            enabled=False,
            next_run_at=past,
        )
        await _add(capability, expired)

        with pytest.raises(ModelRetry, match='expired while paused'):
            await _call(capability, 'resume_schedule', {'schedule_id': 'expired-once'})

        stored = await capability.resolved_store.get('expired-once')
        assert stored is not None
        assert stored.enabled is False
        assert stored.next_run_at == past

    async def test_create_retries_a_colliding_generated_id(self) -> None:
        store = _DuplicateIdOnceStore()
        capability = Scheduling[None](store=store)

        created = await _call(capability, 'create_schedule', {'name': 'n', 'prompt': 'p', 'schedule': 'every 2h'})

        assert created.startswith('Schedule created.')
        assert len(store.attempted_ids) == 2
        assert store.attempted_ids[0] != store.attempted_ids[1]

    async def test_create_id_collision_exhaustion_is_a_model_retry(self) -> None:
        store = _AlwaysDuplicateIdStore()
        capability = Scheduling[None](store=store)

        with pytest.raises(ModelRetry, match='fresh id'):
            await _call(capability, 'create_schedule', {'name': 'n', 'prompt': 'p', 'schedule': 'every 2h'})
        assert len(store.attempted_ids) == 3


def _recurring(schedule_id: str) -> Schedule:
    return Schedule(
        id=schedule_id,
        name=schedule_id,
        prompt='work',
        trigger=IntervalTrigger(every=timedelta(hours=1)),
        next_run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class _DuplicateIdOnceStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self.attempted_ids: list[str] = []

    async def add(self, schedule: Schedule) -> Schedule:
        self.attempted_ids.append(schedule.id)
        if len(self.attempted_ids) == 1:
            raise ValueError(f'A schedule with id {schedule.id!r} already exists.')
        return await super().add(schedule)


class _AlwaysDuplicateIdStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self.attempted_ids: list[str] = []

    async def add(self, schedule: Schedule) -> Schedule:
        self.attempted_ids.append(schedule.id)
        raise ValueError(f'A schedule with id {schedule.id!r} already exists.')


class _ClaimAfterReadStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self._injected = False

    async def get(self, schedule_id: str) -> Schedule | None:
        read = await super().get(schedule_id)
        if read is not None and not self._injected:
            self._injected = True
            claimed = await super().get(schedule_id)
            assert claimed is not None
            claimed.runs_completed += 1
            claimed.next_run_at = claimed.next_run_at + timedelta(hours=1) if claimed.next_run_at is not None else None
            await super().save(claimed)
        return read


class _AlwaysConflictStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self.save_attempts = 0

    async def save(self, schedule: Schedule) -> None:
        del schedule
        self.save_attempts += 1
        raise ScheduleConflictError('forced conflict')


class _ConflictOnceStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self.save_attempts = 0

    async def save(self, schedule: Schedule) -> None:
        self.save_attempts += 1
        if self.save_attempts == 1:
            raise ScheduleConflictError('forced conflict')
        await super().save(schedule)


class TestToolsetIdentity:
    async def test_id_and_registered_tools(self) -> None:
        capability = Scheduling[None](store=InMemoryScheduleStore())
        toolset = capability.get_toolset()
        assert toolset.id == 'scheduling'
        assert set(await toolset.get_tools(_ctx())) == {
            'create_schedule',
            'list_schedules',
            'get_schedule',
            'update_schedule',
            'pause_schedule',
            'resume_schedule',
            'delete_schedule',
            'run_schedule_now',
        }
