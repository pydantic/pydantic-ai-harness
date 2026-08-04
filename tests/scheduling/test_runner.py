"""Tests for scheduled execution semantics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.scheduling import (
    InMemoryScheduleStore,
    IntervalTrigger,
    OnceTrigger,
    Schedule,
    ScheduleConflictError,
    ScheduleResult,
    ScheduleResultCallback,
    ScheduleRunner,
    Scheduling,
)

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 5, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def anyio_backend() -> str:
    """Run agent integration tests on the backend used by Pydantic AI's graph."""
    return 'asyncio'


def _schedule(
    schedule_id: str,
    *,
    trigger: IntervalTrigger | OnceTrigger | None = None,
    due_at: datetime = NOW,
    max_runs: int | None = None,
) -> Schedule:
    return Schedule(
        id=schedule_id,
        name=schedule_id,
        prompt=f'run {schedule_id}',
        trigger=trigger or IntervalTrigger(every=timedelta(hours=1)),
        next_run_at=due_at,
        max_runs=max_runs,
    )


class TestScheduleRunnerExecution:
    async def test_success_advances_before_run_and_records_usage(self) -> None:
        store = InMemoryScheduleStore()
        await store.add(_schedule('success'))
        results = await ScheduleRunner(
            Agent(TestModel(custom_output_text='done')),
            deps=None,
            store=store,
        ).tick(NOW)
        assert len(results) == 1
        assert results[0].status == 'success'
        assert results[0].output == 'done'
        assert results[0].usage is not None
        stored = await store.get('success')
        assert stored is not None
        assert stored.runs_completed == 1
        assert stored.next_run_at == NOW + timedelta(hours=1)
        assert stored.last_run_at == NOW
        assert stored.last_status == 'success'
        assert stored.last_output == 'done'
        assert stored.last_error is None

    async def test_empty_output_is_success(self) -> None:
        def empty_output() -> str:
            return ''

        store = InMemoryScheduleStore()
        await store.add(_schedule('empty'))
        result = (await ScheduleRunner(Agent(TestModel(), output_type=empty_output), deps=None, store=store).tick(NOW))[
            0
        ]
        assert result.status == 'success'
        assert result.output == ''

    async def test_max_runs_exhaustion_completes(self) -> None:
        store = InMemoryScheduleStore()
        await store.add(_schedule('limited', max_runs=1))
        result = (await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(NOW))[0]
        assert result.schedule.runs_completed == 1
        assert result.schedule.next_run_at is None
        assert result.schedule.status == 'completed'

    async def test_future_schedule_is_not_claimed(self) -> None:
        store = InMemoryScheduleStore()
        await store.add(_schedule('future', due_at=NOW + timedelta(hours=1)))
        assert await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(NOW) == []

    async def test_tick_rejects_naive_now(self) -> None:
        runner = ScheduleRunner(Agent(TestModel()), deps=None, store=InMemoryScheduleStore())
        with pytest.raises(ValueError, match='now must be timezone-aware'):
            await runner.tick(datetime(2026, 5, 1, 12))

    async def test_error_records_and_recurring_schedule_continues(self) -> None:
        async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            raise RuntimeError('model broke')

        store = InMemoryScheduleStore()
        await store.add(_schedule('failure'))
        result = (await ScheduleRunner(Agent(FunctionModel(fail)), deps=None, store=store).tick(NOW))[0]
        assert result.status == 'error'
        assert result.error == 'RuntimeError: model broke'
        assert result.schedule.next_run_at == NOW + timedelta(hours=1)
        assert result.schedule.last_output is None

    async def test_schedule_usage_limits_cap_the_run(self) -> None:
        store = InMemoryScheduleStore()
        schedule = _schedule('budget')
        schedule.usage_limits = UsageLimits(request_limit=0)
        await store.add(schedule)
        result = (await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(NOW))[0]
        assert result.status == 'error'
        assert result.error is not None
        assert result.error.startswith('UsageLimitExceeded:')
        stored = await store.get('budget')
        assert stored is not None
        assert stored.next_run_at == NOW + timedelta(hours=1)

    async def test_timeout_is_recorded_without_retry(self) -> None:
        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            await anyio.sleep(1)
            return ModelResponse(parts=[TextPart('late')])  # pragma: no cover - cancelled by run_timeout first

        store = InMemoryScheduleStore()
        await store.add(_schedule('timeout'))
        result = (await ScheduleRunner(Agent(FunctionModel(slow)), deps=None, store=store, run_timeout=0.01).tick(NOW))[
            0
        ]
        assert result.status == 'error'
        assert result.error is not None
        assert result.error.startswith('TimeoutError:')

    async def test_invalid_runner_configuration(self) -> None:
        agent = Agent(TestModel())
        store = InMemoryScheduleStore()
        with pytest.raises(ValueError, match='tick_interval'):
            ScheduleRunner(agent, deps=None, store=store, tick_interval=0)
        with pytest.raises(ValueError, match='misfire_grace'):
            ScheduleRunner(agent, deps=None, store=store, misfire_grace=timedelta(seconds=-1))
        with pytest.raises(ValueError, match='run_timeout'):
            ScheduleRunner(agent, deps=None, store=store, run_timeout=0)

    async def test_partial_claim_failure_drains_started_occurrences_before_raising(self) -> None:
        store = _FailSecondScheduleStore()
        await store.add(_schedule('first'))
        await store.add(_schedule('second'))
        runner = ScheduleRunner(Agent(TestModel()), deps=None, store=store)
        with pytest.raises(RuntimeError, match='save failed'):
            await runner.tick(NOW)
        first = await store.get('first')
        second = await store.get('second')
        assert first is not None
        assert second is not None
        assert first.runs_completed == 1
        assert first.last_status == 'success'
        assert second.runs_completed == 0

        results = await runner.tick(NOW)
        assert [result.schedule.id for result in results] == ['second']
        assert results[0].status == 'success'

    async def test_queued_exhausted_schedule_completes_without_agent_call(self) -> None:
        calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
            nonlocal calls
            del messages, info
            calls += 1
            return ModelResponse(parts=[TextPart('unexpected')])

        store = InMemoryScheduleStore()
        schedule = _schedule('exhausted', max_runs=2)
        schedule.runs_completed = 2
        await store.add(schedule)
        assert await ScheduleRunner(Agent(FunctionModel(model)), deps=None, store=store).tick(NOW) == []
        assert calls == 0
        stored = await store.get('exhausted')
        assert stored is not None
        assert stored.next_run_at is None


class TestMisfires:
    async def test_recurring_misfire_runs_once_and_fast_forwards(self) -> None:
        store = InMemoryScheduleStore()
        await store.add(_schedule('catch-up', due_at=NOW - timedelta(days=3)))
        result = (
            await ScheduleRunner(
                Agent(TestModel()),
                deps=None,
                store=store,
                misfire_grace=timedelta(minutes=10),
            ).tick(NOW)
        )[0]
        assert result.status == 'success'
        assert result.schedule.runs_completed == 1
        assert result.schedule.next_run_at == NOW + timedelta(hours=1)

    async def test_once_misfire_is_missed_without_agent_call(self) -> None:
        calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
            # Never reached when the missed path is correct; `calls` catches a regression.
            nonlocal calls
            del messages, info
            calls += 1
            return ModelResponse(parts=[TextPart('unexpected')])

        store = InMemoryScheduleStore()
        due = NOW - timedelta(hours=1)
        await store.add(_schedule('missed', trigger=OnceTrigger(at=due), due_at=due))
        result = (
            await ScheduleRunner(
                Agent(FunctionModel(model)),
                deps=None,
                store=store,
                misfire_grace=timedelta(minutes=5),
            ).tick(NOW)
        )[0]
        assert calls == 0
        assert result.status == 'missed'
        assert result.schedule.next_run_at is None
        assert result.schedule.runs_completed == 0
        assert result.error is not None
        assert 'runner was not running' in result.error


class TestCallbacks:
    async def test_sync_and_async_callbacks(self) -> None:
        sync_seen: list[str] = []
        async_seen: list[str] = []

        def sync_callback(result: ScheduleResult) -> None:
            sync_seen.append(result.status)

        sync_store = InMemoryScheduleStore()
        await sync_store.add(_schedule('callback-sync'))
        await ScheduleRunner(Agent(TestModel()), deps=None, store=sync_store, on_result=sync_callback).tick(NOW)
        assert sync_seen == ['success']

        async_store = InMemoryScheduleStore()
        await async_store.add(_schedule('callback-async'))
        await ScheduleRunner(
            Agent(TestModel()), deps=None, store=async_store, on_result=_async_callback(async_seen)
        ).tick(NOW)
        assert async_seen == ['success']

    async def test_callback_failure_is_delivery_error_not_run_error(self) -> None:
        def fail(result: ScheduleResult) -> None:
            del result
            raise RuntimeError('delivery broke')

        store = InMemoryScheduleStore()
        await store.add(_schedule('delivery'))
        result = (await ScheduleRunner(Agent(TestModel()), deps=None, store=store, on_result=fail).tick(NOW))[0]
        assert result.status == 'success'
        assert result.schedule.last_delivery_error == 'RuntimeError: delivery broke'
        stored = await store.get('delivery')
        assert stored is not None
        assert stored.last_status == 'success'

    async def test_successful_delivery_clears_previous_error(self) -> None:
        store = InMemoryScheduleStore()
        schedule = _schedule('clear-delivery')
        schedule.last_delivery_error = 'old error'
        await store.add(schedule)
        result = (
            await ScheduleRunner(Agent(TestModel()), deps=None, store=store, on_result=lambda result: None).tick(NOW)
        )[0]
        assert result.schedule.last_delivery_error is None


def _async_callback(seen: list[str]) -> ScheduleResultCallback:
    async def callback(result: ScheduleResult) -> None:
        seen.append(result.status)

    return callback


class _FailSecondScheduleStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    async def save(self, schedule: Schedule) -> None:
        if schedule.id == 'second' and not self._failed:
            self._failed = True
            raise RuntimeError('save failed')
        await super().save(schedule)


class _PauseOnClaimStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self._injected = False

    async def save(self, schedule: Schedule) -> None:
        if schedule.runs_completed == 1 and not self._injected:  # pragma: no branch
            self._injected = True
            current = await super().get(schedule.id)
            assert current is not None
            current.enabled = False
            await super().save(current)
        await super().save(schedule)


class _DeleteOnClaimConflictStore(InMemoryScheduleStore):
    async def save(self, schedule: Schedule) -> None:
        if schedule.runs_completed == 1:  # pragma: no branch
            await self.remove(schedule.id)
            raise ScheduleConflictError('deleted during claim')
        await super().save(schedule)  # pragma: no cover - only claim saves use this test store


class _AlwaysConflictOnClaimStore(InMemoryScheduleStore):
    async def save(self, schedule: Schedule) -> None:
        del schedule
        raise ScheduleConflictError('forced claim conflict')


class _EditOnOutcomeStore(InMemoryScheduleStore):
    def __init__(self) -> None:
        super().__init__()
        self._injected = False

    async def save(self, schedule: Schedule) -> None:
        if schedule.last_status == 'success' and not self._injected:
            self._injected = True
            current = await super().get(schedule.id)
            assert current is not None
            current.name = 'concurrent edit'
            await super().save(current)
        await super().save(schedule)


class _AlwaysConflictOnOutcomeStore(InMemoryScheduleStore):
    async def save(self, schedule: Schedule) -> None:
        if schedule.last_status is not None:
            raise ScheduleConflictError('forced outcome conflict')
        await super().save(schedule)


class TestConcurrencyAndLoop:
    async def test_claim_conflict_rechecks_current_state(self) -> None:
        store = _PauseOnClaimStore()
        await store.add(_schedule('paused-before-claim'))

        results = await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(NOW)

        assert results == []
        stored = await store.get('paused-before-claim')
        assert stored is not None
        assert stored.enabled is False
        assert stored.runs_completed == 0
        assert stored.next_run_at == NOW

    async def test_claim_conflict_skips_a_deleted_schedule(self) -> None:
        store = _DeleteOnClaimConflictStore()
        await store.add(_schedule('deleted-before-retry'))

        assert await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(NOW) == []
        assert await store.get('deleted-before-retry') is None

    async def test_claim_conflict_exhaustion_propagates(self) -> None:
        store = _AlwaysConflictOnClaimStore()
        await store.add(_schedule('contended-claim'))

        with pytest.raises(ScheduleConflictError, match='forced claim conflict'):
            await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(NOW)

    async def test_outcome_conflict_preserves_concurrent_edit(self) -> None:
        store = _EditOnOutcomeStore()
        await store.add(_schedule('edited-during-outcome'))

        result = (await ScheduleRunner(Agent(TestModel(custom_output_text='done')), deps=None, store=store).tick(NOW))[
            0
        ]

        assert result.status == 'success'
        assert result.schedule.name == 'concurrent edit'
        stored = await store.get('edited-during-outcome')
        assert stored is not None
        assert stored.name == 'concurrent edit'
        assert stored.last_status == 'success'
        assert stored.last_output == 'done'

    async def test_outcome_conflict_exhaustion_propagates(self) -> None:
        store = _AlwaysConflictOnOutcomeStore()
        await store.add(_schedule('contended-outcome'))

        with pytest.RaisesGroup(pytest.RaisesExc(ScheduleConflictError, match='forced outcome conflict')):
            await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(NOW)

    async def test_overlap_guard_skips_same_schedule(self) -> None:
        started = anyio.Event()
        release = anyio.Event()

        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart('done')])

        store = InMemoryScheduleStore()
        await store.add(_schedule('overlap'))
        first_results: list[ScheduleResult] = []

        async def first_tick() -> None:
            first_results.extend(await runner.tick(NOW))

        runner = ScheduleRunner(Agent(FunctionModel(slow)), deps=None, store=store)
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(first_tick)
            await started.wait()
            running = await store.get('overlap')
            assert running is not None
            running.next_run_at = NOW
            await store.save(running)
            assert await runner.tick(NOW) == []
            skipped = await store.get('overlap')
            assert skipped is not None
            assert skipped.next_run_at == NOW + timedelta(hours=1)
            release.set()
        assert len(first_results) == 1
        stored = await store.get('overlap')
        assert stored is not None
        assert stored.next_run_at == NOW + timedelta(hours=1)
        assert stored.last_status == 'success'

    async def test_concurrent_pause_survives_run_outcome(self) -> None:
        started = anyio.Event()
        release = anyio.Event()

        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart('done')])

        store = InMemoryScheduleStore()
        await store.add(_schedule('paused-during-run'))
        results: list[ScheduleResult] = []
        runner = ScheduleRunner(Agent(FunctionModel(slow)), deps=None, store=store)

        async def run_tick() -> None:
            results.extend(await runner.tick(NOW))

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_tick)
            await started.wait()
            running = await store.get('paused-during-run')
            assert running is not None
            running.enabled = False
            await store.save(running)
            release.set()

        assert results[0].status == 'success'
        stored = await store.get('paused-during-run')
        assert stored is not None
        assert stored.enabled is False
        assert stored.last_run_at == NOW
        assert stored.last_status == 'success'
        assert stored.last_output == 'done'

    async def test_deletion_during_run_stays_deleted_and_delivers_result(self) -> None:
        started = anyio.Event()
        release = anyio.Event()
        delivered: list[str] = []

        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart('done')])

        store = InMemoryScheduleStore()
        await store.add(_schedule('deleted-during-run'))
        results: list[ScheduleResult] = []
        runner = ScheduleRunner(
            Agent(FunctionModel(slow)),
            deps=None,
            store=store,
            on_result=lambda result: delivered.append(result.status),
        )

        async def run_tick() -> None:
            results.extend(await runner.tick(NOW))

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_tick)
            await started.wait()
            assert await store.remove('deleted-during-run') is True
            release.set()

        assert results[0].status == 'success'
        assert delivered == ['success']
        assert await store.list() == []

    async def test_run_until_stopped_can_stop_from_callback(self) -> None:
        store = InMemoryScheduleStore()
        await store.add(
            _schedule(
                'loop',
                trigger=OnceTrigger(at=datetime.now(timezone.utc) + timedelta(seconds=1)),
                due_at=datetime.now(timezone.utc),
            )
        )
        runner: ScheduleRunner[None]

        def stop(result: ScheduleResult) -> None:
            assert result.status == 'success'
            runner.stop()
            runner.stop()

        runner = ScheduleRunner(
            Agent(TestModel()),
            deps=None,
            store=store,
            on_result=stop,
            tick_interval=0.01,
        )
        with anyio.fail_after(1):
            await runner.run_until_stopped()

    async def test_run_until_stopped_drains_started_occurrences_before_raising(self) -> None:
        store = _FailSecondScheduleStore()
        await store.add(_schedule('first'))
        await store.add(_schedule('second'))
        delivered: list[str] = []
        runner = ScheduleRunner(
            Agent(TestModel()),
            deps=None,
            store=store,
            on_result=lambda result: delivered.append(result.schedule.id),
            tick_interval=0.01,
        )

        with pytest.raises(RuntimeError, match='save failed'):
            await runner.run_until_stopped()
        assert delivered == ['first']

    async def test_run_until_stopped_skips_future_schedules(self) -> None:
        store = InMemoryScheduleStore()
        await store.add(_schedule('future', due_at=datetime.now(timezone.utc) + timedelta(hours=1)))
        runner = ScheduleRunner(Agent(TestModel()), deps=None, store=store, tick_interval=0.01)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(runner.run_until_stopped)
            await anyio.sleep(0.02)
            runner.stop()


class TestScheduledRunRecursionGuard:
    async def test_scheduled_run_has_no_scheduling_tools_or_instructions(self) -> None:
        captured: list[tuple[list[str], str | None]] = []

        def inspect_request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages
            captured.append(([tool.name for tool in info.function_tools], info.instructions))
            return ModelResponse(parts=[TextPart('done')])

        capability = Scheduling[None]()
        await capability.resolved_store.add(_schedule('guard'))
        agent: Agent[None, str] = Agent(FunctionModel(inspect_request), deps_type=type(None), capabilities=[capability])
        results = await ScheduleRunner(agent, deps=None).tick(NOW)
        assert results[0].status == 'success'
        assert captured == [([], None)]

    async def test_ordinary_run_keeps_scheduling_tools_and_instructions(self) -> None:
        captured: list[tuple[list[str], str | None]] = []

        def inspect_request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages
            captured.append(([tool.name for tool in info.function_tools], info.instructions))
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(inspect_request), deps_type=type(None), capabilities=[Scheduling[None]()]
        )
        await agent.run('hello')
        assert 'create_schedule' in captured[0][0]
        assert captured[0][1] is not None
        assert 'ScheduleRunner' in captured[0][1]
