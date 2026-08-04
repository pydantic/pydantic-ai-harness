"""In-process execution loop for due schedules."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Generic, Literal

import anyio
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.scheduling._store import ScheduleConflictError, ScheduleStore
from pydantic_ai_harness.scheduling._types import (
    CronTrigger,
    IntervalTrigger,
    OnceTrigger,
    Schedule,
    ScheduleResult,
    ScheduleResultCallback,
    next_run_time,
)

scheduled_run_var: ContextVar[str | None] = ContextVar('scheduled_run', default=None)
"""Id of the schedule running in the current context, or `None`."""

_SAVE_ATTEMPTS = 3


@dataclass(frozen=True)
class _Claim:
    schedule: Schedule
    execute: bool
    claimed_at: datetime


class ScheduleRunner(Generic[AgentDepsT]):
    """Execute due schedules against an agent.

    A runner assumes no other runner uses its store. Occurrences are advanced
    before agent execution for at-most-once behavior, and `stop()` drains runs
    already in flight before `run_until_stopped()` returns.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, Any],
        *,
        deps: AgentDepsT,
        store: ScheduleStore | None = None,
        on_result: ScheduleResultCallback | None = None,
        tick_interval: float = 60.0,
        misfire_grace: timedelta = timedelta(minutes=10),
        run_timeout: float | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> None:
        """Initialize a schedule runner.

        Store resolution falls back to the agent's `Scheduling` capability. `misfire_grace`
        bounds how late an occurrence may fire, and `usage_limits` is the default when a
        schedule has none.
        """
        if tick_interval <= 0:
            raise ValueError('tick_interval must be greater than zero')
        if misfire_grace < timedelta(0):
            raise ValueError('misfire_grace must not be negative')
        if run_timeout is not None and run_timeout <= 0:
            raise ValueError('run_timeout must be greater than zero')
        self._agent = agent
        self._store = store if store is not None else self._store_from_agent(agent)
        self._deps = deps
        self._on_result = on_result
        self._tick_interval = tick_interval
        self._misfire_grace = misfire_grace
        self._run_timeout = run_timeout
        self._usage_limits = usage_limits
        self._running: set[str] = set()
        self._stop_event = anyio.Event()

    @staticmethod
    def _store_from_agent(agent: AbstractAgent[AgentDepsT, Any]) -> ScheduleStore:
        from pydantic_ai_harness.scheduling._capability import Scheduling

        found: list[Scheduling[AgentDepsT]] = []

        def inspect_capability(capability: AbstractCapability[AgentDepsT]) -> None:
            while isinstance(capability, WrapperCapability):
                capability = capability.wrapped
            if isinstance(capability, Scheduling) and all(match is not capability for match in found):
                found.append(capability)

        agent.root_capability.apply(inspect_capability)
        if len(found) > 1:
            raise ValueError('The agent has multiple Scheduling capabilities; pass `store=` explicitly.')
        if not found:
            raise ValueError('No Scheduling capability was found on the agent; pass `store=` explicitly.')
        return found[0].resolved_store

    async def _claim(self, schedule: Schedule, now: datetime) -> _Claim | None:
        for attempt in range(_SAVE_ATTEMPTS):
            due_at = schedule.next_run_at
            if not schedule.enabled or due_at is None or due_at > now:
                return None

            execute = True
            should_start = True
            if schedule.max_runs is not None and schedule.runs_completed >= schedule.max_runs:
                schedule.next_run_at = None
                execute = False
                should_start = False
            elif schedule.id in self._running:
                if isinstance(schedule.trigger, (CronTrigger, IntervalTrigger)):  # pragma: no branch
                    schedule.next_run_at = next_run_time(schedule.trigger, after=now, timezone=schedule.timezone)
                    execute = False
                    should_start = False
                else:  # pragma: no cover - a one-shot cannot become due again while its only run is in flight
                    return None
            elif now - due_at > self._misfire_grace and isinstance(schedule.trigger, OnceTrigger):
                schedule.next_run_at = None
                schedule.last_status = 'missed'
                schedule.last_error = 'The runner was not running within the allowed grace period.'
                execute = False
            else:
                if isinstance(schedule.trigger, (CronTrigger, IntervalTrigger)):
                    schedule.next_run_at = next_run_time(schedule.trigger, after=now, timezone=schedule.timezone)
                else:
                    schedule.next_run_at = None
                schedule.runs_completed += 1
                if schedule.max_runs is not None and schedule.runs_completed >= schedule.max_runs:
                    schedule.next_run_at = None

            try:
                await self._store.save(schedule)
            except ScheduleConflictError:
                if attempt == _SAVE_ATTEMPTS - 1:
                    raise
                refreshed = await self._store.get(schedule.id)
                if refreshed is None:
                    return None
                schedule = refreshed
                continue

            if should_start:
                self._running.add(schedule.id)
                return _Claim(schedule, execute=execute, claimed_at=now)
            return None
        raise AssertionError('unreachable')  # pragma: no cover

    async def _apply_outcome(self, schedule_id: str, fallback: Schedule, apply: Callable[[Schedule], None]) -> Schedule:
        """Retry against fresh state so outcome fields compose with concurrent edits."""
        for attempt in range(_SAVE_ATTEMPTS):
            current = await self._store.get(schedule_id)
            target = current if current is not None else fallback
            apply(target)
            if current is None:
                return target
            try:
                await self._store.save(target)
            except ScheduleConflictError:
                if attempt == _SAVE_ATTEMPTS - 1:
                    raise
                continue
            return target
        raise AssertionError('unreachable')  # pragma: no cover

    async def _deliver(self, result: ScheduleResult) -> None:
        if self._on_result is None:
            return
        delivery_error: str | None = None
        try:
            callback_result = self._on_result(result)
            if inspect.isawaitable(callback_result):
                await callback_result
        except Exception as exc:
            delivery_error = f'{type(exc).__name__}: {exc}'[:1000]

        def apply_delivery(schedule: Schedule) -> None:
            schedule.last_delivery_error = delivery_error
            result.schedule.last_delivery_error = delivery_error

        await self._apply_outcome(result.schedule.id, result.schedule, apply_delivery)

    async def _execute(self, claim: _Claim) -> ScheduleResult:
        schedule = claim.schedule
        started_at = datetime.now(timezone.utc)
        try:
            if not claim.execute:
                finished_at = datetime.now(timezone.utc)
                result = ScheduleResult(
                    schedule=schedule,
                    status='missed',
                    output=None,
                    error=schedule.last_error,
                    started_at=started_at,
                    finished_at=finished_at,
                    usage=None,
                )
                await self._deliver(result)
                return result

            token = scheduled_run_var.set(schedule.id)
            try:
                run = self._agent.run(
                    schedule.prompt,
                    deps=self._deps,
                    usage_limits=schedule.usage_limits or self._usage_limits,
                )
                if self._run_timeout is None:
                    agent_result = await run
                else:
                    with anyio.fail_after(self._run_timeout):
                        agent_result = await run
            except Exception as exc:
                status: Literal['success', 'error', 'missed'] = 'error'
                output = None
                error = f'{type(exc).__name__}: {exc}'[:1000]
                usage = None
            else:
                status = 'success'
                output = str(agent_result.output)
                error = None
                usage = agent_result.usage
            finally:
                scheduled_run_var.reset(token)

            def apply_outcome(target: Schedule) -> None:
                target.last_run_at = claim.claimed_at
                target.last_status = status
                target.last_error = error
                if status == 'success':
                    target.last_output = output

            updated = await self._apply_outcome(schedule.id, schedule, apply_outcome)
            result = ScheduleResult(
                schedule=updated,
                status=status,
                output=output,
                error=error,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                usage=usage,
            )
            await self._deliver(result)
            return result
        finally:
            self._running.discard(schedule.id)

    async def _execute_collect(self, claim: _Claim, results: list[ScheduleResult]) -> None:
        results.append(await self._execute(claim))

    async def tick(self, now: datetime | None = None) -> list[ScheduleResult]:
        """Claim and concurrently execute schedules due at `now`.

        This method does not require the continuous runner loop, so an external
        scheduler can drive it from system cron, a workflow engine, or serverless
        infrastructure.
        """
        reference = now or datetime.now(timezone.utc)
        if reference.utcoffset() is None:
            raise ValueError('now must be timezone-aware')
        results: list[ScheduleResult] = []
        sweep_error: Exception | None = None
        async with anyio.create_task_group() as task_group:
            try:
                for schedule in await self._store.list():
                    claim = await self._claim(schedule, reference)
                    if claim is not None:
                        task_group.start_soon(self._execute_collect, claim, results)
            except Exception as exc:
                sweep_error = exc
        if sweep_error is not None:
            raise sweep_error
        return results

    async def run_until_stopped(self) -> None:
        """Claim on each interval until stopped, then drain in-flight runs."""
        sweep_error: Exception | None = None
        async with anyio.create_task_group() as task_group:
            while not self._stop_event.is_set():
                try:
                    for schedule in await self._store.list():
                        claim = await self._claim(schedule, datetime.now(timezone.utc))
                        if claim is not None:
                            task_group.start_soon(self._execute, claim)
                except Exception as exc:
                    sweep_error = exc
                    break
                with anyio.move_on_after(self._tick_interval):
                    await self._stop_event.wait()
        if sweep_error is not None:
            raise sweep_error

    def stop(self) -> None:
        """Request an idempotent graceful stop of the continuous loop."""
        self._stop_event.set()
