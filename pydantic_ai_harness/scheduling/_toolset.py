"""Model-facing CRUD tools for the `Scheduling` capability."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone as datetime_timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool

from pydantic_ai_harness.scheduling._runner import scheduled_run_var
from pydantic_ai_harness.scheduling._store import ScheduleConflictError
from pydantic_ai_harness.scheduling._types import (
    CronTrigger,
    IntervalTrigger,
    OnceTrigger,
    Schedule,
    ScheduleTrigger,
    new_schedule_id,
    next_run_time,
    parse_schedule,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.scheduling._capability import Scheduling

CREATE_SCHEDULE_DESCRIPTION = (
    'Create a schedule for a prompt. Schedule text accepts `every 30m`, `every 2h`, '
    '`in 15m`, an ISO 8601 datetime, or a five-field cron expression.'
)
LIST_SCHEDULES_DESCRIPTION = 'List schedules without expanding their prompt or output bodies.'
GET_SCHEDULE_DESCRIPTION = 'Get one schedule by id, including its prompt and most recent successful output.'
UPDATE_SCHEDULE_DESCRIPTION = 'Update selected fields of an existing schedule by id.'
PAUSE_SCHEDULE_DESCRIPTION = 'Pause a schedule so the runner will not claim it.'
RESUME_SCHEDULE_DESCRIPTION = 'Resume a schedule from its next future occurrence without replaying a backlog.'
DELETE_SCHEDULE_DESCRIPTION = 'Delete a schedule permanently by id.'
RUN_SCHEDULE_NOW_DESCRIPTION = 'Queue an enabled schedule for execution on the next runner tick.'
_SAVE_ATTEMPTS = 3
_CONFLICT_RETRY_MESSAGE = 'The schedule changed while this update was being applied. Try again.'


def _trigger_display(trigger: ScheduleTrigger) -> str:
    if isinstance(trigger, CronTrigger):
        return trigger.expr
    if isinstance(trigger, IntervalTrigger):
        seconds = int(trigger.every.total_seconds())
        if seconds % 86_400 == 0:
            return f'every {seconds // 86_400}d'
        if seconds % 3_600 == 0:
            return f'every {seconds // 3_600}h'
        return f'every {seconds // 60}m'
    return trigger.at.isoformat()


def _render_schedule(schedule: Schedule, *, full: bool) -> str:
    next_run = (
        'none'
        if schedule.next_run_at is None
        else schedule.next_run_at.astimezone(ZoneInfo(schedule.timezone)).isoformat()
    )
    lines = [
        f'id: {schedule.id}',
        f'name: {schedule.name}',
        f'status: {schedule.status}',
        f'schedule: {_trigger_display(schedule.trigger)}',
        f'next run ({schedule.timezone}): {next_run}',
    ]
    if schedule.last_status is not None:
        lines.append(f'last status: {schedule.last_status}')
    if schedule.last_error is not None:
        lines.append(f'last error: {schedule.last_error}')
    if schedule.last_delivery_error is not None:
        lines.append(f'last delivery error: {schedule.last_delivery_error}')
    if full:
        lines.extend(
            [
                f'deliver to: {schedule.deliver_to or "none"}',
                f'max runs: {schedule.max_runs if schedule.max_runs is not None else "none"}',
                f'runs completed: {schedule.runs_completed}',
                f'prompt: {schedule.prompt}',
                f'last output: {schedule.last_output if schedule.last_output is not None else "none"}',
            ]
        )
    return '\n'.join(lines)


class SchedulingToolset(FunctionToolset[AgentDepsT]):
    """CRUD tools over a `Scheduling` capability's shared store."""

    def __init__(self, capability: Scheduling[AgentDepsT]) -> None:
        super().__init__(id='scheduling')
        self._capability = capability
        self.add_function(self.create_schedule, description=CREATE_SCHEDULE_DESCRIPTION)
        self.add_function(self.list_schedules, description=LIST_SCHEDULES_DESCRIPTION)
        self.add_function(self.get_schedule, description=GET_SCHEDULE_DESCRIPTION)
        self.add_function(self.update_schedule, description=UPDATE_SCHEDULE_DESCRIPTION)
        self.add_function(self.pause_schedule, description=PAUSE_SCHEDULE_DESCRIPTION)
        self.add_function(self.resume_schedule, description=RESUME_SCHEDULE_DESCRIPTION)
        self.add_function(self.delete_schedule, description=DELETE_SCHEDULE_DESCRIPTION)
        self.add_function(self.run_schedule_now, description=RUN_SCHEDULE_NOW_DESCRIPTION)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Hide scheduling tools while a scheduled prompt is running."""
        if scheduled_run_var.get() is not None:
            return {}
        return await super().get_tools(ctx)

    async def _known_or_retry(self, schedule_id: str) -> Schedule:
        schedule = await self._capability.resolved_store.get(schedule_id)
        if schedule is not None:
            return schedule
        known = ', '.join(item.id for item in await self._capability.resolved_store.list()) or '(none)'
        raise ModelRetry(f'Unknown schedule id {schedule_id!r}. Known ids: {known}.')

    async def _save_or_retry(self, schedule: Schedule, attempt: int) -> bool:
        try:
            await self._capability.resolved_store.save(schedule)
        except ScheduleConflictError as exc:
            if attempt == _SAVE_ATTEMPTS - 1:
                raise ModelRetry(_CONFLICT_RETRY_MESSAGE) from exc
            return False
        return True

    @staticmethod
    def _parse(text: str, *, timezone: str) -> ScheduleTrigger:
        try:
            return parse_schedule(text, timezone=timezone)
        except (ImportError, ValueError) as exc:
            raise ModelRetry(f'Invalid schedule: {exc}') from exc

    @staticmethod
    def _validate_combination(trigger: ScheduleTrigger, max_runs: int | None) -> None:
        if isinstance(trigger, OnceTrigger) and max_runs is not None:
            raise ModelRetry('`max_runs` is only valid for recurring schedules; remove it for a one-shot schedule.')

    async def create_schedule(
        self,
        ctx: RunContext[AgentDepsT],
        name: str,
        prompt: str,
        schedule: str,
        deliver_to: str | None = None,
        max_runs: int | None = None,
    ) -> str:
        """Create a schedule.

        Args:
            ctx: Framework-provided run context.
            name: Short human-readable name.
            prompt: Self-contained prompt executed on every occurrence.
            schedule: Compact schedule text.
            deliver_to: Optional opaque routing hint for the result callback.
            max_runs: Optional attempt limit for a recurring schedule.
        """
        del ctx
        trigger = self._parse(schedule, timezone=self._capability.timezone)
        self._validate_combination(trigger, max_runs)
        now = datetime.now(datetime_timezone.utc)
        next_run_at = next_run_time(trigger, after=now, timezone=self._capability.timezone)
        if isinstance(trigger, OnceTrigger) and next_run_at is None:
            raise ModelRetry('Cannot create this schedule because that time is in the past. Choose a future time.')
        try:
            created = Schedule(
                name=name,
                prompt=prompt,
                trigger=trigger,
                timezone=self._capability.timezone,
                deliver_to=deliver_to,
                max_runs=max_runs,
                next_run_at=next_run_at,
            )
        except ValueError as exc:
            raise ModelRetry(f'Invalid schedule: {exc}') from exc
        for attempt in range(_SAVE_ATTEMPTS):
            try:
                await self._capability.resolved_store.add(created)
            except ValueError as exc:
                if attempt == _SAVE_ATTEMPTS - 1:
                    raise ModelRetry('Could not store the schedule under a fresh id. Try again.') from exc
                created = created.model_copy(update={'id': new_schedule_id()})
                continue
            return f'Schedule created.\n{_render_schedule(created, full=False)}'
        raise AssertionError('unreachable')  # pragma: no cover

    async def list_schedules(self, ctx: RunContext[AgentDepsT]) -> str:
        """List all schedules without prompt or output bodies."""
        del ctx
        schedules = await self._capability.resolved_store.list()
        if not schedules:
            return 'No schedules.'
        return '\n\n'.join(_render_schedule(schedule, full=False) for schedule in schedules)

    async def get_schedule(self, ctx: RunContext[AgentDepsT], schedule_id: str) -> str:
        """Get a schedule's full details.

        Args:
            ctx: Framework-provided run context.
            schedule_id: Id returned by a schedule listing or creation.
        """
        del ctx
        return _render_schedule(await self._known_or_retry(schedule_id), full=True)

    async def update_schedule(
        self,
        ctx: RunContext[AgentDepsT],
        schedule_id: str,
        name: str | None = None,
        prompt: str | None = None,
        schedule: str | None = None,
        deliver_to: str | None = None,
        max_runs: int | None = None,
    ) -> str:
        """Update the provided non-`None` schedule fields.

        Args:
            ctx: Framework-provided run context.
            schedule_id: Id of the schedule to update.
            name: Replacement name.
            prompt: Replacement prompt.
            schedule: Replacement compact schedule text.
            deliver_to: Replacement routing hint; an empty string clears it.
            max_runs: Replacement recurring attempt limit; zero removes the limit.
        """
        del ctx
        for attempt in range(_SAVE_ATTEMPTS):
            existing = await self._known_or_retry(schedule_id)
            if schedule is not None and existing.next_run_at is None:
                raise ModelRetry('This schedule is completed. Create a new schedule instead.')
            trigger = existing.trigger if schedule is None else self._parse(schedule, timezone=existing.timezone)
            inherited_limit_on_once = schedule is not None and isinstance(trigger, OnceTrigger) and max_runs is None
            resulting_max_runs = None if max_runs == 0 or inherited_limit_on_once else max_runs
            if max_runs is None and not inherited_limit_on_once:
                resulting_max_runs = existing.max_runs
            self._validate_combination(trigger, resulting_max_runs)
            updates: dict[str, object] = {}
            if name is not None:
                updates['name'] = name
            if prompt is not None:
                updates['prompt'] = prompt
            if deliver_to is not None:
                updates['deliver_to'] = deliver_to or None
            if max_runs is not None or inherited_limit_on_once:
                updates['max_runs'] = resulting_max_runs
            if schedule is not None:
                now = datetime.now(datetime_timezone.utc)
                next_run_at = next_run_time(trigger, after=now, timezone=existing.timezone)
                if isinstance(trigger, OnceTrigger) and next_run_at is None:
                    raise ModelRetry(
                        'Cannot update this schedule because that time is in the past. Choose a future time.'
                    )
                updates.update(trigger=trigger, next_run_at=next_run_at)
                if trigger != existing.trigger:
                    updates['runs_completed'] = 0
            try:
                data = existing.model_dump()
                data.update(updates)
                updated = Schedule.model_validate(data)
            except ValueError as exc:
                raise ModelRetry(f'Invalid schedule update: {exc}') from exc
            if updated.max_runs is not None and updated.runs_completed >= updated.max_runs:
                updated.next_run_at = None
            if not await self._save_or_retry(updated, attempt):
                continue
            return f'Schedule updated.\n{_render_schedule(updated, full=False)}'
        raise AssertionError('unreachable')  # pragma: no cover

    async def pause_schedule(self, ctx: RunContext[AgentDepsT], schedule_id: str) -> str:
        """Pause a schedule by id."""
        del ctx
        for attempt in range(_SAVE_ATTEMPTS):
            schedule = await self._known_or_retry(schedule_id)
            schedule.enabled = False
            if not await self._save_or_retry(schedule, attempt):
                continue
            return f'Schedule {schedule.id} paused.'
        raise AssertionError('unreachable')  # pragma: no cover

    async def resume_schedule(self, ctx: RunContext[AgentDepsT], schedule_id: str) -> str:
        """Resume from the next occurrence after now instead of replaying a backlog."""
        del ctx
        for attempt in range(_SAVE_ATTEMPTS):
            schedule = await self._known_or_retry(schedule_id)
            if schedule.enabled:
                raise ModelRetry('This schedule is not paused.')
            schedule.enabled = True
            if schedule.next_run_at is not None:
                now = datetime.now(datetime_timezone.utc)
                next_run_at = next_run_time(schedule.trigger, after=now, timezone=schedule.timezone)
                if next_run_at is None:
                    raise ModelRetry('This schedule expired while paused. Create a new schedule instead.')
                schedule.next_run_at = next_run_at
            if not await self._save_or_retry(schedule, attempt):
                continue
            return f'Schedule {schedule.id} resumed.\n{_render_schedule(schedule, full=False)}'
        raise AssertionError('unreachable')  # pragma: no cover

    async def delete_schedule(self, ctx: RunContext[AgentDepsT], schedule_id: str) -> str:
        """Delete a schedule by id."""
        del ctx
        schedule = await self._known_or_retry(schedule_id)
        await self._capability.resolved_store.remove(schedule_id)
        return f"Schedule {schedule.id} ('{schedule.name}') deleted."

    async def run_schedule_now(self, ctx: RunContext[AgentDepsT], schedule_id: str) -> str:
        """Queue an enabled schedule for the next runner tick."""
        del ctx
        for attempt in range(_SAVE_ATTEMPTS):
            schedule = await self._known_or_retry(schedule_id)
            if not schedule.enabled:
                raise ModelRetry('This schedule is paused. Resume it before queuing an immediate run.')
            if schedule.next_run_at is None:
                raise ModelRetry('This schedule is completed. Create a new schedule instead.')
            schedule.next_run_at = datetime.now(datetime_timezone.utc)
            if not await self._save_or_retry(schedule, attempt):
                continue
            return f'Schedule {schedule.id} queued for the next runner tick.'
        raise AssertionError('unreachable')  # pragma: no cover
