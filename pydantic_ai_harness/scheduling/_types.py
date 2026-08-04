"""Types and schedule parsing for the `Scheduling` capability."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone as datetime_timezone
from typing import Annotated, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, Field, field_validator
from pydantic_ai.usage import RunUsage, UsageLimits

_DURATION_RE = re.compile(r'(?P<form>every|in)\s+(?P<count>\d+)(?P<unit>[mhd])', re.IGNORECASE)
_DATE_PREFIX_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
_PARSE_ERROR = (
    'Schedule must be one of: every <N><m|h|d>, in <N><m|h|d>, an ISO 8601 datetime, or a 5-field cron expression.'
)
_INSTALL_HINT = 'Install scheduling support with `pip install pydantic-ai-harness[scheduling]`.'


def _cron_sim(expr: str, start: datetime) -> Iterator[datetime]:
    try:
        from cronsim import CronSim
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return CronSim(expr, start)


def _validate_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (KeyError, ValueError) as exc:
        raise ValueError(f'Unknown IANA timezone: {value!r}') from exc
    return value


class CronTrigger(BaseModel):
    """A recurring five-field cron expression."""

    kind: Literal['cron'] = 'cron'
    expr: str

    @field_validator('expr')
    @classmethod
    def validate_expr(cls, value: str) -> str:
        """Validate the cron expression using `cronsim`."""
        try:
            _cron_sim(value, datetime.now(datetime_timezone.utc))
        except ImportError:
            raise
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        return value


class IntervalTrigger(BaseModel):
    """A recurring fixed interval from one minute through 365 days."""

    kind: Literal['interval'] = 'interval'
    every: timedelta

    @field_validator('every')
    @classmethod
    def validate_every(cls, value: timedelta) -> timedelta:
        """Reject intervals outside the runner's supported cadence."""
        if value < timedelta(minutes=1):
            raise ValueError('Interval schedules must run at least every 1 minute.')
        if value > timedelta(days=365):
            raise ValueError('Interval schedules cannot exceed 365 days. Use a cron expression for longer cadences.')
        return value


class OnceTrigger(BaseModel):
    """A one-shot execution at an aware datetime."""

    kind: Literal['once'] = 'once'
    at: AwareDatetime


ScheduleTrigger = Annotated[CronTrigger | IntervalTrigger | OnceTrigger, Field(discriminator='kind')]
"""A cron, interval, or one-shot schedule trigger."""


def _duration(count: int, unit: str) -> timedelta:
    if unit.lower() == 'm':
        return timedelta(minutes=count)
    if unit.lower() == 'h':
        return timedelta(hours=count)
    return timedelta(days=count)


def parse_schedule(text: str, *, timezone: str = 'UTC', now: datetime | None = None) -> ScheduleTrigger:
    """Parse one of the supported compact schedule forms.

    Accepted forms are `every <N><m|h|d>`, `in <N><m|h|d>`, an ISO 8601
    datetime, or a five-field cron expression. A naive ISO datetime is
    interpreted in `timezone`. Cron expressions raise `ImportError` when the
    scheduling extra is not installed.
    """
    timezone = _validate_timezone(timezone)
    value = text.strip()
    match = _DURATION_RE.fullmatch(value)
    if match is not None:
        try:
            duration = _duration(int(match.group('count')), match.group('unit'))
            if match.group('form').lower() == 'every':
                return IntervalTrigger(every=duration)
            reference = now or datetime.now(tz=ZoneInfo(timezone))
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=ZoneInfo(timezone))
            return OnceTrigger(at=reference + duration)
        except OverflowError as exc:
            raise ValueError('Duration is too large.') from exc

    # Only a leading date marks an ISO datetime: a bare `T` also appears in cron day and
    # month names such as `TUE` or `OCT`.
    if _DATE_PREFIX_RE.match(value) is not None:
        try:
            at = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(_PARSE_ERROR) from exc
        if at.tzinfo is None:
            at = at.replace(tzinfo=ZoneInfo(timezone))
        return OnceTrigger(at=at)

    if len(value.split()) == 5:
        return CronTrigger(expr=' '.join(value.split()))
    raise ValueError(_PARSE_ERROR)


def next_run_time(trigger: ScheduleTrigger, *, after: datetime, timezone: str) -> datetime | None:
    """Return the first occurrence after `after`, or `None` for an expired one-shot."""
    zone = ZoneInfo(_validate_timezone(timezone))
    if isinstance(trigger, CronTrigger):
        return next(_cron_sim(trigger.expr, after.astimezone(zone)))
    if isinstance(trigger, IntervalTrigger):
        return after + trigger.every
    return trigger.at if trigger.at > after else None


def new_schedule_id() -> str:
    """Return a short random schedule id; creators retry on the rare collision."""
    return uuid4().hex[:12]


class Schedule(BaseModel):
    """A persisted scheduled agent run."""

    id: str = Field(default_factory=new_schedule_id)
    version: int = 0
    """Store-managed revision used for optimistic concurrency control."""
    name: str
    prompt: str
    trigger: ScheduleTrigger
    timezone: str = 'UTC'
    deliver_to: str | None = None
    enabled: bool = True
    max_runs: int | None = Field(default=None, ge=1)
    usage_limits: UsageLimits | None = None
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(datetime_timezone.utc))
    next_run_at: AwareDatetime | None = None
    runs_completed: int = 0
    last_run_at: AwareDatetime | None = None
    last_status: Literal['success', 'error', 'missed'] | None = None
    last_error: str | None = None
    last_delivery_error: str | None = None
    last_output: str | None = None

    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Require a resolvable IANA timezone name."""
        return _validate_timezone(value)

    @property
    def status(self) -> Literal['paused', 'completed', 'scheduled']:
        """Return the schedule's user-facing lifecycle status."""
        if not self.enabled:
            return 'paused'
        if self.next_run_at is None:
            return 'completed'
        return 'scheduled'


@dataclass(frozen=True)
class ScheduleResult:
    """The outcome of one scheduled occurrence."""

    schedule: Schedule
    status: Literal['success', 'error', 'missed']
    output: str | None
    error: str | None
    started_at: datetime
    finished_at: datetime
    usage: RunUsage | None


ScheduleResultCallback = Callable[[ScheduleResult], Awaitable[None] | None]
"""A synchronous or asynchronous scheduled-result callback."""
