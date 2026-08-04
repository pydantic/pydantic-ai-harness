"""Cron-only scheduling tests, skipped in the slim dependency matrix."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.scheduling import (
    CronTrigger,
    InMemoryScheduleStore,
    Schedule,
    ScheduleRunner,
    Scheduling,
    parse_schedule,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run agent integration tests on the backend used by Pydantic AI's graph."""
    return 'asyncio'


class TestCronSchedules:
    def test_parse_and_reject(self) -> None:
        assert parse_schedule('0 9 * * *') == CronTrigger(expr='0 9 * * *')
        with pytest.raises(ValidationError):
            CronTrigger(expr='99 99 * * *')

    def test_parse_day_and_month_names_despite_containing_t(self) -> None:
        assert parse_schedule('0 9 * * TUE') == CronTrigger(expr='0 9 * * TUE')
        assert parse_schedule('0 9 * OCT SAT') == CronTrigger(expr='0 9 * OCT SAT')

    async def test_next_run_keeps_wall_clock_hour_across_spring_forward(self) -> None:
        zone = ZoneInfo('America/New_York')
        now = datetime(2026, 3, 7, 10, tzinfo=zone)
        schedule = Schedule(
            id='dst',
            name='DST',
            prompt='work',
            trigger=CronTrigger(expr='0 9 * * *'),
            timezone='America/New_York',
            next_run_at=now.astimezone(timezone.utc),
        )
        store = InMemoryScheduleStore()
        await store.add(schedule)
        result = (await ScheduleRunner(Agent(TestModel()), deps=None, store=store).tick(now.astimezone(timezone.utc)))[
            0
        ]
        assert result.schedule.next_run_at is not None
        local_next = result.schedule.next_run_at.astimezone(zone)
        assert local_next == datetime(2026, 3, 8, 9, tzinfo=zone)
        assert local_next.hour == 9

    async def test_tool_listing_renders_cron_expression(self) -> None:
        capability = Scheduling[None]()
        await capability.resolved_store.add(
            Schedule(
                id='cron',
                name='Cron',
                prompt='work',
                trigger=CronTrigger(expr='0 9 * * *'),
                next_run_at=datetime(2030, 1, 1, 9, tzinfo=timezone.utc),
            )
        )
        toolset = capability.get_toolset()
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        tools = await toolset.get_tools(ctx)
        output = await toolset.call_tool('list_schedules', {}, ctx, tools['list_schedules'])
        assert output == (
            'id: cron\nname: Cron\nstatus: scheduled\nschedule: 0 9 * * *\nnext run (UTC): 2030-01-01T09:00:00+00:00'
        )
