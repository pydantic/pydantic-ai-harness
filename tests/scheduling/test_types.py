"""Tests for public scheduling types and schedule parsing."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pydantic_ai_harness.scheduling import (
    CronTrigger,
    IntervalTrigger,
    OnceTrigger,
    Schedule,
    parse_schedule,
)


class TestScheduleParsing:
    def test_interval_forms_are_case_insensitive(self) -> None:
        assert parse_schedule('every 30m') == IntervalTrigger(every=timedelta(minutes=30))
        assert parse_schedule('EVERY 2H') == IntervalTrigger(every=timedelta(hours=2))
        assert parse_schedule('every 1d') == IntervalTrigger(every=timedelta(days=1))

    def test_relative_once_uses_explicit_now(self) -> None:
        now = datetime(2026, 1, 2, 3, tzinfo=timezone.utc)
        assert parse_schedule('in 2h', now=now) == OnceTrigger(at=now + timedelta(hours=2))

    def test_relative_once_interprets_naive_now_in_timezone(self) -> None:
        now = datetime(2026, 1, 2, 3)
        trigger = parse_schedule('in 1d', timezone='Asia/Kolkata', now=now)
        assert isinstance(trigger, OnceTrigger)
        assert trigger.at == datetime(2026, 1, 3, 3, tzinfo=ZoneInfo('Asia/Kolkata'))

    def test_iso_forms_and_naive_timezone(self) -> None:
        assert parse_schedule('2026-01-02T03:04:05Z') == OnceTrigger(
            at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        )
        trigger = parse_schedule('2026-01-02 03:04:05', timezone='America/New_York')
        assert isinstance(trigger, OnceTrigger)
        assert trigger.at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=ZoneInfo('America/New_York'))

    def test_rejects_unsupported_text_and_bad_iso(self) -> None:
        for value in ('tomorrow morning', '2026-99-99T10:00:00'):
            with pytest.raises(ValueError, match='every <N>'):
                parse_schedule(value)

    def test_rejects_unknown_timezone(self) -> None:
        with pytest.raises(ValueError, match='Unknown IANA timezone'):
            parse_schedule('in 1h', timezone='Mars/Olympus')

    def test_interval_floor(self) -> None:
        with pytest.raises(ValidationError, match='at least every 1 minute'):
            IntervalTrigger(every=timedelta(seconds=59))
        with pytest.raises(ValidationError, match='at least every 1 minute'):
            parse_schedule('every 0m')

    def test_interval_ceiling(self) -> None:
        with pytest.raises(ValidationError, match='cron expression for longer cadences'):
            IntervalTrigger(every=timedelta(days=366))

    @pytest.mark.parametrize('value', ['every 1000000000d', 'in 999999999d'])
    def test_duration_overflow(self, value: str) -> None:
        with pytest.raises(ValueError, match='Duration is too large'):
            parse_schedule(value, now=datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_missing_cronsim_has_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, 'cronsim', None)
        with pytest.raises(ImportError, match=r'pip install pydantic-ai-harness\[scheduling\]'):
            parse_schedule('0 9 * * *')


class TestScheduleModel:
    def test_status(self) -> None:
        schedule = Schedule(name='daily', prompt='work', trigger=IntervalTrigger(every=timedelta(days=1)))
        assert schedule.status == 'completed'
        schedule.next_run_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
        assert schedule.status == 'scheduled'
        schedule.enabled = False
        assert schedule.status == 'paused'

    def test_defaults_and_validation(self) -> None:
        schedule = Schedule(name='daily', prompt='work', trigger=IntervalTrigger(every=timedelta(days=1)))
        assert len(schedule.id) == 12
        assert schedule.created_at.tzinfo is not None
        assert schedule.runs_completed == 0
        with pytest.raises(ValidationError, match='greater than or equal to 1'):
            Schedule(name='bad', prompt='work', trigger=schedule.trigger, max_runs=0)
        with pytest.raises(ValidationError, match='Unknown IANA timezone'):
            Schedule(name='bad', prompt='work', trigger=schedule.trigger, timezone='bad/timezone')

    def test_once_requires_an_aware_datetime(self) -> None:
        with pytest.raises(ValidationError):
            OnceTrigger(at=datetime(2026, 1, 1))


class TestCronTriggerImport:
    def test_public_type_is_available(self) -> None:
        assert CronTrigger.__name__ == 'CronTrigger'
