from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from pydantic_ai.toolsets import FunctionToolset

from pydantic_harness.knows_current_time import (
    KnowsCurrentTime,
    _format_datetime,  # pyright: ignore[reportPrivateUsage]
    _resolve_tz,  # pyright: ignore[reportPrivateUsage]
)

# --- _resolve_tz ---


def test_resolve_tz_utc() -> None:
    assert _resolve_tz('UTC') is timezone.utc  # pyright: ignore[reportPrivateUsage]


def test_resolve_tz_named() -> None:
    tz = _resolve_tz('America/New_York')  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tz, ZoneInfo)
    assert str(tz) == 'America/New_York'


def test_resolve_tz_invalid() -> None:
    with pytest.raises(KeyError):
        _resolve_tz('Not/A/Real/Zone')  # pyright: ignore[reportPrivateUsage]


# --- _format_datetime ---


def test_format_datetime_utc() -> None:
    dt = datetime(2026, 4, 2, 20, 30, 0, tzinfo=timezone.utc)
    result = _format_datetime(dt, '%Y-%m-%dT%H:%M:%SZ')  # pyright: ignore[reportPrivateUsage]
    assert result == 'The current date and time is: 2026-04-02T20:30:00Z (Thursday, April 2, 2026)'


def test_format_datetime_custom_format() -> None:
    dt = datetime(2026, 12, 25, 10, 0, 0, tzinfo=timezone.utc)
    result = _format_datetime(dt, '%Y/%m/%d %H:%M')  # pyright: ignore[reportPrivateUsage]
    assert result == 'The current date and time is: 2026/12/25 10:00 (Friday, December 25, 2026)'


# --- KnowsCurrentTime defaults ---

FIXED_UTC = datetime(2026, 4, 2, 20, 30, 0, tzinfo=timezone.utc)


def test_defaults() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime()
    assert cap.tz == 'UTC'
    assert cap.format == '%Y-%m-%dT%H:%M:%SZ'
    assert cap.include_tool is False


# --- get_instructions ---


def test_get_instructions_returns_callable() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime()
    instructions = cap.get_instructions()
    assert callable(instructions)


def test_get_instructions_default_format() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime()
    instructions = cap.get_instructions()
    assert callable(instructions)
    with patch.object(cap, '_now', return_value=FIXED_UTC):
        result = instructions()  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
    assert result == 'The current date and time is: 2026-04-02T20:30:00Z (Thursday, April 2, 2026)'


def test_get_instructions_custom_format() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime(format='%Y/%m/%d')
    with patch.object(cap, '_now', return_value=FIXED_UTC):
        result = cap._formatted_now()  # pyright: ignore[reportPrivateUsage]
    assert result == 'The current date and time is: 2026/04/02 (Thursday, April 2, 2026)'


def test_get_instructions_with_timezone() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime(tz='America/New_York')
    result = cap._formatted_now()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(result, str)
    assert result.startswith('The current date and time is:')


# --- get_toolset ---


def test_get_toolset_default_none() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime()
    assert cap.get_toolset() is None


def test_get_toolset_with_include_tool() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime(include_tool=True)
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)
    assert 'get_current_time' in toolset.tools


def test_toolset_tool_returns_formatted_time() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime(include_tool=True)
    toolset = cap.get_toolset()
    assert toolset is not None
    assert isinstance(toolset, FunctionToolset)
    tool = toolset.tools['get_current_time']
    with patch.object(cap, '_now', return_value=FIXED_UTC):
        result = tool.function()  # pyright: ignore[reportCallIssue]
    assert result == 'The current date and time is: 2026-04-02T20:30:00Z (Thursday, April 2, 2026)'


# --- serialization name ---


def test_serialization_name() -> None:
    assert KnowsCurrentTime.get_serialization_name() == 'KnowsCurrentTime'


# --- from_spec ---


def test_from_spec_no_args() -> None:
    cap = KnowsCurrentTime.from_spec()
    assert isinstance(cap, KnowsCurrentTime)
    assert cap.tz == 'UTC'


def test_from_spec_with_kwargs() -> None:
    cap = KnowsCurrentTime.from_spec(tz='US/Eastern', format='%H:%M', include_tool=True)
    assert isinstance(cap, KnowsCurrentTime)
    assert cap.tz == 'US/Eastern'
    assert cap.format == '%H:%M'
    assert cap.include_tool is True


# --- timezone validation at init ---


def test_invalid_timezone_raises_value_error() -> None:
    with pytest.raises(ValueError, match="'Not/A/Real/Zone' is not a valid IANA timezone name"):
        KnowsCurrentTime(tz='Not/A/Real/Zone')


def test_invalid_timezone_error_includes_examples() -> None:
    with pytest.raises(ValueError, match='Examples:'):
        KnowsCurrentTime(tz='FakeZone')


def test_valid_timezone_accepted() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime(tz='Europe/London')
    assert cap.tz == 'Europe/London'


def test_utc_timezone_accepted() -> None:
    cap: KnowsCurrentTime[None] = KnowsCurrentTime(tz='UTC')
    assert cap.tz == 'UTC'
