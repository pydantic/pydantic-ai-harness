# KnowsCurrentTime capability

Closes #23.

## Overview

A minimal `AbstractCapability` subclass that injects the current date/time into the system prompt via `get_instructions()`.

## Design

- **`get_instructions()`** returns a zero-arg callable (a `SystemPromptFunc`) that is evaluated on each model request, producing a string like:
  `The current date and time is: 2026-04-02T20:30:00Z (Thursday, April 2, 2026)`

- **Configuration:**
  - `tz: str = 'UTC'` -- IANA timezone name
  - `format: str = '%Y-%m-%dT%H:%M:%SZ'` -- strftime format string
  - `include_tool: bool = False` -- optionally registers a `get_current_time` tool via `get_toolset()`

- **Serialization:** inherits default `get_serialization_name()` and `from_spec()` from `AbstractCapability`, so it supports `- KnowsCurrentTime` and `- KnowsCurrentTime: {tz: 'America/New_York'}` in YAML specs.

## Files

- `src/pydantic_harness/knows_current_time.py` -- implementation
- `src/pydantic_harness/__init__.py` -- re-export
- `tests/test_knows_current_time.py` -- 16 tests, 100% coverage
