"""Shared fakes for the no-Docker tests.

The whole seam is testable without a container: a `FakeEnvironment` records the
commands the agent issues and returns scripted results, standing in for what
`environment.exec` would do against a real Docker container.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pydantic_ai_harness_terminal_bench.tools import CommandResult


@dataclass
class _ExecResult:
    """Mirror of harbor.environments.base.ExecResult (the fields the adapter reads)."""

    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


@dataclass
class FakeEnvironment:
    """Duck-typed stand-in for Harbor's `BaseEnvironment`.

    Only implements what the adapter touches: `exec`. Each call is recorded and
    answered from `responses` (keyed by exact command), defaulting to empty
    success for anything unscripted.
    """

    responses: dict[str, _ExecResult] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> _ExecResult:
        self.calls.append(command)
        return self.responses.get(command, _ExecResult(stdout='', return_code=0))


@dataclass
class RecordingExecutor:
    """A `CommandExecutor` that records commands and replays scripted results."""

    results: dict[str, CommandResult] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    async def __call__(self, command: str, *, timeout_sec: int | None = None) -> CommandResult:
        self.calls.append(command)
        return self.results.get(command, CommandResult(output='', exit_code=0))


@pytest.fixture
def recording_executor() -> RecordingExecutor:
    return RecordingExecutor()


@pytest.fixture
def fake_environment() -> FakeEnvironment:
    return FakeEnvironment()
