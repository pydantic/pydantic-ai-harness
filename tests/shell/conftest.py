"""Fixtures and helpers shared by the shell toolset tests.

`ShellToolset` runs every command inside the run's sandbox, so each test needs a
`RunContext` carrying one. `LocalSandbox(root=tmp_path)` is that sandbox: commands
really run, in a directory the test owns.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import LocalSandbox, Sandbox
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.shell import _toolset as shell_toolset_module
from pydantic_ai_harness.shell._toolset import ShellToolset


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(autouse=True)
def short_kill_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_toolset_module, '_KILL_GRACE_PERIOD', 0.05)


@pytest.fixture
async def sandbox(tmp_path: Path) -> AsyncIterator[Sandbox]:
    async with LocalSandbox(root=tmp_path) as backend:
        yield Sandbox.wrap(backend)


def run_context(sandbox: Sandbox | None = None) -> RunContext[None]:
    """A `RunContext` for calling tools, with `sandbox` attached when given."""
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    if sandbox is not None:
        ctx.sandbox = sandbox
    return ctx


def shell_toolset(
    cwd: Path = Path('.'),
    *,
    allowed_commands: Sequence[str] = (),
    denied_commands: Sequence[str] = (),
    denied_operators: Sequence[str] = (),
    default_timeout: float = 10.0,
    max_timeout: float = 600.0,
    max_output_chars: int = 50_000,
    allow_interactive: bool = False,
    env: Mapping[str, str] | None = None,
    denied_env_patterns: Sequence[str] = (),
) -> ShellToolset[None]:
    return ShellToolset(
        cwd=cwd,
        allowed_commands=allowed_commands,
        denied_commands=denied_commands,
        denied_operators=denied_operators,
        default_timeout=default_timeout,
        max_timeout=max_timeout,
        max_output_chars=max_output_chars,
        allow_interactive=allow_interactive,
        env=env,
        denied_env_patterns=denied_env_patterns,
    )


def background_toolset(
    tmp_path: Path,
    *,
    env: Mapping[str, str] = {},
    max_output_chars: int = 50_000,
) -> ShellToolset[None]:
    """A toolset whose `PATH` reaches a `setsid` shim, for `start_command` tests.

    macOS ships no `setsid` binary, and one code path for every platform keeps the
    background branches covered everywhere.
    """
    setsid = tmp_path / 'setsid'
    setsid.write_text(
        '#!/usr/bin/env python3\nimport os, sys\nos.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n',
        encoding='utf-8',
    )
    setsid.chmod(0o755)
    return shell_toolset(
        tmp_path,
        env={**env, 'PATH': f'{tmp_path}:{os.environ["PATH"]}'},
        max_output_chars=max_output_chars,
    )


async def call_tool(toolset: ShellToolset[None], ctx: RunContext[None], name: str, **tool_args: object) -> str:
    """Invoke a tool the way the agent does, through the toolset's dispatch seam."""
    tools = await toolset.get_tools(ctx)
    result: object = await toolset.call_tool(name, tool_args, ctx, tools[name])
    assert isinstance(result, str)
    return result


def command_id(start_result: str) -> str:
    return start_result.rsplit('ID: ', 1)[1].strip()
