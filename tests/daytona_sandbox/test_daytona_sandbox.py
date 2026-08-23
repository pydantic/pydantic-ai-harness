"""Tests for the public Daytona sandbox capability."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Protocol, TypeGuard, runtime_checkable

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.daytona_sandbox import (
    DaytonaSandbox,
    DaytonaSandboxAuthError,
    DaytonaSandboxError,
    DaytonaSandboxExecResult,
    DaytonaSandboxSession,
    DaytonaSandboxUnavailableError,
)

from .fake_daytona import FakeDaytona

pytestmark = pytest.mark.anyio(backends=['asyncio'])


@runtime_checkable
class _Tools(Protocol):  # pragma: no cover - structural typing only
    id: str | None

    async def __aenter__(self) -> _Tools: ...

    async def __aexit__(self, *args: object) -> None: ...

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str: ...

    async def read_file(self, path: str, *, offset: int | None = None, limit: int | None = None) -> str: ...

    async def write_file(self, path: str, content: str) -> str: ...

    async def list_directory(self, path: str = '.') -> str: ...


def _is_toolset(value: object) -> TypeGuard[AbstractToolset[None]]:
    return isinstance(value, AbstractToolset)


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


@asynccontextmanager
async def _tools(
    *,
    sandbox_id: str | None = None,
    session: DaytonaSandboxSession | None = None,
    snapshot: str | None = None,
    workdir: str | None = None,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = 50 * 1024,
    max_output_lines: int = 2000,
    max_read_bytes: int = 5 * 1024 * 1024,
) -> AsyncGenerator[_Tools]:
    base = DaytonaSandbox[None](
        sandbox_id=sandbox_id,
        session=session,
        snapshot=snapshot,
        workdir=workdir,
        env=env,
        max_output_bytes=max_output_bytes,
        max_output_lines=max_output_lines,
        max_read_bytes=max_read_bytes,
    ).get_toolset()
    if not _is_toolset(base):  # pragma: no cover - capability contract
        raise AssertionError('DaytonaSandbox must return an AbstractToolset')
    run = await base.for_run(_context())
    if not isinstance(run, _Tools):  # pragma: no cover - capability contract
        raise AssertionError('DaytonaSandbox is missing its tools')
    async with run:
        yield run


async def test_agent_runs_command_and_deletes_owned_sandbox(fake_daytona: FakeDaytona) -> None:
    def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'echo hello'})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[None, str] = Agent(FunctionModel(call_then_finish), capabilities=[DaytonaSandbox()])
    fake_daytona.sandbox().responder = lambda command, timeout: ('unused', 0)
    result = await agent.run('run it')

    assert result.output == 'done'
    assert fake_daytona.sandboxes[-1].deleted is True
    assert fake_daytona.delete_calls == [('sb-2', 60, True)]


async def test_attached_sandbox_is_left_running(fake_daytona: FakeDaytona) -> None:
    sandbox = fake_daytona.sandbox()
    sandbox.responder = lambda command, timeout: ('hello\n', 0)
    async with _tools(sandbox_id=sandbox.id) as tools:
        assert await tools.run_command('echo hello') == 'hello'
    assert sandbox.deleted is False
    assert fake_daytona.delete_calls == []


async def test_caller_owned_session_is_reused_across_agent_runs(fake_daytona: FakeDaytona) -> None:
    def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'state.txt', 'content': 'ready'})])
        return ModelResponse(parts=[TextPart('done')])

    async with DaytonaSandboxSession() as session:
        agent: Agent[None, str] = Agent(FunctionModel(call_then_finish), capabilities=[DaytonaSandbox(session=session)])
        assert (await agent.run('first')).output == 'done'
        assert (await agent.run('second')).output == 'done'
        assert len(fake_daytona.sandboxes) == 1
        assert fake_daytona.sandboxes[0].files['state.txt'] == b'ready'
        assert fake_daytona.sandboxes[0].deleted is False
    assert fake_daytona.sandboxes[0].deleted is True
    assert fake_daytona.delete_calls == [('sb-1', 60, True)]


async def test_session_exposes_command_result_and_open_sandbox_id(fake_daytona: FakeDaytona) -> None:
    session = DaytonaSandboxSession(workdir='/workspace', env={'A': 'b'})
    assert session.sandbox_id is None
    async with session:
        fake_daytona.sandboxes[0].responder = lambda command, timeout: ('hello\n', 3)
        result = await session.exec('example', timeout=7)
        assert result == DaytonaSandboxExecResult(output='hello\n', returncode=3)
        assert session.sandbox_id == 'sb-1'
    assert session.sandbox_id is None


async def test_attached_session_is_left_running(fake_daytona: FakeDaytona) -> None:
    sandbox = fake_daytona.sandbox()
    async with DaytonaSandboxSession(sandbox_id=sandbox.id) as session:
        assert session.sandbox_id == sandbox.id
    assert sandbox.deleted is False
    assert fake_daytona.delete_calls == []


async def test_session_exit_without_enter_is_safe() -> None:
    await DaytonaSandboxSession().__aexit__(None, None, None)


@pytest.mark.parametrize('timeout', [0, True, 1.5])
async def test_session_rejects_invalid_exec_timeout(fake_daytona: FakeDaytona, timeout: object) -> None:
    async with DaytonaSandboxSession() as session:
        with pytest.raises(ValueError, match='timeout must be a positive integer'):
            await session.exec('example', timeout=timeout)  # type: ignore[arg-type]


async def test_injected_session_must_be_open(fake_daytona: FakeDaytona) -> None:
    with pytest.raises(DaytonaSandboxError, match='injected session is not open'):
        async with _tools(session=DaytonaSandboxSession()):
            pass  # pragma: no cover


async def test_session_cannot_be_entered_twice(fake_daytona: FakeDaytona) -> None:
    async with DaytonaSandboxSession() as session:
        with pytest.raises(DaytonaSandboxError, match='already open'):
            await session.__aenter__()
    assert len(fake_daytona.sandboxes) == 1


async def test_creation_configuration_reaches_daytona(fake_daytona: FakeDaytona) -> None:
    async with _tools(snapshot='snap-python', workdir='/workspace', env={'A': 'b'}) as tools:
        await tools.write_file('src/main.py', 'print(1)')
    params = fake_daytona.create_params[0]
    assert params.snapshot == 'snap-python'
    assert params.auto_stop_interval == 60
    assert params.auto_delete_interval == 0
    assert params.env_vars == {'A': 'b'}
    sandbox = fake_daytona.sandboxes[0]
    assert sandbox.files['/workspace/src/main.py'] == b'print(1)'
    assert sandbox.exec_calls[0].command == 'mkdir -p -- /workspace/src'


@pytest.mark.parametrize(
    ('output', 'exit_code', 'expected'),
    [
        ('', 0, '(no output)'),
        ('bad\n', 2, 'bad\n[exit code: 2]'),
        ('one\ntwo\nthree', 0, '[... output truncated to the last 2 lines ...]\ntwo\nthree'),
    ],
)
async def test_command_output(fake_daytona: FakeDaytona, output: str, exit_code: int, expected: str) -> None:
    async with _tools(max_output_lines=2) as tools:
        fake_daytona.sandboxes[0].responder = lambda command, timeout: (output, exit_code)
        assert await tools.run_command('example') == expected


async def test_command_timeout_is_clamped(fake_daytona: FakeDaytona) -> None:
    async with _tools() as tools:
        await tools.run_command('slow', timeout_seconds=999)
    assert fake_daytona.sandboxes[0].exec_calls[0].timeout == 300


async def test_command_timeout_is_reported(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaTimeoutError

    async with _tools() as tools:
        fake_daytona.sandboxes[0].exec_error = DaytonaTimeoutError('late')
        assert await tools.run_command('slow', timeout_seconds=2) == '(no output)\n[timed out after 2s]'


async def test_command_sdk_failures(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaAuthenticationError, DaytonaConnectionError, DaytonaNotFoundError

    async with _tools() as tools:
        sandbox = fake_daytona.sandboxes[0]
        sandbox.exec_error = DaytonaConnectionError('offline')
        with pytest.raises(ModelRetry, match='offline'):
            await tools.run_command('example')
        sandbox.exec_error = DaytonaNotFoundError('gone')
        with pytest.raises(DaytonaSandboxUnavailableError):
            await tools.run_command('example')
        sandbox.exec_error = DaytonaAuthenticationError('denied')
        with pytest.raises(DaytonaSandboxAuthError):
            await tools.run_command('example')


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan'), True])
async def test_invalid_command_timeout_is_retryable(fake_daytona: FakeDaytona, timeout: float) -> None:
    async with _tools() as tools:
        with pytest.raises(ModelRetry, match='greater than 0'):
            await tools.run_command('bad', timeout_seconds=timeout)


async def test_file_tools(fake_daytona: FakeDaytona) -> None:
    async with _tools() as tools:
        sandbox = fake_daytona.sandboxes[0]
        sandbox.files['notes.txt'] = b'one\ntwo\nthree\n'
        sandbox.files['src/main.py'] = b'print(1)'
        assert await tools.read_file('notes.txt', offset=2, limit=1) == (
            'two\n\n[1 more lines in file. Use offset=3 to continue.]'
        )
        assert await tools.list_directory() == 'notes.txt\nsrc/'
        assert await tools.write_file('out/result.txt', 'ok') == "Wrote 2 bytes to 'out/result.txt'."
        assert sandbox.files['out/result.txt'] == b'ok'


async def test_absolute_and_parentless_writes(fake_daytona: FakeDaytona) -> None:
    async with _tools(workdir='/workspace') as tools:
        await tools.write_file('/tmp/absolute.txt', 'a')
        await tools.write_file('plain.txt', 'b')
    sandbox = fake_daytona.sandboxes[0]
    assert sandbox.files['/tmp/absolute.txt'] == b'a'
    assert sandbox.files['/workspace/plain.txt'] == b'b'


async def test_write_failures_are_retryable(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaConnectionError

    async with _tools() as tools:
        sandbox = fake_daytona.sandboxes[0]
        sandbox.mkdir_exit_code = 1
        with pytest.raises(ModelRetry, match='Could not create'):
            await tools.write_file('dir/file', 'content')
        sandbox.mkdir_exit_code = 0
        sandbox.fs_error = DaytonaConnectionError('offline')
        with pytest.raises(ModelRetry, match='Could not write'):
            await tools.write_file('file', 'content')
        with pytest.raises(ModelRetry, match='cannot be encoded'):
            await tools.write_file('file', '\ud800')


async def test_empty_directory(fake_daytona: FakeDaytona) -> None:
    async with _tools() as tools:
        assert await tools.list_directory() == '(empty)'


async def test_file_auth_failures_are_terminal(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaAuthorizationError

    async with _tools() as tools:
        fake_daytona.sandboxes[0].fs_error = DaytonaAuthorizationError('denied')
        with pytest.raises(DaytonaSandboxAuthError):
            await tools.read_file('file')
        with pytest.raises(DaytonaSandboxAuthError):
            await tools.write_file('file', 'content')
        with pytest.raises(DaytonaSandboxAuthError):
            await tools.list_directory()


async def test_list_failure_is_retryable(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaConnectionError

    async with _tools() as tools:
        fake_daytona.sandboxes[0].fs_error = DaytonaConnectionError('offline')
        with pytest.raises(ModelRetry, match='Could not list'):
            await tools.list_directory()


async def test_missing_file_is_retryable(fake_daytona: FakeDaytona) -> None:
    async with _tools() as tools:
        with pytest.raises(ModelRetry, match='Could not read'):
            await tools.read_file('missing')


async def test_oversized_file_is_retryable(fake_daytona: FakeDaytona) -> None:
    async with _tools(max_read_bytes=2) as tools:
        fake_daytona.sandboxes[0].files['large'] = b'abc'
        with pytest.raises(ModelRetry, match='over the 2B read limit'):
            await tools.read_file('large')


async def test_file_growth_is_checked_after_download(fake_daytona: FakeDaytona) -> None:
    async with _tools(max_read_bytes=2) as tools:
        sandbox = fake_daytona.sandboxes[0]
        sandbox.files['growing'] = b'abc'
        sandbox.reported_sizes['growing'] = 1
        with pytest.raises(ModelRetry, match='over the 2B read limit'):
            await tools.read_file('growing')


async def test_unavailable_attached_sandbox_is_terminal(fake_daytona: FakeDaytona) -> None:
    with pytest.raises(DaytonaSandboxUnavailableError, match='does not exist'):
        async with _tools(sandbox_id='missing'):
            pass


async def test_sdk_auth_error_is_terminal(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaAuthenticationError

    fake_daytona.create_error = DaytonaAuthenticationError('no')
    with pytest.raises(DaytonaSandboxAuthError, match='DAYTONA_API_KEY'):
        async with _tools():
            pass


async def test_generic_creation_failure_is_reported(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaConnectionError

    fake_daytona.create_error = DaytonaConnectionError('offline')
    with pytest.raises(DaytonaSandboxError, match='offline'):
        async with _tools():
            pass


async def test_cleanup_failure_is_reported(fake_daytona: FakeDaytona) -> None:
    from daytona import DaytonaConnectionError

    fake_daytona.delete_error = DaytonaConnectionError('offline')
    with pytest.raises(DaytonaSandboxError, match='offline'):
        async with _tools():
            pass
    assert fake_daytona.closed_clients == 1


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('auto_stop_minutes', 0),
        ('default_command_timeout', True),
        ('max_command_timeout', -1),
        ('max_output_bytes', 0),
        ('max_output_lines', 0),
        ('max_read_bytes', 0),
    ],
)
def test_positive_integer_configuration(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f'{name} must be a positive integer'):
        DaytonaSandbox(**{name: value})  # type: ignore[arg-type]


def test_configuration_conflicts_and_instructions() -> None:
    with pytest.raises(ValueError, match='cannot exceed'):
        DaytonaSandbox(default_command_timeout=2, max_command_timeout=1)
    with pytest.raises(ValueError, match='snapshot cannot'):
        DaytonaSandbox(sandbox_id='sb', snapshot='snap')
    with pytest.raises(ValueError, match='instructions must'):
        DaytonaSandbox(instructions=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='sandbox_id.*cannot be combined with `session`'):
        DaytonaSandbox(sandbox_id='sb', session=DaytonaSandboxSession())
    with pytest.raises(ValueError, match='snapshot.*cannot be combined with `session`'):
        DaytonaSandbox(snapshot='snap', session=DaytonaSandboxSession())
    with pytest.raises(ValueError, match='workdir.*env.*cannot be combined with `session`'):
        DaytonaSandbox(workdir='/work', env={'A': 'b'}, session=DaytonaSandboxSession())
    with pytest.raises(ValueError, match='auto_stop_minutes.*cannot be combined with `session`'):
        DaytonaSandbox(auto_stop_minutes=30, session=DaytonaSandboxSession())
    for value in (0, True, 1.5):
        with pytest.raises(ValueError, match='auto_stop_minutes must be a positive integer'):
            DaytonaSandboxSession(auto_stop_minutes=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='snapshot cannot'):
        DaytonaSandboxSession(sandbox_id='sb', snapshot='snap')
    assert DaytonaSandbox(instructions='').get_instructions() is None
    assert DaytonaSandbox(instructions='custom').get_instructions() == 'custom'
    assert 'is deleted after this run' in (DaytonaSandbox().get_instructions() or '')
    assert 'persists after this run' in (DaytonaSandbox(sandbox_id='sb').get_instructions() or '')
    assert 'persists after this run' in (DaytonaSandbox(session=DaytonaSandboxSession()).get_instructions() or '')
    toolset = DaytonaSandbox[None](id=None).get_toolset()
    if not isinstance(toolset, _Tools):  # pragma: no cover - capability contract
        raise AssertionError('DaytonaSandbox is missing its tools')
    assert toolset.id == 'daytona_sandbox'


async def test_toolset_requires_run_lifecycle() -> None:
    base = DaytonaSandbox[None]().get_toolset()
    if not isinstance(base, _Tools):  # pragma: no cover - capability contract
        raise AssertionError('DaytonaSandbox is missing its tools')
    async with base:
        with pytest.raises(DaytonaSandboxError, match='session is not open'):
            await base.run_command('echo hello')
    await base.__aexit__(None, None, None)


async def test_run_toolset_cannot_be_entered_twice(fake_daytona: FakeDaytona) -> None:
    base = DaytonaSandbox[None]().get_toolset()
    if not _is_toolset(base):  # pragma: no cover - capability contract
        raise AssertionError('DaytonaSandbox must return an AbstractToolset')
    run = await base.for_run(_context())
    await run.__aenter__()
    try:
        with pytest.raises(DaytonaSandboxError, match='already open'):
            await run.__aenter__()
    finally:
        await run.__aexit__(None, None, None)


def test_public_exports() -> None:
    import pydantic_ai_harness.daytona_sandbox as module

    assert module.DaytonaSandbox is DaytonaSandbox
    assert module.__all__ == (
        'DaytonaSandbox',
        'DaytonaSandboxAuthError',
        'DaytonaSandboxError',
        'DaytonaSandboxExecResult',
        'DaytonaSandboxSession',
        'DaytonaSandboxUnavailableError',
    )


def test_rejects_durable_execution() -> None:
    from pydantic_ai.durable_exec.temporal import TemporalDurability

    with pytest.raises(UserError, match='does not support durable execution'):
        Agent(TestModel(), capabilities=[DaytonaSandbox(), TemporalDurability()])
