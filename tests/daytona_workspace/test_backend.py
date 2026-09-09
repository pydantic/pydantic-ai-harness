"""Protocol and SDK-boundary tests for Daytona sandboxes."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

import anyio
import daytona
import pytest
from daytona import DaytonaAuthenticationError, DaytonaConnectionError
from pydantic_ai.workspaces import (
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness.daytona_workspace import (
    DaytonaWorkspaceBackend,
)

from .fake_daytona import FakeDaytona

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def started(**settings: Any) -> DaytonaWorkspaceBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting the property is that touch.
    """
    backend = DaytonaWorkspaceBackend(**settings)
    await backend.workspace
    return backend


class TestConformance:
    async def test_sandbox_property_is_lazy_and_reuses_handle(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaWorkspaceBackend()
        pending = backend.workspace
        assert not fake_daytona.sandboxes
        sandbox = await pending
        assert await backend.workspace is sandbox

    async def test_run_and_filesystem_protocols(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        assert isinstance(backend, WorkspaceBackend)
        assert isinstance(backend, SupportsFilesystem)

    async def test_shared_run_and_nonzero_result(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[-1].responder = lambda command, timeout: ('', 2)
        result = await backend.run(['false'])
        assert result.exit_code != 0


class TestCommands:
    async def test_argv_output_and_context(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['out', 'put']
        sandbox.process_stderr = ['error']
        sandbox.process_exit_code = 3
        result = await backend.run(['printf', 'a b'], cwd='/work dir', env={'A': 'x y'}, timeout=5)
        assert result == type(result)(exit_code=3, stdout='output', stderr='error')
        assert sandbox.process_command == ("cd -- '/work dir' && env -- 'A=x y' sh -c 'printf '\"'\"'a b'\"'\"''")
        assert sandbox.process_sessions == set()

    async def test_missing_exit_status_is_provider_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_stdout = ['']
        fake_daytona.sandboxes[0].process_exit_code = None
        with pytest.raises(WorkspaceError, match='before reporting an exit status'):
            await backend.run(['true'])

    async def test_log_read_failure_is_provider_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_logs_error = RuntimeError('logs failed')
        with pytest.raises(WorkspaceError, match='logs failed'):
            await backend.run(['true'])

    async def test_deadline_carries_partial_output_and_kills(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['partial out']
        sandbox.process_stderr = ['partial err']
        sandbox.process_hangs = True
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sleep', '30'], timeout=0.01)
        assert (exc_info.value.stdout, exc_info.value.stderr, exc_info.value.timeout) == (
            'partial out',
            'partial err',
            0.01,
        )
        assert sandbox.process_sessions == set()

    async def test_deadline_includes_exit_status_rpc(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['complete output']
        sandbox.process_status_gate = asyncio.Event()
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['true'], timeout=0.01)
        assert (exc_info.value.stdout, exc_info.value.timeout) == ('complete output', 0.01)
        assert sandbox.process_sessions == set()

    async def test_original_error_wins_when_kill_fails(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_status_error = RuntimeError('status failed')
        sandbox.process_delete_error = RuntimeError('delete failed')
        with pytest.raises(WorkspaceError, match='status failed'):
            await backend.run(['false'])

    async def test_cancellation_kills_process(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_hangs = True
        task = asyncio.create_task(backend.run(['sleep', '30']))
        await sandbox.process_logs_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sandbox.process_sessions == set()

    async def test_session_setup_timeout_is_provider_error(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_create_gate = asyncio.Event()
        monkeypatch.setattr('pydantic_ai_harness.daytona_workspace._backend._REQUEST_TIMEOUT', 0.01)
        with pytest.raises(WorkspaceError, match='session setup timed out') as exc_info:
            await backend.run(['true'])
        assert not isinstance(exc_info.value, WorkspaceTimeoutError)

    async def test_session_execution_failure_cleans_up(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.exec_error = DaytonaConnectionError('offline')
        with pytest.raises(WorkspaceError, match='offline'):
            await backend.run(['true'])
        assert sandbox.process_sessions == set()

    @pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
    async def test_invalid_deadline(self, fake_daytona: FakeDaytona, timeout: float) -> None:
        backend = await started()
        with pytest.raises(ValueError, match='positive finite'):
            await backend.run(['true'], timeout=timeout)


class TestErrorsAndFilesystem:
    async def test_error_taxonomy(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_error = DaytonaAuthenticationError('bad key')
        with pytest.raises(WorkspaceUnavailableError) as auth:
            await started()
        assert isinstance(auth.value, WorkspaceUnavailableError)
        fake_daytona.create_error = None
        with pytest.raises(WorkspaceUnavailableError) as unavailable:
            await started(ref=WorkspaceRef(provider='daytona', id='missing'))
        assert isinstance(unavailable.value, WorkspaceUnavailableError)
        fake_daytona.create_error = DaytonaConnectionError('offline')
        with pytest.raises(WorkspaceError) as recoverable:
            await started()
        assert isinstance(recoverable.value, WorkspaceError)
        assert not isinstance(recoverable.value, WorkspaceUnavailableError)

    async def test_concurrent_first_probes_converge(self, fake_daytona: FakeDaytona) -> None:
        # The probe is an idempotent read, so overlapping first calls are allowed to
        # duplicate it: both get the same answer and the cache settles.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.workdir_gate = asyncio.Event()
        first = asyncio.create_task(backend.working_dir())
        await sandbox.workdir_started.wait()
        second = asyncio.create_task(backend.working_dir())
        with anyio.fail_after(5):
            while sandbox.workdir_calls < 2:
                await asyncio.sleep(0)
        sandbox.workdir_gate.set()
        assert await asyncio.gather(first, second) == ['/srv/repo', '/srv/repo']
        assert sandbox.workdir_calls == 2

    async def test_configured_working_dir_is_probed(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(working_dir='/work')
        assert await backend.working_dir() == '/work'
        assert fake_daytona.sandboxes[0].workdir_calls == 1

    async def test_working_dir_error_is_translated(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].workdir_error = RuntimeError('probe failed')
        with pytest.raises(WorkspaceError, match='probe failed'):
            await backend.working_dir()

    async def test_invalid_native_working_dir_is_rejected(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].workdir = 'relative'
        with pytest.raises(WorkspaceError, match='determine the working directory'):
            await backend.working_dir()

    async def test_filesystem_roundtrip_uses_file_entry(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(working_dir='/workspace')
        sandbox = Workspace(backend)
        await sandbox.write_bytes('/workspace/notes/a.txt', b'hello')
        assert await sandbox.read_bytes('/workspace/notes/a.txt') == b'hello'
        entry = await sandbox.stat('/workspace/notes/a.txt')
        assert (entry.path, entry.name, entry.size, entry.is_dir) == (
            '/workspace/notes/a.txt',
            'a.txt',
            5,
            False,
        )
        await sandbox.write_bytes('/root.txt', b'root')
        await sandbox.make_dir('/elsewhere')  # outside the listed directory, must not appear
        await sandbox.make_dir('/workspace/pkg')
        await sandbox.write_bytes('/workspace/pkg/a.py', b'x')
        entries = await sandbox.list_dir('/workspace')
        assert {(entry.name, entry.is_dir, entry.size) for entry in entries} == {
            ('notes', True, None),
            ('pkg', True, None),
        }
        directory = await sandbox.stat('/workspace/pkg')
        assert (directory.is_dir, directory.size) == (True, None)
        assert await sandbox.exists('/workspace/pkg/a.py') is True
        assert await sandbox.exists('/missing') is False
        await sandbox.remove('/workspace/pkg')
        assert await sandbox.exists('/workspace/pkg/a.py') is False

    async def test_filesystem_failures_are_translated(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.mkdir_exit_code = 1
        with pytest.raises(WorkspaceError, match='Could not create') as exc:
            await backend.write_bytes('/pkg/a.py', b'x')
        assert isinstance(exc.value, WorkspaceError)
        sandbox.mkdir_exit_code = 0
        sandbox.fs_error = DaytonaAuthenticationError('denied')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.make_dir('/pkg')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.exists('/pkg')

    async def test_missing_path_uses_builtin_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        with pytest.raises(FileNotFoundError):
            await backend.stat('/missing')


class TestLazyOperations:
    async def test_default_cwd_is_applied_to_commands(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaWorkspaceBackend(working_dir='/work dir')
        await backend.run(['true'])
        assert fake_daytona.sandboxes[0].process_command == "cd -- '/work dir' && true"

    async def test_working_dir_initializes_identity(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaWorkspaceBackend(working_dir='/workspace')
        assert await backend.working_dir() == '/workspace'
        assert backend.ref is not None

    @pytest.mark.parametrize('operation', ['read_bytes', 'exists', 'working_dir'])
    async def test_missing_sandbox_remains_terminal(self, fake_daytona: FakeDaytona, operation: str) -> None:
        backend = DaytonaWorkspaceBackend(ref=WorkspaceRef(provider='daytona', id='missing'))
        with pytest.raises(WorkspaceUnavailableError):
            if operation == 'read_bytes':
                await backend.read_bytes('/note')
            elif operation == 'exists':
                await backend.exists('/note')
            else:
                await backend.working_dir()

    async def test_deadline_bounds_acquisition(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_gate = asyncio.Event()
        with pytest.raises(WorkspaceTimeoutError):
            await DaytonaWorkspaceBackend().run(['true'], timeout=0.01)
        assert fake_daytona.closed_clients == 1

    async def test_deadline_bounds_session_setup(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_create_gate = asyncio.Event()
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['true'], timeout=0.01)


def test_missing_daytona_extra_has_an_install_hint() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            "import sys; sys.modules['daytona'] = None; import pydantic_ai_harness.daytona_workspace",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert 'Install `pydantic-ai-harness[daytona]`' in result.stderr


async def test_argv_rejects_shell_mode(fake_daytona: FakeDaytona) -> None:
    with pytest.raises(TypeError, match='argv sequence'):
        await DaytonaWorkspaceBackend().run(['true'], shell=True)


async def test_client_cleanup_failure_can_be_retried(fake_daytona: FakeDaytona) -> None:
    backend = await started()
    error = RuntimeError('client close failed')
    fake_daytona.close_error = error
    with pytest.raises(WorkspaceError) as caught:
        await backend.disconnect()
    assert caught.value.__cause__ is error
    fake_daytona.close_error = None
    await backend.disconnect()
    assert fake_daytona.closed_clients == 1


async def test_working_directory_is_cached(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaWorkspaceBackend()
    first = await backend.working_dir()
    fake_daytona.sandboxes[0].workdir = '/changed'
    assert await backend.working_dir() == first


async def test_shell_command_is_passed_to_the_session(fake_daytona: FakeDaytona) -> None:
    await DaytonaWorkspaceBackend().run('printf hello | cat', shell=True)
    assert fake_daytona.sandboxes[0].process_command == 'printf hello | cat'


async def test_str_command_requires_shell(fake_daytona: FakeDaytona) -> None:
    with pytest.raises(TypeError, match='requires shell=True'):
        await DaytonaWorkspaceBackend().run('echo hi')


async def test_empty_argv_is_rejected(fake_daytona: FakeDaytona) -> None:
    with pytest.raises(TypeError, match='at least the program'):
        await DaytonaWorkspaceBackend().run([])


async def test_attach_finds_target_among_several(fake_daytona: FakeDaytona) -> None:
    fake_daytona.sandbox('sb-decoy')
    target = fake_daytona.sandbox('sb-target')
    backend = DaytonaWorkspaceBackend(ref=WorkspaceRef(provider='daytona', id=target.id))
    assert (await backend.workspace).id == target.id


async def test_supplied_client_is_used_and_never_closed(fake_daytona: FakeDaytona) -> None:
    client = daytona.AsyncDaytona()
    backend = DaytonaWorkspaceBackend(client=client)
    await backend.workspace
    assert fake_daytona.sandboxes[0].client is client
    await backend.disconnect()
    assert fake_daytona.closed_clients == 0


async def test_supplied_client_survives_acquisition_failure(fake_daytona: FakeDaytona) -> None:
    client = daytona.AsyncDaytona()
    fake_daytona.create_error = DaytonaConnectionError('boom')
    with pytest.raises(WorkspaceError):
        await DaytonaWorkspaceBackend(client=client).workspace
    assert fake_daytona.closed_clients == 0


async def test_create_timeout_is_translated(fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai_harness.daytona_workspace._backend._CREATE_TIMEOUT', 0.05)
    fake_daytona.create_gate = asyncio.Event()
    with pytest.raises(WorkspaceTimeoutError, match='creation did not complete'):
        await DaytonaWorkspaceBackend().workspace
    assert fake_daytona.closed_clients == 1


async def test_attach_timeout_is_translated(fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = fake_daytona.sandbox('sb-existing')
    monkeypatch.setattr('pydantic_ai_harness.daytona_workspace._backend._CREATE_TIMEOUT', 0.05)
    fake_daytona.get_gate = asyncio.Event()
    with pytest.raises(WorkspaceTimeoutError, match='connection did not complete'):
        await DaytonaWorkspaceBackend(ref=WorkspaceRef(provider='daytona', id=existing.id)).workspace


async def test_deadline_bounds_attach(fake_daytona: FakeDaytona) -> None:
    existing = fake_daytona.sandbox('sb-existing')
    fake_daytona.get_gate = asyncio.Event()
    with pytest.raises(WorkspaceTimeoutError):
        await DaytonaWorkspaceBackend(ref=WorkspaceRef(provider='daytona', id=existing.id)).run(['true'], timeout=0.01)


async def test_read_missing_file_uses_builtin_error(fake_daytona: FakeDaytona) -> None:
    backend = await started()
    with pytest.raises(FileNotFoundError):
        await backend.read_bytes('/missing')


async def test_attach_error_is_translated(fake_daytona: FakeDaytona) -> None:
    existing = fake_daytona.sandbox('sb-existing')
    fake_daytona.get_error = DaytonaConnectionError('control plane down')
    backend = DaytonaWorkspaceBackend(ref=WorkspaceRef(provider='daytona', id=existing.id))
    with pytest.raises(WorkspaceError):
        await backend.workspace
