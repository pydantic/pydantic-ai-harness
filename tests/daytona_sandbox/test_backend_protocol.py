"""Protocol and SDK-boundary tests for Daytona sandboxes."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock

import anyio
import pytest
from daytona import DaytonaAuthenticationError, DaytonaConnectionError, DaytonaNotFoundError
from pydantic_ai.sandboxes import (
    Sandbox,
    SandboxBackend,
    SandboxError,
    SandboxRef,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SupportsFilesystem,
)

from pydantic_ai_harness.daytona_sandbox import (
    DaytonaSandboxAuthError,
    DaytonaSandboxBackend,
    DaytonaSandboxError,
    DaytonaSandboxUnavailableError,
)

from ..sandbox_conformance import (
    check_command_validation,
    check_missing_file,
    check_timeout,
)
from .fake_daytona import FakeDaytona

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def started(**settings: Any) -> DaytonaSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting the property is that touch.
    """
    backend = DaytonaSandboxBackend(**settings)
    await backend.sandbox
    return backend


class TestConformance:
    async def test_sandbox_property_is_lazy_and_reuses_handle(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaSandboxBackend()
        pending = backend.sandbox
        assert not fake_daytona.sandboxes
        sandbox = await pending
        assert await backend.sandbox is sandbox

    async def test_run_and_filesystem_protocols(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        assert isinstance(backend, SandboxBackend)
        assert isinstance(backend, SupportsFilesystem)

    async def test_shared_command_validation(self, fake_daytona: FakeDaytona) -> None:
        await check_command_validation(started)

    async def test_shared_missing_file(self, fake_daytona: FakeDaytona) -> None:
        await check_missing_file(started)

    async def test_shared_timeout(self, fake_daytona: FakeDaytona) -> None:
        async def factory() -> DaytonaSandboxBackend:
            backend = await started()
            fake_daytona.sandboxes[-1].process_hangs = True
            return backend

        await check_timeout(factory)

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
        with pytest.raises(DaytonaSandboxError, match='before reporting an exit status'):
            await backend.run(['true'])

    async def test_log_read_failure_is_provider_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_logs_error = RuntimeError('logs failed')
        with pytest.raises(DaytonaSandboxError, match='logs failed'):
            await backend.run(['true'])

    async def test_deadline_carries_partial_output_and_kills(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['partial out']
        sandbox.process_stderr = ['partial err']
        sandbox.process_hangs = True
        with pytest.raises(SandboxTimeoutError) as exc_info:
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
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await backend.run(['true'], timeout=0.01)
        assert (exc_info.value.stdout, exc_info.value.timeout) == ('complete output', 0.01)
        assert sandbox.process_sessions == set()

    async def test_original_error_wins_when_kill_fails(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_status_error = RuntimeError('status failed')
        sandbox.process_delete_error = RuntimeError('delete failed')
        with pytest.raises(DaytonaSandboxError, match='status failed'):
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
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._CREATE_TIMEOUT', 0.01)
        with pytest.raises(DaytonaSandboxError, match='session setup timed out') as exc_info:
            await backend.run(['true'])
        assert not isinstance(exc_info.value, SandboxTimeoutError)

    async def test_session_execution_failure_cleans_up(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.exec_error = DaytonaConnectionError('offline')
        with pytest.raises(DaytonaSandboxError, match='offline'):
            await backend.run(['true'])
        assert sandbox.process_sessions == set()

    @pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
    async def test_invalid_deadline(self, fake_daytona: FakeDaytona, timeout: float) -> None:
        backend = await started()
        with pytest.raises(ValueError, match='positive finite'):
            await backend.run(['true'], timeout=timeout)


class TestLifecycle:
    async def test_create_passes_configuration(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(
            name='stable',
            snapshot='python',
            auto_stop_minutes=15,
            working_dir='/work',
            env={'A': 'b'},
            network_block_all=True,
        )
        params = fake_daytona.create_params[0]
        assert (params.name, params.snapshot, params.auto_stop_interval, params.auto_delete_interval) == (
            'stable',
            'python',
            15,
            -1,
        )
        assert (params.env_vars, params.network_block_all) == ({'A': 'b'}, True)
        await backend.destroy()
        assert fake_daytona.sandboxes[0].deleted is True
        assert fake_daytona.closed_clients == 2

    async def test_connect_accepts_name_and_refs_remain_ids(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox('sb-id')
        sandbox.name = 'stable'
        backend = await started(ref=SandboxRef(sandbox_id='stable'))
        assert backend.ref == SandboxRef(sandbox_id='sb-id')
        assert sandbox.started is True
        await backend.destroy()
        assert sandbox.deleted is True
        assert fake_daytona.closed_clients == 1

    async def test_create_or_connect_connects_first(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox('sb-id')
        sandbox.name = 'stable'
        backend = await started(name='stable')
        assert backend.ref == SandboxRef(sandbox_id='sb-id')
        assert fake_daytona.create_params == []

    async def test_create_or_connect_creates_when_missing(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(name='stable')
        assert backend.ref == SandboxRef(sandbox_id='sb-1')
        assert fake_daytona.create_params[0].name == 'stable'

    async def test_create_or_connect_reconnects_after_lost_race(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        winner = fake_daytona.sandbox('winner')
        winner.name = 'stable'
        connected = await started(ref=SandboxRef(sandbox_id='stable'))
        # The fake cannot change state between attach and create, so script the race directly.
        monkeypatch.setattr(
            DaytonaSandboxBackend,
            '_attach',
            AsyncMock(side_effect=[DaytonaSandboxUnavailableError('missing'), await connected.sandbox]),
        )
        monkeypatch.setattr(DaytonaSandboxBackend, '_create', AsyncMock(side_effect=DaytonaSandboxError('race')))
        assert (await started(name='stable')).ref == SandboxRef(sandbox_id='winner')

    async def test_setup_timeout_is_not_command_timeout(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_daytona.create_gate = asyncio.Event()
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._CREATE_TIMEOUT', 0.01)
        with pytest.raises(DaytonaSandboxError, match='creation did not complete') as exc_info:
            await started()
        assert not isinstance(exc_info.value, SandboxTimeoutError)

    async def test_connect_timeout_is_provider_error(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = fake_daytona.sandbox()
        sandbox.start_gate = asyncio.Event()
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._CREATE_TIMEOUT', 0.01)
        with pytest.raises(DaytonaSandboxError, match='connection did not complete'):
            await started(ref=SandboxRef(sandbox_id=sandbox.id))

    async def test_create_or_connect_preserves_create_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        create_error = DaytonaSandboxError('create failed')
        monkeypatch.setattr(
            DaytonaSandboxBackend,
            '_attach',
            AsyncMock(side_effect=DaytonaSandboxUnavailableError('missing')),
        )
        monkeypatch.setattr(DaytonaSandboxBackend, '_create', AsyncMock(side_effect=create_error))
        with pytest.raises(DaytonaSandboxError) as exc_info:
            await started(name='stable')
        assert exc_info.value is create_error

    async def test_destroy_known_ref_does_not_start_and_not_found_succeeds(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox()
        backend = DaytonaSandboxBackend(ref=SandboxRef(sandbox_id=sandbox.id))
        await backend.destroy()
        assert sandbox.start_calls == []
        assert sandbox.deleted is True
        await DaytonaSandboxBackend(ref=SandboxRef(sandbox_id='missing')).destroy()

    async def test_an_unused_backend_has_nothing_to_destroy_or_disconnect(self, fake_daytona: FakeDaytona) -> None:
        # Building one does no I/O, so releasing it must not open a client either -- resolving
        # here would create the very sandbox being released.
        backend = DaytonaSandboxBackend()
        await backend.destroy()
        await backend.disconnect()
        await backend.pause()
        await backend.stop()

        assert fake_daytona.sandboxes == []
        assert fake_daytona.close_calls == 0

    async def test_destroy_is_idempotent_and_not_found_delete_succeeds(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.delete_error = DaytonaNotFoundError('gone')
        await backend.destroy()
        await backend.destroy()
        assert fake_daytona.close_calls == 2

    async def test_destroy_and_disconnect_failures_are_translated(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.delete_error = RuntimeError('delete failed')
        with pytest.raises(DaytonaSandboxError, match='delete failed'):
            await backend.destroy()
        assert fake_daytona.close_calls == 1

        fake_daytona.delete_error = None
        fake_daytona.close_error = RuntimeError('close failed')
        with pytest.raises(DaytonaSandboxError, match='SDK connection cleanup failed.*close failed'):
            await backend.destroy()

    async def test_pause_and_stop_keep_the_client_for_reacquisition(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]

        await backend.pause()
        assert sandbox.pause_calls == [60.0]
        await backend.sandbox
        await backend.stop()
        assert sandbox.stop_calls == [60.0]
        await backend.sandbox
        assert sandbox.start_calls == [60.0, 60.0]

    async def test_disconnect_keeps_remote_sandbox_and_reconnects(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]

        await backend.disconnect()

        assert sandbox.deleted is False
        await backend.sandbox
        assert sandbox.start_calls == [60.0]

    async def test_stop_rejects_ephemeral_sandbox(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox()
        backend = await started(ref=SandboxRef(sandbox_id=sandbox.id))
        sandbox.remote_auto_delete_interval = 0

        with pytest.raises(DaytonaSandboxError, match='ephemeral'):
            await backend.stop()
        assert sandbox.refresh_data_calls == 1
        assert sandbox.stop_calls == []

    async def test_destroy_waits_for_acquisition(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_gate = asyncio.Event()
        backend = DaytonaSandboxBackend()

        async def acquire() -> object:
            return await backend.sandbox

        acquire_task = asyncio.create_task(acquire())
        await asyncio.sleep(0)
        destroy = asyncio.create_task(backend.destroy())
        await asyncio.sleep(0)
        assert not destroy.done()
        fake_daytona.create_gate.set()
        await asyncio.gather(acquire_task, destroy)
        assert fake_daytona.sandboxes[0].deleted is True

    async def test_destroy_lookup_timeout_is_bounded(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sandbox = fake_daytona.sandbox()
        fake_daytona.get_gate = asyncio.Event()
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._REQUEST_TIMEOUT', 0.01)
        backend = DaytonaSandboxBackend(ref=SandboxRef(sandbox_id=sandbox.id))

        with pytest.raises(DaytonaSandboxError, match='lookup did not complete'):
            await backend.destroy()
        assert sandbox.deleted is False
        assert sandbox.start_calls == []

    async def test_destroy_lookup_failure_closes_client_and_keeps_ref(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.get_error = RuntimeError('lookup failed')
        backend = DaytonaSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))

        with pytest.raises(DaytonaSandboxError, match='lookup failed'):
            await backend.destroy()
        assert backend.ref == SandboxRef(sandbox_id='sbx-keep')
        assert fake_daytona.close_calls == 1

    async def test_missing_destroy_lookup_reports_client_close_failure(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.get_error = DaytonaNotFoundError('gone')
        fake_daytona.close_error = RuntimeError('close failed')
        backend = DaytonaSandboxBackend(ref=SandboxRef(sandbox_id='sbx-gone'))

        with pytest.raises(DaytonaSandboxError, match='close failed'):
            await backend.destroy()

    async def test_pause_failure_can_be_retried(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        cause = RuntimeError('pause failed')
        fake_daytona.sandboxes[0].pause_error = cause
        with pytest.raises(DaytonaSandboxError) as caught:
            await backend.pause()
        assert caught.value.__cause__ is cause
        fake_daytona.sandboxes[0].pause_error = None
        await backend.pause()

    async def test_stop_refresh_failure_is_translated(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].refresh_error = RuntimeError('refresh failed')

        with pytest.raises(DaytonaSandboxError, match='refresh failed'):
            await backend.stop()

    async def test_pause_missing_sandbox_is_unavailable(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].pause_error = DaytonaNotFoundError('gone')
        with pytest.raises(DaytonaSandboxUnavailableError):
            await backend.pause()


class TestErrorsAndFilesystem:
    async def test_error_taxonomy(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_error = DaytonaAuthenticationError('bad key')
        with pytest.raises(DaytonaSandboxAuthError) as auth:
            await started()
        assert isinstance(auth.value, SandboxUnavailableError)
        fake_daytona.create_error = None
        with pytest.raises(DaytonaSandboxUnavailableError) as unavailable:
            await started(ref=SandboxRef(sandbox_id='missing'))
        assert isinstance(unavailable.value, SandboxUnavailableError)
        fake_daytona.create_error = DaytonaConnectionError('offline')
        with pytest.raises(DaytonaSandboxError) as recoverable:
            await started()
        assert isinstance(recoverable.value, SandboxError)
        assert not isinstance(recoverable.value, SandboxUnavailableError)

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
        with pytest.raises(DaytonaSandboxError, match='probe failed'):
            await backend.working_dir()

    async def test_invalid_native_working_dir_is_rejected(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].workdir = 'relative'
        with pytest.raises(DaytonaSandboxError, match='determine the working directory'):
            await backend.working_dir()

    async def test_filesystem_roundtrip_uses_file_entry(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(working_dir='/workspace')
        sandbox = Sandbox(backend)
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
        with pytest.raises(DaytonaSandboxError, match='Could not create') as exc:
            await backend.write_bytes('/pkg/a.py', b'x')
        assert isinstance(exc.value, SandboxError)
        sandbox.mkdir_exit_code = 0
        sandbox.fs_error = DaytonaAuthenticationError('denied')
        with pytest.raises(DaytonaSandboxAuthError):
            await backend.make_dir('/pkg')
        with pytest.raises(DaytonaSandboxAuthError):
            await backend.exists('/pkg')

    async def test_missing_path_uses_builtin_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        with pytest.raises(FileNotFoundError):
            await backend.stat('/missing')


class TestLazyOperations:
    async def test_default_cwd_is_applied_to_commands(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaSandboxBackend(working_dir='/work dir')
        await backend.run(['true'])
        assert fake_daytona.sandboxes[0].process_command == "cd -- '/work dir' && true"

    async def test_working_dir_initializes_identity(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaSandboxBackend(working_dir='/workspace')
        assert await backend.working_dir() == '/workspace'
        assert backend.ref is not None

    @pytest.mark.parametrize('operation', ['read_bytes', 'exists', 'working_dir'])
    async def test_missing_sandbox_remains_terminal(self, fake_daytona: FakeDaytona, operation: str) -> None:
        backend = DaytonaSandboxBackend(ref=SandboxRef(sandbox_id='missing'))
        with pytest.raises(DaytonaSandboxUnavailableError):
            if operation == 'read_bytes':
                await backend.read_bytes('/note')
            elif operation == 'exists':
                await backend.exists('/note')
            else:
                await backend.working_dir()

    async def test_deadline_bounds_acquisition(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_gate = asyncio.Event()
        with pytest.raises(SandboxTimeoutError):
            await DaytonaSandboxBackend().run(['true'], timeout=0.01)
        assert fake_daytona.closed_clients == 1

    async def test_deadline_bounds_session_setup(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_create_gate = asyncio.Event()
        with pytest.raises(SandboxTimeoutError):
            await backend.run(['true'], timeout=0.01)

    async def test_delete_can_be_retried_after_failure(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        cause = RuntimeError('delete failed')
        fake_daytona.delete_error = cause
        with pytest.raises(DaytonaSandboxError) as caught:
            await backend.destroy()
        assert caught.value.__cause__ is cause
        fake_daytona.delete_error = None
        await backend.destroy()
        assert fake_daytona.sandboxes[0].deleted


def test_missing_daytona_extra_has_an_install_hint() -> None:
    result = subprocess.run(
        [sys.executable, '-c', "import sys; sys.modules['daytona'] = None; import pydantic_ai_harness.daytona_sandbox"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert 'Install `pydantic-ai-harness[daytona]`' in result.stderr


async def test_argv_rejects_shell_mode(fake_daytona: FakeDaytona) -> None:
    with pytest.raises(TypeError, match='argv sequence'):
        await DaytonaSandboxBackend().run(['true'], shell=True)


async def test_client_cleanup_failure_can_be_retried(fake_daytona: FakeDaytona) -> None:
    backend = await started()
    error = RuntimeError('client close failed')
    fake_daytona.close_error = error
    with pytest.raises(DaytonaSandboxError) as caught:
        await backend.disconnect()
    assert caught.value.__cause__ is error
    fake_daytona.close_error = None
    await backend.disconnect()
    assert fake_daytona.closed_clients == 1


async def test_working_directory_is_cached(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()
    first = await backend.working_dir()
    fake_daytona.sandboxes[0].workdir = '/changed'
    assert await backend.working_dir() == first


async def test_shell_command_is_passed_to_the_session(fake_daytona: FakeDaytona) -> None:
    await DaytonaSandboxBackend().run('printf hello | cat', shell=True)
    assert fake_daytona.sandboxes[0].process_command == 'printf hello | cat'
