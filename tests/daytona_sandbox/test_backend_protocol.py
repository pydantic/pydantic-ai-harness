"""Protocol and SDK-boundary tests for Daytona sandboxes."""

from __future__ import annotations

import asyncio
import builtins
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
from pydantic_ai_harness.daytona_sandbox._backend import _command_context, _command_line

from ..sandbox_conformance import (
    check_command_validation,
    check_missing_file,
    check_timeout,
)
from .fake_daytona import FakeDaytona

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def _hide_daytona(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_daytona(name: str, *args: object, **kwargs: object) -> object:
        if name == 'daytona':
            raise ImportError('No module named daytona')
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, 'daytona', raising=False)
    monkeypatch.setattr(builtins, '__import__', no_daytona)


async def started(**settings: Any) -> DaytonaSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting the property is that touch.
    """
    backend = DaytonaSandboxBackend(**settings)
    await backend.sandbox
    return backend


class TestConformance:
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
    def test_command_forms_and_context(self) -> None:
        assert _command_line(['printf', 'a b'], False) == "printf 'a b'"
        assert _command_line('printf ok', True) == 'printf ok'
        assert _command_context('run', '/work dir', {'A': 'x y'}) == ("cd -- '/work dir' && env -- 'A=x y' sh -c run")

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
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._REQUEST_TIMEOUT', 0.01)
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
    async def test_missing_package_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hide_daytona(monkeypatch)
        with pytest.raises(DaytonaSandboxError, match='daytona.*required'):
            await started()
        with pytest.raises(DaytonaSandboxError, match='daytona.*required'):
            await started(ref=SandboxRef(sandbox_id='sandbox'))
        with pytest.raises(DaytonaSandboxError, match='daytona.*required'):
            await DaytonaSandboxBackend.delete_by_id('sandbox')

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
            0,
        )
        assert (params.env_vars, params.network_block_all) == ({'A': 'b'}, True)
        await backend.close(terminate=True)
        assert fake_daytona.sandboxes[0].deleted is True
        assert fake_daytona.closed_clients == 2

    async def test_connect_accepts_name_and_refs_remain_ids(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox('sb-id')
        sandbox.name = 'stable'
        backend = await started(ref=SandboxRef(sandbox_id='stable'))
        assert backend.ref == SandboxRef(sandbox_id='sb-id')
        assert sandbox.started is True
        await backend.close(terminate=True)
        assert sandbox.deleted is False
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

    async def test_delete_by_id_does_not_start_and_not_found_succeeds(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox()
        await DaytonaSandboxBackend.delete_by_id(sandbox.id)
        assert sandbox.start_calls == []
        assert sandbox.deleted is True
        await DaytonaSandboxBackend.delete_by_id('missing')

    async def test_an_unused_backend_has_nothing_to_close(self, fake_daytona: FakeDaytona) -> None:
        # Building one does no I/O, so closing it must not open a client either -- resolving
        # here would create the very sandbox being released.
        await DaytonaSandboxBackend().close(terminate=True)

        assert fake_daytona.sandboxes == []
        assert fake_daytona.close_calls == 0

    async def test_close_is_idempotent_and_not_found_delete_succeeds(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.delete_error = DaytonaNotFoundError('gone')
        await backend.close(terminate=True)
        await backend.close(terminate=True)
        assert fake_daytona.close_calls == 1

    async def test_close_and_delete_failures_are_translated(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.delete_error = RuntimeError('delete failed')
        with pytest.raises(DaytonaSandboxError, match='delete failed'):
            await backend.close(terminate=True)
        assert fake_daytona.close_calls == 1

        fake_daytona.delete_error = None
        fake_daytona.close_error = RuntimeError('close failed')
        with pytest.raises(DaytonaSandboxError, match='close failed'):
            assert backend.ref is not None
            await DaytonaSandboxBackend.delete_by_id(backend.ref.sandbox_id)


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

    async def test_concurrent_waits_share_one_outcome(self, fake_daytona: FakeDaytona) -> None:
        # A second waiter arriving mid-settle parks on the lock and receives the first
        # settle's cached result object -- the protocol's concurrent-wait promise. Without
        # the lock it would run its own settle against internals the first one cleans up.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_status_gate = asyncio.Event()
        first = asyncio.create_task(backend.run(['true']))
        await anyio.wait_all_tasks_blocked()  # first is suspended at the status poll, mid-settle
        second = asyncio.create_task(backend.run(['true']))
        await anyio.wait_all_tasks_blocked()
        sandbox.process_status_gate.set()
        one, two = await asyncio.gather(first, second)
        assert one == two

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

    async def test_configured_working_dir_needs_no_probe(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(working_dir='/work')
        assert await backend.working_dir() == '/work'
        assert fake_daytona.sandboxes[0].workdir_calls == 0

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
