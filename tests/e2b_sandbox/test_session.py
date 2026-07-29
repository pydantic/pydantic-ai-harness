"""Tests for the public E2B sandbox session."""

from __future__ import annotations

import importlib
import sys
import types

import anyio
import pytest

from pydantic_ai_harness.e2b_sandbox import (
    E2BSandboxAuthError,
    E2BSandboxError,
    E2BSandboxSession,
    E2BSandboxUnavailableError,
)

from .fake_e2b import (
    AuthenticationException,
    FakeE2B,
    FakeEntryInfo,
    FakeFileType,
    SandboxException,
    SandboxNotFoundException,
    TimeoutException,
)


class TestLifecycle:
    async def test_owned_create_and_kill(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession(
            template='python-data',
            sandbox_timeout=120,
            workdir='/workspace',
            env={'TOKEN': 'secret'},
            metadata={'task': 'test'},
            allow_internet_access=False,
        ) as session:
            assert session.sandbox_id == 'sbx-1'
            assert session.template == 'python-data'
            assert session.mode == 'owned'
            assert session.workdir == '/workspace'
        assert fake_e2b.create_calls == [
            fake_e2b.create_calls[0].__class__(
                'python-data',
                120,
                {'task': 'test'},
                {'TOKEN': 'secret'},
                True,
                False,
            )
        ]
        assert fake_e2b.sandboxes[0].kill_calls == 1
        assert session.sandbox_id is None

    async def test_attach_leaves_sandbox_running(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession(sandbox_id='sbx-existing', workdir='/project') as session:
            assert session.sandbox_id == 'sbx-existing'
            assert session.mode == 'attached'
        assert fake_e2b.connect_calls == [('sbx-existing', None)]
        assert fake_e2b.sandboxes[0].kill_calls == 0
        assert session.sandbox_id is None

    async def test_close_before_enter_is_safe(self) -> None:
        session = E2BSandboxSession()
        assert session.sandbox_id is None
        await session.close()
        await session.__aexit__(None, None, None)

    async def test_enter_twice_rejected(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='already open'):
                await session.__aenter__()
        assert len(fake_e2b.sandboxes) == 1

    async def test_cancel_after_create_still_kills(self, fake_e2b: FakeE2B) -> None:
        session = E2BSandboxSession()
        with anyio.CancelScope() as scope:
            scope.cancel()
            await session.__aenter__()
        assert fake_e2b.sandboxes[0].kill_calls == 1
        assert session.sandbox_id is None

    async def test_create_timeout_is_bounded(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._session._CREATE_TIMEOUT', 0.01)
        fake_e2b.create_hangs = True
        with anyio.fail_after(2):
            with pytest.raises(E2BSandboxError, match='did not complete within'):
                async with E2BSandboxSession():
                    pass  # pragma: no cover

    async def test_kill_timeout_preserves_identity(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._session._TEARDOWN_TIMEOUT', 0.01)
        session = E2BSandboxSession()
        await session.__aenter__()
        fake_e2b.kill_hangs = True
        with pytest.raises(E2BSandboxError, match='Could not kill'):
            await session.close()
        assert session.sandbox_id == 'sbx-1'
        fake_e2b.kill_hangs = False
        await session.close()
        assert session.sandbox_id is None

    async def test_cleanup_failure_warns_during_body_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.kill_error = SandboxException('kill failed')
        with pytest.warns(RuntimeWarning, match='Could not clean up'):
            with pytest.raises(RuntimeError, match='body failed'):
                async with E2BSandboxSession():
                    raise RuntimeError('body failed')

    async def test_cleanup_failure_raises_after_clean_body(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.kill_error = SandboxException('kill failed')
        with pytest.raises(E2BSandboxError, match='Could not kill'):
            async with E2BSandboxSession():
                pass


class TestConfigurationAndErrors:
    @pytest.mark.parametrize('value', [0, -1, True])
    def test_invalid_sandbox_timeout(self, value: int) -> None:
        with pytest.raises(ValueError, match='positive integer'):
            E2BSandboxSession(sandbox_timeout=value)

    @pytest.mark.parametrize('value', ['', 'relative'])
    def test_invalid_workdir(self, value: str) -> None:
        with pytest.raises(ValueError, match='absolute sandbox path'):
            E2BSandboxSession(workdir=value)

    def test_invalid_allow_internet_access(self) -> None:
        with pytest.raises(ValueError, match='allow_internet_access'):
            E2BSandboxSession(allow_internet_access='yes')  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            ({'template': 'custom'}, 'template'),
            ({'sandbox_timeout': 600}, 'sandbox_timeout'),
            ({'env': {'A': 'b'}}, 'env'),
            ({'metadata': {'A': 'b'}}, 'metadata'),
            ({'allow_internet_access': False}, 'allow_internet_access'),
        ],
    )
    def test_attach_rejects_create_only_settings(self, kwargs: dict[str, object], expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            E2BSandboxSession(sandbox_id='sbx', **kwargs)  # type: ignore[arg-type]

    async def test_missing_sdk_has_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = importlib.import_module

        def fail_e2b(name: str, package: str | None = None) -> types.ModuleType:
            if name == 'e2b':
                raise ImportError('missing')
            return real_import(name, package)  # pragma: no cover - only E2B is requested by the subject

        monkeypatch.delitem(sys.modules, 'e2b', raising=False)
        monkeypatch.setattr(importlib, 'import_module', fail_e2b)
        with pytest.raises(E2BSandboxError, match='e2b.*package is required'):
            async with E2BSandboxSession():
                pass  # pragma: no cover

    async def test_incompatible_sdk_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, 'e2b', types.ModuleType('e2b'))
        with pytest.raises(E2BSandboxError, match='does not expose the expected'):
            async with E2BSandboxSession():
                pass  # pragma: no cover

    @pytest.mark.parametrize(
        ('error', 'error_type', 'match'),
        [
            (AuthenticationException('bad key'), E2BSandboxAuthError, 'valid E2B_API_KEY'),
            (SandboxNotFoundException('gone'), E2BSandboxUnavailableError, 'no longer available'),
            (SandboxException('control plane'), E2BSandboxError, 'control plane'),
        ],
    )
    async def test_create_errors_are_classified(
        self,
        fake_e2b: FakeE2B,
        error: Exception,
        error_type: type[E2BSandboxError],
        match: str,
    ) -> None:
        fake_e2b.create_error = error
        with pytest.raises(error_type, match=match):
            async with E2BSandboxSession():
                pass  # pragma: no cover

    async def test_missing_attached_id_is_named(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = SandboxNotFoundException('gone')
        with pytest.raises(E2BSandboxUnavailableError, match="'sbx-missing'"):
            async with E2BSandboxSession(sandbox_id='sbx-missing'):
                pass  # pragma: no cover


class TestExec:
    async def test_success_and_bounded_tail(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('A' * 20 + 'END', 'warning', 0)
        async with E2BSandboxSession(workdir='/project') as session:
            result = await session.exec('echo hello', timeout=7, max_output_bytes=10)
            command_call = fake_e2b.sandboxes[0].commands.calls[0]
            assert command_call.cwd == '/project'
            assert command_call.timeout == 7
        assert result.stdout == 'A' * 7 + 'END'
        assert result.stderr == 'warning'
        assert result.stdout_truncated is True
        assert result.stderr_truncated is False
        assert result.returncode == 0
        assert result.applied_timeout == 7
        assert fake_e2b.sandboxes[0].files.removed

    async def test_nonzero_exit_and_sdk_diagnostics(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('out', 'err', 9)
        fake_e2b.sdk_stdout = 'wrapper out'
        fake_e2b.sdk_stderr = 'wrapper err'
        async with E2BSandboxSession() as session:
            result = await session.exec('exit 9', timeout=3, max_output_bytes=100)
        assert result.stdout == 'out\nwrapper out'
        assert result.stderr == 'err\nwrapper err'
        assert result.returncode == 9
        assert result.timed_out is False

    async def test_timeout_kills_process_group_and_handle(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('partial', '', 0)
        fake_e2b.wait_error = TimeoutException('deadline')
        fake_e2b.invalid_count = True
        fake_e2b.kill_process_error = SandboxException('already gone')
        fake_e2b.handle_kill_error = SandboxException('already gone')
        async with E2BSandboxSession() as session:
            result = await session.exec('sleep 99', timeout=2, max_output_bytes=100)
        assert result.timed_out is True
        assert result.returncode == 124
        assert result.stdout == 'partial'
        assert fake_e2b.sandboxes[0].commands.handles[0].killed is True

    async def test_timeout_without_capture_returns_empty(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.wait_error = TimeoutException('deadline')
        fake_e2b.omit_capture = True
        async with E2BSandboxSession() as session:
            result = await session.exec('sleep 99', timeout=2, max_output_bytes=100)
        assert result.stdout == result.stderr == ''
        assert result.timed_out is True

    async def test_timeout_without_count_returns_available_tail(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('partial', '', 0)
        fake_e2b.wait_error = TimeoutException('deadline')
        fake_e2b.omit_count = True
        async with E2BSandboxSession() as session:
            result = await session.exec('sleep 99', timeout=2, max_output_bytes=100)
        assert result.stdout == 'partial'
        assert result.stdout_truncated is False

    async def test_missing_capture_on_completed_command_is_an_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.omit_capture = True
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='Could not read bounded'):
                await session.exec('true', timeout=2, max_output_bytes=100)

    async def test_missing_count_on_completed_command_is_an_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.omit_count = True
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='Could not read bounded'):
                await session.exec('true', timeout=2, max_output_bytes=100)

    async def test_run_and_wait_failures_are_wrapped(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = SandboxException('run failed')
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='Could not run.*run failed'):
                await session.exec('echo', timeout=2, max_output_bytes=100)
        fake_e2b.run_error = None
        fake_e2b.wait_error = SandboxException('wait failed')
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='Could not read.*wait failed'):
                await session.exec('echo', timeout=2, max_output_bytes=100)

    async def test_invalid_capture_count_is_wrapped(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.invalid_count = True
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='byte count was invalid'):
                await session.exec('echo', timeout=2, max_output_bytes=100)

    async def test_capture_read_failure_is_wrapped(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.read_error = SandboxException('read failed')
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='Could not read bounded.*read failed'):
                await session.exec('echo', timeout=2, max_output_bytes=100)

    async def test_temp_remove_failure_is_best_effort(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.remove_error = SandboxException('remove failed')
        async with E2BSandboxSession() as session:
            assert (await session.exec('true', timeout=2, max_output_bytes=100)).returncode == 0

    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'timeout': 0, 'max_output_bytes': 10}, 'timeout'),
            ({'timeout': 1, 'max_output_bytes': 0}, 'max_output_bytes'),
        ],
    )
    async def test_invalid_limits(self, fake_e2b: FakeE2B, kwargs: dict[str, int], match: str) -> None:
        async with E2BSandboxSession() as session:
            with pytest.raises(ValueError, match=match):
                await session.exec('true', **kwargs)

    async def test_unpaired_surrogate_command_rejected(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='cannot be encoded'):
                await session.exec('\ud800', timeout=1, max_output_bytes=10)

    async def test_exec_before_enter_rejected(self) -> None:
        with pytest.raises(E2BSandboxError, match='session is not open'):
            await E2BSandboxSession().exec('true', timeout=1, max_output_bytes=10)


class TestFiles:
    async def test_file_operations_and_relative_paths(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession(workdir='/project') as session:
            await session.write_bytes('notes.txt', b'abcdef')
            assert await session.file_size('notes.txt') == 6
            assert await session.read_bytes('notes.txt', max_bytes=6) == b'abcdef'
            fake_e2b.sandboxes[0].files.listings['/project'] = [
                FakeEntryInfo('z.txt', '/project/z.txt', FakeFileType('file'), 1),
                FakeEntryInfo('src', '/project/src', FakeFileType('dir'), 0),
                FakeEntryInfo('unknown', '/project/unknown', None, 0),
            ]
            assert await session.list_files('.') == [
                ('z.txt', False),
                ('src', True),
                ('unknown', False),
            ]

    async def test_stream_read_stops_over_limit(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession() as session:
            fake_e2b.sandboxes[0].files.files['/big'] = b'abcdefgh'
            with pytest.raises(E2BSandboxError, match='grew beyond'):
                await session.read_bytes('/big', max_bytes=5)

    async def test_invalid_stream_limit(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession() as session:
            with pytest.raises(ValueError, match='max_bytes'):
                await session.read_bytes('/x', max_bytes=0)

    async def test_missing_file_is_wrapped(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match='Could not inspect'):
                await session.file_size('/missing')

    @pytest.mark.parametrize(
        ('operation', 'match'),
        [
            ('info', 'Could not inspect'),
            ('read', 'Could not read'),
            ('write', 'Could not write'),
            ('list', 'Could not list'),
        ],
    )
    async def test_file_errors_are_wrapped(self, fake_e2b: FakeE2B, operation: str, match: str) -> None:
        error = SandboxException(f'{operation} failed')
        setattr(fake_e2b, f'{operation}_error', error)
        async with E2BSandboxSession() as session:
            with pytest.raises(E2BSandboxError, match=match):
                if operation == 'info':
                    await session.file_size('/x')
                elif operation == 'read':
                    await session.read_bytes('/x', max_bytes=10)
                elif operation == 'write':
                    await session.write_bytes('/x', b'x')
                else:
                    await session.list_files('/x')
