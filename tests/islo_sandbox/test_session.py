"""Tests for the Islo session's lifecycle, I/O, and error semantics."""

from __future__ import annotations

import builtins
import sys

import pytest

from pydantic_ai_harness.islo_sandbox import (
    IsloSandboxAuthError,
    IsloSandboxError,
    IsloSandboxSession,
    IsloSandboxUnavailableError,
)

from .fake_islo import FakeApiError, FakeExecResult, FakeIslo, FakeSandboxResponse


class TestLifecycle:
    async def test_owned_sandbox_forwards_configuration_and_deletes(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.create_response = FakeSandboxResponse(status='starting', workdir='/remote')
        fake_islo.sandboxes.ready_responses = [
            FakeSandboxResponse(status='starting', workdir='/remote'),
            FakeSandboxResponse(status='running', workdir='/ready'),
        ]

        session = IsloSandboxSession(
            image='python:3.12',
            sandbox_timeout=123,
            workdir='/work',
            env={'TOKEN': 'safe'},
            vcpus=2,
            memory_mb=4096,
            disk_gb=20,
            internet_enabled=False,
            gateway_profile='restricted',
            base_url='https://api.example.test',
            compute_url='https://compute.example.test',
            poll_interval=0.001,
        )
        assert session.sandbox_name is None
        assert session.sandbox_id is None
        async with session:
            assert session.sandbox_name == 'sandbox-owned'
            assert session.sandbox_id == 'sb-owned'
            await session.write_bytes('hello.txt', b'hello')

        assert fake_islo.client_init_calls[0]['base_url'] == 'https://api.example.test'
        assert fake_islo.client_init_calls[0]['compute_url'] == 'https://compute.example.test'
        create = fake_islo.sandboxes.create_calls[0]
        assert create['image'] == 'python:3.12'
        assert create['workdir'] == '/work'
        assert create['env'] == {'TOKEN': 'safe'}
        assert create['vcpus'] == 2
        assert create['memory_mb'] == 4096
        assert create['disk_gb'] == 20
        assert create['internet_enabled'] is False
        assert create['gateway_profile'] == 'restricted'
        assert create['lifecycle'].delete_after == 123  # type: ignore[union-attr]
        assert fake_islo.sandboxes.upload_calls[0][1] == '/ready/hello.txt'
        assert fake_islo.sandboxes.delete_calls == ['sandbox-owned']
        assert session.sandbox_name is None

    async def test_attach_does_not_create_or_delete(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.attach_response = FakeSandboxResponse(
            name='existing', id='sb-existing', status='running', workdir='/attached'
        )
        injected_client = fake_islo.module.AsyncIslo()  # type: ignore[attr-defined]

        async with IsloSandboxSession(sandbox_name='existing', client=injected_client) as session:
            assert session.sandbox_id == 'sb-existing'
            await session.write_bytes('file.txt', b'data')

        assert fake_islo.sandboxes.get_calls == ['existing']
        assert fake_islo.sandboxes.create_calls == []
        assert fake_islo.sandboxes.delete_calls == []
        assert fake_islo.sandboxes.upload_calls[0][1] == '/attached/file.txt'

    async def test_exit_without_enter_is_safe(self) -> None:
        await IsloSandboxSession().__aexit__(None, None, None)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'sandbox_timeout': 0},
            {'sandbox_timeout': True},
            {'poll_interval': 0},
            {'poll_interval': True},
            {'poll_interval': float('inf')},
        ],
    )
    def test_invalid_lifecycle_configuration(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            IsloSandboxSession(**kwargs)  # type: ignore[arg-type]

    async def test_missing_remote_and_configured_workdir_falls_back(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.create_response = FakeSandboxResponse(workdir=None)
        async with IsloSandboxSession(workdir=None) as session:
            await session.write_bytes('file', b'data')
        assert fake_islo.sandboxes.upload_calls[0][1] == '/workspace/file'
        assert set(fake_islo.sandboxes.create_calls[0]) == {'image', 'lifecycle'}

    async def test_second_enter_is_rejected(self, fake_islo: FakeIslo) -> None:
        async with IsloSandboxSession() as session:
            with pytest.raises(IsloSandboxError, match='already open'):
                await session.__aenter__()
        assert fake_islo.sandboxes.delete_calls == ['sandbox-owned']

    async def test_terminal_status_during_creation_is_cleaned_up(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.create_response = FakeSandboxResponse(status='failed')
        with pytest.raises(IsloSandboxUnavailableError, match="'failed'"):
            async with IsloSandboxSession():
                pass  # pragma: no cover
        assert fake_islo.sandboxes.delete_calls == ['sandbox-owned']

    @pytest.mark.parametrize('status', ['deleted', 'failed', 'stopped'])
    async def test_terminal_attach_status_is_unavailable(self, fake_islo: FakeIslo, status: str) -> None:
        fake_islo.sandboxes.attach_response = FakeSandboxResponse(name='existing', status=status)
        with pytest.raises(IsloSandboxUnavailableError, match=status):
            async with IsloSandboxSession(sandbox_name='existing'):
                pass  # pragma: no cover

    async def test_cleanup_404_counts_as_success(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.delete_errors = [FakeApiError(404, {'message': 'gone'})]
        session = IsloSandboxSession()
        async with session:
            pass
        assert session.sandbox_name is None

    async def test_cleanup_failure_warns_and_can_be_retried(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.delete_errors = [FakeApiError(500, {'message': 'retry me'})]
        session = IsloSandboxSession()
        with pytest.warns(RuntimeWarning, match='retained for cleanup retry'):
            async with session:
                pass
        assert session.sandbox_name == 'sandbox-owned'
        await session.close()
        assert fake_islo.sandboxes.delete_calls == ['sandbox-owned', 'sandbox-owned']
        assert session.sandbox_name is None


class TestExec:
    async def test_polls_and_normalizes_completed_result(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: [
            FakeExecResult(status='running', stdout='partial'),
            FakeExecResult(status='completed', stdout='complete', stderr='warn', exit_code=7),
        ]
        async with IsloSandboxSession(poll_interval=0.001) as session:
            result = await session.exec(['python', '-V'], timeout=1.2)

        assert result.stdout == 'complete'
        assert result.stderr == 'warn'
        assert result.returncode == 7
        assert result.status == 'completed'
        assert result.timed_out is False
        assert result.applied_timeout == 2
        call = fake_islo.sandboxes.exec_calls[0]
        assert call.command == ['python', '-V']
        assert call.timeout_secs == 2

    async def test_provider_timeout_and_missing_exit_code(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(
            status='timeout', stdout=None, stderr=None, exit_code=None
        )
        async with IsloSandboxSession() as session:
            result = await session.exec(['sleep', '10'], timeout=1)
        assert result.returncode == -1
        assert result.timed_out is True
        assert result.remote_may_be_running is False

    async def test_byte_tail_and_provider_truncation_are_preserved(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(
            stdout='prefix-😀-tail', stderr='server-cut', truncated=True
        )
        async with IsloSandboxSession() as session:
            result = await session.exec(['x'], timeout=1, max_output_bytes=6)
        assert result.stdout == '-tail'
        assert result.stderr == 'er-cut'
        assert result.stdout_truncated is True
        assert result.stderr_truncated is True

    async def test_client_timeout_keeps_last_output_and_warns_remote_may_run(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(status='running', stdout='still here')
        async with IsloSandboxSession(poll_interval=0.001) as session:
            result = await session.exec(['sleep', '99'], timeout=0.003)
        assert result.status == 'client_timeout'
        assert result.stdout == 'still here'
        assert result.returncode == -1
        assert result.timed_out is True
        assert result.remote_may_be_running is True

    @pytest.mark.parametrize(
        ('kwargs', 'error'),
        [
            ({'argv': 'echo hi', 'timeout': 1}, TypeError),
            ({'argv': ['echo'], 'timeout': 0}, ValueError),
            ({'argv': ['echo'], 'timeout': float('inf')}, ValueError),
            ({'argv': ['echo'], 'timeout': 1, 'max_output_bytes': 0}, ValueError),
            ({'argv': ['echo'], 'timeout': 1, 'max_output_bytes': True}, ValueError),
        ],
    )
    async def test_argument_validation(self, kwargs: dict[str, object], error: type[Exception]) -> None:
        session = IsloSandboxSession()
        with pytest.raises(error):
            await session.exec(**kwargs)  # type: ignore[arg-type]

    async def test_closed_session_rejects_exec(self) -> None:
        with pytest.raises(IsloSandboxError, match='not open'):
            await IsloSandboxSession().exec(['true'], timeout=1)


class TestFiles:
    async def test_read_write_and_list_resolve_against_remote_workdir(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.create_response = FakeSandboxResponse(workdir='/repo')
        fake_islo.put_file('/repo/a.txt', b'alpha')
        fake_islo.put_file('/repo/.hidden', b'hidden')
        fake_islo.put_file('/repo/dir/nested.txt', b'nested')

        async with IsloSandboxSession() as session:
            assert await session.read_bytes('a.txt', max_bytes=100) == b'alpha'
            await session.write_bytes('/absolute.bin', b'\x00\x01')
            entries = await session.list_files('.')

        assert fake_islo.sandboxes.download_calls == [('sandbox-owned', '/repo/a.txt')]
        assert fake_islo.sandboxes.files['/absolute.bin'] == b'\x00\x01'
        assert set(entries) == {('a.txt', False), ('.hidden', False), ('dir', True)}

    async def test_read_limit_uses_sentinel_byte(self, fake_islo: FakeIslo) -> None:
        fake_islo.put_file('/workspace/large', b'123456')
        async with IsloSandboxSession() as session:
            with pytest.raises(IsloSandboxError, match='5-byte read limit'):
                await session.read_bytes('large', max_bytes=5)

    @pytest.mark.parametrize('max_bytes', [0, -1, True])
    async def test_read_limit_validation(self, max_bytes: int) -> None:
        with pytest.raises(ValueError, match='positive integer'):
            await IsloSandboxSession().read_bytes('x', max_bytes=max_bytes)

    async def test_listing_reports_timeout_exit_parse_and_truncation(self, fake_islo: FakeIslo) -> None:
        async with IsloSandboxSession() as session:
            fake_islo.sandboxes._directory_result = lambda target: FakeExecResult(status='timeout')  # type: ignore[method-assign]
            with pytest.raises(IsloSandboxError, match='timed out'):
                await session.list_files('.')

            fake_islo.sandboxes._directory_result = lambda target: FakeExecResult(  # type: ignore[method-assign]
                stderr='permission denied', exit_code=2
            )
            with pytest.raises(IsloSandboxError, match='permission denied'):
                await session.list_files('.')

            fake_islo.sandboxes._directory_result = lambda target: FakeExecResult(  # type: ignore[method-assign]
                stdout='f\tok\n', truncated=True
            )
            with pytest.raises(IsloSandboxError, match='provider output limit'):
                await session.list_files('.')

            fake_islo.sandboxes._directory_result = lambda target: FakeExecResult(  # type: ignore[method-assign]
                stdout='bad line\n'
            )
            with pytest.raises(IsloSandboxError, match='Could not parse'):
                await session.list_files('.')


class TestErrors:
    async def test_missing_dependency_is_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == 'islo':
                raise ImportError('No module named islo')
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.delitem(sys.modules, 'islo', raising=False)
        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(IsloSandboxError, match='package is required'):
            async with IsloSandboxSession():
                pass  # pragma: no cover

    @pytest.mark.parametrize('status', [401, 403])
    async def test_auth_errors_are_terminal(self, fake_islo: FakeIslo, status: int) -> None:
        fake_islo.sandboxes.create_error = FakeApiError(status, {'message': 'secret omitted'})
        with pytest.raises(IsloSandboxAuthError, match='ISLO_API_KEY'):
            async with IsloSandboxSession():
                pass  # pragma: no cover

    async def test_attach_404_is_unavailable(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.get_error = FakeApiError(404, {'message': 'missing'})
        with pytest.raises(IsloSandboxUnavailableError, match='not found'):
            async with IsloSandboxSession(sandbox_name='missing'):
                pass  # pragma: no cover

    async def test_file_404_is_recoverable_generic_error(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.download_error = FakeApiError(404, {'message': 'file missing'})
        async with IsloSandboxSession() as session:
            with pytest.raises(IsloSandboxError, match='HTTP 404') as exc:
                await session.read_bytes('missing', max_bytes=10)
        assert not isinstance(exc.value, IsloSandboxUnavailableError)

    async def test_generic_api_and_transport_errors_include_context(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.exec_error = FakeApiError(429, {'message': 'slow down'})
        async with IsloSandboxSession() as session:
            with pytest.raises(IsloSandboxError, match='HTTP 429'):
                await session.exec(['true'], timeout=1)

            fake_islo.sandboxes.exec_error = RuntimeError('network down')
            with pytest.raises(IsloSandboxError, match='RuntimeError: network down'):
                await session.exec(['true'], timeout=1)

    async def test_poll_error_is_wrapped(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.poll_error = RuntimeError('poll failed')
        async with IsloSandboxSession() as session:
            with pytest.raises(IsloSandboxError, match='poll failed'):
                await session.exec(['true'], timeout=1)

    def test_error_mapping_survives_missing_sdk_submodule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == 'islo.core.api_error':
                raise ImportError('missing error module')
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        assert fake_import('builtins') is builtins
        monkeypatch.delitem(sys.modules, 'islo.core.api_error', raising=False)
        monkeypatch.setattr(builtins, '__import__', fake_import)
        mapped = IsloSandboxSession._map_error(RuntimeError('boom'), 'context', unavailable_on_404=False)
        assert 'package is required' in str(mapped)
