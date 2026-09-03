"""Tests for SpriteSandboxSession."""

from __future__ import annotations

import builtins
import logging
from typing import Any

import anyio
import pytest

from pydantic_ai_harness.sprites import (
    SpriteSandboxAuthError,
    SpriteSandboxError,
    SpriteSandboxOwnershipError,
    SpriteSandboxSession,
    SpriteSandboxUnavailableError,
)

from .fake_sprites import (
    FakeAuthenticationError,
    FakeExecResult,
    FakeNotFoundError,
    FakeSpriteError,
    FakeSprites,
    FakeTimeoutError,
)


class TestConfiguration:
    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'base_url': ''}, 'base_url'),
            ({'api_timeout': 0}, 'api_timeout'),
            ({'api_timeout': True}, 'api_timeout'),
            ({'api_timeout': float('inf')}, 'api_timeout'),
            ({'runtime': 'nightly'}, 'runtime'),
            ({'sprite_name': 'kept', 'runtime': 'dev'}, 'runtime only applies'),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SpriteSandboxSession(**kwargs)

    async def test_requires_token(self, monkeypatch: pytest.MonkeyPatch, fake_sprites: FakeSprites) -> None:
        monkeypatch.delenv('SPRITE_TOKEN', raising=False)
        with pytest.raises(SpriteSandboxAuthError, match='SPRITE_TOKEN is not set'):
            await SpriteSandboxSession().__aenter__()
        assert fake_sprites.clients == []

    async def test_exit_before_enter_is_safe(self) -> None:
        await SpriteSandboxSession(token='token').__aexit__(None, None, None)

    async def test_reports_missing_optional_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def missing_sprites(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == 'sprites':
                raise ImportError('not installed')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', missing_sprites)
        with pytest.raises(SpriteSandboxError, match="'sprites-py' package is required"):
            await SpriteSandboxSession(token='token').__aenter__()


class TestLifecycle:
    async def test_creates_labelled_sprite_and_destroys_it(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(
            token='secret', base_url='https://example.test', api_timeout=12, runtime='dev'
        ) as session:
            name = session.sprite_name
            assert name is not None
            assert name.startswith('pydantic-ai-')
            assert session.is_open
            sprite = fake_sprites.sprites[name]
            assert sprite.run_calls[0].argv == ('pwd',)

        create = fake_sprites.create_calls[0]
        assert create['runtime'] == 'dev'
        assert create['labels'][0] == 'pydantic-ai-harness'
        assert fake_sprites.destroy_calls == [name]
        assert fake_sprites.close_calls == 1
        assert not session.is_open
        client = fake_sprites.clients[0]
        assert (client.token, client.base_url, client.timeout) == ('secret', 'https://example.test', 12)

    async def test_attaches_and_leaves_sprite_running(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token', sprite_name='kept', workdir='/app')
        assert session.sprite_name == 'kept'
        fake_sprites.add_sprite('kept')
        async with session:
            assert session.sprite_name == 'kept'
            assert fake_sprites.sprites['kept'].run_calls == []
        assert 'kept' in fake_sprites.sprites
        assert fake_sprites.destroy_calls == []
        assert fake_sprites.close_calls == 1

    async def test_cannot_enter_twice(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(token='token') as session:
            with pytest.raises(SpriteSandboxError, match='already open'):
                await session.__aenter__()
        assert len(fake_sprites.create_calls) == 1

    async def test_maps_auth_failure(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.create_error = FakeAuthenticationError('bad token')
        with pytest.raises(SpriteSandboxAuthError, match='rejected the credentials'):
            async with SpriteSandboxSession(token='bad'):
                pass
        assert fake_sprites.close_calls == 1

    async def test_maps_missing_attached_sprite(self, fake_sprites: FakeSprites) -> None:
        with pytest.raises(SpriteSandboxUnavailableError, match='does not exist'):
            async with SpriteSandboxSession(token='token', sprite_name='gone'):
                pass
        assert fake_sprites.close_calls == 1

    async def test_missing_ownership_label_is_destroyed_and_rejected(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.retain_labels = False
        with pytest.raises(SpriteSandboxOwnershipError, match='did not retain'):
            async with SpriteSandboxSession(token='token'):
                pass
        assert len(fake_sprites.destroy_calls) == 1
        assert fake_sprites.close_calls == 1

    async def test_failed_cwd_probe_destroys_owned_sprite(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.pwd_result = FakeExecResult(returncode=1)
        with pytest.raises(SpriteSandboxError, match='working directory'):
            async with SpriteSandboxSession(token='token'):
                pass
        assert len(fake_sprites.destroy_calls) == 1
        assert fake_sprites.close_calls == 1

    async def test_cwd_probe_removes_only_pwd_record_newline(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.pwd_result = FakeExecResult(stdout=b'/workspace \n\n')
        async with SpriteSandboxSession(token='token') as session:
            await session.exec('pwd', timeout=1, max_output_bytes=100)
            call = fake_sprites.sprites[session.sprite_name or ''].run_calls[-1]
        assert call.cwd == '/workspace \n'

    async def test_failed_enter_reports_cleanup_failure(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.pwd_result = FakeExecResult(returncode=1)
        fake_sprites.destroy_error = FakeSpriteError('destroy failed')
        with pytest.raises(SpriteSandboxError, match='Cleanup also failed: Could not destroy'):
            async with SpriteSandboxSession(token='token'):
                pass

    async def test_cancel_after_create_cleans_up(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        with anyio.CancelScope() as scope:
            scope.cancel()
            await session.__aenter__()
        assert len(fake_sprites.destroy_calls) == 1
        assert not session.is_open

    async def test_cancelled_exit_still_cleans_up(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        with anyio.CancelScope() as scope:
            scope.cancel()
            await session.__aexit__(None, None, None)
        assert len(fake_sprites.destroy_calls) == 1
        assert not session.is_open

    async def test_owned_sprite_already_gone_is_clean_exit(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        fake_sprites.get_error = FakeNotFoundError('already gone')
        await session.__aexit__(None, None, None)
        assert not session.is_open

    async def test_destroy_not_found_is_clean_exit(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        fake_sprites.destroy_error = FakeNotFoundError('already gone')
        await session.__aexit__(None, None, None)
        assert not session.is_open

    async def test_cleanup_verification_error_is_reported(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        fake_sprites.get_error = FakeSpriteError('verification failed')
        with pytest.raises(SpriteSandboxError, match='Could not verify owned Sprite'):
            await session.__aexit__(None, None, None)
        assert session.is_open
        assert fake_sprites.close_calls == 0

    async def test_client_close_error_is_reported(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        fake_sprites.close_error = FakeSpriteError('close failed')
        with pytest.raises(FakeSpriteError, match='close failed'):
            await session.__aexit__(None, None, None)
        assert session.is_open
        fake_sprites.close_error = None
        await session.__aexit__(None, None, None)
        assert not session.is_open

    async def test_generic_open_failure_is_mapped(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.create_error = RuntimeError('unexpected create failure')
        with pytest.raises(SpriteSandboxError, match='Could not start Sprite sandbox'):
            async with SpriteSandboxSession(token='token'):
                pass

    async def test_refuses_to_destroy_replaced_sprite(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        name = session.sprite_name
        assert name is not None
        fake_sprites.sprites[name].labels = []
        with pytest.raises(SpriteSandboxOwnershipError, match='Refusing to destroy'):
            await session.__aexit__(None, None, None)
        assert fake_sprites.destroy_calls == []
        assert fake_sprites.close_calls == 0
        assert session.is_open
        fake_sprites.sprites[name].labels = [fake_sprites.create_calls[-1]['labels'][-1]]
        await session.__aexit__(None, None, None)
        assert not session.is_open

    async def test_failed_destroy_can_be_retried(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        fake_sprites.destroy_error = FakeSpriteError('control plane unavailable')
        with pytest.raises(SpriteSandboxError, match='Could not destroy'):
            await session.__aexit__(None, None, None)
        assert session.is_open
        assert fake_sprites.close_calls == 0

        fake_sprites.destroy_error = None
        await session.__aexit__(None, None, None)
        assert not session.is_open
        assert len(fake_sprites.destroy_calls) == 2

    async def test_cleanup_error_does_not_mask_body_error(
        self, fake_sprites: FakeSprites, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_sprites.destroy_error = FakeSpriteError('control plane unavailable')
        caplog.set_level(logging.ERROR)
        with pytest.raises(RuntimeError, match='body failed'):
            async with SpriteSandboxSession(token='token'):
                raise RuntimeError('body failed')
        assert 'Failed to clean up Sprite' in caplog.text


class TestOperations:
    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'timeout': 0, 'max_output_bytes': 100}, 'timeout'),
            ({'timeout': True, 'max_output_bytes': 100}, 'timeout'),
            ({'timeout': float('inf'), 'max_output_bytes': 100}, 'timeout'),
            ({'timeout': 1, 'max_output_bytes': 0}, 'max_output_bytes'),
            ({'timeout': 1, 'max_output_bytes': True}, 'max_output_bytes'),
        ],
    )
    async def test_exec_rejects_invalid_limits(
        self, fake_sprites: FakeSprites, kwargs: dict[str, Any], match: str
    ) -> None:
        async with SpriteSandboxSession(token='token') as session:
            with pytest.raises(ValueError, match=match):
                await session.exec('echo', **kwargs)
            assert len(fake_sprites.sprites[session.sprite_name or ''].run_calls) == 1

    async def test_exec_passes_bounded_wrapper_and_context(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.responder = lambda call: FakeExecResult(stdout=b'hello\n', returncode=3)
        async with SpriteSandboxSession(token='token', workdir='/app') as session:
            result = await session.exec('echo hello', timeout=4.5, max_output_bytes=100)
            call = fake_sprites.sprites[session.sprite_name or ''].run_calls[-1]
        assert result.output == 'hello\n'
        assert result.returncode == 3
        assert result.applied_timeout == 4.5
        assert call.argv[:4] == ('python3', '-I', '-S', '-c')
        assert "['bash', '-c', command]" in call.argv[4]
        assert "['bash', '-lc', command]" not in call.argv[4]
        assert call.capture_output is False
        assert call.timeout == 9.5
        assert call.cwd == '/app'
        assert call.env == {
            'PYDANTIC_AI_SPRITE_COMMAND': 'echo hello',
            'PYDANTIC_AI_SPRITE_TIMEOUT_SECONDS': '4.5',
        }

    async def test_timeout_control_is_reported_separately_from_output(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.responder = lambda call: FakeExecResult(stdout=b'partial', stderr=b'\0\1', returncode=124)
        async with SpriteSandboxSession(token='token') as session:
            result = await session.exec('sleep 99', timeout=2, max_output_bytes=100)
        assert result.output == 'partial'
        assert result.timed_out is True
        assert result.applied_timeout == 2

    async def test_transport_timeout_is_reported(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(token='token') as session:
            fake_sprites.run_error = FakeTimeoutError('transport deadline')
            result = await session.exec('sleep 99', timeout=2, max_output_bytes=100)
        assert result.returncode == 124
        assert result.timed_out is True

    async def test_truncation_marker_is_detected(self, fake_sprites: FakeSprites) -> None:
        marker = b'\n[... Sprite command output truncated ...]\n'
        fake_sprites.responder = lambda call: FakeExecResult(stdout=b'head' + marker + b'tail', stderr=b'\1\0')
        async with SpriteSandboxSession(token='token') as session:
            result = await session.exec('big', timeout=None, max_output_bytes=100)
            call = fake_sprites.sprites[session.sprite_name or ''].run_calls[-1]
        assert result.truncated is True
        assert call.timeout is None
        assert call.env == {'PYDANTIC_AI_SPRITE_COMMAND': 'big'}

    async def test_timeout_text_in_command_output_is_not_a_status_signal(self, fake_sprites: FakeSprites) -> None:
        marker = b'\n__PYDANTIC_AI_SPRITE_TIMEOUT__\n'
        fake_sprites.responder = lambda call: FakeExecResult(stdout=marker, stderr=b'\0\0')
        async with SpriteSandboxSession(token='token') as session:
            result = await session.exec('printf marker', timeout=2, max_output_bytes=100)
        assert result.output == marker.decode()
        assert result.timed_out is False

    async def test_decoding_replacement_cannot_expand_past_byte_limit(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.responder = lambda call: FakeExecResult(stdout=b'\xff', stderr=b'\0\0')
        async with SpriteSandboxSession(token='token') as session:
            result = await session.exec('printf invalid', timeout=2, max_output_bytes=1)
        assert len(result.output.encode()) <= 1
        assert result.truncated is True

    async def test_sdk_stream_is_aborted_when_helper_exceeds_transport_limit(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.responder = lambda call: FakeExecResult(stdout=b'unbounded')
        async with SpriteSandboxSession(token='token') as session:
            with pytest.raises(SpriteSandboxError, match='transport limit'):
                await session.exec('big', timeout=2, max_output_bytes=2)

    async def test_file_roundtrip_and_listing(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(token='token') as session:
            await session.write_bytes('sub/file.txt', b'hello')
            assert await session.read_bytes('sub/file.txt', max_bytes=5) == b'hello'
            listing = await session.list_files('.', max_entries=10, max_output_bytes=100)
            assert listing.entries == [('sub', True)]
            assert listing.truncated is False

    async def test_directory_transport_metadata_counts_toward_byte_limit(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(token='token') as session:
            await session.write_bytes('a', b'x')
            listing = await session.list_files('.', max_entries=10, max_output_bytes=1)
        assert listing.entries == []
        assert listing.truncated is True

    async def test_transient_sdk_failure_is_recoverable(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(token='token') as session:
            fake_sprites.filesystem_error = FakeSpriteError('temporary failure')
            with pytest.raises(SpriteSandboxError, match='temporary failure') as exc_info:
                await session.read_bytes('x', max_bytes=10)
        assert not isinstance(exc_info.value, SpriteSandboxUnavailableError)

    async def test_operation_detects_destroyed_sprite(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.add_sprite('gone')
        async with SpriteSandboxSession(token='token', sprite_name='gone') as session:
            fake_sprites.filesystem_error = FakeSpriteError('request failed')
            fake_sprites.get_error = FakeNotFoundError('gone')
            with pytest.raises(SpriteSandboxUnavailableError, match='no longer exists'):
                await session.read_bytes('x', max_bytes=10)

    async def test_operation_detects_expired_auth(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.add_sprite('kept')
        async with SpriteSandboxSession(token='token', sprite_name='kept') as session:
            fake_sprites.filesystem_error = FakeSpriteError('request failed')
            fake_sprites.get_error = FakeAuthenticationError('expired')
            with pytest.raises(SpriteSandboxAuthError, match='rejected the credentials'):
                await session.read_bytes('x', max_bytes=10)

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (FakeAuthenticationError('expired'), SpriteSandboxAuthError),
            (FakeNotFoundError('gone'), SpriteSandboxUnavailableError),
        ],
    )
    async def test_direct_terminal_operation_error(
        self, fake_sprites: FakeSprites, error: Exception, expected: type[Exception]
    ) -> None:
        async with SpriteSandboxSession(token='token') as session:
            fake_sprites.filesystem_error = error
            with pytest.raises(expected):
                await session.read_bytes('x', max_bytes=10)

    async def test_non_sdk_operation_error_is_recoverable(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(token='token') as session:
            fake_sprites.filesystem_error = RuntimeError('unexpected')
            with pytest.raises(SpriteSandboxError, match='unexpected'):
                await session.read_bytes('x', max_bytes=10)

    async def test_probe_failure_preserves_original_operation_error(self, fake_sprites: FakeSprites) -> None:
        session = SpriteSandboxSession(token='token')
        await session.__aenter__()
        try:
            fake_sprites.filesystem_error = FakeSpriteError('disk issue')
            fake_sprites.get_error = FakeSpriteError('probe issue')
            with pytest.raises(SpriteSandboxError, match='disk issue'):
                await session.read_bytes('x', max_bytes=10)
        finally:
            fake_sprites.get_error = None
            await session.__aexit__(None, None, None)

    async def test_unopened_operations_raise(self) -> None:
        with pytest.raises(SpriteSandboxError, match='not open'):
            await SpriteSandboxSession(token='token').read_bytes('x', max_bytes=10)
