"""Tests for the public SpriteSandbox capability and toolset."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Protocol, TypeGuard, runtime_checkable

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

import pydantic_ai_harness
from pydantic_ai_harness.sprites import (
    SpriteSandbox,
    SpriteSandboxAuthError,
    SpriteSandboxError,
    SpriteSandboxSession,
)
from pydantic_ai_harness.sprites._toolset import SpriteSandboxToolset

from .fake_sprites import FakeAuthenticationError, FakeExecResult, FakeSpriteError, FakeSprites


@runtime_checkable
class _SpriteTools(Protocol):
    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str: ...
    async def read_file(self, path: str, *, offset: int | None = None, limit: int | None = None) -> str: ...
    async def write_file(self, path: str, content: str) -> str: ...
    async def list_directory(self, path: str = '.') -> str: ...


def _is_toolset(value: object) -> TypeGuard[AbstractToolset[None]]:
    return isinstance(value, AbstractToolset)


def _context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)


@asynccontextmanager
async def _toolset(**kwargs: Any) -> AsyncGenerator[_SpriteTools]:
    if 'session' not in kwargs:
        kwargs['token'] = 'token'
    toolset = SpriteSandbox[None](**kwargs).get_toolset()
    assert _is_toolset(toolset)
    run_toolset = await toolset.for_run(_context())
    assert isinstance(run_toolset, _SpriteTools)
    async with run_toolset:
        yield run_toolset


class TestCapability:
    def test_public_exports(self) -> None:
        assert pydantic_ai_harness.SpriteSandbox is SpriteSandbox

    def test_default_and_reused_instructions(self) -> None:
        owned = SpriteSandbox[None]().get_instructions()
        reused = SpriteSandbox[None](sprite_name='kept').get_instructions()
        assert owned is not None and 'destroyed when the run ends' in owned
        assert reused is not None and 'reused across runs' in reused
        assert SpriteSandbox[None](instructions='custom').get_instructions() == 'custom'
        assert SpriteSandbox[None](instructions='').get_instructions() is None

    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'max_output_bytes': 0}, 'max_output_bytes'),
            ({'max_output_lines': True}, 'max_output_lines'),
            ({'max_read_bytes': -1}, 'max_read_bytes'),
            ({'api_timeout': float('nan')}, 'api_timeout'),
            ({'default_command_timeout': 0}, 'default_command_timeout'),
            ({'max_command_timeout': float('inf')}, 'max_command_timeout'),
            ({'default_command_timeout': 10, 'max_command_timeout': 5}, 'cannot exceed'),
            ({'base_url': ''}, 'base_url'),
            ({'runtime': 'nightly'}, 'runtime'),
            ({'sprite_name': 'kept', 'runtime': 'dev'}, 'runtime only applies'),
            ({'instructions': 1}, 'instructions'),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SpriteSandbox(**kwargs)

    @pytest.mark.parametrize('field', ['token', 'sprite_name', 'base_url', 'api_timeout', 'runtime', 'workdir'])
    def test_injected_session_rejects_connection_settings(self, field: str) -> None:
        defaults: dict[str, Any] = {
            'token': 'token',
            'sprite_name': 'kept',
            'base_url': 'https://example.test',
            'api_timeout': 2,
            'runtime': 'dev',
            'workdir': '/app',
        }
        with pytest.raises(ValueError, match=field):
            SpriteSandbox(session=SpriteSandboxSession(token='token'), **{field: defaults[field]})


class TestTools:
    async def test_base_toolset_enter_is_a_noop(self, fake_sprites: FakeSprites) -> None:
        toolset = SpriteSandbox[None](token='token').get_toolset()
        assert _is_toolset(toolset)
        async with toolset:
            pass
        assert fake_sprites.clients == []

    async def test_runs_command_and_reports_exit(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.responder = lambda call: FakeExecResult(stdout=b'failed\n', returncode=2)
        async with _toolset(default_command_timeout=10, max_command_timeout=20) as tools:
            result = await tools.run_command('false')
            call = next(iter(fake_sprites.sprites.values())).run_calls[-1]
        assert result == 'failed\n[exit code: 2]'
        assert call.env is not None and call.env['PYDANTIC_AI_SPRITE_TIMEOUT_SECONDS'] == '10'

    async def test_reports_no_output_and_timeout(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.responder = lambda call: FakeExecResult(stderr=b'\0\1', returncode=124)
        async with _toolset() as tools:
            assert await tools.run_command('sleep', timeout_seconds=3) == '(no output)\n[timed out after 3s]'

    @pytest.mark.parametrize('timeout', [0, -1, float('inf'), True])
    async def test_rejects_invalid_tool_timeout(self, fake_sprites: FakeSprites, timeout: float) -> None:
        async with _toolset() as tools:
            with pytest.raises(ModelRetry, match='must be greater than 0'):
                await tools.run_command('echo', timeout_seconds=timeout)

    async def test_clamps_timeout(self, fake_sprites: FakeSprites) -> None:
        async with _toolset(default_command_timeout=2, max_command_timeout=4) as tools:
            await tools.run_command('echo', timeout_seconds=99)
            call = next(iter(fake_sprites.sprites.values())).run_calls[-1]
        assert call.env is not None and call.env['PYDANTIC_AI_SPRITE_TIMEOUT_SECONDS'] == '4'

    async def test_truncates_lines_and_marks_session_cut(self, fake_sprites: FakeSprites) -> None:
        marker = b'\n[... Sprite command output truncated ...]\n'
        fake_sprites.responder = lambda call: FakeExecResult(
            stdout=b'one\ntwo\nthree' + marker + b'end', stderr=b'\1\0'
        )
        async with _toolset(max_output_lines=1) as tools:
            result = await tools.run_command('big')
        assert result.endswith('end')

    async def test_final_command_result_honors_tiny_caps(self, fake_sprites: FakeSprites) -> None:
        fake_sprites.responder = lambda call: FakeExecResult(stdout=b'x', stderr=b'\1\0', returncode=2)
        async with _toolset(max_output_bytes=1, max_output_lines=1) as tools:
            result = await tools.run_command('big')
        assert len(result.encode()) <= 1
        assert len(result.splitlines()) <= 1

    async def test_sdk_error_becomes_model_retry(self, fake_sprites: FakeSprites) -> None:
        async with _toolset() as tools:
            fake_sprites.run_error = FakeSpriteError('temporary')
            with pytest.raises(ModelRetry, match='temporary'):
                await tools.run_command('echo')

    async def test_terminal_command_error_propagates(self, fake_sprites: FakeSprites) -> None:
        async with _toolset() as tools:
            fake_sprites.run_error = FakeAuthenticationError('expired')
            with pytest.raises(SpriteSandboxAuthError):
                await tools.run_command('echo')

    async def test_file_tools(self, fake_sprites: FakeSprites) -> None:
        async with _toolset(max_output_lines=4) as tools:
            assert await tools.write_file('dir/file.txt', 'one\ntwo\nthree') == "Wrote 13 bytes to 'dir/file.txt'."
            assert await tools.read_file('dir/file.txt', offset=2, limit=1) == (
                'two\n\n[1 more lines in file. Use offset=3 to continue.]'
            )
            assert await tools.list_directory('.') == 'dir/'

    async def test_read_limit_is_enforced_during_the_read(self, fake_sprites: FakeSprites) -> None:
        async with _toolset(max_read_bytes=10) as tools:
            await tools.write_file('growing.txt', 'x' * 100)
            with pytest.raises(ModelRetry, match='exceeded the 10-byte read limit'):
                await tools.read_file('growing.txt')

    async def test_directory_listing_is_bounded_before_transfer(self, fake_sprites: FakeSprites) -> None:
        async with _toolset(max_output_lines=2, max_output_bytes=100) as tools:
            for name in ('a', 'b', 'c'):
                await tools.write_file(name, name)
            result = await tools.list_directory('.')
        assert result.startswith('[... directory listing truncated ...]')
        assert len(result.encode()) <= 100
        assert len(result.splitlines()) <= 2

    async def test_empty_directory(self, fake_sprites: FakeSprites) -> None:
        async with _toolset() as tools:
            assert await tools.list_directory('/empty') == '(empty)'

    async def test_file_tool_results_honor_tiny_final_caps(self, fake_sprites: FakeSprites) -> None:
        async with _toolset(max_output_bytes=1, max_output_lines=1) as tools:
            write_result = await tools.write_file('multi.txt', 'one\ntwo')
            read_result = await tools.read_file('multi.txt')
            empty_result = await tools.list_directory('/empty')
        for result in (write_result, read_result, empty_result):
            assert len(result.encode()) <= 1
            assert len(result.splitlines()) <= 1

    async def test_directory_entry_delimiters_are_escaped(self, fake_sprites: FakeSprites) -> None:
        async with _toolset() as tools:
            await tools.write_file('old\nreport.txt', 'content')
            result = await tools.list_directory('.')
        assert result == r'old\nreport.txt'
        assert len(result.splitlines()) == 1

    async def test_file_error_becomes_model_retry(self, fake_sprites: FakeSprites) -> None:
        async with _toolset() as tools:
            fake_sprites.filesystem_error = FakeSpriteError('disk issue')
            with pytest.raises(ModelRetry, match="Could not read 'x': disk issue"):
                await tools.read_file('x')
            with pytest.raises(ModelRetry, match="Could not write 'x': disk issue"):
                await tools.write_file('x', 'a')
            with pytest.raises(ModelRetry, match="Could not list 'x': disk issue"):
                await tools.list_directory('x')

    @pytest.mark.parametrize('operation', ['read', 'write', 'list'])
    async def test_terminal_file_error_propagates(self, fake_sprites: FakeSprites, operation: str) -> None:
        async with _toolset() as tools:
            fake_sprites.filesystem_error = FakeAuthenticationError('expired')
            with pytest.raises(SpriteSandboxAuthError):
                if operation == 'read':
                    await tools.read_file('x')
                elif operation == 'write':
                    await tools.write_file('x', 'content')
                else:
                    await tools.list_directory('x')

    async def test_unencodable_write_is_retryable(self, fake_sprites: FakeSprites) -> None:
        async with _toolset() as tools:
            with pytest.raises(ModelRetry, match='cannot be encoded'):
                await tools.write_file('x', '\ud800')

    async def test_tool_on_unentered_toolset_raises(self) -> None:
        toolset = SpriteSandbox[None]().get_toolset()
        assert isinstance(toolset, _SpriteTools)
        with pytest.raises(SpriteSandboxError, match='not open'):
            await toolset.run_command('echo')

    async def test_injected_session_is_reused(self, fake_sprites: FakeSprites) -> None:
        async with SpriteSandboxSession(token='token') as session:
            name = session.sprite_name
            async with _toolset(session=session) as tools:
                await tools.run_command('echo')
            assert session.is_open
            assert name in fake_sprites.sprites
        assert name not in fake_sprites.sprites

    async def test_unopened_injected_session_is_rejected(self) -> None:
        toolset = SpriteSandbox[None](session=SpriteSandboxSession(token='token')).get_toolset()
        assert _is_toolset(toolset)
        run_toolset = await toolset.for_run(_context())
        with pytest.raises(SpriteSandboxError, match='not open'):
            await run_toolset.__aenter__()

    async def test_owned_session_is_retained_when_cleanup_fails(self, fake_sprites: FakeSprites) -> None:
        toolset = SpriteSandbox[None](token='token').get_toolset()
        assert _is_toolset(toolset)
        run_toolset = await toolset.for_run(_context())
        assert isinstance(run_toolset, SpriteSandboxToolset)
        await run_toolset.__aenter__()
        fake_sprites.destroy_error = FakeSpriteError('control plane unavailable')
        with pytest.raises(SpriteSandboxError, match='Could not destroy'):
            await run_toolset.__aexit__(None, None, None)
        assert run_toolset._session is not None
        assert run_toolset._session.is_open

        fake_sprites.destroy_error = None
        await run_toolset.__aexit__(None, None, None)
        assert run_toolset._session is None
