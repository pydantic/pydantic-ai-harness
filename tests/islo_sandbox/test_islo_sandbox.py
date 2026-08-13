"""Tests for the public Islo sandbox capability and model-facing tools."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Protocol, TypeGuard, runtime_checkable

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.islo_sandbox import (
    IsloSandbox,
    IsloSandboxError,
    IsloSandboxSession,
    IsloSandboxTerminalError,
    IsloSandboxUnavailableError,
)

from .fake_islo import FakeApiError, FakeExecResult, FakeIslo


@runtime_checkable
class _IsloSandboxTools(Protocol):  # pragma: no cover - structural typing only
    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str: ...

    async def read_file(self, path: str, *, offset: int | None = None, limit: int | None = None) -> str: ...

    async def write_file(self, path: str, content: str) -> str: ...

    async def list_directory(self, path: str = '.') -> str: ...


def _is_abstract_toolset(value: object) -> TypeGuard[AbstractToolset[None]]:
    return isinstance(value, AbstractToolset)


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


@asynccontextmanager
async def _toolset(
    *,
    sandbox_name: str | None = None,
    sandbox_timeout: int = 900,
    max_command_timeout: int | None = None,
    max_output_bytes: int = 50_000,
    max_output_lines: int = 2000,
    max_read_bytes: int = 5 * 1024 * 1024,
    env: Mapping[str, str] | None = None,
    session: IsloSandboxSession | None = None,
) -> AsyncGenerator[_IsloSandboxTools]:
    toolset = IsloSandbox[None](
        sandbox_name=sandbox_name,
        sandbox_timeout=sandbox_timeout,
        default_command_timeout=30,
        max_command_timeout=max_command_timeout,
        max_output_bytes=max_output_bytes,
        max_output_lines=max_output_lines,
        max_read_bytes=max_read_bytes,
        env=env,
        poll_interval=0.5 if session is not None else 0.001,
        session=session,
    ).get_toolset()
    if not _is_abstract_toolset(toolset):  # pragma: no cover - capability contract
        raise AssertionError('IsloSandbox must return an AbstractToolset')
    run_toolset = await toolset.for_run(_run_context())
    if not isinstance(run_toolset, _IsloSandboxTools):  # pragma: no cover - capability contract
        raise AssertionError('IsloSandbox toolset is missing its public tools')
    async with run_toolset:
        yield run_toolset


class TestRunCommand:
    async def test_combines_streams_and_exit_code(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(stdout='out\n', stderr='err\n', exit_code=2)
        async with _toolset() as toolset:
            result = await toolset.run_command('false')
        assert result == '[stdout]\nout\n[stderr]\nerr\n[exit code: 2]'
        call = fake_islo.sandboxes.exec_calls[0]
        assert call.command == ['sh', '-c', 'false']
        assert call.timeout_secs == 30

    async def test_no_output(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult()
        async with _toolset() as toolset:
            assert await toolset.run_command('true') == '(no output)'

    @pytest.mark.parametrize(
        ('requested', 'ceiling', 'expected'),
        [(0.1, None, 1), (12.1, None, 13), (9999, None, 900), (9999, 50, 50)],
    )
    async def test_timeout_rounding_and_ceiling(
        self, fake_islo: FakeIslo, requested: float, ceiling: int | None, expected: int
    ) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(stdout=str(call.timeout_secs))
        async with _toolset(max_command_timeout=ceiling) as toolset:
            await toolset.run_command('echo', timeout_seconds=requested)
        assert fake_islo.sandboxes.exec_calls[0].timeout_secs == expected

    @pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
    async def test_bad_timeout_is_a_model_retry(self, fake_islo: FakeIslo, timeout: float) -> None:
        async with _toolset() as toolset:
            with pytest.raises(ModelRetry, match='greater than 0'):
                await toolset.run_command('echo', timeout_seconds=timeout)
        assert fake_islo.sandboxes.exec_calls == []

    async def test_provider_timeout_and_client_timeout_are_distinct(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(status='timeout', stdout='partial\n', exit_code=-1)
        async with _toolset() as toolset:
            provider = await toolset.run_command('sleep 9', timeout_seconds=2)
        assert provider == '[stdout]\npartial\n[timed out after 2s]'

        fake_islo.sandboxes.responder = lambda call: FakeExecResult(status='running', stdout='still running')
        async with _toolset() as toolset:
            client = await toolset.run_command('sleep 9', timeout_seconds=0.003)
        assert 'remote command may still be running' in client

    async def test_output_is_bounded_and_provider_cut_is_marked(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(stdout='A' * 200 + 'END', truncated=True)
        async with _toolset(max_output_bytes=30) as toolset:
            result = await toolset.run_command('flood')
        assert 'output truncated to the last 30B' in result
        assert result.endswith('END')

    async def test_line_cap_preserves_tail(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(stdout='first\nsecond\nthird')
        async with _toolset(max_output_lines=1) as toolset:
            result = await toolset.run_command('lines')
        assert 'last 1 lines' in result
        assert result.endswith('third')

    async def test_transient_and_terminal_errors_are_classified(self, fake_islo: FakeIslo) -> None:
        async with _toolset() as toolset:
            fake_islo.sandboxes.exec_error = FakeApiError(500, {'message': 'retry'})
            with pytest.raises(ModelRetry, match='HTTP 500'):
                await toolset.run_command('echo')

            fake_islo.sandboxes.exec_error = FakeApiError(404, {'message': 'gone'})
            with pytest.raises(IsloSandboxUnavailableError, match='not found'):
                await toolset.run_command('echo')

    async def test_unentered_toolset_rejects_direct_call(self) -> None:
        toolset = IsloSandbox[None]().get_toolset()
        assert isinstance(toolset, _IsloSandboxTools)
        with pytest.raises(IsloSandboxError, match='session is not open'):
            await toolset.run_command('echo')


class TestFiles:
    async def test_read_file_supports_windows(self, fake_islo: FakeIslo) -> None:
        fake_islo.put_file('/workspace/file.txt', b'a\nb\nc\nd\n')
        fake_islo.put_file('/workspace/empty.txt', b'')
        async with _toolset() as toolset:
            assert await toolset.read_file('file.txt', offset=2, limit=2) == (
                'b\nc\n\n[1 more lines in file. Use offset=4 to continue.]'
            )
            assert await toolset.read_file('empty.txt') == ''

    async def test_read_file_caps_output_and_rejects_binary(self, fake_islo: FakeIslo) -> None:
        fake_islo.put_file('/workspace/large.txt', b'line1\nline2\nline3')
        fake_islo.put_file('/workspace/image.bin', b'\xff\xfe')
        async with _toolset(max_output_lines=2) as toolset:
            result = await toolset.read_file('large.txt')
            assert 'Showing lines 1-2 of 3' in result
            with pytest.raises(ModelRetry, match='not valid UTF-8'):
                await toolset.read_file('image.bin')

    async def test_read_limit_and_missing_file_are_model_retries(self, fake_islo: FakeIslo) -> None:
        fake_islo.put_file('/workspace/large', b'123456')
        async with _toolset(max_read_bytes=5) as toolset:
            with pytest.raises(ModelRetry, match='5-byte read limit'):
                await toolset.read_file('large')
            with pytest.raises(ModelRetry, match='Could not read'):
                await toolset.read_file('missing')

    async def test_write_and_list(self, fake_islo: FakeIslo) -> None:
        fake_islo.put_file('/workspace/z.txt', b'z')
        fake_islo.put_file('/workspace/sub/nested', b'n')
        async with _toolset() as toolset:
            assert await toolset.write_file('a.txt', 'hé') == "Wrote 3 bytes to 'a.txt'."
            assert await toolset.list_directory() == 'a.txt\nsub/\nz.txt'
            assert await toolset.list_directory('/empty') == '(empty)'
        assert fake_islo.sandboxes.files['/workspace/a.txt'] == 'hé'.encode()

    async def test_unpaired_surrogate_is_model_retry(self, fake_islo: FakeIslo) -> None:
        async with _toolset() as toolset:
            with pytest.raises(ModelRetry, match='unpaired surrogates'):
                await toolset.write_file('bad', '\ud800')

    async def test_file_api_errors_are_model_retries(self, fake_islo: FakeIslo) -> None:
        async with _toolset() as toolset:
            fake_islo.sandboxes.upload_error = FakeApiError(500, 'upload failed')
            with pytest.raises(ModelRetry, match='Could not write'):
                await toolset.write_file('x', 'x')

            fake_islo.sandboxes.upload_error = None
            fake_islo.sandboxes._directory_result = lambda target: FakeExecResult(exit_code=2)  # type: ignore[method-assign]
            with pytest.raises(ModelRetry, match='Could not list'):
                await toolset.list_directory('.')

    async def test_terminal_file_errors_end_the_run(self, fake_islo: FakeIslo) -> None:
        terminal = IsloSandboxUnavailableError('sandbox gone')
        async with _toolset() as toolset:
            fake_islo.sandboxes.download_error = terminal
            with pytest.raises(IsloSandboxTerminalError, match='sandbox gone'):
                await toolset.read_file('x')

            fake_islo.sandboxes.download_error = None
            fake_islo.sandboxes.upload_error = terminal
            with pytest.raises(IsloSandboxTerminalError, match='sandbox gone'):
                await toolset.write_file('x', 'x')

            fake_islo.sandboxes.upload_error = None

            def fail_listing(target: str) -> FakeExecResult:
                raise terminal

            fake_islo.sandboxes._directory_result = fail_listing  # type: ignore[method-assign]
            with pytest.raises(IsloSandboxTerminalError, match='sandbox gone'):
                await toolset.list_directory('.')

    async def test_listing_is_head_bounded(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.files['/workspace/'] = b''
        fake_islo.put_file('/workspace/a', b'')
        fake_islo.put_file('/workspace/b', b'')
        fake_islo.put_file('/workspace/c', b'')
        async with _toolset(max_output_lines=2) as toolset:
            result = await toolset.list_directory()
        assert result == 'a\nb\n[... output truncated to the first 2 lines ...]'


class TestLifecycle:
    async def test_for_run_copies_config_and_base_toolset_is_inert(self, fake_islo: FakeIslo) -> None:
        original = IsloSandbox[None](image='python:custom', sandbox_timeout=99, poll_interval=0.001).get_toolset()
        assert _is_abstract_toolset(original)
        async with original:
            pass
        assert fake_islo.sandboxes.create_calls == []

        fresh = await original.for_run(_run_context())
        assert fresh is not original
        async with fresh:
            pass
        assert fake_islo.sandboxes.create_calls[0]['image'] == 'python:custom'
        assert fake_islo.sandboxes.create_calls[0]['lifecycle'].delete_after == 99  # type: ignore[union-attr]

    async def test_attached_sandbox_is_left_running(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.attach_response.name = 'keep'
        async with _toolset(sandbox_name='keep') as toolset:
            await toolset.run_command('true')
        assert fake_islo.sandboxes.create_calls == []
        assert fake_islo.sandboxes.delete_calls == []

    async def test_injected_open_session_is_reused(self, fake_islo: FakeIslo) -> None:
        async with IsloSandboxSession(poll_interval=0.001) as session:
            async with _toolset(session=session) as toolset:
                assert await toolset.run_command('true')
            assert len(fake_islo.sandboxes.create_calls) == 1
            assert fake_islo.sandboxes.delete_calls == []
        assert fake_islo.sandboxes.delete_calls == ['sandbox-owned']

    async def test_unopened_injected_session_fails_at_run_start(self) -> None:
        with pytest.raises(IsloSandboxError, match='injected session is not open'):
            async with _toolset(session=IsloSandboxSession()):
                pass  # pragma: no cover

    async def test_error_exit_still_deletes_owned_sandbox(self, fake_islo: FakeIslo) -> None:
        with pytest.raises(RuntimeError, match='body failed'):
            async with _toolset():
                raise RuntimeError('body failed')
        assert fake_islo.sandboxes.delete_calls == ['sandbox-owned']


class TestCapability:
    def test_defaults_and_exports(self) -> None:
        import pydantic_ai_harness
        import pydantic_ai_harness.islo_sandbox as islo_sandbox

        capability = IsloSandbox()
        assert capability.image == 'ghcr.io/islo-labs/islo-runner:latest'
        assert capability.sandbox_timeout == 900
        assert capability.default_command_timeout == 60
        assert isinstance(capability.get_toolset(), AbstractToolset)
        assert IsloSandbox.get_serialization_name() == 'IsloSandbox'
        assert 'IsloSandbox' in islo_sandbox.__all__
        assert 'IsloSandboxToolset' not in islo_sandbox.__all__
        assert 'IsloSandbox' not in pydantic_ai_harness.__all__

    def test_configuration_is_keyword_only(self) -> None:
        parameters = inspect.signature(IsloSandbox).parameters.values()
        assert parameters
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters)

    @pytest.mark.parametrize(
        ('name', 'value'),
        [
            ('sandbox_timeout', 0),
            ('sandbox_timeout', True),
            ('max_output_bytes', -1),
            ('max_output_lines', 0),
            ('max_read_bytes', -1),
            ('vcpus', 0),
            ('memory_mb', True),
            ('disk_gb', -1),
            ('default_command_timeout', 0),
            ('default_command_timeout', True),
            ('default_command_timeout', float('nan')),
            ('default_command_timeout', float('inf')),
            ('max_command_timeout', 0),
            ('poll_interval', 0),
            ('poll_interval', float('inf')),
            ('instructions', 123),
        ],
    )
    def test_invalid_configuration_is_rejected(self, name: str, value: object) -> None:
        with pytest.raises(ValueError, match=name):
            IsloSandbox(**{name: value})  # type: ignore[arg-type]

    def test_env_is_copied(self) -> None:
        env = {'A': 'one'}
        capability = IsloSandbox(env=env)
        env['A'] = 'mutated'
        assert capability.env == {'A': 'one'}

    @pytest.mark.parametrize(
        ('kwargs', 'field'),
        [
            ({'image': 'custom'}, 'image'),
            ({'sandbox_timeout': 901}, 'sandbox_timeout'),
            ({'workdir': '/tmp'}, 'workdir'),
            ({'env': {'A': 'b'}}, 'env'),
            ({'vcpus': 2}, 'vcpus'),
            ({'memory_mb': 1024}, 'memory_mb'),
            ({'disk_gb': 10}, 'disk_gb'),
            ({'internet_enabled': False}, 'internet_enabled'),
            ({'gateway_profile': 'locked'}, 'gateway_profile'),
        ],
    )
    def test_attach_rejects_creation_only_settings(self, kwargs: dict[str, object], field: str) -> None:
        with pytest.raises(ValueError, match=f'{field} only apply'):
            IsloSandbox(sandbox_name='keep', **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ('kwargs', 'field'),
        [
            ({'sandbox_name': 'keep'}, 'sandbox_name'),
            ({'base_url': 'https://api.test'}, 'base_url'),
            ({'compute_url': 'https://compute.test'}, 'compute_url'),
            ({'poll_interval': 0.1}, 'poll_interval'),
            ({'env': {'A': 'b'}}, 'env'),
        ],
    )
    def test_session_rejects_connection_and_creation_settings(
        self, fake_islo: FakeIslo, kwargs: dict[str, object], field: str
    ) -> None:
        with pytest.raises(ValueError, match=f'{field} cannot be combined'):
            IsloSandbox(session=IsloSandboxSession(), **kwargs)  # type: ignore[arg-type]

    def test_owned_command_ceiling_cannot_outlive_sandbox(self) -> None:
        with pytest.raises(ValueError, match='cannot exceed sandbox_timeout'):
            IsloSandbox(sandbox_timeout=30, max_command_timeout=31)
        assert IsloSandbox(sandbox_name='keep', max_command_timeout=901).max_command_timeout == 901

    def test_instructions_cover_lifecycle_timeout_override_and_disable(self) -> None:
        owned = IsloSandbox(default_command_timeout=45.1, sandbox_timeout=120).get_instructions()
        reused = IsloSandbox(sandbox_name='keep', max_command_timeout=999).get_instructions()
        assert owned is not None and 'up to 46s' in owned and 'reset between runs' in owned
        assert reused is not None and 'up to 999s' in reused and 'persists across runs' in reused
        assert IsloSandbox(instructions='Custom.').get_instructions() == 'Custom.'
        assert IsloSandbox(instructions='').get_instructions() is None

    async def test_agent_can_call_command_and_cleans_up(self, fake_islo: FakeIslo) -> None:
        fake_islo.sandboxes.responder = lambda call: FakeExecResult(stdout='hello\n')

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'echo hello'}, 'run-1')])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(call_then_finish), capabilities=[IsloSandbox(poll_interval=0.001)]
        )
        result = await agent.run('run a command')
        assert result.output == 'done'
        tool_returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_command'
        ]
        assert tool_returns == ['[stdout]\nhello']
        assert fake_islo.sandboxes.delete_calls == ['sandbox-owned']
