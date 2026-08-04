"""Tests for E2BSandbox through its public capability API."""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol, TypeGuard, runtime_checkable

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage
from typing_extensions import Self

from pydantic_ai_harness.e2b_sandbox import (
    E2BSandbox,
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


class BaseDurabilityCapability(AbstractCapability[Any]):
    """Match the stable base-class identity without importing Prefect."""

    __module__ = 'pydantic_ai.durable_exec._base'


class PrefectDurability(BaseDurabilityCapability):
    """Dependency-free representative of Prefect's concrete wrapper."""


@runtime_checkable
class _E2BSandboxTools(Protocol):  # pragma: no cover - structural typing only
    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *args: object) -> None: ...

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str: ...

    async def read_file(self, path: str, *, offset: int | None = None, limit: int | None = None) -> str: ...

    async def write_file(self, path: str, content: str) -> str: ...

    async def list_directory(self, path: str = '.') -> str: ...


def _is_abstract_toolset(value: object) -> TypeGuard[AbstractToolset[None]]:
    return isinstance(value, AbstractToolset)


def _run_context(*, tracer: Tracer | None = None) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        tracer=tracer or NoOpTracer(),
    )


@asynccontextmanager
async def _toolset(
    *,
    sandbox_id: str | None = None,
    template: str | None = None,
    sandbox_timeout: int = 300,
    workdir: str = '/home/user',
    default_command_timeout: float = 30.0,
    max_command_timeout: int | None = None,
    max_output_bytes: int = 50_000,
    max_output_lines: int = 2000,
    max_read_bytes: int = 5 * 1024 * 1024,
    env: Mapping[str, str] | None = None,
    metadata: Mapping[str, str] | None = None,
    allow_internet_access: bool = True,
    session: E2BSandboxSession | None = None,
    tracer: Tracer | None = None,
) -> AsyncGenerator[_E2BSandboxTools]:
    toolset = E2BSandbox[None](
        template=template,
        sandbox_id=sandbox_id,
        sandbox_timeout=sandbox_timeout,
        workdir=workdir,
        default_command_timeout=default_command_timeout,
        max_command_timeout=max_command_timeout,
        max_output_bytes=max_output_bytes,
        max_output_lines=max_output_lines,
        max_read_bytes=max_read_bytes,
        env=env,
        metadata=metadata,
        allow_internet_access=allow_internet_access,
        session=session,
    ).get_toolset()
    if not _is_abstract_toolset(toolset):  # pragma: no cover - capability contract
        raise AssertionError('E2BSandbox must return an AbstractToolset')
    run_toolset = await toolset.for_run(_run_context(tracer=tracer))
    if not isinstance(run_toolset, _E2BSandboxTools):  # pragma: no cover - capability contract
        raise AssertionError('E2BSandbox toolset is missing its public tools')
    async with run_toolset:
        yield run_toolset


class TestRunCommand:
    async def test_stdout_stderr_nonzero_and_no_output(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('out\n', 'err\n', 2)
        async with _toolset() as tools:
            assert await tools.run_command('false') == ('[stdout]\nout\n[stderr]\nerr\n[exit code: 2]')
            fake_e2b.responder = lambda command, timeout: ('', '', 0)
            assert await tools.run_command('true') == '(no output)'

    async def test_timeout_is_reported_and_killed(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('partial\n', '', 0)
        fake_e2b.wait_error = TimeoutException('deadline')
        async with _toolset() as tools:
            result = await tools.run_command('sleep 99', timeout_seconds=2)
        assert result == '[stdout]\npartial\n[timed out after 2s]'
        assert fake_e2b.sandboxes[0].commands.handles[0].killed is True

    async def test_timeout_output_is_marked_as_prefix(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('A' * 100 + 'END', '', 0)
        fake_e2b.wait_error = TimeoutException('deadline')
        async with _toolset(max_output_bytes=20) as tools:
            result = await tools.run_command('flood', timeout_seconds=2)
        assert result == f'[stdout]\n{"A" * 20}\n[... output truncated to the first 20B ...]\n[timed out after 2s]'

    async def test_timeout_rounds_and_clamps(self, fake_e2b: FakeE2B) -> None:
        async with _toolset(sandbox_timeout=20, default_command_timeout=4.1) as tools:
            await tools.run_command('default')
            await tools.run_command('fractional', timeout_seconds=0.2)
            await tools.run_command('clamped', timeout_seconds=100)
        timeouts = [call.timeout for call in fake_e2b.sandboxes[0].commands.calls if call.background]
        assert timeouts == [5, 1, 20]

    async def test_owned_default_ceiling_tracks_longer_sandbox_lifetime(self, fake_e2b: FakeE2B) -> None:
        async with _toolset(sandbox_timeout=600) as tools:
            await tools.run_command('long', timeout_seconds=500)
        command = next(call for call in fake_e2b.sandboxes[0].commands.calls if call.background)
        assert command.timeout == 500

    async def test_explicit_ceiling_for_reused_sandbox(self, fake_e2b: FakeE2B) -> None:
        async with _toolset(sandbox_id='sbx-existing', max_command_timeout=600) as tools:
            await tools.run_command('long', timeout_seconds=900)
        command = next(call for call in fake_e2b.sandboxes[0].commands.calls if call.background)
        assert command.timeout == 600
        assert fake_e2b.sandboxes[0].kill_calls == 0

    @pytest.mark.parametrize('timeout', [0.0, -1.0, float('inf'), float('nan')])
    async def test_bad_model_timeout_retries(self, fake_e2b: FakeE2B, timeout: float) -> None:
        async with _toolset() as tools:
            with pytest.raises(ModelRetry, match='greater than 0'):
                await tools.run_command('echo', timeout_seconds=timeout)
        assert fake_e2b.sandboxes[0].commands.calls == []

    async def test_output_is_bounded_per_stream_and_by_lines(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: (
            'A' * 100 + 'END',
            'first\nsecond\nthird',
            0,
        )
        async with _toolset(max_output_bytes=20, max_output_lines=1) as tools:
            result = await tools.run_command('flood')
        assert result.startswith('[stdout]\n[... output truncated')
        assert 'END' in result
        assert '[stderr]\n[... output truncated to the last 1 lines ...]\nthird' in result

    async def test_recoverable_and_terminal_errors(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = SandboxException('transient')
        async with _toolset() as tools:
            with pytest.raises(ModelRetry, match='transient'):
                await tools.run_command('echo')
        fake_e2b.run_error = SandboxNotFoundException('gone')
        async with _toolset() as tools:
            with pytest.raises(E2BSandboxUnavailableError):
                await tools.run_command('echo')

    async def test_unentered_base_toolset_rejected(self) -> None:
        toolset = E2BSandbox[None]().get_toolset()
        assert isinstance(toolset, _E2BSandboxTools)
        async with toolset:
            with pytest.raises(E2BSandboxError, match='session is not open'):
                await toolset.run_command('echo')


class TestFiles:
    async def test_read_write_and_list(self, fake_e2b: FakeE2B) -> None:
        async with _toolset() as tools:
            assert await tools.write_file('notes.txt', 'alpha\nbeta\ngamma') == ("Wrote 16 bytes to 'notes.txt'.")
            assert await tools.read_file('notes.txt', offset=2, limit=1) == (
                'beta\n\n[1 more lines in file. Use offset=3 to continue.]'
            )
            fake_e2b.sandboxes[0].files.listings['/home/user'] = [
                FakeEntryInfo('z.txt', '/home/user/z.txt', FakeFileType('file'), 1),
                FakeEntryInfo('src', '/home/user/src', FakeFileType('dir'), 0),
            ]
            assert await tools.list_directory() == 'src/\nz.txt'

    async def test_empty_and_truncated_directory(self, fake_e2b: FakeE2B) -> None:
        async with _toolset(max_output_lines=1) as tools:
            assert await tools.list_directory() == '(empty)'
            fake_e2b.sandboxes[0].files.listings['/home/user'] = [
                FakeEntryInfo('a', '/home/user/a', FakeFileType('file'), 1),
                FakeEntryInfo('b', '/home/user/b', FakeFileType('file'), 1),
            ]
            assert (await tools.list_directory()).startswith('a\n[... output truncated')

    async def test_read_refuses_large_file_and_growth_race(self, fake_e2b: FakeE2B) -> None:
        async with _toolset(max_read_bytes=5) as tools:
            fake_e2b.sandboxes[0].files.stat_sizes['/large'] = 6
            with pytest.raises(ModelRetry, match='over the 5B read limit'):
                await tools.read_file('/large')
            fake_e2b.sandboxes[0].files.stat_sizes['/growing'] = 5
            fake_e2b.sandboxes[0].files.files['/growing'] = b'123456'
            with pytest.raises(ModelRetry, match='grew beyond'):
                await tools.read_file('/growing')

    @pytest.mark.parametrize(
        ('size_bytes', 'max_read_bytes', 'match'),
        [
            (2 * 1024, 1024, r'2\.0KB.*1\.0KB'),
            (2 * 1024 * 1024, 1024 * 1024, r'2\.0MB.*1\.0MB'),
        ],
    )
    async def test_read_limit_formats_large_sizes(
        self,
        fake_e2b: FakeE2B,
        size_bytes: int,
        max_read_bytes: int,
        match: str,
    ) -> None:
        async with _toolset(max_read_bytes=max_read_bytes) as tools:
            fake_e2b.sandboxes[0].files.stat_sizes['/large'] = size_bytes
            with pytest.raises(ModelRetry, match=match):
                await tools.read_file('/large')

    @pytest.mark.parametrize(
        ('offset', 'limit', 'match'),
        [
            (0, None, 'offset must be >= 1'),
            (None, 0, 'limit must be >= 1'),
            (4, None, 'beyond end of file'),
        ],
    )
    async def test_read_rejects_invalid_windows(
        self,
        fake_e2b: FakeE2B,
        offset: int | None,
        limit: int | None,
        match: str,
    ) -> None:
        async with _toolset() as tools:
            fake_e2b.sandboxes[0].files.files['/notes'] = b'alpha\nbeta\ngamma'
            with pytest.raises(ModelRetry, match=match):
                await tools.read_file('/notes', offset=offset, limit=limit)

    async def test_read_handles_trailing_newline_and_safety_caps(self, fake_e2b: FakeE2B) -> None:
        async with _toolset(max_output_lines=1, max_output_bytes=5) as tools:
            files = fake_e2b.sandboxes[0].files.files
            files['/newline'] = b'alpha\n'
            assert await tools.read_file('/newline') == 'alpha'
            files['/line-cap'] = b'a\nb'
            assert await tools.read_file('/line-cap') == ('a\n\n[Showing lines 1-1 of 2. Use offset=2 to continue.]')

        async with _toolset(max_output_bytes=5) as tools:
            files = fake_e2b.sandboxes[1].files.files
            files['/byte-cap'] = b'aa\nbbb'
            assert await tools.read_file('/byte-cap') == (
                'aa\n\n[Showing lines 1-1 of 2 (5B limit). Use offset=2 to continue.]'
            )

    @pytest.mark.parametrize(
        ('data', 'expected'),
        [
            (
                b'123456\nnext',
                '[Line 1 is 6B, exceeds the 5B limit and was omitted. Use offset=2 to continue.]',
            ),
            (
                b'123456',
                '[Line 1 is 6B, exceeds the 5B limit and was omitted.]',
            ),
        ],
    )
    async def test_read_omits_oversized_first_line(
        self,
        fake_e2b: FakeE2B,
        data: bytes,
        expected: str,
    ) -> None:
        async with _toolset(max_output_bytes=5) as tools:
            fake_e2b.sandboxes[0].files.files['/wide'] = data
            assert await tools.read_file('/wide') == expected

    async def test_list_omits_oversized_first_entry(self, fake_e2b: FakeE2B) -> None:
        async with _toolset(max_output_bytes=5) as tools:
            fake_e2b.sandboxes[0].files.listings['/home/user'] = [
                FakeEntryInfo('123456', '/home/user/123456', FakeFileType('file'), 0),
            ]
            assert await tools.list_directory() == '[... first line exceeds the 5B limit, output omitted ...]'

    async def test_binary_and_unpaired_surrogate_retry(self, fake_e2b: FakeE2B) -> None:
        async with _toolset() as tools:
            fake_e2b.sandboxes[0].files.files['/binary'] = b'\xff'
            with pytest.raises(ModelRetry, match='not valid UTF-8'):
                await tools.read_file('/binary')
            with pytest.raises(ModelRetry, match='unpaired surrogates'):
                await tools.write_file('/x', '\ud800')

    @pytest.mark.parametrize(
        ('operation', 'match'),
        [
            ('read', 'Could not read'),
            ('write', 'Could not write'),
            ('list', 'Could not list'),
        ],
    )
    async def test_recoverable_file_errors(self, fake_e2b: FakeE2B, operation: str, match: str) -> None:
        setattr(fake_e2b, f'{operation}_error', SandboxException('transient'))
        async with _toolset() as tools:
            if operation == 'read':
                fake_e2b.sandboxes[0].files.files['/x'] = b'x'
            with pytest.raises(ModelRetry, match=match):
                if operation == 'read':
                    await tools.read_file('/x')
                elif operation == 'write':
                    await tools.write_file('/x', 'x')
                else:
                    await tools.list_directory('/x')

    @pytest.mark.parametrize('operation', ['read', 'write', 'list'])
    async def test_terminal_file_errors(self, fake_e2b: FakeE2B, operation: str) -> None:
        error = SandboxNotFoundException('gone')
        if operation == 'read':
            fake_e2b.info_error = error
        else:
            setattr(fake_e2b, f'{operation}_error', error)
        async with _toolset() as tools:
            with pytest.raises(E2BSandboxUnavailableError):
                if operation == 'read':
                    await tools.read_file('/x')
                elif operation == 'write':
                    await tools.write_file('/x', 'x')
                else:
                    await tools.list_directory('/x')


class TestLifecycleAndTracing:
    async def test_injected_open_session_is_reused(self, fake_e2b: FakeE2B) -> None:
        async with E2BSandboxSession(template='python') as session:
            async with _toolset(session=session) as tools:
                assert await tools.run_command('true') == '(no output)'
            assert fake_e2b.sandboxes[0].kill_calls == 0
        assert fake_e2b.sandboxes[0].kill_calls == 1

    async def test_injected_session_must_be_open(self) -> None:
        session = E2BSandboxSession()
        toolset = E2BSandbox[None](session=session).get_toolset()
        assert _is_abstract_toolset(toolset)
        run_toolset = await toolset.for_run(_run_context())
        with pytest.raises(E2BSandboxError, match='injected session is not open'):
            async with run_toolset:
                pass  # pragma: no cover

    async def test_spans_have_identity_outcomes_and_no_content(self, fake_e2b: FakeE2B) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer('e2b-test')
        fake_e2b.responder = lambda command, timeout: ('secret output', '', 3)
        async with _toolset(template='python', tracer=tracer) as tools:
            await tools.run_command('secret command')

        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert set(spans) == {
            'e2b.sandbox.create',
            'e2b.sandbox.run_command',
            'e2b.sandbox.kill',
        }
        command_span = spans['e2b.sandbox.run_command']
        assert command_span.attributes is not None
        assert command_span.attributes['e2b.sandbox.id'] == 'sbx-1'
        assert command_span.attributes['e2b.sandbox.template'] == 'python'
        assert command_span.attributes['e2b.sandbox.mode'] == 'owned'
        assert command_span.attributes['e2b.outcome'] == 'nonzero_exit'
        assert command_span.attributes['e2b.command.exit_code'] == 3
        assert 'secret' not in repr(command_span.attributes)

    @pytest.mark.parametrize(
        ('error', 'outcome'),
        [
            (SandboxException('transient secret path /workspace/private.txt'), 'retry'),
            (AuthenticationException('bad key'), 'terminal_error'),
        ],
    )
    async def test_error_span_outcomes(self, fake_e2b: FakeE2B, error: Exception, outcome: str) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        fake_e2b.run_error = error
        with pytest.raises((ModelRetry, E2BSandboxAuthError)):
            async with _toolset(tracer=provider.get_tracer('test')) as tools:
                await tools.run_command('echo')
        span = next(span for span in exporter.get_finished_spans() if span.name == 'e2b.sandbox.run_command')
        assert span.attributes is not None
        assert span.attributes['e2b.outcome'] == outcome
        assert not span.events
        assert 'secret' not in repr(span)


class TestCapability:
    def test_defaults_exports_and_keyword_only(self) -> None:
        cap = E2BSandbox()
        assert cap.template is None
        assert cap.sandbox_timeout == 300
        assert cap.workdir == '/home/user'
        assert cap.default_command_timeout == 60.0
        assert isinstance(cap.get_toolset(), AbstractToolset)
        assert E2BSandbox.get_serialization_name() == 'E2BSandbox'
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(E2BSandbox).parameters.values()
        )

        import pydantic_ai_harness
        import pydantic_ai_harness.e2b_sandbox as e2b_sandbox

        assert 'E2BSandbox' in e2b_sandbox.__all__
        assert 'E2BSandboxSession' in e2b_sandbox.__all__
        assert 'E2BSandboxToolset' not in e2b_sandbox.__all__
        assert 'E2BSandbox' not in pydantic_ai_harness.__all__

    @pytest.mark.parametrize(
        ('name', 'value'),
        [
            ('sandbox_timeout', 0),
            ('max_output_bytes', True),
            ('max_output_lines', -1),
            ('max_read_bytes', 0),
            ('default_command_timeout', 0),
            ('default_command_timeout', float('nan')),
            ('default_command_timeout', float('inf')),
            ('max_command_timeout', 0),
            ('workdir', 'relative'),
            ('allow_internet_access', 'yes'),
            ('instructions', 123),
        ],
    )
    def test_invalid_configuration(self, name: str, value: object) -> None:
        with pytest.raises(ValueError, match=name):
            E2BSandbox(**{name: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            ({'template': 'python'}, 'template'),
            ({'sandbox_timeout': 600}, 'sandbox_timeout'),
            ({'env': {'A': 'b'}}, 'env'),
            ({'metadata': {'A': 'b'}}, 'metadata'),
            ({'allow_internet_access': False}, 'allow_internet_access'),
        ],
    )
    def test_attach_rejects_create_only_settings(self, kwargs: dict[str, object], expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            E2BSandbox(sandbox_id='sbx', **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            ({'sandbox_id': 'sbx'}, 'sandbox_id'),
            ({'template': 'python'}, 'template'),
            ({'workdir': '/project'}, 'workdir'),
        ],
    )
    def test_injected_rejects_other_lifecycle_settings(self, kwargs: dict[str, object], expected: str) -> None:
        with pytest.raises(ValueError, match=expected):
            E2BSandbox(session=E2BSandboxSession(), **kwargs)  # type: ignore[arg-type]

    def test_owned_ceiling_cannot_outlive_sandbox(self) -> None:
        with pytest.raises(ValueError, match='cannot exceed sandbox_timeout'):
            E2BSandbox(sandbox_timeout=10, max_command_timeout=11)
        assert E2BSandbox(sandbox_id='sbx', max_command_timeout=600).max_command_timeout == 600

    def test_mapping_inputs_are_copied(self) -> None:
        env = {'A': 'one'}
        metadata = {'task': 'test'}
        cap = E2BSandbox(env=env, metadata=metadata)
        env['A'] = 'two'
        metadata['task'] = 'changed'
        assert cap.env == {'A': 'one'}
        assert cap.metadata == {'task': 'test'}

    def test_mode_aware_instructions_and_override(self) -> None:
        owned = E2BSandbox(default_command_timeout=4.1, sandbox_timeout=20).get_instructions()
        reused = E2BSandbox(
            sandbox_id='sbx',
            default_command_timeout=500,
            max_command_timeout=600,
        ).get_instructions()
        injected = E2BSandbox(session=E2BSandboxSession()).get_instructions()
        assert owned is not None and 'after 5s' in owned and 'up to 20s' in owned
        assert reused is not None and 'after 500s' in reused and 'persists across runs' in reused
        assert injected is not None and 'persists across runs' in injected
        assert E2BSandbox(instructions='Custom').get_instructions() == 'Custom'
        assert E2BSandbox(instructions='').get_instructions() is None

    @pytest.mark.parametrize(
        ('module_name', 'class_name', 'engine_name'),
        [
            ('pydantic_ai.durable_exec.temporal', 'TemporalDurability', 'Temporal'),
            ('pydantic_ai.durable_exec.dbos', 'DBOSDurability', 'DBOS'),
        ],
    )
    def test_durable_execution_is_rejected_before_run(
        self,
        module_name: str,
        class_name: str,
        engine_name: str,
    ) -> None:
        module = pytest.importorskip(module_name, exc_type=ImportError)
        durability_type = getattr(module, class_name)

        with pytest.raises(UserError, match=rf'E2BSandbox.*{engine_name}'):
            Agent(
                TestModel(),
                name=f'e2b_{engine_name.lower()}',
                capabilities=[E2BSandbox(), durability_type()],
            )

    def test_prefect_durability_is_rejected_before_run_without_prefect_dependency(self) -> None:
        with pytest.raises(UserError, match=r'E2BSandbox.*Prefect'):
            Agent(
                TestModel(),
                name='e2b_prefect',
                capabilities=[E2BSandbox(), PrefectDurability()],
            )

    @pytest.mark.parametrize(
        ('module_name', 'class_name', 'engine_name'),
        [
            ('pydantic_ai.durable_exec.temporal', 'TemporalDurability', 'Temporal'),
            ('pydantic_ai.durable_exec.dbos', 'DBOSDurability', 'DBOS'),
        ],
    )
    @pytest.mark.anyio(backends=['asyncio'])
    async def test_run_level_durable_execution_is_rejected(
        self,
        fake_e2b: FakeE2B,
        module_name: str,
        class_name: str,
        engine_name: str,
    ) -> None:
        module = pytest.importorskip(module_name, exc_type=ImportError)
        durability_type = getattr(module, class_name)
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[E2BSandbox()])

        with pytest.raises(UserError, match=rf'E2BSandbox.*{engine_name}'):
            await agent.run('hi', capabilities=[durability_type()])
        assert fake_e2b.sandboxes == []

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_run_level_prefect_durability_is_rejected_without_prefect_dependency(self, fake_e2b: FakeE2B) -> None:
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[E2BSandbox()])

        with pytest.raises(UserError, match=r'E2BSandbox.*Prefect'):
            await agent.run('hi', capabilities=[PrefectDurability()])
        assert fake_e2b.sandboxes == []

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_run_level_non_durability_capability_is_allowed(self, fake_e2b: FakeE2B) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[E2BSandbox()])
        result = await agent.run('hi', capabilities=[Capability(id='run_extra', description='run extra')])
        assert result.output
        assert fake_e2b.sandboxes[0].kill_calls == 1

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_can_call_command(self, fake_e2b: FakeE2B) -> None:
        import sniffio

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        fake_e2b.responder = lambda command, timeout: ('hello\n', '', 0)

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(
                    parts=[ToolCallPart('run_command', {'command': 'echo hello'}, tool_call_id='run-1')]
                )
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(call_then_finish),
            capabilities=[E2BSandbox()],
        )
        result = await agent.run('run a command')
        assert result.output == 'done'
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert returns == ['[stdout]\nhello']
        assert fake_e2b.sandboxes[0].kill_calls == 1
