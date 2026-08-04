"""Tests for the public Belgie Sandbox capability."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Protocol, TypeGuard, runtime_checkable

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.belgie_sandbox import (
    BelgieSandbox,
    BelgieSandboxExecutionError,
    BelgieSandboxSession,
)
from pydantic_ai_harness.code_mode import CodeMode

from .fake_belgie import BelgieJavaScriptError, FakeBelgie

pytestmark = pytest.mark.anyio


def echo(value: str) -> str:
    return value  # pragma: no cover - model only inspects this tool's schema


@runtime_checkable
class _BelgieSandboxTools(Protocol):  # pragma: no cover - structural typing only
    async def run_typescript(self, code: str) -> ToolReturn: ...


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
async def _toolset(capability: BelgieSandbox[None] | None = None) -> AsyncGenerator[_BelgieSandboxTools]:
    toolset = (capability or BelgieSandbox[None]()).get_toolset()
    if not _is_abstract_toolset(toolset):  # pragma: no cover - capability contract
        raise AssertionError('BelgieSandbox must return an AbstractToolset')
    run_toolset = await toolset.for_run(_run_context())
    if not isinstance(run_toolset, _BelgieSandboxTools):  # pragma: no cover - capability contract
        raise AssertionError('BelgieSandbox toolset is missing run_typescript')
    async with run_toolset:
        yield run_toolset


class TestBelgieSandbox:
    def test_public_exports(self) -> None:
        import pydantic_ai_harness.belgie_sandbox as belgie_sandbox

        assert set(belgie_sandbox.__all__) == {
            'BelgieSandbox',
            'BelgieSandboxError',
            'BelgieSandboxExecutionError',
            'BelgieSandboxSession',
            'BelgieSandboxTimeoutError',
            'BelgieSandboxUnavailableError',
        }

    async def test_tool_schema_metadata_and_result(self, fake_belgie: FakeBelgie) -> None:
        capability = BelgieSandbox[None]()
        toolset = capability.get_toolset()
        assert _is_abstract_toolset(toolset)
        run_toolset = await toolset.for_run(_run_context())

        async with run_toolset:
            tools = await run_toolset.get_tools(_run_context())
            assert list(tools) == ['run_typescript']
            tool_def = tools['run_typescript'].tool_def
            assert tool_def.sequential is True
            assert tool_def.metadata == {'code_arg_name': 'code', 'code_arg_language': 'typescript'}
            assert tool_def.parameters_json_schema['required'] == ['code']
            assert 'TypeScript' in (tool_def.description or '')

            assert isinstance(run_toolset, _BelgieSandboxTools)
            result = await run_toolset.run_typescript('export default () => ({ ok: true })')

        assert result.return_value == {'ok': True}
        assert result.metadata == {
            'belgie_sandbox': True,
            'code_language': 'typescript',
            'output_bytes': 11,
        }
        assert fake_belgie.scripts == ['export default () => ({ ok: true })']

    async def test_session_is_lazy_and_run_scoped(self, fake_belgie: FakeBelgie) -> None:
        async with _toolset() as toolset:
            assert fake_belgie.runtimes == []
            await toolset.run_typescript('first')
            await toolset.run_typescript('second')
            assert len(fake_belgie.runtimes) == 1
            assert fake_belgie.runtimes[0].exited is False

        assert fake_belgie.runtimes[0].exited is True

        async with _toolset() as toolset:
            await toolset.run_typescript('third')
        assert len(fake_belgie.runtimes) == 2

    async def test_toolset_retains_owned_session_when_cleanup_fails(self, fake_belgie: FakeBelgie) -> None:
        toolset = BelgieSandbox[None]().get_toolset()
        assert _is_abstract_toolset(toolset)
        run_toolset = await toolset.for_run(_run_context())
        await run_toolset.__aenter__()
        assert isinstance(run_toolset, _BelgieSandboxTools)
        await run_toolset.run_typescript('code')
        fake_belgie.runtime_exit_error = RuntimeError('cleanup failed')

        with pytest.raises(RuntimeError, match='cleanup failed'):
            await run_toolset.__aexit__(None, None, None)

        assert fake_belgie.runtimes[0].exit_calls == 1
        assert not fake_belgie.runtimes[0].exited
        fake_belgie.runtime_exit_error = None
        await run_toolset.__aexit__(None, None, None)
        assert fake_belgie.runtimes[0].exit_calls == 2
        assert fake_belgie.runtimes[0].exited
        with pytest.raises(BelgieSandboxExecutionError, match='not active'):
            await run_toolset.run_typescript('after close')

    async def test_script_failure_and_output_limit_are_retries(self, fake_belgie: FakeBelgie) -> None:
        async with _toolset(BelgieSandbox(max_output_bytes=4)) as toolset:
            fake_belgie.script_error = BelgieJavaScriptError('bad syntax')
            with pytest.raises(ModelRetry, match='bad syntax'):
                await toolset.run_typescript('bad')

            fake_belgie.script_error = None
            fake_belgie.result = 'large'
            with pytest.raises(ModelRetry, match='exceeding the 4-byte limit'):
                await toolset.run_typescript('large')

            fake_belgie.result = {1, 2}
            with pytest.raises(ModelRetry, match='invalid JSON'):
                await toolset.run_typescript('invalid')

    async def test_unentered_base_toolset_is_a_caller_error(self, fake_belgie: FakeBelgie) -> None:
        toolset = BelgieSandbox[None]().get_toolset()
        assert _is_abstract_toolset(toolset)
        assert await toolset.__aenter__() is toolset
        assert isinstance(toolset, _BelgieSandboxTools)
        with pytest.raises(BelgieSandboxExecutionError, match='not active'):
            await toolset.run_typescript('code')

    async def test_injected_session_is_reused_and_not_closed(self, fake_belgie: FakeBelgie) -> None:
        session = BelgieSandboxSession()
        async with session:
            async with _toolset(BelgieSandbox(session=session)) as toolset:
                await toolset.run_typescript('one')
            assert session.is_open
            async with _toolset(BelgieSandbox(session=session)) as toolset:
                await toolset.run_typescript('two')
            assert len(fake_belgie.runtimes) == 1

        assert not session.is_open

    async def test_rejects_unopened_injected_session(self, fake_belgie: FakeBelgie) -> None:
        with pytest.raises(BelgieSandboxExecutionError, match='not open'):
            async with _toolset(BelgieSandbox(session=BelgieSandboxSession())):
                pass  # pragma: no cover - entering the toolset raises

    async def test_concurrent_runs_have_separate_runtimes(self, fake_belgie: FakeBelgie) -> None:
        async def run_once(source: str) -> None:
            async with _toolset() as toolset:
                await toolset.run_typescript(source)
                await asyncio.sleep(0)

        await asyncio.gather(run_once('alpha'), run_once('beta'))
        assert len(fake_belgie.runtimes) == 2
        assert all(runtime.exited for runtime in fake_belgie.runtimes)

    async def test_agent_executes_tool_and_preserves_other_tools(self, fake_belgie: FakeBelgie) -> None:
        seen_tools: list[set[str]] = []

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen_tools.append({tool.name for tool in info.function_tools})
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(
                    parts=[ToolCallPart('run_typescript', {'code': 'export default () => 42'}, tool_call_id='run-1')]
                )
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(call_then_finish),
            tools=[echo],
            capabilities=[BelgieSandbox()],
        )
        result = await agent.run('run TypeScript')

        assert result.output == 'done'
        assert seen_tools[0] == {'echo', 'run_typescript'}
        assert len(fake_belgie.runtimes) == 1
        assert fake_belgie.runtimes[0].exited

    async def test_unused_capability_does_not_start_runtime(self, fake_belgie: FakeBelgie) -> None:
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[BelgieSandbox()])

        assert (await agent.run('no code needed')).output == 'done'
        assert fake_belgie.runtimes == []

    async def test_deferred_loading_has_stable_routing_metadata(self, fake_belgie: FakeBelgie) -> None:
        model = TestModel(custom_output_text='done', call_tools=[])
        capability = BelgieSandbox(defer_loading=True)
        agent: Agent[None, str] = Agent(model, capabilities=[capability])

        await agent.run('no code needed')

        assert capability.id == 'belgie_sandbox'
        assert capability.description is not None
        assert model.last_model_request_parameters is not None
        tool_names = {tool.name for tool in model.last_model_request_parameters.function_tools}
        assert 'load_capability' in tool_names
        assert 'run_typescript' not in tool_names

    async def test_composes_as_peer_of_code_mode(self, fake_belgie: FakeBelgie) -> None:
        seen_tools: set[str] = set()

        def finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen_tools.update(tool.name for tool in info.function_tools)
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(finish),
            tools=[echo],
            capabilities=[CodeMode(), BelgieSandbox()],
        )

        assert (await agent.run('done')).output == 'done'
        assert seen_tools == {'run_code', 'run_typescript'}

    async def test_rejects_durable_execution(self, fake_belgie: FakeBelgie) -> None:
        try:
            from pydantic_ai.durable_exec.dbos import DBOSDurability
        except ImportError:  # pragma: no cover - exercised by the minimal dependency CI job
            pytest.skip('DBOS is not installed')  # pragma: no cover

        with pytest.raises(UserError, match='does not support durable execution.*DBOSDurability'):
            Agent(
                TestModel(),
                name='belgie-durable-test',
                capabilities=[BelgieSandbox(), DBOSDurability()],
            )

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'allow_package_imports': 1}, 'allow_package_imports must be a bool'),
            ({'allow_network': 1}, 'allow_network must be a bool'),
            ({'enable_rendering': 1}, 'enable_rendering must be a bool'),
            ({'max_old_generation_size_mb': 0}, 'must be a positive integer or None'),
            ({'timeout': 0}, 'timeout must be a positive finite number'),
            ({'timeout': float('inf')}, 'timeout must be a positive finite number'),
            ({'max_output_bytes': 0}, 'max_output_bytes must be a positive integer'),
            ({'max_retries': -1}, 'max_retries must be a non-negative integer'),
            ({'instructions': 1}, 'instructions must be a string or None'),
        ],
    )
    async def test_validates_configuration(
        self, fake_belgie: FakeBelgie, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            BelgieSandbox(**kwargs)  # pyright: ignore[reportArgumentType]

    async def test_session_rejects_owned_runtime_settings(self, fake_belgie: FakeBelgie) -> None:
        with pytest.raises(ValueError, match='cannot be combined with `session`'):
            BelgieSandbox(session=BelgieSandboxSession(), enable_rendering=True)

    async def test_instructions_reflect_configuration(self, fake_belgie: FakeBelgie) -> None:
        strict = BelgieSandbox().get_instructions()
        assert strict is not None
        assert 'imports are disabled' in strict
        assert 'fetch` is disabled' in strict
        assert '@belgie/render' not in strict

        open_profile = BelgieSandbox(
            allow_package_imports=True,
            allow_network=True,
            enable_rendering=True,
            timeout=12,
            max_output_bytes=100,
        ).get_instructions()
        assert open_profile is not None
        assert 'imports are enabled' in open_profile
        assert 'network access is enabled' in open_profile
        assert '@belgie/render' in open_profile
        assert 'plugins: []' in open_profile
        assert '12s deadline' in open_profile

        rendering_only = BelgieSandbox(enable_rendering=True).get_instructions()
        assert rendering_only is not None
        assert 'imports are enabled' in rendering_only
        assert '@belgie/render' in rendering_only

        assert BelgieSandbox(instructions='Custom.').get_instructions() == 'Custom.'
        assert BelgieSandbox(instructions='').get_instructions() is None

        session_instructions = BelgieSandbox(session=BelgieSandboxSession()).get_instructions()
        assert session_instructions is not None
        assert 'caller-managed' in session_instructions
