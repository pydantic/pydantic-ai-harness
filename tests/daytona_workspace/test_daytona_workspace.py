"""Public Daytona workspace capability tests."""

from __future__ import annotations

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceRef

from pydantic_ai_harness.daytona_workspace import DaytonaWorkspace, DaytonaWorkspaceBackend

from .fake_daytona import FakeDaytona

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def test_capability_is_lazy_and_forwards_options(fake_daytona: FakeDaytona) -> None:
    capability = DaytonaWorkspace(
        snapshot='python', auto_stop_minutes=0, workdir='/work', env={'A': 'b'}, network_block_all=True
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, DaytonaWorkspaceBackend)
    assert not fake_daytona.create_params
    await backend.workspace
    params = fake_daytona.create_params[0]
    assert (params.snapshot, params.auto_stop_interval, params.env_vars, params.network_block_all) == (
        'python',
        0,
        {'A': 'b'},
        True,
    )


async def test_explicit_ref_attaches_without_creation(fake_daytona: FakeDaytona) -> None:
    existing = fake_daytona.sandbox('existing')
    backend = DaytonaWorkspaceBackend(ref=WorkspaceRef(provider='daytona', id=existing.id))
    await backend.workspace
    assert backend.ref == WorkspaceRef(provider='daytona', id=existing.id)
    assert not fake_daytona.create_params


async def test_foreign_ref_is_declined_and_backend_rejects_it() -> None:
    assert DaytonaWorkspace().get_workspace(_ctx(), ref=WorkspaceRef(provider='other', id='x')) is None
    with pytest.raises(ValueError, match="expected 'daytona'"):
        DaytonaWorkspaceBackend(ref=WorkspaceRef(provider='other', id='x'))


async def test_native_ref_conflict_and_native_identity(fake_daytona: FakeDaytona) -> None:
    seed = DaytonaWorkspaceBackend()
    native = await seed.workspace
    backend = DaytonaWorkspaceBackend(workspace=native)
    assert await backend.workspace is native
    assert backend.ref == WorkspaceRef(provider='daytona', id=native.id)
    with pytest.raises(ValueError, match='either `workspace` or `ref`'):
        DaytonaWorkspaceBackend(workspace=native, ref=backend.ref)


async def test_agent_without_workspace_use_does_not_create(fake_daytona: FakeDaytona) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[DaytonaWorkspace()]).run('hello')
    assert result.output == 'done'
    assert not fake_daytona.create_params


async def test_agent_runs_create_fresh_resources_without_history(fake_daytona: FakeDaytona) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[DaytonaWorkspace()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['echo', 'ok'])).stdout

    await agent.run('run')
    await agent.run('run')
    assert len(fake_daytona.create_params) == 2


async def test_agent_history_attaches_persisted_workspace(fake_daytona: FakeDaytona) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[DaytonaWorkspace()])

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('/marker'):
            return (await ctx.workspace.read_bytes('/marker')).decode()
        await ctx.workspace.write_bytes('/marker', b'saved')
        return 'created'

    first = await agent.run('remember')
    second = await agent.run('continue', message_history=first.all_messages())
    assert 'done' in first.output and 'done' in second.output
    assert fake_daytona.create_params and len(fake_daytona.create_params) == 1
    assert len(fake_daytona.sandboxes) == 1
    assert fake_daytona.sandboxes[0].start_calls == [60]


async def test_read_only_facade_identity_and_denied_command(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaWorkspaceBackend()
    facade = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(call_tools=['check']), capabilities=[DaytonaWorkspace()])

    @agent.tool
    async def check(ctx: RunContext[object]) -> str:
        assert ctx.workspace is facade
        with pytest.raises(UserError, match='read-only'):
            await ctx.workspace.run(['echo', 'blocked'])
        return 'ok'

    result = await agent.run('check', workspace=facade)
    assert result.workspace is facade
    assert 'ok' in result.output
    assert not fake_daytona.create_params


async def test_concurrent_acquisition_and_operation_overlap(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaWorkspaceBackend()
    results: list[object] = []

    async def acquire() -> None:
        results.append(await backend.workspace)

    async with anyio.create_task_group() as tg:
        tg.start_soon(acquire)
        tg.start_soon(acquire)
    assert len(fake_daytona.create_params) == 1
    assert results[0] is results[1]
