"""Public E2B workspace capability tests."""

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
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceError, WorkspaceRef

from pydantic_ai_harness.e2b_workspace import E2BWorkspace, E2BWorkspaceBackend

from .fake_e2b import FakeE2B

pytestmark = pytest.mark.anyio


def test_capability_uses_workspace_contract() -> None:
    capability = E2BWorkspace()
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    backend = capability.get_workspace(ctx, ref=WorkspaceRef(provider='e2b', id='known'))
    assert isinstance(backend, E2BWorkspaceBackend)
    assert backend.ref == WorkspaceRef(provider='e2b', id='known')


def test_foreign_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 'e2b'"):
        E2BWorkspaceBackend(ref=WorkspaceRef(provider='modal', id='other'))


def test_capability_declines_foreign_reference() -> None:
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert E2BWorkspace().get_workspace(ctx, ref=WorkspaceRef(provider='modal', id='other')) is None


async def test_native_handle_is_exposed_without_creation(fake_e2b: FakeE2B) -> None:
    seed_backend = E2BWorkspaceBackend()
    native = await seed_backend.workspace
    fake_e2b.create_calls.clear()
    backend = E2BWorkspaceBackend(workspace=native)
    assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
    assert await backend.workspace is native
    assert not fake_e2b.create_calls
    with pytest.raises(ValueError, match='either `workspace` or `ref`'):
        E2BWorkspaceBackend(workspace=native, ref=WorkspaceRef(provider='e2b', id='other'))


async def test_agent_without_workspace_use_does_not_create(fake_e2b: FakeE2B) -> None:
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[E2BWorkspace()])
    result = await agent.run('hello')
    assert result.output == 'done'
    assert not fake_e2b.create_calls


async def test_concurrent_first_use_creates_once(fake_e2b: FakeE2B) -> None:
    backend = E2BWorkspaceBackend()
    results: list[object] = []

    async def acquire() -> None:
        results.append(await backend.workspace)

    async with anyio.create_task_group() as tg:
        tg.start_soon(acquire)
        tg.start_soon(acquire)

    assert len(fake_e2b.create_calls) == 1
    assert results[0] is results[1]


async def test_capability_forwards_creation_options_and_run_does_not_kill(fake_e2b: FakeE2B) -> None:
    capability = E2BWorkspace(
        template='base',
        sandbox_timeout=120,
        workdir='/work',
        env={'FOO': 'bar'},
        metadata={'owner': 'test'},
        allow_internet_access=False,
    )
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[capability])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['echo', 'ok'])).stdout

    result = await agent.run('run', deps=None)
    assert 'ok' in result.output
    call = fake_e2b.create_calls[0]
    assert (call.template, call.timeout, call.envs, call.metadata, call.allow_internet_access) == (
        'base',
        120,
        {'FOO': 'bar'},
        {'owner': 'test'},
        False,
    )
    assert fake_e2b.sandboxes[0].killed is False
    assert isinstance(result.workspace, Workspace)
    backend = result.workspace.backend
    assert isinstance(backend, E2BWorkspaceBackend)
    native = await backend.workspace
    await native.kill()
    assert fake_e2b.sandboxes[0].killed is True


async def test_explicit_ref_uses_attach_even_with_creation_options(fake_e2b: FakeE2B) -> None:
    capability = E2BWorkspace(template='base', env={'FOO': 'bar'}, metadata={'owner': 'test'})
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    backend = capability.get_workspace(ctx, ref=WorkspaceRef(provider='e2b', id='existing'))
    assert isinstance(backend, E2BWorkspaceBackend)
    await backend.workspace
    assert fake_e2b.connect_calls == [('existing', None)]
    assert not fake_e2b.create_calls


async def test_post_acquisition_operations_overlap(fake_e2b: FakeE2B) -> None:
    backend = E2BWorkspaceBackend()
    await backend.workspace
    fake_e2b.command_hangs = True

    async def run_command() -> None:
        await backend.run(['sleep', '1'])

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_command)
        tg.start_soon(run_command)
        with anyio.fail_after(1):
            while len(fake_e2b.sandboxes[0].commands.calls) < 2:
                await anyio.sleep(0)
        tg.cancel_scope.cancel()
    assert len(fake_e2b.sandboxes[0].commands.calls) == 2


async def test_agent_preserves_explicit_read_only_workspace(fake_e2b: FakeE2B) -> None:
    backend = E2BWorkspaceBackend()
    facade = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(call_tools=['check_workspace']), capabilities=[E2BWorkspace()])

    @agent.tool
    async def check_workspace(ctx: RunContext[object]) -> str:
        assert ctx.workspace is facade
        with pytest.raises(UserError, match='read-only'):
            await ctx.workspace.run(['echo', 'blocked'])
        return 'checked'

    result = await agent.run('check', workspace=facade)
    assert 'checked' in result.output
    assert result.workspace is facade
    assert not fake_e2b.create_calls


async def test_failed_acquisition_can_retry(fake_e2b: FakeE2B) -> None:
    backend = E2BWorkspaceBackend()
    fake_e2b.create_error = fake_e2b.error_type('temporary')
    with pytest.raises(WorkspaceError, match='temporary'):
        await backend.workspace
    fake_e2b.create_error = None
    assert await backend.workspace is fake_e2b.sandboxes[0]
    assert len(fake_e2b.create_calls) == 2


async def test_cancelled_acquisition_can_retry(fake_e2b: FakeE2B) -> None:
    backend = E2BWorkspaceBackend()
    fake_e2b.create_hangs = True

    async def acquire() -> None:
        await backend.workspace

    async with anyio.create_task_group() as tg:
        tg.start_soon(acquire)
        with anyio.fail_after(1):
            while not fake_e2b.create_calls:
                await anyio.sleep(0)
        tg.cancel_scope.cancel()

    fake_e2b.create_hangs = False
    assert await backend.workspace is fake_e2b.sandboxes[0]
    assert len(fake_e2b.create_calls) == 2


async def test_agent_runs_without_history_create_fresh_workspaces(fake_e2b: FakeE2B) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[E2BWorkspace()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['printf', 'ok'])).stdout

    await agent.run('go')
    await agent.run('go')
    assert len(fake_e2b.sandboxes) == 2


async def test_agent_history_attaches_same_workspace(fake_e2b: FakeE2B) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        if isinstance(request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='run_command', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[E2BWorkspace()])
    tool_results: list[str] = []

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('/marker.txt'):
            result = await ctx.workspace.read_text('/marker.txt')
        else:
            await ctx.workspace.write_text('/marker.txt', 'persisted')
            result = 'created'
        tool_results.append(result)
        return result

    first = await agent.run('go')
    second = await agent.run('go', message_history=first.all_messages())
    assert first.output == 'done'
    assert second.output == 'done'
    assert tool_results == ['created', 'persisted']
    assert len(fake_e2b.sandboxes) == 1
    assert first.workspace.ref is not None
    assert fake_e2b.connect_calls == [('sbx-1', None)]
