"""Focused public tests for the Modal workspace capability."""

from __future__ import annotations

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import Workspace, WorkspaceError, WorkspaceRef, WorkspaceTimeoutError

from pydantic_ai_harness.modal_workspace import ModalWorkspace, ModalWorkspaceBackend

from .fake_modal import FakeModal

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def test_backend_acquires_fresh_workspace_and_records_ref(fake_modal: FakeModal) -> None:
    backend = ModalWorkspaceBackend()
    assert not fake_modal.sandboxes
    native = await backend.workspace
    assert native is fake_modal.sandboxes[0]
    assert backend.ref == WorkspaceRef(provider='modal', id=native.object_id)


async def test_backend_attaches_explicit_ref_without_create(fake_modal: FakeModal) -> None:
    backend = ModalWorkspaceBackend(ref=WorkspaceRef(provider='modal', id='existing'))
    await backend.workspace
    assert fake_modal.attach_ids == ['existing']
    assert not fake_modal.create_kwargs


async def test_native_workspace_identity_is_immediate(fake_modal: FakeModal) -> None:
    native = await ModalWorkspaceBackend().workspace
    backend = ModalWorkspaceBackend(workspace=native)
    assert backend.ref == WorkspaceRef(provider='modal', id=native.object_id)
    assert await backend.workspace is native


async def test_native_workspace_and_ref_conflict(fake_modal: FakeModal) -> None:
    native = await ModalWorkspaceBackend().workspace
    with pytest.raises(ValueError, match='either `workspace` or `ref`'):
        ModalWorkspaceBackend(workspace=native, ref=WorkspaceRef(provider='modal', id='other'))


async def test_filesystem_directory_error_uses_builtin_exception(fake_modal: FakeModal) -> None:
    backend = ModalWorkspaceBackend()
    await backend.workspace
    fake_modal.sandboxes[0].directories.add('/directory')
    with pytest.raises(IsADirectoryError, match='Is a directory'):
        await backend.read_bytes('/directory')


async def test_command_start_timeout_is_bounded(fake_modal: FakeModal) -> None:
    fake_modal.exec_hangs = True
    backend = ModalWorkspaceBackend()
    with pytest.raises(WorkspaceTimeoutError, match='before the command could start') as exc_info:
        with anyio.fail_after(0.2):
            await backend.run(['echo', 'hello'], timeout=0.01)
    assert exc_info.value.timeout == 0.01


async def test_command_timeout_keeps_captured_output(fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai_harness.modal_workspace._backend._RESULT_GRACE', 0.01)
    fake_modal.responder = lambda argv, timeout: ('partial stdout', 'partial stderr', 0)
    fake_modal.wait_hangs = True
    backend = ModalWorkspaceBackend()
    with pytest.raises(WorkspaceTimeoutError) as exc_info:
        await backend.run(['echo', 'hello'], timeout=0.01)
    assert exc_info.value.stdout == 'partial stdout'
    assert exc_info.value.stderr == 'partial stderr'


async def test_capability_declines_foreign_ref(fake_modal: FakeModal) -> None:
    capability = ModalWorkspace()
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert capability.get_workspace(ctx, ref=WorkspaceRef(provider='other', id='foreign')) is None
    assert not fake_modal.sandboxes


async def test_backend_rejects_foreign_ref(fake_modal: FakeModal) -> None:
    with pytest.raises(ValueError, match="expected 'modal'"):
        ModalWorkspaceBackend(ref=WorkspaceRef(provider='other', id='foreign'))
    assert not fake_modal.sandboxes


async def test_agent_without_workspace_tool_does_not_create(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(), capabilities=[ModalWorkspace()])
    result = await agent.run('go')
    assert result.output
    assert not fake_modal.sandboxes


async def test_agent_runs_without_history_create_fresh_workspaces(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[ModalWorkspace()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['printf', 'ok'])).stdout

    await agent.run('go')
    await agent.run('go')
    assert len(fake_modal.sandboxes) == 2


async def test_agent_history_attaches_same_workspace(fake_modal: FakeModal) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        if isinstance(request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='run_command', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[ModalWorkspace()])
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
    assert len(fake_modal.sandboxes) == 1
    assert first.workspace.ref is not None
    assert fake_modal.attach_ids == [first.workspace.ref.id]


async def test_concurrent_operations_share_one_acquisition(fake_modal: FakeModal) -> None:
    backend = ModalWorkspaceBackend()
    fake_modal.create_gate = anyio.Event()

    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['printf', 'one'])
        tg.start_soon(backend.run, ['printf', 'two'])
        while not fake_modal.create_started:
            await anyio.sleep(0)
        fake_modal.create_gate.set()
    assert fake_modal.owned_creates == 1
    assert len(fake_modal.sandboxes[0].exec_calls) == 2


async def test_filesystem_operation_is_not_blocked_by_command_wait(
    fake_modal: FakeModal,
) -> None:
    fake_modal.wait_hangs = True
    backend = ModalWorkspaceBackend()
    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['sleep', '1'])
        while not fake_modal.sandboxes:
            await anyio.sleep(0)
        fake_modal.sandboxes[0].files['/marker.txt'] = b'marker'
        with anyio.fail_after(0.2):
            assert await Workspace(backend).read_text('/marker.txt') == 'marker'
        tg.cancel_scope.cancel()


async def test_failed_acquisition_can_retry(fake_modal: FakeModal) -> None:
    backend = ModalWorkspaceBackend()
    fake_modal.create_error = fake_modal.error_type('temporary')
    with pytest.raises(WorkspaceError):
        await backend.workspace
    fake_modal.create_error = None
    await backend.workspace
    assert fake_modal.owned_creates == 1


async def test_cancelled_acquisition_can_retry(fake_modal: FakeModal) -> None:
    backend = ModalWorkspaceBackend()
    fake_modal.create_gate = anyio.Event()

    async def acquire() -> None:
        await backend.workspace

    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as tg:
            tg.start_soon(acquire)
            while not fake_modal.create_started:
                await anyio.sleep(0)
            scope.cancel()
            fake_modal.create_gate.set()
    fake_modal.create_gate = None
    await backend.workspace
    assert fake_modal.owned_creates == 1


async def test_agent_uses_modal_workspace(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[ModalWorkspace()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        result = await ctx.workspace.run(['printf', 'hello'])
        return result.stdout

    result = await agent.run('go')
    assert 'run_command' in result.output
    assert fake_modal.sandboxes[0].exec_calls[0].argv == ['printf', 'hello']
