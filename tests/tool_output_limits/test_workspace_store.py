"""Spilled tool results live in the run's workspace by default."""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    Workspace,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceFileEntry,
    WorkspaceRef,
)

from pydantic_ai_harness.tool_output_limits import (
    READ_TOOL_NAME,
    Band,
    Spill,
    ToolOutputLimits,
    Truncate,
    WorkspaceStore,
)

from .._workspace import HOST_ENV, local_workspace

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _returns(messages: Sequence[ModelMessage], tool_name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


class _FilesystemOnly:
    """A workspace backend with file operations but no command execution, like a mounted bucket."""

    def __init__(self, working_dir: Path) -> None:
        self._local = local_workspace(working_dir)

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._local.ref  # pragma: no cover - not used by the store

    async def working_dir(self) -> str:
        return await self._local.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self._local.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._local.write_bytes(path, data)

    async def stat(self, path: str) -> WorkspaceFileEntry:
        return await self._local.stat(path)  # pragma: no cover - not used by the store

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        return await self._local.list_dir(path)  # pragma: no cover - not used by the store

    async def make_dir(self, path: str) -> None:
        await self._local.make_dir(path)  # pragma: no cover - not used by the store

    async def remove(self, path: str) -> None:
        await self._local.remove(path)  # pragma: no cover - not used by the store

    async def exists(self, path: str) -> bool:
        return await self._local.exists(path)  # pragma: no cover - not used by the store


class _TempDirCommand(LocalWorkspaceBackend):
    """A local backend whose commands all return one fixed result."""

    def __init__(self, working_dir: Path, result: CommandResult) -> None:
        super().__init__(working_dir)
        self.result = result

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        return self.result


class _FailingRead(LocalWorkspaceBackend):
    async def read_bytes(self, path: str) -> bytes:
        raise WorkspaceError('sandbox refused the read')


@dataclasses.dataclass
class _Ctx:
    workspace: Workspace


class TestDefaultStore:
    async def test_spill_and_read_back_through_the_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        work = tmp_path / 'work'
        work.mkdir()
        sandbox_tmp = tmp_path / 'sandbox-tmp'
        monkeypatch.setenv('TMPDIR', str(sandbox_tmp))
        payload = '\n'.join(f'line {i}' for i in range(500))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if read := _returns(messages, READ_TOOL_NAME):
                return ModelResponse(parts=[TextPart(str(read[0].content))])
            if spilled := _returns(messages, 'big_tool'):
                assert spilled[0].metadata is not None
                handle = spilled[0].metadata['overflow_handle']
                return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': handle, 'limit': 2})])
            return ModelResponse(parts=[ToolCallPart('big_tool', {})])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[ToolOutputLimits(bands=[Band(over=100, action=Spill())]), LocalWorkspace(work, env=HOST_ENV)],
        )

        @agent.tool_plain
        def big_tool() -> str:
            return payload

        result = await agent.run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert spilled.metadata is not None
        handle = spilled.metadata['overflow_handle']
        assert handle.startswith(f'{sandbox_tmp}/pydantic-ai-harness/tool-output/')
        assert Path(handle).read_text() == payload
        assert oct((sandbox_tmp / 'pydantic-ai-harness' / 'tool-output').stat().st_mode & 0o777) == '0o700'
        assert result.output == f'[handle {handle!r}: 500 matching line(s); showing 2]\nline 0\nline 1'

    async def test_no_workspace_spills_to_the_host(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
        payload = '\n'.join(f'line {i}' for i in range(500))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if read := _returns(messages, READ_TOOL_NAME):
                return ModelResponse(parts=[TextPart(str(read[0].content))])
            if spilled := _returns(messages, 'big_tool'):
                assert spilled[0].metadata is not None
                handle = spilled[0].metadata['overflow_handle']
                return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': handle, 'limit': 2})])
            return ModelResponse(parts=[ToolCallPart('big_tool', {})])

        agent = Agent(FunctionModel(respond), capabilities=[ToolOutputLimits(bands=[Band(over=100, action=Spill())])])

        @agent.tool_plain
        def big_tool() -> str:
            return payload

        result = await agent.run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert spilled.metadata is not None
        handle = spilled.metadata['overflow_handle']
        assert not handle.startswith('/')
        assert result.output == f'[handle {handle!r}: 500 matching line(s); showing 2]\nline 0\nline 1'

    async def test_explicit_workspace_store_without_workspace_warns_and_falls_back(self):
        agent = Agent(
            FunctionModel(
                lambda messages, info: ModelResponse(
                    parts=[TextPart('done') if _returns(messages, 'big_tool') else ToolCallPart('big_tool', {})]
                )
            ),
            capabilities=[
                ToolOutputLimits(
                    bands=[Band(over=100, action=Spill(then=Truncate(max_chars=150)))], store=WorkspaceStore()
                )
            ],
        )

        @agent.tool_plain
        def big_tool() -> str:
            return 'x' * 1_000

        with pytest.warns(UserWarning, match="could not spill a 'big_tool' result: No workspace is attached"):
            result = await agent.run('go')

        [part] = _returns(result.all_messages(), 'big_tool')
        assert isinstance(part.content, str) and 'truncated' in part.content

    async def test_read_without_workspace_guides_the_model(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if read := _returns(messages, READ_TOOL_NAME):
                return ModelResponse(parts=[TextPart(str(read[0].content))])
            return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': 'call-1'})])

        result = await Agent(FunctionModel(respond), capabilities=[ToolOutputLimits()]).run('go')
        assert result.output.startswith("[No stored tool result for handle 'call-1'.")

    async def test_workspace_failure_on_read_is_a_failed_tool_call(self, tmp_path: Path):
        toolset = ToolOutputLimits(store=WorkspaceStore(directory='spills')).get_toolset()
        assert toolset is not None
        tool = toolset.tools[READ_TOOL_NAME]  # type: ignore[union-attr]
        ctx = _Ctx(workspace=Workspace(_FailingRead(tmp_path)))
        with pytest.raises(ToolFailed, match='sandbox refused the read'):
            await tool.function(ctx, 'run/call.0')  # type: ignore[attr-defined]


class TestWorkspaceStore:
    async def test_directory_relative_to_working_dir(self, tmp_path: Path):
        workspace = Workspace(local_workspace(tmp_path))
        store = WorkspaceStore(directory='spills')
        handle = await store.write(workspace, 'run-1/../call 1.0', b'payload')
        assert handle == f'{tmp_path.resolve()}/spills/run-1/_/call_1.0'
        assert await store.read(workspace, handle) == b'payload'
        assert await store.read(workspace, 'run-1/_/call_1.0') == b'payload'
        assert await store.write(workspace, '', b'empty') == f'{tmp_path.resolve()}/spills/_'

    @pytest.mark.parametrize('handle', ['../secret.txt', '/etc/passwd', '.'])
    async def test_read_outside_directory_is_refused(self, tmp_path: Path, handle: str):
        (tmp_path / 'secret.txt').write_text('secret')
        workspace = Workspace(local_workspace(tmp_path))
        with pytest.raises(PermissionError, match='outside the store directory'):
            await WorkspaceStore(directory='spills').read(workspace, handle)

    async def test_missing_handle_raises_file_not_found(self, tmp_path: Path):
        workspace = Workspace(local_workspace(tmp_path))
        with pytest.raises(FileNotFoundError):
            await WorkspaceStore(directory='spills').read(workspace, 'run/missing.0')

    async def test_filesystem_only_workspace_uses_working_dir(self, tmp_path: Path):
        workspace = Workspace(_FilesystemOnly(tmp_path))
        store = WorkspaceStore()
        handle = await store.write(workspace, 'run/call.0', b'data')
        assert handle == f'{tmp_path.resolve()}/.pydantic-ai-harness/tool-output/run/call.0'
        assert await store.read(workspace, handle) == b'data'

    @pytest.mark.parametrize(
        ('result', 'message'),
        [
            (CommandResult(exit_code=1, stdout='', stderr='mkdir: denied\n'), 'mkdir: denied'),
            (CommandResult(exit_code=0, stdout='relative/dir', stderr=''), 'no writable temporary directory'),
        ],
    )
    async def test_unusable_temp_dir_raises(self, tmp_path: Path, result: CommandResult, message: str):
        workspace = Workspace(_TempDirCommand(tmp_path, result))
        with pytest.raises(WorkspaceError, match=message):
            await WorkspaceStore().write(workspace, 'run/call.0', b'data')
