from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import anyio
import anyio.to_thread
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from sprites import Sprite, SpritesClient
from sprites.exceptions import AuthenticationError, NotFoundError, SpriteError
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from pydantic_ai_harness.sprites import SpriteWorkspace, SpriteWorkspaceBackend

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class FakeOperation:
    def __init__(self, transport: SpriteTransport, args: list[str]) -> None:
        self.transport = transport
        self.args = args
        self.stdout = b''
        self.stderr = b''
        self.exit_override = transport.run_exit_override if len(args) == 6 else transport.cancel_exit_override
        self._task = asyncio.create_task(asyncio.to_thread(self._execute))

    def _execute(self) -> int:
        self.transport.commands.append(self.args)
        if len(self.args) == 6:
            self.transport.controls.add(self.args[4])
            self.transport.started.set()
            if self.transport.release_start is not None:
                assert self.transport.release_start.wait(5)
        process = subprocess.Popen(
            [sys.executable, *self.args[1:]], cwd=self.transport.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert process.stdout is not None and process.stderr is not None
        stdout = bytearray()
        stderr = bytearray()

        def read_stream(source: BinaryIO, target: bytearray, output: str) -> None:
            while chunk := source.read(1):
                target.extend(chunk)
                setattr(self, output, bytes(target))
            source.close()

        readers = [
            threading.Thread(target=read_stream, args=(process.stdout, stdout, 'stdout')),
            threading.Thread(target=read_stream, args=(process.stderr, stderr, 'stderr')),
        ]
        for reader in readers:
            reader.start()
        code = process.wait(timeout=10)
        for reader in readers:
            reader.join()
        self.stdout, self.stderr = bytes(stdout), bytes(stderr)
        return code

    async def wait(self) -> int:
        result = await asyncio.shield(self._task)
        if len(self.args) == 6 and self.transport.run_stderr:
            self.stderr = self.transport.run_stderr
        return self.exit_override if self.exit_override is not None else result

    def get_stdout(self) -> bytes:
        return self.stdout

    def get_stderr(self) -> bytes:
        return self.stderr


class FakeControlConnection:
    transport: SpriteTransport

    def __init__(self, sprite: Sprite) -> None:
        self.sprite = sprite
        self.close_error = self.transport.control_close_error
        self.closed = False

    async def connect(self) -> None:
        if self.transport.connect_error is not None:
            raise self.transport.connect_error

    async def start_op(self, op: str, *, cmd: list[str], stdin: bool) -> FakeOperation:
        assert op == 'exec'
        assert stdin is False
        return FakeOperation(self.transport, cmd)

    async def close(self) -> None:
        if self.transport.control_close_hang:
            await anyio.sleep(1)
        if self.transport.control_close_error is not None:
            raise self.transport.control_close_error
        self.closed = True


class SpriteTransport:
    """SDK acquisition fake; public control payloads execute in local subprocesses."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.names: set[str] = set()
        self.created: list[str] = []
        self.clients: list[SpritesClient] = []
        self.controls: set[str] = set()
        self.commands: list[list[str]] = []
        self.creation_error: SpriteError | None = None
        self.get_error: SpriteError | None = None
        self.close_error: Exception | None = None
        self.close_calls = 0
        self.connect_error: Exception | None = None
        self.control_close_error: Exception | None = None
        self.control_close_hang = False
        self.run_exit_override: int | None = None
        self.cancel_exit_override: int | None = None
        self.run_stderr = b''
        self.started = threading.Event()
        self.release_start: threading.Event | None = None

    def client(self, token: str, base_url: str, timeout: float) -> SpritesClient:
        client = SpritesClient(token=token, base_url=base_url, timeout=timeout)
        self.clients.append(client)
        return client

    def get(self, client: SpritesClient, name: str) -> Sprite:
        if self.get_error is not None:
            raise self.get_error
        if name not in self.names:
            raise NotFoundError(name)
        return Sprite(name, client)

    def create(self, client: SpritesClient, name: str, *, runtime: str | None) -> Sprite:
        if self.creation_error is not None:
            raise self.creation_error
        self.names.add(name)
        self.created.append(name)
        return Sprite(name, client)

    def close(self, client: SpritesClient) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            error = self.close_error
            self.close_error = None
            raise error


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[SpriteTransport]:
    transport = SpriteTransport(tmp_path)
    monkeypatch.setenv('SPRITE_TOKEN', 'test-token')
    monkeypatch.setattr('pydantic_ai_harness.sprites._backend.SpritesClient', transport.client)
    FakeControlConnection.transport = transport
    monkeypatch.setattr('pydantic_ai_harness.sprites._backend.ControlConnection', FakeControlConnection)

    def get(client: SpritesClient, name: str) -> Sprite:
        return transport.get(client, name)

    def create(client: SpritesClient, name: str, *, runtime: str | None) -> Sprite:
        return transport.create(client, name, runtime=runtime)

    monkeypatch.setattr(SpritesClient, 'get_sprite', get)
    monkeypatch.setattr(SpritesClient, 'create_sprite', create)

    def close(client: SpritesClient) -> None:
        transport.close(client)

    monkeypatch.setattr(SpritesClient, 'close', close)
    yield transport
    for client in transport.clients:
        client.close()
    for control in transport.controls:
        shutil.rmtree(control, ignore_errors=True)


def context(conversation: str = 'chat') -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), conversation_id=conversation, run_id='run')


class TestSpriteWorkspace:
    async def test_construction_is_lazy_and_first_use_is_shared(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspace[None]().get_workspace(context(), ref=None)
        assert isinstance(backend, SpriteWorkspaceBackend)
        assert transport.clients == []
        first, second = await asyncio.gather(backend.workspace, backend.workspace)
        assert first is second
        assert transport.created == [first.name]
        assert backend.ref == WorkspaceRef(provider='sprites', id=first.name)

    def test_foreign_reference_is_declined_and_backend_rejects_it(self, transport: SpriteTransport) -> None:
        assert SpriteWorkspace[None]().get_workspace(context(), ref=WorkspaceRef(provider='other', id='x')) is None
        with pytest.raises(ValueError, match="expected 'sprites'"):
            SpriteWorkspaceBackend(ref=WorkspaceRef(provider='other', id='x'))

    async def test_native_handle_conflict_and_identity(self, transport: SpriteTransport) -> None:
        seed = SpriteWorkspaceBackend()
        native = await seed.workspace
        backend = SpriteWorkspaceBackend(workspace=native)
        assert await backend.workspace is native
        assert backend.ref == WorkspaceRef(provider='sprites', id=native.name)
        with pytest.raises(ValueError, match='either `workspace` or `ref`'):
            SpriteWorkspaceBackend(workspace=native, ref=backend.ref)

    async def test_agent_without_workspace_use_does_not_create(self, transport: SpriteTransport) -> None:
        result = await Agent(TestModel(custom_output_text='done'), capabilities=[SpriteWorkspace()]).run('go')
        assert result.output == 'done'
        assert transport.created == []

    async def test_agent_files_and_result_survive_run_end(self, transport: SpriteTransport) -> None:
        agent = Agent(TestModel(), capabilities=[SpriteWorkspace()])

        @agent.tool
        async def write(ctx: RunContext[object]) -> str:
            await ctx.workspace.write_bytes('result.bin', b'\x00\xff\n')
            return 'written'

        result = await agent.run('write')
        assert result.workspace is not None
        assert await result.workspace.read_bytes('result.bin') == b'\x00\xff\n'
        assert len(transport.names) == 1

    async def test_missing_reference_does_not_recreate(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='missing'))
        with pytest.raises(WorkspaceUnavailableError):
            await backend.workspace
        assert transport.created == []

    @pytest.mark.parametrize('error_type', [AuthenticationError, SpriteError])
    async def test_creation_error_preserves_cause(
        self, transport: SpriteTransport, error_type: type[SpriteError]
    ) -> None:
        error = error_type('failed')
        transport.creation_error = error
        with pytest.raises(WorkspaceError) as caught:
            await SpriteWorkspaceBackend().workspace
        assert caught.value.__cause__ is error

    async def test_missing_token(self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SPRITE_TOKEN')
        with pytest.raises(WorkspaceUnavailableError, match='SPRITE_TOKEN'):
            await SpriteWorkspaceBackend().workspace

    async def test_connection_settings_and_disconnect(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend(token='sentinel', base_url='https://example.invalid', api_timeout=7)
        native = await backend.workspace
        assert native.client.token == 'sentinel'
        assert native.client.base_url == 'https://example.invalid'
        await backend.disconnect()
        await backend.disconnect()
        assert native.name in transport.names
        assert (await backend.run(['true'])).exit_code == 0

    async def test_disconnect_before_use_does_not_create(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.disconnect()
        assert transport.clients == []

    async def test_disconnect_retries_owned_client_and_preserves_remote(self, transport: SpriteTransport) -> None:
        transport.names.add('remote')
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='remote'))
        await backend.workspace
        transport.close_error = RuntimeError('temporary')
        with pytest.raises(WorkspaceError):
            await backend.disconnect()
        assert backend.ref == WorkspaceRef(provider='sprites', id='remote')
        await backend.disconnect()
        assert 'remote' in transport.names
        assert (await backend.workspace).name == 'remote'

    async def test_injected_client_is_never_closed(self, transport: SpriteTransport) -> None:
        transport.names.add('owned-by-caller')
        client = transport.client('test-token', 'https://api.sprites.dev', 30)
        backend = SpriteWorkspaceBackend(client=client, ref=WorkspaceRef(provider='sprites', id='owned-by-caller'))
        await backend.workspace
        await backend.disconnect()
        assert transport.close_calls == 0
        assert (await backend.workspace).name == 'owned-by-caller'

    @pytest.mark.parametrize(
        'status_code,expected_type',
        [
            (401, WorkspaceUnavailableError),
            (404, WorkspaceUnavailableError),
            (500, WorkspaceError),
        ],
    )
    async def test_cached_control_handshake_errors_are_typed(
        self, transport: SpriteTransport, status_code: int, expected_type: type[WorkspaceError]
    ) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        error = InvalidStatus(Response(status_code, 'status', Headers()))
        transport.connect_error = error
        with pytest.raises(expected_type) as caught:
            await backend.run(['true'])
        assert caught.value.__cause__ is error
        if status_code in (401, 404):
            assert isinstance(caught.value, WorkspaceUnavailableError)
        else:
            assert not isinstance(caught.value, WorkspaceUnavailableError)

    async def test_control_close_failure_after_exit_preserves_cause(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        error = RuntimeError('close failed')
        transport.control_close_error = error
        with pytest.raises(WorkspaceError) as caught:
            await backend.run(['true'])
        assert caught.value.__cause__ is error
        assert 'close Sprite command connection' in str(caught.value)

    async def test_control_close_timeout_is_provider_error(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        transport.control_close_hang = True
        monkeypatch.setattr('pydantic_ai_harness.sprites._backend._CONTROL_TIMEOUT', 0.01)
        with pytest.raises(WorkspaceError) as caught:
            await backend.run(['true'])
        assert isinstance(caught.value.__cause__, TimeoutError)
        assert 'cleanup bound' in str(caught.value)

    @pytest.mark.parametrize('run_stderr', [b'connection closed', b''])
    async def test_transport_loss_preserves_cause_and_requests_cancel(
        self, transport: SpriteTransport, run_stderr: bytes
    ) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        error = RuntimeError('control connection lost')
        transport.run_exit_override = -1
        transport.run_stderr = run_stderr
        transport.control_close_error = error
        with pytest.raises(WorkspaceError) as caught:
            await backend.run(['true'])
        assert caught.value.__cause__ is error
        if run_stderr:
            assert run_stderr.decode() in str(caught.value)
        assert any(len(command) == 5 for command in transport.commands)

    @pytest.mark.parametrize(
        'lookup_error,expected_type',
        [
            (SpriteError('lookup failed'), WorkspaceError),
            (AuthenticationError('bad token'), WorkspaceUnavailableError),
        ],
    )
    async def test_run_acquisition_error_preserves_type(
        self, transport: SpriteTransport, lookup_error: SpriteError, expected_type: type[WorkspaceError]
    ) -> None:
        transport.get_error = lookup_error
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='target'))
        with pytest.raises(expected_type) as caught:
            await backend.run(['true'])
        assert caught.value.__cause__ is lookup_error
        if expected_type is WorkspaceError:
            assert type(lookup_error).__name__ in str(caught.value)

    async def test_timeout_cancels_before_original_close_finishes(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        transport.control_close_hang = True
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run('sleep .5; touch escaped', shell=True, timeout=0.01)
        assert not (transport.root / 'escaped').exists()

    async def test_cancel_failure_and_primary_error_are_both_reported(
        self, transport: SpriteTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        transport.cancel_exit_override = 1
        transport.control_close_error = RuntimeError('close failed')
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run('sleep 1', shell=True, timeout=0.01)
        assert 'Could not confirm remote Sprite command termination' in caplog.text
        assert 'Could not close Sprite cancellation connection' in caplog.text
        assert 'Could not close original Sprite command connection' in caplog.text

    async def test_argv_shell_environment_and_nonzero_exit(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        result = await backend.run(['/bin/echo', 'a; echo injected'])
        assert result.stdout == 'a; echo injected\n'
        result = await backend.run('printf "$VALUE"; printf error >&2; exit 124', shell=True, env={'VALUE': 'hello'})
        assert (result.exit_code, result.stdout, result.stderr) == (124, 'hello', 'error')

    async def test_canonical_working_directory_preserves_spaces(self, transport: SpriteTransport) -> None:
        target = transport.root / ' directory '
        target.mkdir()
        link = transport.root / 'link'
        link.symlink_to(target)
        backend = SpriteWorkspaceBackend(working_dir=str(link))
        assert await backend.working_dir() == str(target.resolve())
        assert await backend.working_dir() == str(target.resolve())

    async def test_missing_working_directory(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend(working_dir=str(transport.root / 'absent'))
        with pytest.raises(WorkspaceError, match='working directory'):
            await backend.working_dir()

    async def test_filesystem_fallback_handles_directories_and_binary(self, transport: SpriteTransport) -> None:
        sandbox = Workspace(SpriteWorkspaceBackend())
        await sandbox.make_dir('folder')
        await sandbox.write_bytes('folder/a\nb', b'\x00\xff')
        assert await sandbox.read_bytes('folder/a\nb') == b'\x00\xff'
        assert (await sandbox.stat('folder')).is_dir
        assert [entry.name for entry in await sandbox.list_dir('folder')] == ['a\nb']
        await sandbox.remove('folder')
        assert not await sandbox.exists('folder')

    async def test_deadline_kills_child_and_preserves_partial_output(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        with pytest.raises(WorkspaceTimeoutError) as caught:
            await backend.run('printf ready; sleep 1; touch escaped', shell=True, timeout=0.3)
        await anyio.sleep(1)
        assert not (transport.root / 'escaped').exists()
        assert caught.value.stdout == 'ready'

    async def test_cancellation_before_remote_start_prevents_command(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.workspace
        transport.release_start = threading.Event()
        task = asyncio.create_task(backend.run(['touch', 'escaped']))
        try:
            assert await anyio.to_thread.run_sync(transport.started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            transport.release_start.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await anyio.sleep(0.2)
        assert not (transport.root / 'escaped').exists()

    @pytest.mark.parametrize('timeout', [0, -1, float('inf')])
    async def test_invalid_timeout(self, transport: SpriteTransport, timeout: float) -> None:
        with pytest.raises(ValueError, match='timeout'):
            await SpriteWorkspaceBackend().run(['true'], timeout=timeout)

    @pytest.mark.parametrize('command,shell', [('true', False), ([], False), (['true'], True)])
    async def test_invalid_command(self, transport: SpriteTransport, command: str | list[str], shell: bool) -> None:
        with pytest.raises(TypeError):
            await SpriteWorkspaceBackend().run(command, shell=shell)

    def test_relative_working_dir_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='absolute'):
            SpriteWorkspaceBackend(working_dir='relative')
