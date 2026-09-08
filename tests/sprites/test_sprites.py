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
from pydantic_ai.sandboxes import Sandbox, SandboxRef, SandboxTimeoutError
from pydantic_ai.usage import RunUsage
from sprites import Sprite, SpritesClient
from sprites.exceptions import AuthenticationError, NotFoundError, SpriteError

from pydantic_ai_harness.sprites import (
    SpriteSandbox,
    SpriteSandboxAuthError,
    SpriteSandboxBackend,
    SpriteSandboxError,
    SpriteSandboxUnavailableError,
)

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
        return await asyncio.shield(self._task)

    def get_stdout(self) -> bytes:
        return self.stdout

    def get_stderr(self) -> bytes:
        return self.stderr


class FakeControlConnection:
    transport: SpriteTransport

    def __init__(self, sprite: Sprite) -> None:
        self.sprite = sprite
        self.close_error: Exception | None = None
        self.closed = False

    async def connect(self) -> None:
        return None

    async def start_op(self, op: str, *, cmd: list[str], stdin: bool) -> FakeOperation:
        assert op == 'exec'
        assert stdin is False
        return FakeOperation(self.transport, cmd)

    async def close(self) -> None:
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
        self.create_then_error = False
        self.destroyed: list[str] = []
        self.destroy_error: SpriteError | None = None
        self.close_error: Exception | None = None
        self.close_calls = 0
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
        if self.creation_error is not None and not self.create_then_error:
            raise self.creation_error
        self.names.add(name)
        self.created.append(name)
        if self.creation_error is not None:
            raise self.creation_error
        return Sprite(name, client)

    def destroy(self, client: SpritesClient, name: str) -> None:
        if self.destroy_error is not None:
            error = self.destroy_error
            self.destroy_error = None
            raise error
        if name not in self.names:
            raise NotFoundError(name)
        self.names.remove(name)
        self.destroyed.append(name)

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

    def destroy(client: SpritesClient, name: str) -> None:
        transport.destroy(client, name)

    def close(client: SpritesClient) -> None:
        transport.close(client)

    monkeypatch.setattr(SpritesClient, 'destroy_sprite', destroy)
    monkeypatch.setattr(SpritesClient, 'close', close)
    yield transport
    for client in transport.clients:
        client.close()
    for control in transport.controls:
        shutil.rmtree(control, ignore_errors=True)


def context(conversation: str = 'chat') -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), conversation_id=conversation, run_id='run')


class TestSpriteSandbox:
    async def test_construction_is_lazy_and_first_use_is_shared(self, transport: SpriteTransport) -> None:
        backend = SpriteSandbox[None]().get_sandbox(context(), ref=None)
        assert isinstance(backend, SpriteSandboxBackend)
        assert transport.clients == []
        first, second = await asyncio.gather(backend.sandbox, backend.sandbox)
        assert first is second
        assert transport.created == [first.name]
        assert backend.ref == SandboxRef(sandbox_id=first.name)

    async def test_conversation_reuse_and_explicit_reference_precedence(self, transport: SpriteTransport) -> None:
        capability = SpriteSandbox[None](sprite_name='configured')
        transport.names.add('reference')
        selected = capability.get_sandbox(context(), ref=SandboxRef(sandbox_id='reference'))
        await selected.working_dir()
        default = SpriteSandbox[None]()
        a = default.get_sandbox(context(), ref=None)
        b = default.get_sandbox(context(), ref=None)
        await a.working_dir()
        await b.working_dir()
        assert a.ref == b.ref
        assert len(transport.created) == 1

    async def test_agent_files_and_result_survive_run_end(self, transport: SpriteTransport) -> None:
        agent = Agent[None, str](TestModel(), deps_type=type(None), capabilities=[SpriteSandbox[None]()])

        @agent.tool
        async def write(ctx: RunContext[None]) -> str:
            await ctx.sandbox.write_bytes('result.bin', b'\x00\xff\n')
            return 'written'

        result = await agent.run('write')
        assert result.sandbox is not None
        assert await result.sandbox.read_bytes('result.bin') == b'\x00\xff\n'
        assert len(transport.names) == 1

    async def test_missing_reference_does_not_recreate(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend(ref=SandboxRef(sandbox_id='missing'))
        with pytest.raises(SpriteSandboxUnavailableError):
            await backend.sandbox
        assert transport.created == []

    @pytest.mark.parametrize('error_type', [AuthenticationError, SpriteError])
    async def test_creation_error_preserves_cause(
        self, transport: SpriteTransport, error_type: type[SpriteError]
    ) -> None:
        error = error_type('failed')
        transport.creation_error = error
        with pytest.raises(SpriteSandboxError) as caught:
            await SpriteSandboxBackend().sandbox
        assert caught.value.__cause__ is error

    async def test_lost_creation_reply_recovers_same_name(self, transport: SpriteTransport) -> None:
        transport.creation_error = SpriteError('lost reply')
        transport.create_then_error = True
        backend = SpriteSandboxBackend(name='stable')
        assert (await backend.sandbox).name == 'stable'
        assert transport.created == ['stable']

    async def test_missing_token(self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SPRITE_TOKEN')
        with pytest.raises(SpriteSandboxAuthError, match='SPRITE_TOKEN'):
            await SpriteSandboxBackend().sandbox

    async def test_connection_settings_and_disconnect(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend(token='sentinel', base_url='https://example.invalid', api_timeout=7)
        native = await backend.sandbox
        assert native.client.token == 'sentinel'
        assert native.client.base_url == 'https://example.invalid'
        await backend.disconnect()
        await backend.disconnect()
        assert native.name in transport.names
        assert (await backend.run(['true'])).exit_code == 0

    async def test_lifecycle_before_use_does_not_create(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend()
        await backend.disconnect()
        await backend.destroy()
        assert transport.clients == []

    async def test_destroy_saved_reference_without_acquiring(self, transport: SpriteTransport) -> None:
        transport.names.add('known')
        backend = SpriteSandboxBackend(ref=SandboxRef(sandbox_id='known'))
        await backend.destroy()
        assert transport.destroyed == ['known']
        with pytest.raises(SpriteSandboxUnavailableError):
            await backend.run(['true'])

    async def test_destroy_missing_reference_succeeds_and_retry_retains_ref(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend(ref=SandboxRef(sandbox_id='missing'))
        await backend.destroy()
        assert backend.ref == SandboxRef(sandbox_id='missing')
        transport.names.add('retry')
        backend = SpriteSandboxBackend(ref=SandboxRef(sandbox_id='retry'))
        transport.destroy_error = SpriteError('temporary')
        with pytest.raises(SpriteSandboxError) as caught:
            await backend.destroy()
        assert caught.value.__cause__ is not None
        assert backend.ref == SandboxRef(sandbox_id='retry')
        await backend.destroy()
        assert transport.destroyed[-1] == 'retry'

    async def test_disconnect_retries_owned_client_and_preserves_remote(self, transport: SpriteTransport) -> None:
        transport.names.add('remote')
        backend = SpriteSandboxBackend(ref=SandboxRef(sandbox_id='remote'))
        await backend.sandbox
        transport.close_error = RuntimeError('temporary')
        with pytest.raises(SpriteSandboxError):
            await backend.disconnect()
        assert backend.ref == SandboxRef(sandbox_id='remote')
        await backend.disconnect()
        assert 'remote' in transport.names
        assert (await backend.sandbox).name == 'remote'

    async def test_injected_client_is_never_closed(self, transport: SpriteTransport) -> None:
        transport.names.add('owned-by-caller')
        client = transport.client('test-token', 'https://api.sprites.dev', 30)
        backend = SpriteSandboxBackend(client=client, ref=SandboxRef(sandbox_id='owned-by-caller'))
        await backend.sandbox
        await backend.disconnect()
        assert transport.close_calls == 0
        assert (await backend.sandbox).name == 'owned-by-caller'

    async def test_argv_shell_environment_and_nonzero_exit(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend()
        result = await backend.run(['/bin/echo', 'a; echo injected'])
        assert result.stdout == 'a; echo injected\n'
        result = await backend.run('printf "$VALUE"; printf error >&2; exit 124', shell=True, env={'VALUE': 'hello'})
        assert (result.exit_code, result.stdout, result.stderr) == (124, 'hello', 'error')

    async def test_canonical_working_directory_preserves_spaces(self, transport: SpriteTransport) -> None:
        target = transport.root / ' directory '
        target.mkdir()
        link = transport.root / 'link'
        link.symlink_to(target)
        backend = SpriteSandboxBackend(working_dir=str(link))
        assert await backend.working_dir() == str(target.resolve())
        assert await backend.working_dir() == str(target.resolve())

    async def test_missing_working_directory(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend(working_dir=str(transport.root / 'absent'))
        with pytest.raises(SpriteSandboxError, match='working directory'):
            await backend.working_dir()

    async def test_filesystem_fallback_handles_directories_and_binary(self, transport: SpriteTransport) -> None:
        sandbox = Sandbox(SpriteSandboxBackend())
        await sandbox.make_dir('folder')
        await sandbox.write_bytes('folder/a\nb', b'\x00\xff')
        assert await sandbox.read_bytes('folder/a\nb') == b'\x00\xff'
        assert (await sandbox.stat('folder')).is_dir
        assert [entry.name for entry in await sandbox.list_dir('folder')] == ['a\nb']
        await sandbox.remove('folder')
        assert not await sandbox.exists('folder')

    async def test_deadline_kills_child_and_preserves_partial_output(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend()
        await backend.sandbox
        with pytest.raises(SandboxTimeoutError) as caught:
            await backend.run('printf ready; sleep 1; touch escaped', shell=True, timeout=0.3)
        await anyio.sleep(1)
        assert not (transport.root / 'escaped').exists()
        assert caught.value.stdout == 'ready'

    async def test_cancellation_before_remote_start_prevents_command(self, transport: SpriteTransport) -> None:
        backend = SpriteSandboxBackend()
        await backend.sandbox
        transport.release_start = threading.Event()
        task = asyncio.create_task(backend.run(['touch', 'escaped']))
        await anyio.to_thread.run_sync(transport.started.wait)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        transport.release_start.set()
        await anyio.sleep(0.2)
        assert not (transport.root / 'escaped').exists()

    @pytest.mark.parametrize('timeout', [0, -1, float('inf')])
    async def test_invalid_timeout(self, transport: SpriteTransport, timeout: float) -> None:
        with pytest.raises(ValueError, match='timeout'):
            await SpriteSandboxBackend().run(['true'], timeout=timeout)

    @pytest.mark.parametrize('command,shell', [('true', False), ([], False), (['true'], True)])
    async def test_invalid_command(self, transport: SpriteTransport, command: str | list[str], shell: bool) -> None:
        with pytest.raises(TypeError):
            await SpriteSandboxBackend().run(command, shell=shell)

    def test_invalid_configuration(self) -> None:
        with pytest.raises(ValueError, match='runtime'):
            SpriteSandbox(sprite_name='existing', runtime='dev')
        with pytest.raises(ValueError, match='absolute'):
            SpriteSandbox(workdir='relative')
        with pytest.raises(ValueError, match='api_timeout'):
            SpriteSandboxBackend(api_timeout=0)
