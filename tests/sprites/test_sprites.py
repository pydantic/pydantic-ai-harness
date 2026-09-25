from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator

import anyio
import anyio.to_thread
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from sprites import AsyncSprite
from sprites.exceptions import AuthenticationError, NetworkError, NotFoundError, SpriteError
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites import SpriteWorkspace, SpriteWorkspaceBackend

from .conftest import live_token
from .fake_sprites import SpriteTransport

pytestmark = pytest.mark.anyio


def context(conversation: str = 'chat') -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), conversation_id=conversation, run_id='run')


def _handshake(status: int) -> InvalidStatus:
    return InvalidStatus(Response(status, 'status', Headers()))


class TestSpriteWorkspace:
    async def test_construction_is_lazy_and_first_use_is_shared(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspace[None]().get_workspace(context(), ref=None)
        assert isinstance(backend, SpriteWorkspaceBackend)
        assert transport.clients == []
        first, second = await asyncio.gather(backend.get_client(), backend.get_client())
        assert first is second
        assert transport.created == [first.name]
        assert backend.ref == WorkspaceRef(provider='sprites', id=first.name)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'working_dir': 'relative'},
            {'api_timeout': 0},
            {'api_timeout': float('inf')},
            {'api_timeout': True},
            {'defer_loading': True},
        ],
    )
    def test_invalid_configuration_fails_at_construction(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(UserError, match=f'^{next(iter(kwargs))} must be'):
            SpriteWorkspace[None](**kwargs)  # pyright: ignore[reportArgumentType]

    async def test_cancelled_creation_still_names_the_sprite_and_a_retry_attaches(
        self, transport: SpriteTransport
    ) -> None:
        backend = SpriteWorkspaceBackend()
        transport.release_create = asyncio.Event()

        async def acquire() -> AsyncSprite:
            return await backend.get_client()

        task = asyncio.create_task(acquire())
        await transport.create_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        transport.release_create.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.ref == WorkspaceRef(provider='sprites', id=transport.created[0])
        assert (await backend.get_client()).name == transport.created[0]
        assert len(transport.created) == 1

    def test_foreign_reference_is_declined_and_backend_rejects_it(self) -> None:
        assert SpriteWorkspace[None]().get_workspace(context(), ref=WorkspaceRef(provider='other', id='x')) is None
        with pytest.raises(ValueError, match="expected 'sprites'"):
            SpriteWorkspaceBackend(ref=WorkspaceRef(provider='other', id='x'))

    async def test_native_handle_conflict_and_identity(self, transport: SpriteTransport) -> None:
        seed = SpriteWorkspaceBackend()
        native = await seed.get_client()
        backend = SpriteWorkspaceBackend(workspace=native)
        assert await backend.get_client() is native
        assert backend.ref == WorkspaceRef(provider='sprites', id=native.name)
        with pytest.raises(ValueError, match='either `workspace` or `ref`'):
            SpriteWorkspaceBackend(workspace=native, ref=backend.ref)

    async def test_agent_without_workspace_use_does_not_create(self, transport: SpriteTransport) -> None:
        result = await Agent(TestModel(custom_output_text='done'), capabilities=[SpriteWorkspace()]).run('go')
        assert result.output == 'done'
        assert transport.created == []

    async def test_run_end_closes_the_owned_client_and_the_result_reattaches(self, transport: SpriteTransport) -> None:
        agent = Agent(TestModel(), capabilities=[SpriteWorkspace()])

        @agent.tool
        async def write(ctx: RunContext[object]) -> str:
            await ctx.workspace.write_bytes('result.bin', b'\x00\xff\n')
            return 'written'

        result = await agent.run('write')
        assert transport.close_calls == 1

        assert await result.workspace.read_bytes('result.bin') == b'\x00\xff\n'
        assert len(transport.clients) == 2
        assert len(transport.names) == 1

    async def test_cancelled_run_still_closes_the_owned_client(self, transport: SpriteTransport) -> None:
        agent = Agent(TestModel(), capabilities=[SpriteWorkspace()])
        used = anyio.Event()

        @agent.tool
        async def hang(ctx: RunContext[object]) -> None:
            await ctx.workspace.run(['true'])
            used.set()
            await anyio.sleep_forever()

        async with anyio.create_task_group() as group:
            group.start_soon(agent.run, 'hang')
            await used.wait()
            group.cancel_scope.cancel()

        assert transport.close_calls == 1

    async def test_run_end_leaves_caller_clients_and_backends_open(self, transport: SpriteTransport) -> None:
        client = transport.client('test-token', 'https://api.sprites.dev', 30)
        agent = Agent(TestModel(), capabilities=[SpriteWorkspace(client=client)])

        @agent.tool
        async def touch(ctx: RunContext[object]) -> str:
            await ctx.workspace.write_text('touched', '')
            return 'touched'

        await agent.run('touch')
        explicit = SpriteWorkspaceBackend()
        await agent.run('touch', workspace=explicit)

        assert transport.close_calls == 0
        assert len(transport.clients) == 2

    async def test_coder_tools_run_in_the_sprite(self, transport: SpriteTransport) -> None:
        async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if not returns:
                yield {0: DeltaToolCall('write_file', json.dumps({'path': 'notes.txt', 'content': 'hello'}))}
            elif len(returns) == 1:
                yield {0: DeltaToolCall('shell', json.dumps({'command': 'cat notes.txt'}))}
            else:
                yield str(returns[-1].content)

        # Coder's members stream, so the scripted model is a stream function.
        agent = Agent(FunctionModel(stream_function=model), capabilities=[SpriteWorkspace(), Coder()])
        result = await agent.run('go')

        assert result.output.startswith('hello\n')
        assert '"exit_code": 0' in result.output
        assert (transport.root / 'notes.txt').read_text() == 'hello'

    async def test_missing_reference_does_not_recreate(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='missing'))
        with pytest.raises(WorkspaceUnavailableError, match="Sprite 'missing' no longer exists"):
            await backend.get_client()
        assert transport.created == []

    @pytest.mark.parametrize(
        'stage,error,expected',
        [
            ('acquire', AuthenticationError('bad token'), WorkspaceUnavailableError),
            ('acquire', NotFoundError('gone'), WorkspaceUnavailableError),
            ('acquire', SpriteError('Failed get sprite (status 400)'), WorkspaceError),
            ('acquire', SpriteError('Failed get sprite (status 429): rate limited'), None),
            ('acquire', NetworkError('reset'), None),
            ('connect', _handshake(401), WorkspaceUnavailableError),
            ('connect', _handshake(404), WorkspaceUnavailableError),
            ('connect', _handshake(500), None),
            ('connect', ConnectionResetError('reset'), None),
        ],
    )
    async def test_errors_are_mapped_or_propagate(
        self,
        transport: SpriteTransport,
        stage: str,
        error: Exception,
        expected: type[WorkspaceError] | None,
    ) -> None:
        transport.names.add('target')
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='target'))
        if stage == 'acquire':
            transport.get_error = error
        else:
            transport.connect_error = error

        with pytest.raises(Exception) as caught:
            await backend.run(['true'])

        if expected is None:
            assert caught.value is error
        else:
            assert type(caught.value) is expected
            assert caught.value.__cause__ is error

    async def test_missing_token(self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SPRITE_TOKEN')
        with pytest.raises(WorkspaceUnavailableError, match='SPRITE_TOKEN'):
            await SpriteWorkspaceBackend().get_client()

    @pytest.mark.parametrize('attach', [False, True], ids=['create', 'attach'])
    async def test_stalled_acquisition_propagates_as_a_transport_timeout(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch, attach: bool
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.sprites._backend._ACQUIRE_TIMEOUT', 0.05)
        transport.names.add('remote')
        transport.release_create = asyncio.Event()
        transport.release_get = asyncio.Event()
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='remote') if attach else None)

        with pytest.raises(TimeoutError, match='control plane may be unreachable') as caught:
            await backend.run(['true'], timeout=30)

        assert not isinstance(caught.value, WorkspaceError)
        assert ('connection' if attach else 'creation') in str(caught.value)
        assert transport.commands == []

    async def test_run_deadline_starts_once_the_sprite_is_acquired(self, transport: SpriteTransport) -> None:
        transport.names.add('remote')
        transport.release_get = asyncio.Event()
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='remote'))
        # Attaching outlasts the timeout; the command itself fits in it comfortably.
        task = asyncio.create_task(backend.run(['echo', 'ok'], timeout=1))
        await asyncio.sleep(1.1)
        transport.release_get.set()

        assert (await task).stdout == 'ok\n'

    async def test_failed_acquisition_keeps_the_client_for_a_retry(self, transport: SpriteTransport) -> None:
        transport.get_error = SpriteError('lookup failed')
        transport.names.add('remote')
        backend = SpriteWorkspaceBackend(ref=WorkspaceRef(provider='sprites', id='remote'))
        with pytest.raises(WorkspaceError):
            await backend.get_client()
        transport.get_error = None
        assert (await backend.get_client()).name == 'remote'
        assert (len(transport.clients), transport.close_calls) == (1, 0)

    async def test_connection_settings_reach_the_owned_client(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend(token='sentinel', base_url='https://example.invalid', api_timeout=7)
        native = await backend.get_client()
        assert native.client.token == 'sentinel'
        assert native.client.base_url == 'https://example.invalid'
        assert native.client.timeout == 7

    async def test_aclose_leaves_an_injected_client_open(self, transport: SpriteTransport) -> None:
        injected = SpriteWorkspaceBackend(client=transport.client('test-token', 'https://api.sprites.dev', 30))
        await injected.get_client()
        await injected.aclose()
        assert transport.close_calls == 0

    async def test_failed_close_is_logged_and_retried_by_the_next_aclose(
        self, transport: SpriteTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        owned = SpriteWorkspaceBackend()
        await owned.get_client()
        transport.close_error = RuntimeError('close failed')
        await owned.aclose()
        assert 'Could not close Sprites SDK client' in caplog.text
        await owned.aclose()
        await owned.aclose()
        assert transport.close_calls == 2

    async def test_cancelled_aclose_finishes_the_close_first(self, transport: SpriteTransport) -> None:
        owned = SpriteWorkspaceBackend()
        await owned.get_client()
        transport.release_close = asyncio.Event()
        task = asyncio.create_task(owned.aclose())
        await transport.close_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        transport.release_close.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        await owned.aclose()
        assert transport.close_calls == 1

    async def test_aclose_waits_for_an_in_flight_creation(self, transport: SpriteTransport) -> None:
        owned = SpriteWorkspaceBackend()
        transport.release_create = asyncio.Event()
        creating = asyncio.create_task(owned.get_client())
        await transport.create_started.wait()
        closing = asyncio.create_task(owned.aclose())
        await asyncio.sleep(0)
        assert transport.close_calls == 0
        transport.release_create.set()
        await creating
        await closing

        assert transport.close_calls == 1
        assert (await owned.get_client()).client is transport.clients[1]

    async def test_cancelled_command_finishes_closing_its_connection(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.get_client()
        transport.release_control_close = asyncio.Event()
        task = asyncio.create_task(backend.run(['true']))
        await transport.control_close_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        transport.release_control_close.set()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.control_closes == 1

    async def test_deleted_sprite_is_unavailable_to_attach_commands_and_files(self, transport: SpriteTransport) -> None:
        owner = SpriteWorkspaceBackend()
        native = await owner.get_client()
        assert owner.ref is not None
        await native.delete()

        with pytest.raises(WorkspaceUnavailableError):
            await SpriteWorkspaceBackend(ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError, match=native.name):
            await owner.run(['true'])
        with pytest.raises(WorkspaceUnavailableError):
            await Workspace(owner).read_bytes('/tmp/anything')

    @pytest.mark.parametrize('failure', ['error', 'hang'])
    async def test_control_close_failure_after_exit_returns_the_result_and_aborts_the_socket(
        self,
        transport: SpriteTransport,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        failure: str,
    ) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.get_client()
        if failure == 'error':
            transport.control_close_error = RuntimeError('close failed')
        else:
            transport.control_close_hang = True
            monkeypatch.setattr('pydantic_ai_harness.sprites._backend._CONTROL_TIMEOUT', 0.01)

        result = await backend.run(['echo', 'ok'])

        assert (result.exit_code, result.stdout) == (0, 'ok\n')
        assert transport.aborted == 1
        assert 'Could not close a Sprite control connection' in caplog.text

    async def test_transport_loss_propagates_the_sdk_error_and_requests_cancel(
        self, transport: SpriteTransport
    ) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.get_client()
        error = ConnectionResetError('control connection lost')
        transport.run_exit_override = -1
        transport.connection_lost = error
        with pytest.raises(ConnectionResetError) as caught:
            await backend.run(['true'])
        assert caught.value is error
        assert any(len(command) == 5 for command in transport.commands)

    async def test_transport_loss_without_an_sdk_error_is_a_connection_error(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.get_client()
        transport.run_exit_override = -1
        transport.run_stderr = b'Error: op failed'
        with pytest.raises(ConnectionError, match='before reporting an exit status. Error: op failed$'):
            await backend.run(['true'])

    async def test_timeout_cancels_before_original_close_finishes(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.get_client()
        transport.control_close_hang = True
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run('sleep .5; touch escaped', shell=True, timeout=0.01)
        assert not (transport.root / 'escaped').exists()

    async def test_cancel_failure_is_logged_and_the_timeout_still_raised(
        self, transport: SpriteTransport, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.get_client()
        transport.cancel_exit_override = 1
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run('sleep 1', shell=True, timeout=0.01)
        assert 'Could not confirm remote Sprite command termination' in caplog.text

    async def test_argv_shell_environment_and_nonzero_exit(
        self, transport: SpriteTransport, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Commands inherit the Sprite's exec environment, which the fake runs on this process's.
        monkeypatch.setenv('BASH_ENV', '/startup')
        monkeypatch.setenv('ENV', '/startup')
        backend = SpriteWorkspace[None](env={'BASE': 'base', 'LAYERED': 'base'}).get_workspace(context(), ref=None)
        assert isinstance(backend, SpriteWorkspaceBackend)

        result = await backend.run(['/bin/echo', 'a; echo injected'])
        assert result.stdout == 'a; echo injected\n'
        result = await backend.run(
            'printf "$0 $BASE $LAYERED ${BASH_ENV-unset} ${ENV-unset}"; printf error >&2; exit 124',
            shell=True,
            env={'LAYERED': 'command'},
        )
        assert (result.exit_code, result.stdout, result.stderr) == (124, '/bin/sh base command unset unset', 'error')

    async def test_canonical_working_directory_preserves_spaces(self, transport: SpriteTransport) -> None:
        target = transport.root / ' directory '
        target.mkdir()
        link = transport.root / 'link'
        link.symlink_to(target)
        backend = SpriteWorkspaceBackend(working_dir=str(link))
        assert await backend.working_dir() == str(target.resolve())
        assert await backend.working_dir() == str(target.resolve())

    async def test_unexpected_working_directory_output(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        transport.run_exit_override = 1
        with pytest.raises(WorkspaceError, match='working directory'):
            await backend.working_dir()

    @pytest.mark.parametrize(
        'command,cwd,expected',
        [
            (['true'], 'absent', FileNotFoundError),
            (['missing-program'], None, FileNotFoundError),
            (['true'], 'file', NotADirectoryError),
            (['./file'], None, PermissionError),
            (['./garbage'], None, WorkspaceError),
        ],
    )
    async def test_command_that_cannot_start_raises(
        self, transport: SpriteTransport, command: list[str], cwd: str | None, expected: type[Exception]
    ) -> None:
        (transport.root / 'file').write_text('')
        garbage = transport.root / 'garbage'
        garbage.write_bytes(b'\x00\x01')
        garbage.chmod(0o755)
        backend = SpriteWorkspaceBackend(working_dir=str(transport.root))
        with pytest.raises(expected) as caught:
            await backend.run(command, cwd=None if cwd is None else str(transport.root / cwd))
        assert type(caught.value) is expected

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
        await backend.get_client()
        with pytest.raises(WorkspaceTimeoutError, match='Command timed out after 0.3 seconds') as caught:
            await backend.run('printf ready; sleep 1; touch escaped', shell=True, timeout=0.3)
        await anyio.sleep(1)
        assert not (transport.root / 'escaped').exists()
        assert caught.value.stdout == 'ready'

    async def test_cancellation_before_remote_start_prevents_command(self, transport: SpriteTransport) -> None:
        backend = SpriteWorkspaceBackend()
        await backend.get_client()
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
        assert await anyio.to_thread.run_sync(transport.finished.wait, 5)
        assert not (transport.root / 'escaped').exists()

    @pytest.mark.parametrize('timeout', [0, -1, float('inf')])
    async def test_invalid_timeout(self, transport: SpriteTransport, timeout: float) -> None:
        with pytest.raises(ValueError, match='timeout'):
            await SpriteWorkspaceBackend().run(['true'], timeout=timeout)

    @pytest.mark.parametrize('command,shell', [('true', False), (['true'], True)])
    async def test_invalid_command(self, transport: SpriteTransport, command: str | list[str], shell: bool) -> None:
        with pytest.raises(TypeError):
            await SpriteWorkspaceBackend().run(command, shell=shell)

    def test_relative_working_dir_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='absolute'):
            SpriteWorkspaceBackend(working_dir='relative')


@pytest.mark.parametrize(
    'env,outcome',
    [
        ({'PYDANTIC_AI_HARNESS_SPRITES_LIVE': '1', 'SPRITE_TOKEN': 'token'}, 'run'),
        ({'PYDANTIC_AI_HARNESS_SPRITES_LIVE': '1', 'SPRITE_TOKEN': ''}, 'skip'),
        ({'SPRITE_TOKEN': 'token'}, 'skip'),
        ({'PYDANTIC_AI_HARNESS_SPRITES_LIVE': '1', 'SPRITES_REQUIRE_LIVE': '1'}, 'fail'),
    ],
)
def test_live_tier_gate(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], outcome: str) -> None:
    for name in ('PYDANTIC_AI_HARNESS_SPRITES_LIVE', 'SPRITE_TOKEN', 'SPRITES_REQUIRE_LIVE'):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if outcome == 'run':
        assert live_token() == 'token'
    else:
        with pytest.raises(pytest.skip.Exception if outcome == 'skip' else pytest.fail.Exception):
            live_token()
