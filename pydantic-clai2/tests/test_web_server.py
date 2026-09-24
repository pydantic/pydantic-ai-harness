"""Exercise socket streaming and server teardown without a model provider."""

import asyncio
import signal
import socket
import sys
from pathlib import Path
from typing import Self

import anyio
import httpx2
import pytest
import uvicorn
from anyio.lowlevel import checkpoint
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel
from test_web import CHAT, install_plugin

from pydantic_clai2 import web
from pydantic_clai2.auth import CodexAuth
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.web import serve_web


async def test_server_drains_stream_before_closing_plugins_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='coder', factory='unused', enabled=False))
    ready, stopping, tool_started, release = (anyio.Event() for _ in range(4))
    servers: list[uvicorn.Server] = []
    events: list[str] = []
    loop = asyncio.get_running_loop()

    class Server(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets)
            servers.append(self)
            ready.set()

        async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
            stopping.set()
            await super().shutdown(sockets)

    class Model(TestModel):
        async def __aenter__(self) -> Self:
            assert asyncio.get_running_loop() is loop
            events.append('model enter')
            return self

        async def __aexit__(self, *args: object) -> None:
            await checkpoint()
            assert asyncio.get_running_loop() is loop
            events.append('model exit')

    async def resolve(name: str, *, auth: CodexAuth) -> TestModel:
        return Model(call_tools=['echo'])

    def activate(host: PluginHost[None]) -> None:
        @host.on('session_start')
        async def start(event: SessionStart) -> None:
            assert asyncio.get_running_loop() is loop
            events.append('start')

        @host.on('session_end')
        async def end(event: SessionEnd) -> None:
            await checkpoint()
            events.append(event.reason)

        async def echo() -> str:
            assert asyncio.get_running_loop() is loop
            tool_started.set()
            await release.wait()
            events.append('tool done')
            return 'stream complete'

        host.add(Capability(tools=[echo]))

    install_plugin(monkeypatch, store, activate)
    monkeypatch.setattr(web.uvicorn, 'Server', Server)
    monkeypatch.setattr(web, 'resolve_model', resolve)

    async def serve() -> None:
        await serve_web(settings=Settings(model='test'), store=store, project=ProjectSettings(), port=0)

    with anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(serve)
            await ready.wait()
            server = servers[0]
            address, port = server.servers[0].sockets[0].getsockname()
            assert address == '127.0.0.1'
            try:
                async with httpx2.AsyncClient(base_url=f'http://127.0.0.1:{port}') as client:
                    assert (await client.get('/api/health')).status_code == 200
                    async with client.stream('POST', '/api/chat', json=CHAT) as response:
                        assert response.status_code == 200
                        lines = response.aiter_lines()
                        assert (await anext(lines)).startswith('data: ')
                        await tool_started.wait()
                        server.should_exit = True
                        await stopping.wait()
                        assert 'tool done' not in events
                        assert 'exit' not in events
                        release.set()
                        stream = '\n'.join([line async for line in lines])
                        assert 'stream complete' in stream
                        assert 'data: [DONE]' in stream
            finally:
                release.set()
                server.should_exit = True
    server = servers[0]
    assert events.index('tool done') < events.index('exit')
    assert events[-2:] == ['exit', 'model exit']
    assert events.count('model enter') == events.count('model exit')
    assert not server.server_state.tasks
    assert all(not listener.is_serving() for listener in server.servers)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX SIGINT delivery to a child process')
async def test_ctrl_c_closes_plugins_before_uvicorn_reraises_sigint(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='coder', factory='unused', enabled=False))
    marker = tmp_path / 'lifecycle.txt'
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'lifecycle.py').write_text(
        'from pathlib import Path\n'
        'from anyio.lowlevel import checkpoint\n'
        'def activate(host):\n'
        "    @host.on('session_start')\n"
        '    async def start(event):\n'
        f"        Path({str(marker)!r}).write_text('start')\n"
        "    @host.on('session_end')\n"
        '    async def end(event):\n'
        '        await checkpoint()\n'
        f"        Path({str(marker)!r}).write_text('end')\n"
    )
    with socket.socket() as available:
        available.bind(('127.0.0.1', 0))
        port = available.getsockname()[1]
    with anyio.fail_after(30):
        async with await anyio.open_process(
            [
                sys.executable,
                '-m',
                'pydantic_clai2',
                '--database',
                str(store.path),
                '--web',
                '--model',
                'test',
                '--port',
                str(port),
            ]
        ) as process:
            assert process.stderr is not None
            await BufferedByteReceiveStream(process.stderr).receive_until(b'Uvicorn running on ', 65536)
            assert marker.read_text() == 'start'
            process.send_signal(signal.SIGINT)
            assert await process.wait() == 0
    assert marker.read_text() == 'end'


async def test_bind_failure_closes_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    events: list[str] = []

    def activate(host: PluginHost[None]) -> None:
        @host.on('session_end')
        async def end(event: SessionEnd) -> None:
            await checkpoint()
            events.append(event.reason)

    install_plugin(monkeypatch, store, activate)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        with pytest.raises(SystemExit) as exc:
            await serve_web(
                settings=Settings(model='test'),
                store=store,
                project=ProjectSettings(),
                port=occupied.getsockname()[1],
            )
    assert exc.value.code != 0
    assert events == ['error']
