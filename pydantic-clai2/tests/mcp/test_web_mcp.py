"""A browser disconnect closes the real stdio tool through core's streaming runner."""

import socket
from pathlib import Path

import anyio
import httpx2
import pytest
import uvicorn
from anyio.abc import SocketAttribute, SocketStream
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic_ai import RunContext
from pydantic_ai.mcp import MCPToolset, load_mcp_toolsets
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, PrefixedToolset
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send
from test_mcp import assert_stopped, write_config
from test_web import CHAT, install_plugin

from pydantic_clai2 import mcp, web
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.project_settings import ProjectSettings
from pydantic_clai2.settings_store import SettingsStore


async def test_disconnect_closes_mcp_stdio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready, tool_ready, request_done, stopped = (anyio.Event() for _ in range(4))
    servers: list[uvicorn.Server] = []
    clients: list[MCPToolset[None]] = []
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='coder', factory='unused', enabled=False))

    def load(path: Path) -> list[AbstractToolset[None]]:
        toolsets: list[AbstractToolset[None]] = load_mcp_toolsets(path)
        for prefixed in toolsets:
            assert isinstance(prefixed, PrefixedToolset)
            assert isinstance(prefixed.wrapped, MCPToolset)
            clients.append(prefixed.wrapped)
        return toolsets

    class Observed:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            try:
                await self.app(scope, receive, send)
            finally:
                if scope['type'] == 'http' and scope['path'] == '/api/chat':
                    request_done.set()

    class Server(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            assert isinstance(config.app, Starlette)
            config.app.add_middleware(Observed)
            super().__init__(config)

        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets)
            servers.append(self)
            ready.set()

    def activate(host: PluginHost[None]) -> None:
        @host.on('prepare_tools')
        async def prepare(ctx: RunContext[None], tools: list[ToolDefinition]) -> list[ToolDefinition]:
            return [tool for tool in tools if tool.name == 'probe_wait']

    async def receive(stream: SocketStream) -> None:
        async with stream:
            expected = (tmp_path / 'server.pid').read_bytes()
            assert await BufferedByteReceiveStream(stream).receive_exactly(len(expected)) == expected
            tool_ready.set()

    async def serve() -> None:
        try:
            await web.serve_web(settings=Settings(model='test'), store=store, project=ProjectSettings(), port=0)
        finally:
            stopped.set()

    install_plugin(monkeypatch, store, activate)
    monkeypatch.setattr(mcp, 'load_mcp_toolsets', load)
    monkeypatch.setattr(web.uvicorn, 'Server', Server)
    with anyio.fail_after(30):
        async with await anyio.create_tcp_listener(local_host='127.0.0.1') as listener:
            port = listener.extra(SocketAttribute.local_address)[1]
            assert isinstance(port, int)
            config = write_config(tmp_path, prefix='probe', port=port)
            store.save_plugin(
                PluginSettings(id='mcp', factory='pydantic_clai2.mcp', settings={'config_path': str(config)})
            )
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(listener.serve, receive)
                tasks.start_soon(serve)
                await ready.wait()
                port = servers[0].servers[0].sockets[0].getsockname()[1]
                try:
                    async with httpx2.AsyncClient(base_url=f'http://127.0.0.1:{port}') as client:
                        async with client.stream('POST', '/api/chat', json=CHAT) as response:
                            assert response.status_code == 200
                            await tool_ready.wait()
                            assert clients[0].is_running
                    await request_done.wait()
                    assert not clients[0].is_running
                    assert_stopped(tmp_path)
                finally:
                    with anyio.CancelScope(shield=True):
                        for toolset in clients:
                            await toolset.client.close()
                    servers[0].should_exit = True
                    await stopped.wait()
                    tasks.cancel_scope.cancel()
