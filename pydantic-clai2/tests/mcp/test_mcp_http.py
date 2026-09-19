"""Exercise the configured HTTP endpoint and headers through a real MCP connection."""

import json
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from starlette.applications import Starlette
from test_plugin_loader import Harness
from uvicorn import Config, Server

from pydantic_clai2 import DEFAULT_PLUGINS


async def test_http_tools_use_configured_endpoint_and_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ready = anyio.Event()
    mcp: FastMCP[None] = FastMCP('local-http', streamable_http_path='/configured-mcp')

    @mcp.tool()
    def context() -> str:
        request = mcp.get_context().request_context.request
        assert request is not None
        return f'{request.url.path}: {request.headers["x-clai-key"]}'

    app = mcp.streamable_http_app()
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None]:
        async with original_lifespan(app):
            ready.set()
            yield

    app.router.lifespan_context = lifespan
    plugin = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'mcp')
    harness = Harness(tmp_path, builtin=(plugin,))
    await harness.loader.load_all()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        (tmp_path / '.mcp.json').write_text(
            json.dumps(
                {
                    'mcpServers': {
                        'remote': {
                            'url': f'http://127.0.0.1:{port}/configured-mcp',
                            'headers': {'X-Clai-Key': 'sentinel'},
                        }
                    }
                }
            )
        )
        server = Server(Config(app, log_config=None, access_log=False))

        async def serve() -> None:
            await server.serve(sockets=[listener])

        try:
            with anyio.fail_after(30):
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(serve)
                    await ready.wait()
                    try:
                        assert 'Loaded 1' in await harness.commands.execute_async('/mcp load --approve')
                        result = await Agent(TestModel(call_tools=['remote_context']), deps_type=type(None)).run(
                            'Read the HTTP context', capabilities=harness.loader.capabilities()
                        )
                        assert '/configured-mcp: sentinel' in result.output
                    finally:
                        server.should_exit = True
        finally:
            await harness.loader.close('exit')
