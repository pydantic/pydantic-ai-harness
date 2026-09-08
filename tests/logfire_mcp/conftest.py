"""Fixtures for the Logfire MCP capability tests.

Logfire's hosted server is stood in for by a FastMCP server on real HTTP, so the credential and the
tool annotations travel the same path they do in production. Each fake tool reports the
`Authorization` header it was called with.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import pytest

# `fastmcp-slim` imports but raises ImportError for server support, so widen the skip.
pytest.importorskip('fastmcp.server', exc_type=ImportError)
pytest.importorskip('mcp')

import uvicorn
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_request
from mcp.types import ToolAnnotations


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _tool(name: str) -> Callable[[], dict[str, str | None]]:
    def tool() -> dict[str, str | None]:
        """A Logfire tool that reports the credential it was called with."""
        return {'tool': name, 'authorization': get_http_request().headers.get('authorization')}

    tool.__name__ = name
    return tool


@pytest.fixture(scope='session')
def logfire_url() -> Iterator[str]:
    """Serve a stand-in Logfire MCP server on localhost and yield its `/mcp` URL."""
    server = FastMCP('logfire-fake')
    server.tool(_tool('query_run'), annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    server.tool(_tool('dashboard_create'), annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    server.tool(_tool('unannotated_tool'))
    # A JSON response per request, rather than an SSE stream, so no session outlives the server.
    http = uvicorn.Server(
        uvicorn.Config(
            server.http_app(path='/mcp', stateless_http=True, json_response=True),
            host='127.0.0.1',
            port=0,
            log_level='error',
        )
    )
    thread = threading.Thread(target=http.run, daemon=True)
    thread.start()
    while not http.started:
        time.sleep(0.01)
    try:
        yield f'http://127.0.0.1:{http.servers[0].sockets[0].getsockname()[1]}/mcp'
    finally:
        http.should_exit = True
        thread.join(timeout=5)
