"""Offline Streamable HTTP MCP server used by the Slack capability tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass

import anyio
import httpx
import pytest
from anyio.abc import TaskGroup
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.types import Receive, Scope, Send


@dataclass
class MCPCall:
    token: str
    name: str
    arguments: dict[str, object]


@dataclass
class _Session:
    token: str
    transport: StreamableHTTPServerTransport
    started: anyio.Event
    closed: anyio.Event


class OfflineMCP:
    """A real MCP Streamable HTTP server exposed through an in-process ASGI transport."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tools: list[types.Tool] = []
        self.instructions = 'Offline MCP server instructions.'
        self.calls: list[MCPCall] = []
        self.authorization_headers: list[str] = []
        self.http_clients: list[httpx.AsyncClient] = []
        self.session_starts: list[str] = []
        self.session_closes: list[str] = []
        self.result: types.CallToolResult | None = None
        self.error: Exception | None = None
        self.error_once = False
        self.block_calls = False
        self.call_started = anyio.Event()
        self.call_release = anyio.Event()
        self._sessions: dict[str, _Session] = {}
        self._task_group: TaskGroup | None = None
        self._stack: AsyncExitStack | None = None

        def create_client(
            *,
            headers: dict[str, str] | None = None,
            timeout: httpx.Timeout | None = None,
            auth: httpx.Auth | None = None,
            follow_redirects: bool = True,
            **_: object,
        ) -> httpx.AsyncClient:
            del auth
            client_headers = headers or {}
            token = client_headers.get('Authorization')
            if token is not None:  # pragma: no branch - every fixture MCP client has Authorization
                self.authorization_headers.append(token)
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self),
                headers=client_headers,
                follow_redirects=follow_redirects,
                timeout=timeout,
            )
            self.http_clients.append(client)
            return client

        monkeypatch.setattr('fastmcp.client.transports.http.create_mcp_http_client', create_client)

    async def __aenter__(self) -> OfflineMCP:
        self._stack = AsyncExitStack()
        self._task_group = await self._stack.enter_async_context(anyio.create_task_group())
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._stack is not None
        await self._terminate_sessions()
        if self._task_group is not None:  # pragma: no branch - __aenter__ initializes the task group
            self._task_group.cancel_scope.cancel()
        await self._stack.aclose()
        self._task_group = None

    async def _terminate_sessions(self) -> None:
        for session in self._sessions.values():
            await session.transport.terminate()
        with anyio.move_on_after(2):
            for session in self._sessions.values():
                await session.closed.wait()

    async def _create_session(self, token: str) -> _Session:
        if self._task_group is None:  # pragma: no cover - fixture always enters first
            raise RuntimeError('OfflineMCP must be used as an async context manager')
        transport = StreamableHTTPServerTransport(None, is_json_response_enabled=True)
        session = _Session(token, transport, anyio.Event(), anyio.Event())
        self._sessions[token] = session
        server = Server('offline-slack-mcp', instructions=self.instructions)

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return self.tools

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, object]) -> types.CallToolResult:
            self.calls.append(MCPCall(token, name, arguments))
            if self.block_calls:
                self.call_started.set()
                await self.call_release.wait()
            if self.error is not None:
                error = self.error
                if self.error_once:  # pragma: no branch - the fixture uses one-shot errors only
                    self.error = None
                raise error
            if self.result is None:
                return types.CallToolResult(content=[types.TextContent(type='text', text='ok')])
            return self.result

        async def run_server() -> None:
            try:
                async with transport.connect() as streams:
                    session.started.set()
                    self.session_starts.append(token)
                    await server.run(*streams, server.create_initialization_options())
            finally:
                session.closed.set()
                self.session_closes.append(token)

        self._task_group.start_soon(run_server)
        await session.started.wait()
        return session

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        headers = dict(scope.get('headers', []))  # type: ignore[arg-type]
        token_header = headers.get(b'authorization', b'').decode()
        session = self._sessions.get(token_header)
        if session is None or session.closed.is_set():
            session = await self._create_session(token_header)
        await session.transport.handle_request(scope, receive, send)


@pytest.fixture
async def offline_mcp(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[OfflineMCP]:
    async with OfflineMCP(monkeypatch) as server:
        yield server
