"""Fixtures for Google Workspace capability tests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

import pytest

pytest.importorskip('fastmcp')
pytest.importorskip('mcp')

from mcp.server.fastmcp.server import FastMCP, Settings


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def gmail_server() -> FastMCP:
    Settings.model_rebuild()
    server = FastMCP('gmail-fake')

    @server.tool()
    def search_threads(query: str = '') -> list[dict[str, str]]:
        """Search Gmail threads."""
        return [{'id': 'thread-1', 'subject': 'Launch plan', 'query': query}]

    @server.tool()
    def create_draft(to: str, body: str) -> dict[str, str]:
        """Create a Gmail draft."""
        return {'id': 'draft-1', 'to': to, 'body': body}

    return server


@pytest.fixture
def gmail_server_factory() -> Callable[[str, Callable[[], Awaitable[None]] | None], FastMCP]:
    def make_server(identity: str, rendezvous: Callable[[], Awaitable[None]] | None) -> FastMCP:
        Settings.model_rebuild()
        server = FastMCP(f'gmail-{identity}-fake')

        @server.tool()
        async def search_threads(query: str = '') -> list[dict[str, str]]:
            """Search Gmail threads for one identity."""
            if rendezvous is not None:
                await rendezvous()
            return [{'id': f'{identity}-thread-1', 'subject': identity, 'query': query}]

        return server

    return make_server


@pytest.fixture
def calendar_server() -> FastMCP:
    Settings.model_rebuild()
    server = FastMCP('calendar-fake')

    @server.tool()
    def list_events(query: str = '') -> list[dict[str, str]]:
        """List Calendar events."""
        return [{'id': 'event-1', 'summary': 'Planning', 'query': query}]

    @server.tool()
    def create_event(summary: str) -> dict[str, str]:
        """Create a Calendar event."""
        return {'id': 'event-2', 'summary': summary}

    return server


@pytest.fixture
def workspace_server_factory() -> Callable[[Sequence[str], Sequence[str]], FastMCP]:
    def make_server(read_tools: Sequence[str], write_tools: Sequence[str]) -> FastMCP:
        Settings.model_rebuild()
        server = FastMCP(f'{read_tools[0]}-fake')

        for tool_name in read_tools:

            @server.tool(name=tool_name)
            def read(resource: str = '') -> dict[str, str]:
                """Read a resource."""
                return {'resource': resource}

        for tool_name in write_tools:

            @server.tool(name=tool_name)
            def write(value: str = '') -> dict[str, str]:
                """Change a resource."""
                return {'value': value}

        return server

    return make_server
