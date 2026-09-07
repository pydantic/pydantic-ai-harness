"""Shared fixtures for the Logfire MCP capability tests."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

collect_ignore = (
    ['test_logfire_mcp.py']
    if importlib.util.find_spec('mcp') is None or importlib.util.find_spec('fastmcp') is None
    else []
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def logfire_calls() -> list[tuple[str, dict[str, object]]]:
    return []


@pytest.fixture
def logfire_server(logfire_calls: list[tuple[str, dict[str, object]]]) -> FastMCP:
    """In-process stand-in for Logfire's hosted MCP endpoint."""
    from mcp.server.fastmcp.server import FastMCP, Settings

    Settings.model_rebuild()
    server = FastMCP('logfire-fake')

    @server.tool()
    def project_list() -> list[str]:
        """List the projects this credential can reach."""
        logfire_calls.append(('project_list', {}))
        return ['acme/production']

    @server.tool()
    def query_run(query: str, project: str) -> list[dict[str, object]]:
        """Run SQL against one Logfire project."""
        logfire_calls.append(('query_run', {'query': query, 'project': project}))
        return [{'count': 3}]

    return server
