"""Shared fixtures for Linear capability tests."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


collect_ignore = (
    ['test_linear.py'] if importlib.util.find_spec('mcp') is None or importlib.util.find_spec('fastmcp') is None else []
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def linear_server() -> FastMCP:
    """In-process stand-in for Linear's hosted MCP server."""
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('linear-fake')

    @server.tool()
    def list_issues(query: str = '') -> list[dict[str, str]]:
        """List matching issues."""
        return [{'identifier': 'HAR-787', 'title': 'Add integrations'}]

    @server.tool()
    def create_issue(title: str) -> dict[str, str]:
        """Create an issue."""
        return {'identifier': 'HAR-900', 'title': title}

    return server
