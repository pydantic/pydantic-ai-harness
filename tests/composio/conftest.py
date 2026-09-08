"""Collect MCP tests only when the optional dependencies are installed."""

from importlib.util import find_spec

import pytest

collect_ignore = ['test_capability.py'] if find_spec('mcp') is None or find_spec('fastmcp') is None else []


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
