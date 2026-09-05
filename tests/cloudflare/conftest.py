"""Fixtures for the Cloudflare MCP capability tests."""

from __future__ import annotations

from typing import Annotated

import pytest
from mcp.server.fastmcp.server import FastMCP, Settings
from mcp.types import ToolAnnotations
from pydantic import Field
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

SmallLimit = Annotated[int, Field(ge=1, le=3)]
LargeLimit = Annotated[int, Field(ge=10, le=20)]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def run_context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _server(name: str) -> FastMCP:
    Settings.model_rebuild()
    return FastMCP(name, instructions='Use the fake Cloudflare server.')


@pytest.fixture
def focused_server() -> FastMCP:
    server = _server('cloudflare-focused')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def list_records(account_id: str, zoneId: str, limit: Annotated[int, Field(ge=1, le=100)] = 50) -> str:
        """List DNS records."""
        return '\n'.join(f'{account_id}:{zoneId}:{index}' for index in range(limit))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def zone_details(account_id: str, zoneId: str) -> str:
        """Show zone details."""
        return f'zone:{account_id}:{zoneId}'

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    def delete_record(account_id: str, zoneId: str, record_id: str) -> str:
        """Delete a DNS record."""
        return f'deleted:{account_id}:{zoneId}:{record_id}'

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    def failing_delete(record_id: str) -> str:
        """Fail after accepting a delete request."""
        raise ValueError('x' * 100 + 'SECRET')

    @server.tool()
    def ambiguous_tool() -> str:
        """Tool without safety annotations."""
        return 'ambiguous'  # pragma: no cover - hidden by the read-safe policy

    return server


@pytest.fixture
def api_server() -> FastMCP:
    server = _server('cloudflare-api')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def docs(query: str) -> str:
        """Search Cloudflare docs."""
        return query

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def search(code: str) -> str:
        """Search the API schema."""
        return code  # pragma: no cover - catalog-only fake

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    def execute(code: str, account_id: str | None = None) -> str:
        """Execute Cloudflare API code."""
        return f'{account_id}:{code}'  # pragma: no cover - approval metadata fake

    return server


@pytest.fixture
def alternate_schema_server() -> FastMCP:
    server = _server('cloudflare-alternate-schema')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def camel_scope(accountId: str, zone: str) -> str:
        """Read data using alternate resource-key spellings."""
        return f'{accountId}:{zone}'  # pragma: no cover - network call is intercepted

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def limited_records(limit: Annotated[int | None, Field(ge=1, le=3)] = None) -> str:
        """Return a server-limited result page."""
        return ','.join(str(index) for index in range(limit or 3))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def simple_limited(limit: Annotated[int, Field(ge=1, le=3)] = 3) -> str:
        """Return a page with top-level numeric constraints."""
        return str(limit)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def minimum_too_large(limit: Annotated[int | None, Field(ge=5)] = None) -> str:
        """Require more results than a restrictive client permits."""
        return str(limit or 5)  # pragma: no cover - hidden by the result policy

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def structured_read() -> dict[str, int]:
        """Return structured data."""
        return {'count': 200}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def emoji_read() -> str:
        """Return multi-byte text."""
        return '😀' * 20

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def exact_text() -> str:
        """Return text with significant line endings."""
        return 'a\r\nb\n'

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def ambiguous_limit(limit: SmallLimit | LargeLimit = 1) -> str:
        """Use disjoint numeric ranges."""
        return str(limit)  # pragma: no cover - hidden by the result policy

    return server


@pytest.fixture
def untrusted_api_server() -> FastMCP:
    server = _server('cloudflare-untrusted-api')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
    def search(code: str) -> str:
        """Use a familiar name without a read-only contract."""
        return code  # pragma: no cover - hidden by the untrusted-client policy

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def claimed_read() -> str:
        """Claim a read-only contract from an untrusted server."""
        return 'read'  # pragma: no cover - catalog-only fake

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=True))
    def contradictory() -> str:
        """Publish contradictory safety annotations."""
        return 'conflict'  # pragma: no cover - hidden by the safety policy

    return server


@pytest.fixture
def error_server() -> FastMCP:
    server = _server('cloudflare-error')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def failing_read() -> str:
        """Return a long provider error."""
        raise ValueError('provider failure: ' + 'sensitive detail ' * 30)

    return server
