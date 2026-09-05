"""In-process Supabase MCP stand-in."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import pytest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

collect_ignore = (
    ['test_supabase.py']
    if importlib.util.find_spec('mcp') is None or importlib.util.find_spec('fastmcp') is None
    else []
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def connections() -> list[tuple[str, object]]:
    return []


@pytest.fixture
def failures() -> set[str]:
    return set()


def _register_database_tools(server: FastMCP, calls: list[str], failures: set[str]) -> None:
    @server.tool()
    def list_tables() -> list[str]:
        """List database tables."""
        if 'list_tables' in failures:
            raise RuntimeError('Supabase unavailable')
        calls.append('list_tables')
        return ['public.todos']

    @server.tool()
    def list_extensions() -> list[str]:
        """List database extensions."""
        calls.append('list_extensions')
        return []

    @server.tool()
    def list_migrations() -> list[str]:
        """List database migrations."""
        calls.append('list_migrations')
        return []

    @server.tool()
    def execute_sql(query: str) -> str:
        """Execute SQL against the database."""
        calls.append(f'execute_sql:{query}')
        return 'ok'

    @server.tool()
    def apply_migration(name: str, query: str) -> str:
        """Apply a database migration."""
        calls.append(f'apply_migration:{name}:{query}')
        return 'applied'


def _register_default_tools(server: FastMCP, calls: list[str]) -> None:
    @server.tool()
    def query_logs(query: str = 'select 1') -> list[str]:
        """Query project logs."""
        calls.append(f'query_logs:{query}')
        return []

    @server.tool()
    def get_advisors() -> list[str]:
        """Get project security and performance advisors."""
        calls.append('get_advisors')
        return []

    @server.tool()
    def get_project_url() -> str:
        """Get the project URL."""
        calls.append('get_project_url')
        return 'https://example.supabase.co'

    @server.tool()
    def get_publishable_keys() -> list[str]:
        """Get project publishable keys."""
        calls.append('get_publishable_keys')
        return []

    @server.tool()
    def generate_typescript_types() -> str:
        """Generate TypeScript types."""
        calls.append('generate_typescript_types')
        return 'export type Database = unknown'

    @server.tool()
    def search_docs(query: str = 'database') -> list[str]:
        """Search Supabase documentation."""
        calls.append(f'search_docs:{query}')
        return []


def _register_optional_tools(server: FastMCP, calls: list[str]) -> None:
    @server.tool()
    def list_edge_functions() -> list[str]:
        """List Edge Functions."""
        calls.append('list_edge_functions')
        return []

    @server.tool()
    def get_edge_function(name: str = 'hello') -> str:
        """Get an Edge Function."""
        calls.append(f'get_edge_function:{name}')
        return name

    @server.tool()
    def deploy_edge_function(name: str) -> str:
        """Deploy an Edge Function."""
        calls.append(f'deploy_edge_function:{name}')
        return 'deployed'

    @server.tool()
    def list_storage_buckets() -> list[str]:
        """List Storage buckets."""
        calls.append('list_storage_buckets')
        return []

    @server.tool()
    def get_storage_config() -> str:
        """Get Storage configuration."""
        calls.append('get_storage_config')
        return 'default'

    @server.tool()
    def update_storage_config() -> str:
        """Update Storage configuration."""
        calls.append('update_storage_config')
        return 'updated'


def _register_branching_tools(server: FastMCP, calls: list[str]) -> None:
    @server.tool()
    def create_branch(name: str) -> str:  # pragma: no cover - capability must never expose this incomplete flow
        """Create a database branch."""
        calls.append(f'create_branch:{name}')
        return name

    @server.tool()
    def list_branches() -> list[str]:
        """List database branches."""
        calls.append('list_branches')
        return []

    @server.tool()
    def delete_branch() -> str:
        """Delete a database branch."""
        calls.append('delete_branch')
        return 'deleted'

    @server.tool()
    def merge_branch() -> str:
        """Merge a database branch."""
        calls.append('merge_branch')
        return 'merged'

    @server.tool()
    def reset_branch() -> str:
        """Reset a database branch."""
        calls.append('reset_branch')
        return 'reset'

    @server.tool()
    def rebase_branch() -> str:
        """Rebase a database branch."""
        calls.append('rebase_branch')
        return 'rebased'

    @server.tool()
    def future_mutation() -> str:
        """Represent a server tool unknown to this capability version."""
        calls.append('future_mutation')  # pragma: no cover
        return 'changed'  # pragma: no cover


@pytest.fixture
def supabase_server(
    calls: list[str], connections: list[tuple[str, object]], failures: set[str], monkeypatch: pytest.MonkeyPatch
) -> FastMCP:
    from mcp.server.fastmcp.server import FastMCP, Settings  # noqa: PLC0415

    Settings.model_rebuild()
    server = FastMCP('supabase-fake')
    _register_database_tools(server, calls, failures)
    _register_default_tools(server, calls)
    _register_optional_tools(server, calls)
    _register_branching_tools(server, calls)

    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient  # noqa: PLC0415

    def local_toolset(
        client: MCPToolsetClient,
        *,
        id: str | None = None,
        auth: object = None,
    ) -> MCPToolset[None]:
        assert isinstance(client, str)
        assert urlsplit(client)._replace(query='').geturl() == 'https://mcp.supabase.com/mcp'
        parameters = parse_qs(urlsplit(client).query)
        assert parameters['project_ref']
        assert parameters['features']
        assert auth == 'oauth'
        connections.append((client, auth))
        return MCPToolset(server, id=id)

    monkeypatch.setattr('pydantic_ai_harness.supabase._capability.MCPToolset', local_toolset)

    return server
