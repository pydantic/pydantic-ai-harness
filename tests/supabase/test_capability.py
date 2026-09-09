"""Test Supabase's connection settings and tool selection through an agent."""

from __future__ import annotations

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.supabase import Supabase

# MCP's test server leaves its lifespan annotation unresolved with pydantic-settings 2.15.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def server() -> FastMCP:
    server = FastMCP('provider', instructions='Provider instructions.')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def read_resource() -> str:
        return 'read'

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def write_resource() -> str:
        return 'written'

    @server.tool()
    def unmarked_resource() -> str:
        return 'unmarked'

    return server


def transport(capability: Supabase[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    result = toolset.client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


class TestSupabase:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"read_resource":"read","write_resource":"written","unmarked_resource":"unmarked"}'),
            (True, '{"read_resource":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(self, server: FastMCP, read_only: bool, expected: str) -> None:
        agent = Agent(TestModel(), capabilities=[Supabase(client=server, read_only=read_only)])
        result = await agent.run('Use the tools')
        assert result.output == expected

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Supabase(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    def test_custom_client_owns_authentication(self) -> None:
        client = StreamableHttpTransport('https://example.com/mcp', auth=httpx.BasicAuth('user', 'secret'))
        assert transport(Supabase(client=client, auth='ignored')).auth is client.auth

    def test_auth_reaches_default_connection(self) -> None:
        auth = httpx.BasicAuth('user', 'secret')
        assert transport(Supabase(auth=auth)).auth is auth

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Supabase(auth='secret-token'))

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SUPABASE_ACCESS_TOKEN', 'environment-token')
        auth = transport(Supabase()).auth
        assert isinstance(auth, httpx.Auth)
        request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
        assert request.headers['Authorization'] == 'Bearer environment-token'

    def test_native_project_configuration(self) -> None:
        connection = transport(Supabase(auth='token', project_ref='my-project', features=['database'], read_only=True))
        assert connection.url == 'https://mcp.supabase.com/mcp?project_ref=my-project&features=database&read_only=true'

    def test_default_retains_server_configuration(self) -> None:
        assert transport(Supabase(auth='token')).url == 'https://mcp.supabase.com/mcp'

    def test_oauth_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SUPABASE_ACCESS_TOKEN', raising=False)
        with pytest.warns(UserWarning, match='in-memory token storage'):
            assert transport(Supabase()).auth is not None
