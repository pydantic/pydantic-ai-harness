"""Test AWS's connection settings and tool selection through an agent."""

from __future__ import annotations

from typing import Literal

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.aws import AWS

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


def transport(capability: AWS[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    result = toolset.client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


class TestAWS:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"read_resource":"read","write_resource":"written","unmarked_resource":"unmarked"}'),
            (True, '{"read_resource":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(self, server: FastMCP, read_only: bool, expected: str) -> None:
        agent = Agent(TestModel(), capabilities=[AWS(client=server, read_only=read_only)])
        result = await agent.run('Use the tools')
        assert result.output == expected

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[AWS(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    def test_custom_client_owns_authentication(self) -> None:
        client = StreamableHttpTransport('https://example.com/mcp', auth=httpx.BasicAuth('user', 'secret'))
        assert transport(AWS(client=client, auth='ignored')).auth is client.auth

    def test_auth_reaches_default_connection(self) -> None:
        auth = httpx.BasicAuth('user', 'secret')
        assert transport(AWS(auth=auth)).auth is auth

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(AWS(auth='secret-token'))

    @pytest.mark.parametrize('region', ['us-east-1', 'eu-central-1'])
    def test_regional_endpoint(self, region: Literal['us-east-1', 'eu-central-1']) -> None:
        assert transport(AWS(region=region, auth='token')).url == f'https://aws-mcp.{region}.api.aws/mcp'

    def test_default_uses_oauth(self) -> None:
        with pytest.warns(UserWarning, match='in-memory token storage'):
            assert transport(AWS()).auth is not None
