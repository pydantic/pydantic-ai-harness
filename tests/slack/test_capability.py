"""Test Slack's connection settings and tool selection through an agent."""

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

from pydantic_ai_harness.slack import Slack


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


def transport(capability: Slack[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    result = toolset.client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


class TestSlack:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"read_resource":"read","write_resource":"written","unmarked_resource":"unmarked"}'),
            (True, '{"read_resource":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(self, server: FastMCP, read_only: bool, expected: str) -> None:
        agent = Agent(TestModel(), capabilities=[Slack(client=server, read_only=read_only)])
        result = await agent.run('Use the tools')
        assert result.output == expected

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Slack(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    def test_custom_client_owns_authentication(self) -> None:
        client = StreamableHttpTransport('https://example.com/mcp', auth=httpx.BasicAuth('user', 'secret'))
        assert transport(Slack(client=client, auth='ignored')).auth is client.auth

    def test_auth_reaches_default_connection(self) -> None:
        auth = httpx.BasicAuth('user', 'secret')
        assert transport(Slack(auth=auth)).auth is auth

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Slack(auth='secret-token'))

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_USER_TOKEN', 'environment-token')
        auth = transport(Slack()).auth
        assert isinstance(auth, httpx.Auth)
        request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
        assert request.headers['Authorization'] == 'Bearer environment-token'

    def test_hosted_endpoint(self) -> None:
        assert transport(Slack(auth='token')).url == 'https://mcp.slack.com/mcp'
