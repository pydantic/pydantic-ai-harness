"""Test Stripe's connection settings and tools through an agent."""

from __future__ import annotations

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.stripe import Stripe

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

    @server.tool()
    def read_resource() -> str:
        return 'read'

    return server


def transport(capability: Stripe[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    result = toolset.client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


class TestStripe:
    async def test_agent_executes_tools(self, server: FastMCP) -> None:
        agent = Agent(TestModel(), capabilities=[Stripe(client=server)])
        result = await agent.run('Use the tools')
        assert result.output == '{"read_resource":"read"}'

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Stripe(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    def test_custom_client_owns_authentication(self) -> None:
        client = StreamableHttpTransport('https://example.com/mcp', auth=httpx.BasicAuth('user', 'secret'))
        assert transport(Stripe(client=client, auth='ignored')).auth is client.auth

    def test_auth_reaches_default_connection(self) -> None:
        auth = httpx.BasicAuth('user', 'secret')
        assert transport(Stripe(auth=auth)).auth is auth

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Stripe(auth='secret-token'))

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('STRIPE_API_KEY', 'environment-token')
        auth = transport(Stripe()).auth
        assert isinstance(auth, httpx.Auth)
        request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
        assert request.headers['Authorization'] == 'Bearer environment-token'

    def test_native_connected_account(self) -> None:
        assert transport(Stripe(auth='token', connected_account='acct_example')).headers == {
            'Stripe-Account': 'acct_example'
        }

    def test_hosted_endpoint(self) -> None:
        assert transport(Stripe(auth='token')).url == 'https://mcp.stripe.com'

    def test_oauth_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('STRIPE_API_KEY', raising=False)
        with pytest.warns(UserWarning, match='in-memory token storage'):
            assert transport(Stripe()).auth is not None
