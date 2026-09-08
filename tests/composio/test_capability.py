"""Exercise Composio's MCP wiring through an agent."""

import pytest
from fastmcp import FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.composio import Composio


class TestComposio:
    @pytest.mark.parametrize('include', [True, False])
    async def test_session_tools_and_instructions(self, include: bool) -> None:
        server = FastMCP('session', instructions='Search before executing an action.')

        @server.tool()
        def search_tools() -> str:
            return 'gmail'

        capability = Composio(client=server, include_instructions=include)
        result = await Agent(TestModel(), capabilities=[capability]).run('Find email tools')
        assert result.output == '{"search_tools":"gmail"}'
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Search before executing an action.' in (request.instructions or '')) is include

    @pytest.mark.parametrize('headers', [None, {'x-api-key': 'secret', 'unset': None}])
    def test_session_connection(self, headers: dict[str, str | None] | None) -> None:
        capability = Composio(url='https://example.com/session/mcp', headers=headers)
        transport = capability.get_toolset().client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'https://example.com/session/mcp'
        assert transport.headers == ({'x-api-key': 'secret'} if headers else {})

    def test_headers_are_not_in_repr(self) -> None:
        assert 'secret' not in repr(Composio(url='https://example.com/mcp', headers={'x-api-key': 'secret'}))

    def test_connection_is_required(self) -> None:
        with pytest.raises(ValueError, match='session URL or a configured client'):
            Composio().get_toolset()
