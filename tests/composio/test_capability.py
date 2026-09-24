"""Exercise Composio's MCP wiring through an agent."""

from typing import Any

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.composio import Composio

# MCP's test server leaves its lifespan annotation unresolved with some pydantic-settings versions.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
)


def session_server(user: str) -> FastMCP:
    server = FastMCP(f'{user}-session')

    @server.tool()
    def whoami() -> str:
        return user

    return server


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
        toolset = Composio(url='https://example.com/session/mcp', headers=headers).get_toolset()
        assert isinstance(toolset, MCPToolset)
        transport = toolset.client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'https://example.com/session/mcp'
        assert transport.headers == ({'x-api-key': 'secret'} if headers else {})

    def test_headers_are_not_in_repr(self) -> None:
        assert 'secret' not in repr(Composio(url='https://example.com/mcp', headers={'x-api-key': 'secret'}))

    def test_connection_is_required(self) -> None:
        with pytest.raises(UserError, match='Pass `url` from `session.mcp.url`, or your own `client`'):
            Composio()

    def test_custom_client_owns_the_connection(self) -> None:
        client = StreamableHttpTransport('https://example.com/mcp')
        toolset = Composio(client=client).get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.client.transport is client

    @pytest.mark.parametrize(
        'settings', [{'url': 'https://example.com/session/mcp'}, {'headers': {'x-api-key': 'key'}}]
    )
    def test_client_cannot_be_combined_with_connection_settings(self, settings: dict[str, Any]) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            Composio(client='https://example.com/mcp', **settings)

    def test_defer_loading_needs_no_id(self) -> None:
        Agent(TestModel(), capabilities=[Composio(client=session_server('ada'), defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(UserError, match="Two `Composio` capabilities share the id 'composio'"):
            Agent(
                TestModel(),
                capabilities=[
                    Composio(url='https://example.com/first/mcp'),
                    Composio(url='https://example.com/second/mcp'),
                ],
            )


class TestDynamicCapability:
    """A dynamic capability builds each user's own `Composio` session, as the docs show."""

    async def test_each_run_uses_its_own_users_session(self) -> None:
        sessions = {user: session_server(user) for user in ('alice', 'bob')}

        def composio_session(ctx: RunContext[str]) -> Composio[str]:
            return Composio(client=sessions[ctx.deps])

        agent = Agent(TestModel(), deps_type=str, capabilities=[DynamicCapability(composio_session, id='composio')])
        alice = await agent.run('Who am I?', deps='alice')
        bob = await agent.run('Who am I?', deps='bob')
        assert (alice.output, bob.output) == ('{"whoami":"alice"}', '{"whoami":"bob"}')

    async def test_async_factory_returning_none_omits_tools(self) -> None:
        async def composio_session(ctx: RunContext[str]) -> Composio[str]:
            return Composio(client=session_server(ctx.deps))

        agent = Agent(TestModel(), deps_type=str, capabilities=[DynamicCapability(composio_session, id='composio')])
        alice = await agent.run('Who am I?', deps='alice')
        assert alice.output == '{"whoami":"alice"}'

        def no_session(ctx: RunContext[object]) -> None:
            return None

        agent = Agent(TestModel(), capabilities=[DynamicCapability(no_session, id='composio')])
        nobody = await agent.run('Who am I?')
        assert nobody.output == 'success (no tool calls)'
