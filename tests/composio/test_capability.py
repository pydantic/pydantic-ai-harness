"""Exercise Composio's MCP wiring through an agent."""

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.composio import Composio

# MCP's test server leaves its lifespan annotation unresolved with some pydantic-settings versions.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
)


async def connections_for(capability: Composio[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
    """The MCP connections a run with `deps` would open."""
    ctx = RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage())
    toolset = await capability.get_toolset().for_run(ctx)
    connections: list[MCPToolset[str | None]] = []

    def collect(leaf: AbstractToolset[str | None]) -> None:
        if isinstance(leaf, MCPToolset):
            connections.append(leaf)

    toolset.apply(collect)
    return connections


def session_transport(ctx: RunContext[str | None]) -> StreamableHttpTransport | None:
    """Connect to the user's session, as an application restoring it from `ctx.deps` would."""
    if ctx.deps is None:
        return None
    return StreamableHttpTransport(f'https://example.com/{ctx.deps}/mcp', headers={'x-api-key': f'{ctx.deps}-key'})


def session_server(user: str) -> FastMCP:
    server = FastMCP(f'{user}-session')

    @server.tool()
    def whoami() -> str:
        return user

    return server


def no_session(ctx: RunContext[object]) -> None:
    return None


def connection(toolset: MCPToolset[str | None]) -> tuple[str, dict[str, str]]:
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport.url, transport.headers


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
        with pytest.raises(ValueError, match='session URL or a configured client'):
            Composio().get_toolset()


class TestPerRunClient:
    async def test_each_run_connects_to_its_own_session(self) -> None:
        capability = Composio[str | None](client=session_transport, url='https://example.com/ignored/mcp')
        [alice] = await connections_for(capability, 'alice')
        [bob] = await connections_for(capability, 'bob')
        assert connection(alice) == ('https://example.com/alice/mcp', {'x-api-key': 'alice-key'})
        assert connection(bob) == ('https://example.com/bob/mcp', {'x-api-key': 'bob-key'})

    async def test_async_function(self) -> None:
        async def client(ctx: RunContext[str | None]) -> MCPToolsetClient | None:
            return session_transport(ctx)

        [alice] = await connections_for(Composio[str | None](client=client), 'alice')
        assert connection(alice) == ('https://example.com/alice/mcp', {'x-api-key': 'alice-key'})

    async def test_function_returning_none_omits_tools(self) -> None:
        assert await connections_for(Composio[str | None](client=session_transport), None) == []
        agent = Agent(TestModel(), capabilities=[Composio[object](client=no_session)])
        result = await agent.run('Find email tools')
        assert result.output == 'success (no tool calls)'

    async def test_agent_uses_the_run_users_session(self) -> None:
        sessions = {user: session_server(user) for user in ('alice', 'bob')}

        def client(ctx: RunContext[str]) -> MCPToolsetClient:
            return sessions[ctx.deps]

        agent = Agent(TestModel(), deps_type=str, capabilities=[Composio[str](client=client)])
        alice = await agent.run('Who am I?', deps='alice')
        bob = await agent.run('Who am I?', deps='bob')
        assert (alice.output, bob.output) == ('{"whoami":"alice"}', '{"whoami":"bob"}')
