"""Behavioral tests for Ordinal through `Agent(capabilities=[...])`."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp.server import FastMCP, Settings
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.ordinal import Ordinal

# The MCP SDK leaves a settings annotation unresolved in some supported dependency
# combinations. Rebuild it before warnings are escalated by the test suite.
Settings.model_rebuild()


def _http_transport(toolset: MCPToolset[Any]) -> StreamableHttpTransport:
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


async def connections_for(capability: Ordinal[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
    """The MCP connections a run with `deps` would open."""
    ctx = RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage())
    toolset = await capability.get_toolset().for_run(ctx)
    connections: list[MCPToolset[str | None]] = []

    def collect(leaf: AbstractToolset[str | None]) -> None:
        if isinstance(leaf, MCPToolset):
            connections.append(leaf)

    toolset.apply(collect)
    return connections


def no_credential(ctx: RunContext[object]) -> None:
    return None


def bearer(connection: MCPToolset[str | None]) -> str:
    transport = connection.client.transport
    assert isinstance(transport, StreamableHttpTransport) and transport.auth is not None
    request = next(transport.auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


class TestOrdinal:
    def test_agent_accepts_capability(self) -> None:
        capability = Ordinal(auth='ordinal-token')
        agent = Agent(TestModel(), capabilities=[capability])

        assert capability in agent.root_capability.capabilities

    @pytest.mark.anyio
    async def test_agent_runs_with_ordinal_tools(self) -> None:
        server = FastMCP('ordinal-fake')

        @server.tool()
        def ordinal_get_workspace_context() -> dict[str, str]:
            """List Ordinal workspaces."""
            return {'slug': 'acme'}

        class FakeOrdinal(Ordinal):
            def get_toolset(self) -> MCPToolset[Any]:
                return MCPToolset(server)

        agent = Agent(
            TestModel(call_tools=['ordinal_get_workspace_context']),
            capabilities=[FakeOrdinal()],
        )

        result = await agent.run('List my workspaces')

        assert 'acme' in result.output

    def test_serialization_name(self) -> None:
        assert Ordinal.get_serialization_name() == 'Ordinal'

    def test_hosted_url_and_forwards_instructions(self) -> None:
        toolset = Ordinal(auth='ordinal-token').get_toolset()
        assert isinstance(toolset, MCPToolset)
        transport = _http_transport(toolset)

        assert transport.url == 'https://app.tryordinal.com/mcp'
        assert toolset.include_instructions is True
        assert toolset.id == 'ordinal'

    def test_custom_id_is_forwarded(self) -> None:
        toolset = Ordinal(auth='ordinal-token', id='tenant-ordinal').get_toolset()

        assert isinstance(toolset, MCPToolset)
        assert toolset.id == 'tenant-ordinal'

    def test_missing_auth_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ORDINAL_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `ORDINAL_ACCESS_TOKEN`'):
            Ordinal().get_toolset()

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORDINAL_ACCESS_TOKEN', 'environment-token')
        assert isinstance(Ordinal().get_toolset(), MCPToolset)

    def test_fixed_token_is_sent_as_bearer(self) -> None:
        toolset = Ordinal(auth='ordinal-token').get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert bearer(toolset) == 'Bearer ordinal-token'


class TestPerRunAuth:
    @pytest.mark.anyio
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = Ordinal[str | None](auth=lambda ctx: ctx.deps)
        [alice] = await connections_for(capability, 'alice-token')
        [bob] = await connections_for(capability, 'bob-token')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-token', 'Bearer bob-token')

    @pytest.mark.anyio
    async def test_async_provider(self) -> None:
        async def token(ctx: RunContext[str | None]) -> str | None:
            return ctx.deps

        [connection] = await connections_for(Ordinal[str | None](auth=token), 'alice-token')
        assert bearer(connection) == 'Bearer alice-token'

    @pytest.mark.anyio
    async def test_provider_returning_none_omits_tools(self) -> None:
        capability = Ordinal[str | None](auth=lambda ctx: ctx.deps)
        assert await connections_for(capability, None) == []
        agent = Agent(TestModel(), capabilities=[Ordinal[object](auth=no_credential)])
        result = await agent.run('List my workspaces')
        assert result.output == 'success (no tool calls)'
