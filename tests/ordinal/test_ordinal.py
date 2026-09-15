"""Behavioral tests for Ordinal through `Agent(capabilities=[...])`."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp.server import FastMCP, Settings
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.ordinal import Ordinal

# The MCP SDK leaves a settings annotation unresolved in some supported dependency
# combinations. Rebuild it before warnings are escalated by the test suite.
Settings.model_rebuild()


def _http_transport(toolset: MCPToolset[Any]) -> StreamableHttpTransport:
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestOrdinal:
    def test_agent_accepts_capability(self) -> None:
        capability = Ordinal()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
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

    def test_hosted_url_uses_oauth_and_forwards_instructions(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            toolset = Ordinal().get_toolset()
        transport = _http_transport(toolset)

        assert transport.url == 'https://app.tryordinal.com/mcp'
        assert transport.auth is not None
        assert toolset.include_instructions is True
        assert toolset.id == 'ordinal'

    def test_custom_id_is_forwarded(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            toolset = Ordinal(id='tenant-ordinal').get_toolset()

        assert isinstance(toolset, MCPToolset)
        assert toolset.id == 'tenant-ordinal'
