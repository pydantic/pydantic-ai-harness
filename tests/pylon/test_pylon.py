"""Behavioral tests for Pylon through `Agent(capabilities=[...])`."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp.server import FastMCP, Settings
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.pylon import Pylon

# The MCP SDK leaves a settings annotation unresolved in some supported dependency
# combinations. Rebuild it before warnings are escalated by the test suite.
Settings.model_rebuild()


def _http_transport(toolset: AbstractToolset[Any]) -> StreamableHttpTransport:
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


def bearer(toolset: AbstractToolset[Any]) -> str:
    auth = _http_transport(toolset).auth
    assert auth is not None
    request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


async def connections_for(capability: Pylon[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
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


class TestPylon:
    @pytest.mark.anyio
    async def test_agent_runs_with_pylon_tools(self) -> None:
        server = FastMCP('pylon-fake')

        @server.tool()
        def search_issues() -> dict[str, str]:
            """Search Pylon issues."""
            return {'title': 'Login fails for Acme'}

        agent = Agent(TestModel(call_tools=['search_issues']), capabilities=[Pylon(client=server)])
        result = await agent.run('Summarize my open issues')
        assert 'Login fails for Acme' in result.output

    @pytest.mark.anyio
    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, include: bool) -> None:
        server = FastMCP('pylon-fake', instructions='Pylon instructions.')
        agent = Agent(TestModel(call_tools=[]), capabilities=[Pylon(client=server, include_instructions=include)])
        request = (await agent.run('Hello')).all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Pylon instructions.' in (request.instructions or '')) is include

    def test_connects_to_pylon_with_the_token(self) -> None:
        toolset = Pylon(auth='pylon-token').get_toolset()
        assert _http_transport(toolset).url == 'https://mcp.usepylon.com'
        assert bearer(toolset) == 'Bearer pylon-token'

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PYLON_ACCESS_TOKEN', 'environment-token')
        assert bearer(Pylon().get_toolset()) == 'Bearer environment-token'

    def test_missing_auth_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PYLON_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `PYLON_ACCESS_TOKEN`'):
            Pylon().get_toolset()

    @pytest.mark.filterwarnings('ignore:Using in-memory token storage')
    def test_oauth_uses_browser_login(self) -> None:
        assert isinstance(_http_transport(Pylon(auth='oauth').get_toolset()).auth, OAuth)

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Pylon(auth='secret-token'))

    @pytest.mark.parametrize('auth', ['pylon-token', no_credential])
    def test_client_cannot_be_combined_with_auth(self, auth: Any) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            Pylon(client='https://example.com/mcp', auth=auth)

    @pytest.mark.parametrize(
        'settings', [{'auth': 'pylon-token'}, {'auth': no_credential}, {'client': 'https://example.com/mcp'}]
    )
    def test_custom_id_is_forwarded(self, settings: dict[str, Any]) -> None:
        assert Pylon(id='tenant-pylon', **settings).get_toolset().id == 'tenant-pylon'

    def test_defer_loading_needs_no_id(self) -> None:
        Agent(TestModel(), capabilities=[Pylon(auth='pylon-token', defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError, match="Capability id 'pylon' is used by multiple Pylon capabilities that disagree on 'auth'"
        ):
            Agent(TestModel(), capabilities=[Pylon(auth='a'), Pylon(auth='b')])


class TestPerRunAuth:
    @pytest.mark.anyio
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = Pylon[str | None](auth=lambda ctx: ctx.deps)
        [alice] = await connections_for(capability, 'alice-token')
        [bob] = await connections_for(capability, 'bob-token')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-token', 'Bearer bob-token')

    @pytest.mark.anyio
    @pytest.mark.parametrize('missing', [None, ''])
    async def test_no_credential_means_no_tools(self, missing: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        # The environment token is set to show a function never falls back to it.
        monkeypatch.setenv('PYLON_ACCESS_TOKEN', 'deployment-token')
        capability = Pylon[str | None](auth=lambda ctx: ctx.deps)
        assert await connections_for(capability, missing) == []

    @pytest.mark.anyio
    async def test_function_returning_oauth_raises(self) -> None:
        capability = Pylon[str | None](auth=lambda ctx: ctx.deps)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')
