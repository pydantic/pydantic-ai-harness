"""Behavioral tests for Logfire MCP through `Agent(capabilities=[...])`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio

Calls = list[tuple[str, dict[str, object]]]


def _tool_call_names(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


def _instructions(messages: list[ModelMessage]) -> str:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    return first.instructions or ''


def _http_transport(capability: LogfireMCP[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestLogfireMCP:
    def test_agent_spec_schema_excludes_runtime_client(self):
        schema = AgentSpec.model_json_schema_with_capabilities([LogfireMCP])
        properties = schema['$defs']['spec_params_LogfireMCP']['properties']
        assert 'client' not in properties
        assert {'url', 'auth', 'allowed_tools'} <= set(properties)
        assert LogfireMCP.get_serialization_name() == 'LogfireMCP'

    def test_from_spec_forwards_serializable_options(self):
        capability = LogfireMCP.from_spec(
            id='prod-logfire',
            description='Production telemetry',
            defer_loading=True,
            url=LOGFIRE_EU_MCP_URL,
            auth='token',
            allowed_tools=['query_run'],
            include_instructions=False,
        )
        assert capability.id == 'prod-logfire'
        assert capability.description == 'Production telemetry'
        assert capability.defer_loading is True
        assert capability.url == LOGFIRE_EU_MCP_URL
        assert capability.auth == 'token'
        assert capability.allowed_tools == ['query_run']
        assert capability.include_instructions is False

    def test_defaults_to_us_endpoint_with_oauth(self):
        with pytest.warns(UserWarning, match='in-memory token storage'):
            transport = _http_transport(LogfireMCP[None]())

        assert transport.url == LOGFIRE_US_MCP_URL
        assert transport.auth is not None

    def test_api_key_and_url_reach_transport_and_stay_out_of_repr(self):
        capability = LogfireMCP[None](url='https://logfire.acme.example/mcp', auth='secret-key')
        transport = _http_transport(capability)

        assert transport.url == 'https://logfire.acme.example/mcp'
        assert transport.auth is not None
        assert 'secret-key' not in repr(capability)

    def test_injected_client_ignores_auth(self):
        transport = StreamableHttpTransport('https://logfire.acme.example/mcp')
        toolset = LogfireMCP[None](client=transport, auth='ignored-key').get_toolset()

        assert isinstance(toolset, MCPToolset)
        assert toolset.client.transport is transport

    async def test_server_instructions_reach_the_model(self, logfire_server: FastMCP, logfire_calls: Calls):
        agent = Agent(
            TestModel(call_tools=['project_list', 'query_run']),
            capabilities=[LogfireMCP(client=logfire_server)],
        )
        result = await agent.run('Count recent errors')

        assert [name for name, _ in logfire_calls] == ['project_list', 'query_run']
        assert 'Call project_list before other Logfire tools.' in _instructions(result.all_messages())

    async def test_server_instructions_can_be_left_out(self, logfire_server: FastMCP):
        capability = LogfireMCP(client=logfire_server, include_instructions=False)
        result = await Agent(TestModel(call_tools=[]), capabilities=[capability]).run('hello')

        assert 'project_list' not in _instructions(result.all_messages())

    @pytest.mark.parametrize(('allowed', 'expected'), [(['query_run'], {'query_run'}), (['query'], set[str]())])
    async def test_allowed_tools_filters_by_exact_name(
        self, logfire_server: FastMCP, allowed: list[str], expected: set[str]
    ):
        capability = LogfireMCP(client=logfire_server, allowed_tools=allowed)
        result = await Agent(TestModel(), capabilities=[capability]).run('Count errors')

        assert _tool_call_names(result.all_messages()) == expected
