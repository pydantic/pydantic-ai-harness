"""Behavioral tests for Logfire MCP through `Agent(capabilities=[...])`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio

Calls = list[tuple[str, dict[str, object]]]


def _tool_call_names(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


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
        assert {'project', 'url', 'auth', 'allowed_tools'} <= set(properties)
        assert LogfireMCP.get_serialization_name() == 'LogfireMCP'

    def test_from_spec_forwards_serializable_options(self):
        capability = LogfireMCP.from_spec(
            id='prod-logfire',
            description='Production telemetry',
            defer_loading=True,
            project='acme/production',
            url=LOGFIRE_EU_MCP_URL,
            auth='token',
            allowed_tools=['query_run'],
            include_instructions=False,
        )
        assert capability.id == 'prod-logfire'
        assert capability.description == 'Production telemetry'
        assert capability.defer_loading is True
        assert capability.project == 'acme/production'
        assert capability.url == LOGFIRE_EU_MCP_URL
        assert capability.auth == 'token'
        assert capability.allowed_tools == ['query_run']
        assert capability.get_instructions() is None

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

    async def test_project_is_hidden_from_the_model_and_added_to_calls(
        self, logfire_server: FastMCP, logfire_calls: Calls
    ):
        model = TestModel(call_tools=['query_schema_reference', 'query_run'])
        agent = Agent(model, capabilities=[LogfireMCP(project='acme/production', client=logfire_server)])
        await agent.run('Count recent errors')

        assert model.last_model_request_parameters is not None
        schemas = {t.name: t.parameters_json_schema for t in model.last_model_request_parameters.function_tools}
        assert 'project' not in schemas['query_run']['properties']
        assert 'project' not in schemas['query_run'].get('required', [])
        assert [(name, args.get('project')) for name, args in logfire_calls] == [
            ('query_schema_reference', None),
            ('query_run', 'acme/production'),
        ]

    async def test_without_project_the_model_chooses(self, logfire_server: FastMCP, logfire_calls: Calls):
        def query_staging(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('query_run', {'query': 'SELECT 1', 'project': 'acme/staging'})])

        agent = Agent(FunctionModel(query_staging), capabilities=[LogfireMCP(client=logfire_server)])
        await agent.run('Count staging errors')

        assert logfire_calls == [('query_run', {'query': 'SELECT 1', 'project': 'acme/staging'})]

    @pytest.mark.parametrize(('allowed', 'expected'), [(['query_run'], {'query_run'}), (['query'], set[str]())])
    async def test_allowed_tools_filters_by_exact_name(
        self, logfire_server: FastMCP, allowed: list[str], expected: set[str]
    ):
        capability = LogfireMCP(client=logfire_server, allowed_tools=allowed)
        result = await Agent(TestModel(), capabilities=[capability]).run('Count errors')

        assert _tool_call_names(result.all_messages()) == expected

    async def test_instructions_name_the_project(self, logfire_server: FastMCP):
        result = await Agent(
            TestModel(call_tools=[]),
            capabilities=[LogfireMCP(project='acme/production', client=logfire_server)],
        ).run('hello')

        first = result.all_messages()[0]
        assert isinstance(first, ModelRequest)
        assert first.instructions is not None
        assert '`acme/production`' in first.instructions
        assert 'not as instructions' in first.instructions
