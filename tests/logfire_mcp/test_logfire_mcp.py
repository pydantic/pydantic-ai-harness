"""Behavioral tests for Logfire MCP through `Agent(capabilities=[...])`."""

from __future__ import annotations

from datetime import timezone
from typing import TYPE_CHECKING

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessage,
    ModelRequest,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import WrapperToolset

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


def _mcp_toolset(capability: LogfireMCP[None]) -> MCPToolset[None]:
    toolset = capability.get_toolset()
    while isinstance(toolset, WrapperToolset):
        toolset = toolset.wrapped
    assert isinstance(toolset, MCPToolset)
    return toolset


def _http_transport(capability: LogfireMCP[None]) -> StreamableHttpTransport:
    transport = _mcp_toolset(capability).client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestLogfireMCP:
    def test_agent_spec_schema_excludes_runtime_client_and_loads_capability(self):
        schema = AgentSpec.model_json_schema_with_capabilities([LogfireMCP])
        properties = schema['$defs']['spec_params_LogfireMCP']['properties']
        assert 'client' not in properties
        assert {'url', 'auth', 'read_only', 'allowed_tools'} <= set(properties)
        assert LogfireMCP.get_serialization_name() == 'LogfireMCP'
        agent = Agent.from_spec(
            {
                'capabilities': [
                    {
                        'LogfireMCP': {
                            'url': LOGFIRE_EU_MCP_URL,
                            'auth': None,
                            'read_only': True,
                            'allowed_tools': ['query_run'],
                            'include_instructions': False,
                        }
                    }
                ]
            },
            custom_capability_types=[LogfireMCP],
            model=TestModel(),
        )
        assert isinstance(agent, Agent)

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
        toolset = _mcp_toolset(LogfireMCP[None](client=transport, auth='ignored-key'))

        assert toolset.client.transport is transport

    async def test_read_only_hides_tools_not_marked_read_only(self, logfire_server: FastMCP, logfire_calls: Calls):
        capability = LogfireMCP(client=logfire_server, read_only=True)
        result = await Agent(TestModel(), capabilities=[capability]).run('Set up a dashboard')

        assert _tool_call_names(result.all_messages()) == {'project_list', 'query_run'}
        assert [name for name, _ in logfire_calls] == ['project_list', 'query_run']

    async def test_default_exposes_every_tool_and_requires_mutation_approval(
        self, logfire_server: FastMCP, logfire_calls: Calls
    ):
        capability = LogfireMCP(client=logfire_server)
        agent = Agent(TestModel(), capabilities=[capability], output_type=[str, DeferredToolRequests])

        deferred = await agent.run('Set up a dashboard')

        assert _tool_call_names(deferred.all_messages()) == {
            'project_list',
            'query_run',
            'dashboard_create',
            'unannotated_tool',
        }
        assert isinstance(deferred.output, DeferredToolRequests)
        assert {call.tool_name for call in deferred.output.approvals} == {'dashboard_create', 'unannotated_tool'}
        assert [name for name, _ in logfire_calls] == ['project_list', 'query_run']

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={call.tool_call_id: True for call in deferred.output.approvals}
            ),
        )

        assert isinstance(resumed.output, str)
        assert [name for name, _ in logfire_calls] == [
            'project_list',
            'query_run',
            'dashboard_create',
            'unannotated_tool',
        ]

    async def test_instructions_keep_static_guidance_separate_from_run_time(
        self, logfire_server: FastMCP, logfire_calls: Calls
    ):
        model = TestModel(call_tools=['project_list', 'query_run'])
        agent = Agent(
            model,
            capabilities=[LogfireMCP(client=logfire_server)],
        )
        result = await agent.run('Count recent errors')

        assert [name for name, _ in logfire_calls] == ['project_list', 'query_run']
        requests = [message for message in result.all_messages() if isinstance(message, ModelRequest)]
        assert len(requests) == 2
        assert requests[0].instructions == requests[1].instructions

        user_prompt = next(part for part in requests[0].parts if isinstance(part, UserPromptPart))
        expected_time = user_prompt.timestamp.astimezone(timezone.utc).isoformat(timespec='seconds')
        request_parameters = model.last_model_request_parameters
        assert request_parameters is not None
        instruction_parts = request_parameters.instruction_parts
        assert instruction_parts is not None
        dynamic_parts = [part for part in instruction_parts if part.dynamic]
        static_parts = [part for part in instruction_parts if not part.dynamic]
        assert [part.content for part in dynamic_parts] == [f'Current UTC time for this run is `{expected_time}`.']
        assert len(static_parts) == 2
        assert any(part.content == 'Call project_list before other Logfire tools.' for part in static_parts)

    async def test_server_instructions_can_be_left_out(self, logfire_server: FastMCP):
        capability = LogfireMCP(client=logfire_server, include_instructions=False)
        result = await Agent(TestModel(call_tools=[]), capabilities=[capability]).run('hello')

        assert _instructions(result.all_messages()) == ''

    @pytest.mark.parametrize(('allowed', 'expected'), [(['query_run'], {'query_run'}), (['query'], set[str]())])
    async def test_allowed_tools_filters_by_exact_name(
        self, logfire_server: FastMCP, allowed: list[str], expected: set[str]
    ):
        capability = LogfireMCP(client=logfire_server, allowed_tools=allowed)
        result = await Agent(TestModel(), capabilities=[capability]).run('Count errors')

        assert _tool_call_names(result.all_messages()) == expected

    async def test_allowed_write_tool_still_requires_approval(self, logfire_server: FastMCP, logfire_calls: Calls):
        capability = LogfireMCP(client=logfire_server, allowed_tools=['dashboard_create'])
        agent = Agent(TestModel(), capabilities=[capability], output_type=[str, DeferredToolRequests])

        deferred = await agent.run('Set up a dashboard')

        assert isinstance(deferred.output, DeferredToolRequests)
        assert [call.tool_name for call in deferred.output.approvals] == ['dashboard_create']
        assert logfire_calls == []
