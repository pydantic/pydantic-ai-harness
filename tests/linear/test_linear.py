"""Tests for Linear through the public capability and MCP boundaries."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.auth import BearerAuth, OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import AnyUrl
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FilteredToolset

from pydantic_ai_harness.linear import LINEAR_MCP_URL, LINEAR_READ_ONLY_MCP_URL, Linear

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio


def _tool_calls(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


def _transport(capability: Linear[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


def _tool_names(model: TestModel) -> list[str]:
    parameters = model.last_model_request_parameters
    assert parameters is not None
    return [tool.name for tool in parameters.function_tools]


class _Auth(httpx.Auth):
    def auth_flow(self, request: httpx.Request):
        yield request


class TestLinear:
    def test_default_uses_read_only_endpoint_and_oauth(self):
        with pytest.warns(UserWarning, match='in-memory token storage'):
            transport = _transport(Linear())
        assert transport.url == LINEAR_READ_ONLY_MCP_URL
        assert isinstance(transport.auth, OAuth)

    def test_read_write_uses_standard_endpoint(self):
        with pytest.warns(UserWarning, match='in-memory token storage'):
            transport = _transport(Linear(read_only=False))
        assert transport.url == LINEAR_MCP_URL

    def test_bearer_token_is_forwarded_and_hidden(self):
        capability = Linear(auth='linear-secret')
        transport = _transport(capability)
        assert isinstance(transport.auth, BearerAuth)
        assert transport.auth.token.get_secret_value() == 'linear-secret'
        assert 'linear-secret' not in repr(capability)

    def test_custom_httpx_auth_is_forwarded(self):
        auth = _Auth()
        assert _transport(Linear(auth=auth)).auth is auth

    def test_url_client_uses_configured_auth(self):
        transport = _transport(Linear(client='https://linear.example.test/mcp', auth='linear-secret'))
        assert transport.url == 'https://linear.example.test/mcp'
        assert isinstance(transport.auth, BearerAuth)
        assert transport.auth.token.get_secret_value() == 'linear-secret'

    def test_url_client_defaults_to_oauth(self):
        with pytest.warns(UserWarning, match='in-memory token storage'):
            transport = _transport(Linear(client='https://linear.example.test/mcp'))
        assert isinstance(transport.auth, OAuth)

    def test_any_url_client_uses_configured_auth(self):
        transport = _transport(Linear(client=AnyUrl('https://linear.example.test/mcp'), auth='linear-secret'))
        assert isinstance(transport.auth, BearerAuth)

    @pytest.mark.parametrize('client', ['http://linear.example.test/mcp', AnyUrl('http://linear.example.test/mcp')])
    def test_url_client_requires_https(self, client: str | AnyUrl):
        with pytest.raises(UserError, match='must use HTTPS'):
            Linear(client=client, auth='linear-secret').get_toolset()

    def test_prebuilt_toolset_is_returned_unchanged(self, linear_server: FastMCP):
        toolset = MCPToolset(linear_server, id='linear-test')
        capability = Linear(client=toolset)
        assert capability.get_toolset() is toolset

    def test_prebuilt_client_is_wrapped_without_default_auth(self, linear_server: FastMCP):
        client = Client(linear_server)
        capability = Linear(client=client)
        toolset = capability.get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.client is client

    def test_prebuilt_values_reject_separate_auth(self, linear_server: FastMCP):
        with pytest.raises(UserError, match='cannot be combined with a prebuilt MCP client or toolset'):
            Linear(client=Client(linear_server), auth='linear-secret')
        with pytest.raises(UserError, match='cannot be combined with a prebuilt MCP client or toolset'):
            Linear(client=MCPToolset(linear_server), auth='linear-secret')

    def test_runtime_escape_hatches_are_hidden_from_spec(self):
        schema = json.dumps(AgentSpec.model_json_schema_with_capabilities([Linear]), sort_keys=True)
        assert '"client"' not in schema
        assert '"httpx.Auth"' not in schema

    def test_agent_spec_loads_non_default_configuration(self):
        arguments = {
            'read_only': False,
            'allowed_tools': ['create_comment'],
            'auth': 'linear-secret',
            'include_instructions': False,
            'id': 'linear-workspace',
            'description': 'Project tracker',
            'defer_loading': True,
        }
        spec = {
            'capabilities': [
                {
                    'Linear': arguments,
                }
            ]
        }
        agent = Agent.from_spec(spec, custom_capability_types=[Linear], model=TestModel())
        assert isinstance(agent, Agent)
        capability = Linear.from_spec(
            read_only=False,
            allowed_tools=['create_comment'],
            auth='linear-secret',
            include_instructions=False,
            id='linear-workspace',
            description='Project tracker',
            defer_loading=True,
        )
        assert capability.read_only is False
        assert capability.allowed_tools == ['create_comment']
        assert capability.auth == 'linear-secret'
        assert capability.include_instructions is False
        assert capability.id == 'linear-workspace'
        assert capability.description == 'Project tracker'
        assert capability.defer_loading is True

    async def test_agent_run_calls_linear_tool(self, linear_server: FastMCP):
        agent = Agent(TestModel(call_tools=['list_issues']), capabilities=[Linear(client=linear_server)])
        result = await agent.run('Find the integrations issue')
        assert _tool_calls(result.all_messages()) == {'list_issues'}
        assert 'HAR-787' in result.output

    async def test_allowed_tools_are_exact(self, linear_server: FastMCP):
        capability = Linear(client=linear_server, allowed_tools=['list_issues'])
        toolset = capability.get_toolset()
        assert isinstance(toolset, FilteredToolset)

        model = TestModel()
        agent = Agent(model, capabilities=[capability])
        await agent.run('What tools are available?')
        assert _tool_names(model) == ['list_issues']

    async def test_prebuilt_toolset_is_filtered(self, linear_server: FastMCP):
        model = TestModel()
        mcp = MCPToolset(linear_server)
        agent = Agent(model, capabilities=[Linear(client=mcp, allowed_tools=['list_issues'])])
        await agent.run('What tools are available?')
        assert _tool_names(model) == ['list_issues']

    async def test_empty_allowed_tools_exposes_no_tools(self, linear_server: FastMCP):
        model = TestModel()
        agent = Agent(model, capabilities=[Linear(client=linear_server, allowed_tools=[])])
        await agent.run('What tools are available?')
        assert _tool_names(model) == []

    async def test_two_instances_collide_on_server_tool_names(self, linear_server: FastMCP):
        agent = Agent(
            TestModel(),
            capabilities=[Linear(client=linear_server), Linear(client=linear_server)],
        )
        with pytest.raises(UserError, match="defines a tool whose name conflicts.*'list_issues'"):
            await agent.run('What tools are available?')

    async def test_short_instructions_are_injected(self, linear_server: FastMCP):
        agent = Agent(TestModel(), capabilities=[Linear(client=linear_server)])
        result = await agent.run('What tools are available?')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert request.instructions == 'Use Linear tools to find and manage work. Read before changing anything.'

    def test_instructions_can_be_disabled(self):
        assert Linear(include_instructions=False).get_instructions() is None

    def test_serialization_name(self):
        assert Linear.get_serialization_name() == 'Linear'
