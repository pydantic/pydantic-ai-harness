"""Behavioral tests for Linear through `Agent(capabilities=[...])`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport, StreamableHttpTransport
from pydantic import AnyUrl
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FilteredToolset

from pydantic_ai_harness.linear import LINEAR_MCP_URL, LINEAR_READ_ONLY_MCP_URL, Linear

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio


def _tool_call_names(messages: list[ModelMessage]) -> set[str]:
    return {part.tool_name for message in messages for part in message.parts if isinstance(part, ToolCallPart)}


def _request_instructions(messages: list[ModelMessage]) -> str:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    return first.instructions or ''


def _http_transport(linear: Linear[None]) -> StreamableHttpTransport:
    toolset = linear.get_toolset()
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestLinear:
    def test_agent_spec_schema_excludes_runtime_client(self):
        schema = AgentSpec.model_json_schema_with_capabilities([Linear])
        linear_schema = schema['$defs']['spec_params_Linear']
        assert 'client' not in linear_schema['properties']

    def test_serialization_name(self):
        assert Linear.get_serialization_name() == 'Linear'

    def test_from_spec_forwards_serializable_options(self):
        capability = Linear.from_spec(
            id='tenant-linear',
            description='Tenant issues',
            defer_loading=True,
            read_only=False,
            auth='token',
            allowed_tools=['create_issue'],
            include_instructions=False,
        )
        assert capability.id == 'tenant-linear'
        assert capability.description == 'Tenant issues'
        assert capability.defer_loading is True
        assert capability.read_only is False
        assert capability.auth == 'token'
        assert capability.allowed_tools == ['create_issue']
        assert capability.include_instructions is False

    async def test_agent_from_spec_runs_linear(self, linear_server: FastMCP, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            'pydantic_ai_harness.linear._capability.LINEAR_READ_ONLY_MCP_URL',
            linear_server,
        )
        agent = Agent.from_spec(
            {
                'capabilities': [
                    {
                        'Linear': {
                            'allowed_tools': ['get_issue'],
                        }
                    }
                ]
            },
            custom_capability_types=[Linear],
            model=TestModel(),
        )
        result = await agent.run('Read an issue')
        assert _tool_call_names(result.all_messages()) == {'get_issue'}
        assert 'identifiers' in _request_instructions(result.all_messages())
        assert 'Before changing Linear data' not in _request_instructions(result.all_messages())

    def test_default_uses_documented_read_only_endpoint(self):
        transport = _http_transport(Linear())
        assert transport.url == LINEAR_READ_ONLY_MCP_URL
        assert transport.auth is None

    def test_oauth_is_forwarded(self):
        with pytest.warns(UserWarning, match='in-memory token storage'):
            transport = _http_transport(Linear(auth='oauth'))
        assert transport.auth is not None

    def test_read_write_is_explicit(self):
        transport = _http_transport(Linear(read_only=False))
        assert transport.url == LINEAR_MCP_URL

    def test_bearer_auth_is_forwarded_and_hidden_from_repr(self):
        capability = Linear(auth='lin_api_secret')
        transport = _http_transport(capability)
        assert transport.auth is not None
        assert 'lin_api_secret' not in repr(capability)

    def test_url_client_receives_auth(self):
        transport = _http_transport(Linear(client='https://proxy.example/mcp', auth='lin_api_secret'))
        assert transport.url == 'https://proxy.example/mcp'
        assert transport.auth is not None

    def test_url_client_normalizes_uppercase_scheme(self):
        transport = _http_transport(Linear(client='HTTPS://proxy.example/mcp'))
        assert transport.url == 'https://proxy.example/mcp'

    def test_any_url_client_receives_auth_and_custom_id(self):
        capability = Linear(client=AnyUrl('https://proxy.example/mcp'), auth='token', id='tenant-linear')
        toolset = capability.get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.id == 'tenant-linear'
        assert isinstance(toolset.client.transport, StreamableHttpTransport)
        assert toolset.client.transport.auth is not None

    @pytest.mark.parametrize('client', ['http://proxy.example/mcp', AnyUrl('http://proxy.example/mcp')])
    def test_authenticated_http_url_is_rejected(self, client: str | AnyUrl):
        with pytest.raises(UserError, match='must use HTTPS'):
            Linear(client=client, auth='lin_api_secret').get_toolset()

    @pytest.mark.parametrize(
        'client',
        [
            'http://user:secret@proxy.example/mcp',
            AnyUrl('http://user:secret@proxy.example/mcp'),
            'https://user:secret@proxy.example/mcp',
            AnyUrl('https://user:secret@proxy.example/mcp'),
        ],
    )
    def test_url_credentials_are_rejected(self, client: str | AnyUrl):
        with pytest.raises(UserError, match='must not contain credentials'):
            Linear(client=client).get_toolset()

    def test_falsey_injected_client_is_preserved(self, linear_server: FastMCP):
        class FalseyFastMCPTransport(FastMCPTransport):
            def __bool__(self) -> bool:
                return False

        client = FalseyFastMCPTransport(linear_server)
        assert not client
        toolset = Linear(client=client).get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.client.transport is client

    def test_no_auth_is_supported(self):
        transport = _http_transport(Linear(auth=None))
        assert transport.auth is None

    async def test_agent_calls_read_only_server_tool(self, linear_server: FastMCP):
        agent = Agent(TestModel(call_tools=['get_issue']), capabilities=[Linear(client=linear_server)])
        result = await agent.run('Read ENG-123')
        assert _tool_call_names(result.all_messages()) == {'get_issue'}
        assert 'Fix the build' in result.output

    async def test_agent_lists_issues(self, linear_server: FastMCP):
        agent = Agent(TestModel(call_tools=['list_issues']), capabilities=[Linear(client=linear_server)])
        result = await agent.run('List issues')
        assert _tool_call_names(result.all_messages()) == {'list_issues'}
        assert 'ENG-123' in result.output

    async def test_server_failure_reaches_model_as_retry(self, linear_server: FastMCP):
        def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            retry = next(
                (
                    part
                    for message in messages
                    for part in message.parts
                    if isinstance(part, RetryPromptPart) and part.tool_name == 'fail_issue'
                ),
                None,
            )
            if retry is None:
                return ModelResponse(parts=[ToolCallPart(tool_name='fail_issue', args={})])
            assert 'provider exploded' in str(retry.content)
            return ModelResponse(parts=[TextPart('recovered')])

        agent = Agent(FunctionModel(model_fn), capabilities=[Linear(client=linear_server)])
        result = await agent.run('Read the issue')
        assert result.output == 'recovered'

    async def test_allowed_tools_are_exact_names(self, linear_server: FastMCP):
        agent = Agent(
            TestModel(),
            capabilities=[Linear(client=linear_server, allowed_tools=['get_issue'])],
        )
        result = await agent.run('Read issues')
        assert _tool_call_names(result.all_messages()) == {'get_issue'}

    async def test_single_allowed_tool_string_is_one_exact_name(self, linear_server: FastMCP):
        agent = Agent(TestModel(), capabilities=[Linear(client=linear_server, allowed_tools='get_issue')])
        result = await agent.run('Read issues')
        assert _tool_call_names(result.all_messages()) == {'get_issue'}

    async def test_empty_allowed_tools_exposes_no_tools(self, linear_server: FastMCP):
        model = TestModel()
        agent = Agent(model, capabilities=[Linear(client=linear_server, allowed_tools=[])])
        await agent.run('Do not use tools')
        params = model.last_model_request_parameters
        assert params is not None
        assert params.function_tools == []

    def test_allowed_tools_uses_public_core_wrapper(self, linear_server: FastMCP):
        toolset = Linear(client=linear_server, allowed_tools=['get_issue']).get_toolset()
        assert isinstance(toolset, FilteredToolset)
        assert isinstance(toolset.wrapped, MCPToolset)

    @pytest.mark.parametrize('prebuild_toolset', [False, True])
    async def test_prebuilt_connection_runs_through_agent(self, linear_server: FastMCP, prebuild_toolset: bool):
        client = Client(linear_server)
        connection = MCPToolset(client, id='tenant-linear') if prebuild_toolset else client
        capability = Linear(client=connection, allowed_tools=['get_issue'])
        toolset = capability.get_toolset()
        assert isinstance(toolset, FilteredToolset)
        assert isinstance(toolset.wrapped, MCPToolset)
        assert toolset.wrapped.client is client
        assert isinstance(client.transport, FastMCPTransport)

        result = await Agent(TestModel(), capabilities=[capability]).run('Read an issue')
        assert _tool_call_names(result.all_messages()) == {'get_issue'}

    @pytest.mark.parametrize('prebuild_toolset', [False, True])
    def test_auth_with_prebuilt_connection_is_rejected(self, linear_server: FastMCP, prebuild_toolset: bool):
        client = Client(linear_server)
        connection = MCPToolset(client) if prebuild_toolset else client
        with pytest.raises(UserError, match='configure auth on'):
            Linear(client=connection, auth='token').get_toolset()

    def test_non_url_client_without_auth_is_supported(self, tmp_path: Path):
        script = tmp_path / 'server.py'
        script.write_text('', encoding='utf-8')
        toolset = Linear(client=script).get_toolset()
        assert isinstance(toolset, MCPToolset)

    async def test_injected_connection_instructions_cover_possible_mutations(self, linear_server: FastMCP):
        result = await Agent(TestModel(call_tools=['get_issue']), capabilities=[Linear(client=linear_server)]).run(
            'Read ENG-123'
        )
        instructions = _request_instructions(result.all_messages())
        assert 'Linear' in instructions
        assert 'identifiers' in instructions
        assert 'Before changing Linear data' in instructions

    async def test_read_write_instructions_cover_mutations(
        self, linear_server: FastMCP, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr('pydantic_ai_harness.linear._capability.LINEAR_MCP_URL', linear_server)
        result = await Agent(TestModel(), capabilities=[Linear(read_only=False, allowed_tools=['create_issue'])]).run(
            'Create an issue'
        )
        assert 'Before changing Linear data' in _request_instructions(result.all_messages())
        assert 'search for an existing match when a read tool is available' in _request_instructions(
            result.all_messages()
        )

    async def test_approval_wrapper_defers_and_resumes_mutation(self, linear_server: FastMCP):
        responses = iter(
            [
                ModelResponse(
                    parts=[ToolCallPart('create_issue', {'title': 'Fix CI'}, tool_call_id='create-approval')]
                ),
                ModelResponse(parts=[TextPart('created')]),
            ]
        )
        agent = Agent(
            FunctionModel(lambda _messages, _info: next(responses)),
            toolsets=[Linear(client=linear_server, read_only=False).get_toolset().approval_required()],
            output_type=[str, DeferredToolRequests],
        )

        deferred = await agent.run('Create an issue')
        assert isinstance(deferred.output, DeferredToolRequests)
        assert [approval.tool_name for approval in deferred.output.approvals] == ['create_issue']

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'create-approval': True}),
        )
        assert resumed.output == 'created'
        assert any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'create_issue'
            for message in resumed.all_messages()
            for part in message.parts
        )

    async def test_instructions_can_be_disabled(self, linear_server: FastMCP):
        result = await Agent(
            TestModel(call_tools=['get_issue']),
            capabilities=[Linear(client=linear_server, include_instructions=False)],
        ).run('Read issues')
        assert 'Linear tools' not in _request_instructions(result.all_messages())
