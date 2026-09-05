"""Behavioral tests for `AWS` through its public capability boundary."""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Callable
from pathlib import Path

import pytest
from fastmcp.client.transports import FastMCPTransport, StreamableHttpTransport
from mcp.server.fastmcp.server import FastMCP
from mcp.types import ImageContent, ToolAnnotations
from pydantic_ai import Agent, DeferredToolResults, ToolDenied
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    BinaryContent,
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
from pydantic_ai.tools import DeferredToolRequests

from pydantic_ai_harness.aws import AWS

pytestmark = pytest.mark.anyio


def _visible_tools(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    names = ','.join(tool.name for tool in info.function_tools)
    return ModelResponse(parts=[TextPart(names)])


def _call_once(tool_name: str, args: dict[str, object]) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id='call-1')])

    return model


def _request_instructions(messages: list[ModelMessage]) -> str:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    return first.instructions or ''


def _single_tool_return(messages: list[ModelMessage]) -> ToolReturnPart:
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    assert len(returns) == 1
    return returns[0]


class TestAWS:
    def test_serialization_name(self):
        assert AWS.get_serialization_name() == 'AWS'

    def test_agent_spec_excludes_runtime_client(self):
        schema = json.dumps(AgentSpec.model_json_schema_with_capabilities([AWS]), sort_keys=True)
        assert '"managed_transport"' not in schema
        assert '"authentication"' not in schema
        assert 'approval_required' in schema

    def test_from_spec_builds_public_knowledge_connection(self):
        capability = AWS.from_spec('123456789012', 'us-west-2', id='aws-docs')
        assert capability.id == 'aws-docs'
        assert capability.authentication == 'unauthenticated'

    def test_agent_loads_public_knowledge_spec(self, tmp_path: Path):
        spec = tmp_path / 'agent.yaml'
        spec.write_text(
            """\
capabilities:
  - AWS:
      account_id: '123456789012'
      region: us-west-2
""",
            encoding='utf-8',
        )
        agent = Agent.from_file(spec, custom_capability_types=[AWS], model=TestModel())
        assert isinstance(agent, Agent)

    def test_from_spec_preserves_output_limits(self):
        capability = AWS.from_spec('123456789012', 'us-west-2', max_output_bytes=1000, max_output_lines=25)
        assert capability.max_output_bytes == 1000
        assert capability.max_output_lines == 25

    def test_output_limit_defaults(self):
        capability = AWS('123456789012', 'us-west-2')
        assert capability.max_output_bytes == 50 * 1024
        assert capability.max_output_lines == 2000

    @pytest.mark.parametrize('account_id', ['', '1234', '12345678901x', 123456789012])
    def test_rejects_invalid_account_id(self, account_id: object):
        with pytest.raises(UserError, match='12-digit AWS account ID'):
            AWS(account_id, 'us-west-2')  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('region', ['', 'US West', 'us_west_2', 1])
    def test_rejects_invalid_target_region(self, region: object):
        with pytest.raises(UserError, match='AWS Region identifier'):
            AWS('123456789012', region)  # pyright: ignore[reportArgumentType]

    def test_rejects_unsupported_endpoint_region(self):
        with pytest.raises(UserError, match='`us-east-1`.*`eu-central-1`'):
            AWS(
                '123456789012',
                'ap-south-1',
                endpoint_region='ap-south-1',  # pyright: ignore[reportArgumentType]
            )

    def test_rejects_invalid_access_at_runtime(self):
        with pytest.raises(UserError, match='`access` must be'):
            AWS('123456789012', 'us-west-2', access='write')  # pyright: ignore[reportArgumentType]

    def test_rejects_invalid_authentication_at_runtime(self):
        with pytest.raises(UserError, match='`authentication` must be'):
            AWS('123456789012', 'us-west-2', authentication='token')  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('value', [0, -1, True])
    def test_rejects_non_positive_output_bytes(self, value: int):
        with pytest.raises(UserError, match='`max_output_bytes` must be a positive integer'):
            AWS('123456789012', 'us-west-2', max_output_bytes=value)

    def test_rejects_output_byte_limit_smaller_than_marker(self):
        with pytest.raises(UserError, match='`max_output_bytes` must be at least'):
            AWS('123456789012', 'us-west-2', max_output_bytes=1)

    @pytest.mark.parametrize('value', [0, -1, True])
    def test_rejects_non_positive_output_lines(self, value: int):
        with pytest.raises(UserError, match='`max_output_lines` must be a positive integer'):
            AWS('123456789012', 'us-west-2', max_output_lines=value)

    @pytest.mark.parametrize('authentication', ['oauth', 'sigv4'])
    def test_authenticated_modes_require_caller_owned_transport(self, authentication: str):
        with pytest.raises(UserError, match=f'`authentication="{authentication}"` requires'):
            AWS(
                '123456789012',
                'us-west-2',
                authentication=authentication,  # pyright: ignore[reportArgumentType]
            )

    def test_unauthenticated_mode_rejects_transport(self, aws_server: tuple[FastMCP, list[str]]):
        server, _ = aws_server
        with pytest.raises(UserError, match='`managed_transport` requires'):
            AWS('123456789012', 'us-west-2', managed_transport=server)  # pyright: ignore[reportArgumentType]

    def test_authenticated_mode_rejects_unconfigured_transport_input(self):
        with pytest.raises(UserError, match='FastMCP `ClientTransport`'):
            AWS(
                '123456789012',
                'us-west-2',
                authentication='oauth',
                managed_transport='https://aws-mcp.us-east-1.api.aws/mcp',  # pyright: ignore[reportArgumentType]
            )

    def test_scope_derives_stable_id(self):
        capability = AWS('123456789012', 'us-west-2')
        assert capability.id == 'aws-123456789012-us-west-2-us-east-1'

    def test_two_scope_ids_stay_distinct(self):
        first = AWS('123456789012', 'us-west-2')
        second = AWS('210987654321', 'eu-west-1')
        assert (first.id, second.id) == (
            'aws-123456789012-us-west-2-us-east-1',
            'aws-210987654321-eu-west-1-us-east-1',
        )

    async def test_prefixed_scopes_route_to_separate_transports(self, aws_server: tuple[FastMCP, list[str]]):
        first_server, first_calls = aws_server
        second_server = FastMCP('aws-managed-second')
        second_calls: list[str] = []

        @second_server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___list_regions() -> list[str]:
            second_calls.append('list')
            return ['eu-west-1']

        def call_both_scopes(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if not returns:
                descriptions = {tool.name: tool.description for tool in info.function_tools}
                first_description = descriptions['first_aws___list_regions']
                second_description = descriptions['second_aws___list_regions']
                assert first_description is not None
                assert second_description is not None
                assert first_description.startswith('AWS scope: account 123456789012, target Region us-west-2.')
                assert second_description.startswith('AWS scope: account 210987654321, target Region eu-west-1.')
                return ModelResponse(parts=[ToolCallPart('first_aws___list_regions', {}, tool_call_id='call-1')])
            if len(returns) == 1:
                return ModelResponse(parts=[ToolCallPart('second_aws___list_regions', {}, tool_call_id='call-2')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(call_both_scopes),
            capabilities=[
                PrefixTools(
                    AWS(
                        '123456789012',
                        'us-west-2',
                        authentication='sigv4',
                        managed_transport=FastMCPTransport(first_server),
                    ),
                    prefix='first',
                ),
                PrefixTools(
                    AWS(
                        '210987654321',
                        'eu-west-1',
                        authentication='sigv4',
                        managed_transport=FastMCPTransport(second_server),
                    ),
                    prefix='second',
                ),
            ],
        )
        result = await agent.run('list both scopes')

        assert result.output == 'done'
        assert first_calls == ['list']
        assert second_calls == ['list']

    def test_direct_unauthenticated_transport(self):
        toolset = AWS('123456789012', 'us-west-2', endpoint_region='eu-central-1').get_toolset()
        assert isinstance(toolset, MCPToolset)
        transport = toolset.client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'https://aws-mcp.eu-central-1.api.aws/mcp'
        assert transport.auth is None

    def test_oauth_transport(self):
        with pytest.warns(UserWarning, match='in-memory token storage'):
            transport = StreamableHttpTransport('https://aws-mcp.us-east-1.api.aws/mcp', auth='oauth')
        toolset = AWS('123456789012', 'us-west-2', authentication='oauth', managed_transport=transport).get_toolset()
        assert isinstance(toolset, MCPToolset)
        transport = toolset.client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'https://aws-mcp.us-east-1.api.aws/mcp'
        assert transport.auth is not None

    def test_preserves_caller_owned_transport(self, aws_server: tuple[FastMCP, list[str]]):
        server, _ = aws_server
        transport = FastMCPTransport(server)
        toolset = AWS('123456789012', 'us-west-2', authentication='sigv4', managed_transport=transport).get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.client.transport is transport
        assert transport.server is server

    def test_transport_hidden_from_repr(self, aws_server: tuple[FastMCP, list[str]]):
        server, _ = aws_server
        capability = AWS(
            '123456789012', 'us-west-2', authentication='oauth', managed_transport=FastMCPTransport(server)
        )
        assert 'aws-managed-fake' not in repr(capability)

    async def test_default_access_exposes_only_read_only_tools(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        capability = AWS(
            '123456789012', 'us-west-2', authentication='oauth', managed_transport=FastMCPTransport(server)
        )
        agent = Agent(FunctionModel(_visible_tools), capabilities=[capability])
        result = await agent.run('inspect AWS')
        assert result.output == 'aws___list_regions,aws___failing_read'
        assert calls == []

    async def test_output_limit_preserves_small_structured_result(self):
        server = FastMCP('aws-managed-small-structured')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___list_regions() -> dict[str, list[str]]:
            return {'regions': ['us-east-1', 'us-west-2']}

        agent = Agent(
            FunctionModel(_call_once('aws___list_regions', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=1000,
                    max_output_lines=20,
                )
            ],
        )
        result = await agent.run('list regions')

        assert _single_tool_return(result.all_messages()).content == {'regions': ['us-east-1', 'us-west-2']}

    async def test_output_limit_bounds_large_structured_result(self):
        server = FastMCP('aws-managed-large-structured')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___list_regions() -> dict[str, list[str]]:
            return {'regions': ['x' * 100]}

        agent = Agent(
            FunctionModel(_call_once('aws___list_regions', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=50,
                    max_output_lines=2,
                )
            ],
        )
        result = await agent.run('list regions')
        content = _single_tool_return(result.all_messages()).content

        assert isinstance(content, str)
        assert content.endswith('[... AWS MCP output truncated ...]')
        assert len(content.encode()) <= 50

    async def test_output_limit_bounds_multiline_result(self):
        server = FastMCP('aws-managed-multiline')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___list_regions() -> str:
            return 'first\nsecond\nthird'

        agent = Agent(
            FunctionModel(_call_once('aws___list_regions', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=100,
                    max_output_lines=2,
                )
            ],
        )
        result = await agent.run('list regions')
        content = _single_tool_return(result.all_messages()).content

        assert isinstance(content, str)
        assert content == 'first\n[... AWS MCP output truncated ...]'
        assert len(content.encode()) <= 100
        assert len(content.splitlines()) <= 2

    async def test_output_limit_clips_multibyte_text_without_partial_codepoint(self):
        server = FastMCP('aws-managed-multibyte')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___list_regions() -> str:
            return 'é' * 50

        agent = Agent(
            FunctionModel(_call_once('aws___list_regions', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=40,
                    max_output_lines=2,
                )
            ],
        )
        result = await agent.run('list regions')
        content = _single_tool_return(result.all_messages()).content

        assert isinstance(content, str)
        assert content == 'éé\n[... AWS MCP output truncated ...]'
        assert len(content.encode()) == 39

    @pytest.mark.parametrize(('max_output_bytes', 'max_output_lines'), [(34, 2), (100, 1)])
    async def test_output_limit_uses_marker_only_when_no_preview_fits(
        self, max_output_bytes: int, max_output_lines: int
    ):
        server = FastMCP('aws-managed-marker-only')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___list_regions() -> str:
            return 'x' * 200

        agent = Agent(
            FunctionModel(_call_once('aws___list_regions', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=max_output_bytes,
                    max_output_lines=max_output_lines,
                )
            ],
        )
        result = await agent.run('list regions')

        assert _single_tool_return(result.all_messages()).content == '[... AWS MCP output truncated ...]'

    async def test_output_limit_drops_oversized_binary_result(self):
        server = FastMCP('aws-managed-binary')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___download_diagram() -> ImageContent:
            return ImageContent(type='image', data=b64encode(b'x' * 100).decode(), mimeType='image/png')

        agent = Agent(
            FunctionModel(_call_once('aws___download_diagram', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=50,
                    max_output_lines=2,
                )
            ],
        )
        result = await agent.run('download diagram')

        assert _single_tool_return(result.all_messages()).content == '[... AWS MCP output truncated ...]'

    async def test_output_limit_counts_binary_history_serialization(self):
        server = FastMCP('aws-managed-serialized-binary')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___download_diagram() -> ImageContent:
            return ImageContent(type='image', data=b64encode(b'x' * 40).decode(), mimeType='image/png')

        agent = Agent(
            FunctionModel(_call_once('aws___download_diagram', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=100,
                    max_output_lines=20,
                )
            ],
        )
        result = await agent.run('download diagram')

        assert _single_tool_return(result.all_messages()).content == '[... AWS MCP output truncated ...]'

    async def test_output_limit_counts_binary_metadata(self):
        server = FastMCP('aws-managed-binary-metadata')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___download_diagram() -> ImageContent:
            return ImageContent(type='image', data=b64encode(b'x').decode(), mimeType=f'image/{"x" * 200}')

        agent = Agent(
            FunctionModel(_call_once('aws___download_diagram', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=100,
                    max_output_lines=20,
                )
            ],
        )
        result = await agent.run('download diagram')

        assert _single_tool_return(result.all_messages()).content == '[... AWS MCP output truncated ...]'

    async def test_output_limit_preserves_small_binary_result(self):
        server = FastMCP('aws-managed-small-binary')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___download_diagram() -> ImageContent:
            return ImageContent(type='image', data=b64encode(b'png').decode(), mimeType='image/png')

        agent = Agent(
            FunctionModel(_call_once('aws___download_diagram', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=1000,
                    max_output_lines=20,
                )
            ],
        )
        result = await agent.run('download diagram')

        assert isinstance(_single_tool_return(result.all_messages()).content, BinaryContent)

    async def test_default_access_rejects_hidden_write_call(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        turns = 0

        def attempt_hidden_write(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal turns
            turns += 1
            if turns == 1:
                return ModelResponse(parts=[ToolCallPart('aws___run_script', {'code': 'create_bucket()'})])
            return ModelResponse(parts=[TextPart('blocked')])

        agent = Agent(
            FunctionModel(attempt_hidden_write),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        result = await agent.run('create a bucket')

        assert result.output == 'blocked'
        assert calls == []
        retries = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
        ]
        assert len(retries) == 1
        assert 'Unknown tool name' in str(retries[0].content)

    async def test_empty_managed_catalog_fails_explicitly(self):
        server = FastMCP('aws-managed-empty')
        agent = Agent(
            TestModel(),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )

        with pytest.raises(UserError, match='returned no tools.*throttled initialization'):
            await agent.run('inspect AWS')

    async def test_read_only_fails_if_server_has_no_safe_tools(self):
        server = FastMCP('aws-managed-no-safe-tools')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
        def aws___run_script() -> str:  # pragma: no cover - fail-closed filtering must make this unreachable
            return 'created'  # pragma: no cover

        agent = Agent(
            TestModel(),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )

        with pytest.raises(UserError, match='no tools explicitly marked read-only'):
            await agent.run('inspect AWS')

    async def test_approval_required_defers_non_read_tool(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        agent = Agent(
            FunctionModel(_call_once('aws___run_script', {'code': 'create_bucket()'})),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    access='approval_required',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        result = await agent.run('create a bucket')
        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['aws___run_script']
        assert calls == []

    async def test_approved_tool_executes_once_on_resume(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        code = 'create_bucket(' + 'x' * 100 + ')'
        agent = Agent(
            FunctionModel(_call_once('aws___run_script', {'code': code})),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    access='approval_required',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=50,
                    max_output_lines=2,
                )
            ],
        )
        deferred = await agent.run('create a bucket')
        assert isinstance(deferred.output, DeferredToolRequests)

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': True}),
        )

        assert resumed.output == 'done'
        assert calls == [f'run:{code}']
        content = _single_tool_return(resumed.all_messages()).content
        assert isinstance(content, str)
        assert content.endswith('[... AWS MCP output truncated ...]')
        assert len(content.encode()) <= 50

    async def test_denied_tool_does_not_execute_on_resume(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        agent = Agent(
            FunctionModel(_call_once('aws___run_script', {'code': 'create_bucket()'})),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    access='approval_required',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        deferred = await agent.run('create a bucket')
        assert isinstance(deferred.output, DeferredToolRequests)

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': ToolDenied('not approved')}),
        )

        assert resumed.output == 'done'
        assert calls == []

    async def test_failed_mutation_requires_fresh_approval_before_retry(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server

        def retry_mutation(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            retry_seen = any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts)
            call_id = 'call-2' if retry_seen else 'call-1'
            return ModelResponse(parts=[ToolCallPart('aws___failing_write', {}, tool_call_id=call_id)])

        agent = Agent(
            FunctionModel(retry_mutation),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    access='approval_required',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        deferred = await agent.run('change AWS')
        assert isinstance(deferred.output, DeferredToolRequests)

        retried = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': True}),
        )

        assert isinstance(retried.output, DeferredToolRequests)
        assert [call.tool_call_id for call in retried.output.approvals] == ['call-2']
        assert calls == ['failing-write']

    async def test_approval_required_allows_read_tool(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        agent = Agent(
            FunctionModel(_call_once('aws___list_regions', {})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    access='approval_required',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        result = await agent.run('list regions')
        assert result.output == 'done'
        assert calls == ['list']

    async def test_approval_required_treats_missing_annotation_as_non_read(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        agent = Agent(
            FunctionModel(_call_once('aws___future_tool', {})),
            output_type=[str, DeferredToolRequests],
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    access='approval_required',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        result = await agent.run('use the future tool')
        assert isinstance(result.output, DeferredToolRequests)
        assert calls == []

    async def test_unrestricted_executes_non_read_tool(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        agent = Agent(
            FunctionModel(_call_once('aws___run_script', {'code': 'create_bucket()'})),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    access='unrestricted',
                    authentication='sigv4',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        result = await agent.run('create a bucket')
        assert result.output == 'done'
        assert calls == ['run:create_bucket()']

    async def test_managed_server_failure_surfaces_through_retry(self, aws_server: tuple[FastMCP, list[str]]):
        server, calls = aws_server
        turns = 0

        def call_failing_tool(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal turns
            turns += 1
            if turns == 1:
                return ModelResponse(parts=[ToolCallPart('aws___failing_read', {}, tool_call_id='call-1')])
            return ModelResponse(parts=[TextPart('failure observed')])

        agent = Agent(
            FunctionModel(call_failing_tool),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                )
            ],
        )
        result = await agent.run('read AWS')

        assert result.output == 'failure observed'
        assert calls == ['failing-read']
        retries = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
        ]
        assert len(retries) == 1
        assert 'managed AWS boundary failed' in str(retries[0].content)

    async def test_output_limit_bounds_managed_server_failure(self):
        server = FastMCP('aws-managed-large-failure')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
        def aws___failing_read() -> str:
            raise RuntimeError('x' * 5000)

        turns = 0

        def call_failing_tool(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal turns
            turns += 1
            if turns == 1:
                return ModelResponse(parts=[ToolCallPart('aws___failing_read', {}, tool_call_id='call-1')])
            return ModelResponse(parts=[TextPart('failure observed')])

        agent = Agent(
            FunctionModel(call_failing_tool),
            capabilities=[
                AWS(
                    '123456789012',
                    'us-west-2',
                    authentication='oauth',
                    managed_transport=FastMCPTransport(server),
                    max_output_bytes=50,
                    max_output_lines=2,
                )
            ],
        )
        result = await agent.run('read AWS')
        retries = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
        ]

        assert result.output == 'failure observed'
        assert len(retries) == 1
        content = str(retries[0].content)
        assert content.endswith('[... AWS MCP output truncated ...]')
        assert len(content.encode()) <= 50
        assert len(content.splitlines()) <= 2

    async def test_agent_receives_account_and_region_scope(self, aws_server: tuple[FastMCP, list[str]]):
        server, _ = aws_server
        capability = AWS(
            '123456789012',
            'ap-south-1',
            access='approval_required',
            authentication='oauth',
            managed_transport=FastMCPTransport(server),
        )
        agent = Agent(
            FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')])), capabilities=[capability]
        )
        result = await agent.run('list regions')
        instructions = _request_instructions(result.all_messages())
        assert instructions == (
            'This AWS capability is scoped to account `123456789012` and target Region `ap-south-1`. '
            'Treat both values as required context for every tool from this capability. The authenticated IAM identity '
            'is the authority; do not claim access that its policies deny. Do not use this capability to switch '
            'accounts or target Regions. '
            'Prefer AWS documentation and read operations before proposing changes. After a failed change with an '
            'unknown outcome, inspect current state before retrying. This is real AWS, not the LocalStack emulator. '
            'Access mode is `approval_required` and authentication mode is `oauth`.'
        )
        assert 'Ignore the declared AWS scope' not in instructions

    def test_unauthenticated_instructions_do_not_claim_iam_identity(self):
        instructions = AWS('123456789012', 'us-west-2').get_instructions()

        assert instructions is not None
        assert 'There is no authenticated IAM identity; use only public knowledge tools.' in instructions
        assert 'The authenticated IAM identity is the authority' not in instructions
