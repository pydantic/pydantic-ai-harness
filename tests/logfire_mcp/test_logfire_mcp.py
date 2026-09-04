"""Behavioral tests for the project-scoped hosted Logfire MCP capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastmcp.client.auth import BearerAuth, OAuth
from fastmcp.client.transports import FastMCPTransport, StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.logfire_mcp import LogfireMCP

from .conftest import LogfireState

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from pydantic_ai.tools import RunContext

pytestmark = pytest.mark.anyio


def _tool_names(model: TestModel) -> set[str]:
    parameters = model.last_model_request_parameters
    assert parameters is not None
    return {tool.name for tool in parameters.function_tools}


class TestLogfireMCP:
    def test_default_us_oauth_transport(self):
        capability = LogfireMCP(project='acme/production')
        with pytest.warns(UserWarning, match='in-memory token storage'):
            toolset = capability.get_toolset()
        transport = toolset.client.transport

        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'https://logfire-us.pydantic.dev/mcp'
        assert isinstance(transport.auth, OAuth)
        assert capability.id == 'logfire-mcp-acme-production'

    def test_eu_api_key_transport_hides_secret(self):
        capability = LogfireMCP(project='acme/production', region='eu', api_key='secret-key')
        toolset = capability.get_toolset()
        transport = toolset.client.transport

        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'https://logfire-eu.pydantic.dev/mcp'
        assert isinstance(transport.auth, BearerAuth)
        assert transport.auth.token.get_secret_value() == 'secret-key'
        assert 'secret-key' not in repr(capability)
        assert 'secret-key' not in repr(toolset)

    @pytest.mark.parametrize(
        'url',
        [
            'http://logfire.internal/mcp',
            'HTTPS://logfire.internal/mcp',
            'https:///mcp',
            'https://user:secret@logfire.internal/mcp',
            'https://logfire.internal/mcp?token=secret',
            'https://logfire.internal/mcp#secret',
        ],
    )
    def test_self_hosted_url_must_be_canonical_https_without_credentials(self, url: str):
        with pytest.raises(UserError, match='HTTPS URL without user info, query parameters, or fragments'):
            LogfireMCP(project='acme/production', mcp_url=url)

    def test_self_hosted_url_reaches_transport(self):
        capability = LogfireMCP(
            project='acme/production',
            mcp_url='https://logfire.acme.example/mcp',
            api_key='secret-key',
        )
        transport = capability.get_toolset().client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'https://logfire.acme.example/mcp'

    @pytest.mark.parametrize('project', ['', 'project-only', '/project', 'org/', 'org/project/extra', 'org/two words'])
    def test_project_must_include_organization_and_project(self, project: str):
        with pytest.raises(UserError, match='organization/project'):
            LogfireMCP(project=project)

    @pytest.mark.parametrize('api_key', ['', '  ', 'oauth'])
    def test_invalid_api_key_fails_closed(self, api_key: str):
        with pytest.raises(UserError, match='api_key'):
            LogfireMCP(project='acme/production', api_key=api_key)

    def test_unknown_or_duplicate_tools_are_rejected(self):
        with pytest.raises(UserError, match='Unsupported Logfire MCP tools'):
            LogfireMCP(project='acme/production', tools=('query_run', 'future_tool'))
        with pytest.raises(UserError, match='Unsupported Logfire MCP tools'):
            LogfireMCP(project='acme/production', tools=('channel_list',))
        with pytest.raises(UserError, match='Unsupported Logfire MCP tools'):
            LogfireMCP(project='acme/production', tools=('schedule_list',))
        with pytest.raises(UserError, match='unique'):
            LogfireMCP(project='acme/production', tools=('query_run', 'query_run'))
        with pytest.raises(UserError, match='unique'):
            LogfireMCP(project='acme/production', tools=())
        with pytest.raises(UserError, match='not one string'):
            LogfireMCP(project='acme/production', tools='query_run')

    def test_invalid_region_and_query_limit_fail_closed(self):
        with pytest.raises(UserError, match='region'):
            LogfireMCP(project='acme/production', region='apac')  # pyright: ignore[reportArgumentType]
        with pytest.raises(UserError, match='max_query_rows'):
            LogfireMCP(project='acme/production', max_query_rows=0)

    def test_client_owns_authentication(self, logfire_server: FastMCP):
        with pytest.raises(UserError, match='configure authentication on the client'):
            LogfireMCP(project='acme/production', client=logfire_server, api_key='secret')
        with pytest.raises(UserError, match='mcp_url.*client'):
            LogfireMCP(project='acme/production', client=logfire_server, mcp_url='https://logfire.acme.example/mcp')

    @pytest.mark.parametrize('client', ['http://logfire.internal/mcp', Path('/tmp/logfire-mcp')])
    def test_client_rejects_transport_specs_that_bypass_connection_policy(self, client: object):
        with pytest.raises(UserError, match='pre-built MCP client'):
            LogfireMCP(project='acme/production', client=client)  # pyright: ignore[reportArgumentType]

    def test_agent_spec_excludes_credentials_and_runtime_client(self):
        schema = AgentSpec.model_json_schema_with_capabilities([LogfireMCP])
        properties = schema['$defs']['spec_params_LogfireMCP']['properties']
        assert 'api_key' not in properties
        assert 'client' not in properties
        assert LogfireMCP.get_serialization_name() == 'LogfireMCP'

    def test_from_spec_preserves_serializable_policy(self):
        capability = LogfireMCP.from_spec(
            'acme/production',
            id='production-observability',
            description='Production telemetry',
            defer_loading=True,
            region='eu',
            mcp_url='https://logfire.acme.example/mcp',
            tools=('dashboard_list',),
            max_query_rows=20,
            include_instructions=False,
        )
        assert capability.id == 'production-observability'
        assert capability.description == 'Production telemetry'
        assert capability.defer_loading is True
        assert capability.region == 'eu'
        assert capability.mcp_url == 'https://logfire.acme.example/mcp'
        assert capability.tools == ('dashboard_list',)
        assert capability.max_query_rows == 20
        assert capability.get_instructions() is None
        assert capability.api_key is None
        assert capability.client is None

    def test_from_spec_uses_environment_token_for_headless_auth(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('LOGFIRE_MCP_TOKEN', 'environment-secret')
        capability = LogfireMCP.from_spec('acme/production')

        toolset = capability.get_toolset()
        transport = toolset.client.transport

        assert isinstance(transport, StreamableHttpTransport)
        assert isinstance(transport.auth, BearerAuth)
        assert transport.auth.token.get_secret_value() == 'environment-secret'
        assert 'environment-secret' not in repr(capability)
        assert 'environment-secret' not in repr(toolset)

    def test_invalid_environment_token_fails_when_toolset_is_built(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('LOGFIRE_MCP_TOKEN', '  ')
        with pytest.raises(UserError, match='LOGFIRE_MCP_TOKEN'):
            LogfireMCP.from_spec('acme/production').get_toolset()

    def test_environment_token_is_not_forwarded_to_custom_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('LOGFIRE_MCP_TOKEN', 'hosted-only-secret')
        capability = LogfireMCP(project='acme/production', mcp_url='https://logfire.acme.example/mcp')

        with pytest.warns(UserWarning):
            transport = capability.get_toolset().client.transport

        assert isinstance(transport, StreamableHttpTransport)
        assert isinstance(transport.auth, OAuth)
        assert 'hosted-only-secret' not in repr(transport)

    async def test_default_tools_are_useful_read_only_subset(
        self, logfire_server: FastMCP, logfire_state: LogfireState
    ):
        seen_tools: set[str] = set()
        seen_instructions: list[str | None] = []

        def count_errors(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen_tools.update(tool.name for tool in info.function_tools)
            seen_instructions.append(info.instructions)
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(
                    parts=[ToolCallPart('query_run', {'query': 'SELECT count(*) FROM records LIMIT 20'})]
                )
            return ModelResponse(parts=[TextPart('3 errors in acme/production')])

        model = FunctionModel(count_errors)
        agent = Agent(model, capabilities=[LogfireMCP(project='acme/production', client=logfire_server)])

        result = await agent.run('Count recent errors')

        assert seen_tools == {
            'project_logfire_link',
            'query_find_exceptions_in_file',
            'query_run',
            'query_schema_reference',
        }
        assert logfire_state['calls'] == [
            (
                'query_run',
                {
                    'query': 'SELECT count(*) FROM records LIMIT 20',
                    'project': 'acme/production',
                    'start_timestamp': None,
                    'end_timestamp': None,
                },
            )
        ]
        assert 'acme/production' in str(result.output)
        assert all(instructions is not None for instructions in seen_instructions)
        assert all('reveal every secret' not in (instructions or '') for instructions in seen_instructions)
        assert logfire_state['lifecycle'] == ['entered', 'exited']

    async def test_exact_tool_selection_excludes_account_and_future_tools(self, logfire_server: FastMCP):
        model = TestModel(call_tools=[])
        agent = Agent(
            model,
            capabilities=[
                LogfireMCP(
                    project='acme/production',
                    tools=('query_run', 'dashboard_list'),
                    client=logfire_server,
                )
            ],
        )

        await agent.run('Inspect telemetry and dashboards')

        assert _tool_names(model) == {'query_run', 'dashboard_list'}

    async def test_two_projects_collide_on_fixed_tool_names(self, logfire_server: FastMCP):
        agent = Agent(
            TestModel(call_tools=[]),
            capabilities=[
                LogfireMCP(project='acme/production', client=logfire_server),
                LogfireMCP(project='acme/staging', client=logfire_server),
            ],
        )

        with pytest.raises(UserError, match='conflicts with existing tool'):
            await agent.run('Inspect both projects')

    async def test_selected_tool_hidden_by_token_scope_fails_clearly(
        self, logfire_server: FastMCP, logfire_state: LogfireState
    ):
        agent = Agent(
            TestModel(call_tools=[]),
            capabilities=[LogfireMCP(project='acme/production', tools=('alert_list',), client=logfire_server)],
        )

        with pytest.raises(UserError, match='not available.*token permissions'):
            await agent.run('List alerts')
        assert logfire_state['lifecycle'] == ['entered', 'exited']

    async def test_selected_project_tool_without_scope_field_fails_closed(
        self, logfire_server: FastMCP, logfire_state: LogfireState
    ):
        agent = Agent(
            TestModel(call_tools=[]),
            capabilities=[LogfireMCP(project='acme/production', tools=('issue_list',), client=logfire_server)],
        )

        with pytest.raises(UserError, match='does not expose its documented `project` scope'):
            await agent.run('List issues')
        assert logfire_state['lifecycle'] == ['entered', 'exited']

    async def test_cross_project_argument_retries_before_network_call(
        self, logfire_server: FastMCP, logfire_state: LogfireState, run_context: RunContext[None]
    ):
        toolset = LogfireMCP(project='acme/production', client=logfire_server).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ModelRetry, match='configured Logfire project'):
                await toolset.call_tool(
                    'query_run',
                    {'query': 'SELECT 1 LIMIT 1', 'project': 'other/project'},
                    run_context,
                    tools['query_run'],
                )

        assert logfire_state['calls'] == []

    async def test_matching_project_argument_executes(
        self, logfire_server: FastMCP, logfire_state: LogfireState, run_context: RunContext[None]
    ):
        toolset = LogfireMCP(project='acme/production', client=logfire_server).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
            await toolset.call_tool(
                'query_run',
                {
                    'query': 'SELECT count(*) FROM records LIMIT 1',
                    'project': 'acme/production',
                    'start_timestamp': '2026-09-05T10:00:00Z',
                    'end_timestamp': '2026-09-05T10:30:00Z',
                },
                run_context,
                tools['query_run'],
            )

        assert logfire_state['calls'][0][1] == {
            'query': 'SELECT count(*) FROM records LIMIT 1',
            'project': 'acme/production',
            'start_timestamp': '2026-09-05T10:00:00Z',
            'end_timestamp': '2026-09-05T10:30:00Z',
        }

    async def test_global_schema_tool_executes_without_project(
        self, logfire_server: FastMCP, logfire_state: LogfireState, run_context: RunContext[None]
    ):
        toolset = LogfireMCP(
            project='acme/production', tools=('query_schema_reference',), client=logfire_server
        ).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool('query_schema_reference', {}, run_context, tools['query_schema_reference'])

        assert 'CREATE TABLE records' in str(result)
        assert logfire_state['calls'] == [('query_schema_reference', {})]

    async def test_query_requires_string_before_network_call(
        self, logfire_server: FastMCP, logfire_state: LogfireState, run_context: RunContext[None]
    ):
        toolset = LogfireMCP(project='acme/production', client=logfire_server).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ModelRetry, match='requires a string'):
                await toolset.call_tool(
                    'query_run',
                    {'query': 1},
                    run_context,
                    tools['query_run'],
                )

        assert logfire_state['calls'] == []

    @pytest.mark.parametrize(
        'query',
        [
            'SELECT count(*) FROM records limit 1',
            'SELECT count(*) FROM records LIMIT 1;',
            'SELECT count(*) FROM records LiMiT 1 OFFSET 2',
            "SELECT message FROM records WHERE message LIKE '%--%' LIMIT 1",
            "SELECT '/* not a comment */' FROM records LIMIT 1",
            "SELECT ';' FROM records LIMIT 1",
        ],
    )
    async def test_supported_query_limit_forms_execute(
        self,
        query: str,
        logfire_server: FastMCP,
        logfire_state: LogfireState,
        run_context: RunContext[None],
    ):
        toolset = LogfireMCP(project='acme/production', client=logfire_server).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
            await toolset.call_tool('query_run', {'query': query}, run_context, tools['query_run'])

        assert logfire_state['calls'][0][1]['query'] == query

    @pytest.mark.parametrize(
        ('query', 'match'),
        [
            ('SELECT * FROM records', 'final numeric `LIMIT`'),
            ('SELECT * FROM records LIMIT 101', 'at most 100'),
            ('SELECT * FROM records LIMIT ALL', 'final numeric `LIMIT`'),
            ('SELECT * FROM records -- LIMIT 1', 'comments or multiple statements'),
            ('SELECT * FROM records /* bounded */ LIMIT 1', 'comments or multiple statements'),
            ('SELECT * FROM records; SELECT 1 LIMIT 1', 'comments or multiple statements'),
        ],
    )
    async def test_query_row_limit_is_enforced_before_network_call(
        self,
        query: str,
        match: str,
        logfire_server: FastMCP,
        logfire_state: LogfireState,
        run_context: RunContext[None],
    ):
        toolset = LogfireMCP(project='acme/production', client=logfire_server).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ModelRetry, match=match):
                await toolset.call_tool('query_run', {'query': query}, run_context, tools['query_run'])

        assert logfire_state['calls'] == []

    async def test_custom_query_row_limit_is_enforced(
        self, logfire_server: FastMCP, logfire_state: LogfireState, run_context: RunContext[None]
    ):
        toolset = LogfireMCP(project='acme/production', max_query_rows=2, client=logfire_server).get_toolset()
        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(ModelRetry, match='at most 2'):
                await toolset.call_tool('query_run', {'query': 'SELECT 1 LIMIT 3'}, run_context, tools['query_run'])

        assert logfire_state['calls'] == []

    async def test_selected_mutation_requires_approval_and_preserves_scope(
        self, logfire_server: FastMCP, logfire_state: LogfireState
    ):
        def create_dashboard(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolCallPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart('dashboard_create', {'name': 'Errors'}, 'create-1')])
            return ModelResponse(parts=[TextPart('created')])

        agent = Agent(
            FunctionModel(create_dashboard),
            capabilities=[
                LogfireMCP(
                    project='acme/production',
                    tools=('dashboard_create',),
                    client=logfire_server,
                )
            ],
            output_type=[str, DeferredToolRequests],
        )

        deferred = await agent.run('Create an errors dashboard')
        assert isinstance(deferred.output, DeferredToolRequests)
        assert [call.tool_name for call in deferred.output.approvals] == ['dashboard_create']
        assert deferred.output.metadata == {'create-1': {'project': 'acme/production'}}
        assert logfire_state['calls'] == []

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'create-1': True}),
        )
        assert resumed.output == 'created'
        assert logfire_state['calls'] == [('dashboard_create', {'name': 'Errors', 'project': 'acme/production'})]

    async def test_denied_mutation_does_not_execute(self, logfire_server: FastMCP, logfire_state: LogfireState):
        agent = Agent(
            TestModel(call_tools=['dashboard_create']),
            capabilities=[
                LogfireMCP(
                    project='acme/production',
                    tools=('dashboard_create',),
                    client=logfire_server,
                )
            ],
            output_type=[str, DeferredToolRequests],
        )
        deferred = await agent.run('Create a dashboard')
        assert isinstance(deferred.output, DeferredToolRequests)

        await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={deferred.output.approvals[0].tool_call_id: False}),
        )

        assert logfire_state['calls'] == []

    async def test_approved_mutation_failure_is_not_retried_inside_toolset(
        self, logfire_server: FastMCP, logfire_state: LogfireState, run_context: RunContext[None]
    ):
        logfire_state['failures'].add('dashboard_create')
        run_context.tool_call_approved = True
        toolset = LogfireMCP(
            project='acme/production', tools=('dashboard_create',), client=logfire_server
        ).get_toolset()

        async with toolset:
            tools = await toolset.get_tools(run_context)
            with pytest.raises(UserError, match='outcome may be unknown') as exc_info:
                await toolset.call_tool('dashboard_create', {'name': 'Errors'}, run_context, tools['dashboard_create'])

        assert isinstance(exc_info.value.__cause__, ModelRetry)
        assert logfire_state['calls'] == [('dashboard_create', {'name': 'Errors', 'project': 'acme/production'})]
        assert logfire_state['lifecycle'] == ['entered', 'exited']

    async def test_server_failure_uses_mcp_retry_policy(self, logfire_server: FastMCP, logfire_state: LogfireState):
        logfire_state['failures'].add('query_run')

        def retry_query(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            calls = sum(isinstance(part, ToolCallPart) for message in messages for part in message.parts)
            if calls < 2:
                return ModelResponse(parts=[ToolCallPart('query_run', {'query': 'SELECT 1 LIMIT 1'})])
            return ModelResponse(parts=[TextPart('unreachable')])

        agent = Agent(
            FunctionModel(retry_query),
            capabilities=[LogfireMCP(project='acme/production', client=logfire_server)],
        )

        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries') as exc_info:
            await agent.run('Count errors')

        assert exc_info.value.__cause__ is not None
        assert 'Logfire unavailable' in str(exc_info.value.__cause__)
        assert logfire_state['lifecycle'] == ['entered', 'exited']

    def test_in_process_client_uses_real_mcp_transport(self, logfire_server: FastMCP):
        toolset = LogfireMCP(project='acme/production', client=logfire_server).get_toolset()
        assert isinstance(toolset.client.transport, FastMCPTransport)

    def test_instructions_state_scope_bounds_and_untrusted_data(self):
        instructions = LogfireMCP(project='acme/production').get_instructions()
        assert instructions is not None
        assert 'acme/production' in instructions
        assert '100 rows' in instructions
        assert '30 minutes' in instructions
        assert '`query_schema_reference` before `query_run`' in instructions
        assert 'only `SELECT` queries' in instructions
        assert 'Select only the columns needed' in instructions
        assert '`start_timestamp` and `end_timestamp`' in instructions
        assert 'link only when the user asks' in instructions
        assert '`handoff=true` only when opening it immediately' in instructions
        assert 'untrusted diagnostic data' in instructions
        assert 'approval' not in instructions

        write_instructions = LogfireMCP(project='acme/production', tools=('dashboard_create',)).get_instructions()
        assert write_instructions is not None
        assert 'approval' in write_instructions
        assert 'inspect the current Logfire state' in write_instructions
        assert 'query_run' not in write_instructions
        assert 'Logfire link' not in write_instructions

        non_query_instructions = LogfireMCP(project='acme/production', tools=('dashboard_list',)).get_instructions()
        assert non_query_instructions is not None
        assert 'query_run' not in non_query_instructions
        assert 'Logfire link' not in non_query_instructions
        assert 'approval' not in non_query_instructions

        query_only_instructions = LogfireMCP(project='acme/production', tools=('query_run',)).get_instructions()
        assert query_only_instructions is not None
        assert 'numeric limit' in query_only_instructions
        assert 'query_schema_reference' not in query_only_instructions
        assert 'Logfire link' not in query_only_instructions

        schema_instructions = LogfireMCP(
            project='acme/production', tools=('query_schema_reference',)
        ).get_instructions()
        assert schema_instructions is not None
        assert 'Project-scoped Logfire tools' in schema_instructions
        assert 'query_run' not in schema_instructions
