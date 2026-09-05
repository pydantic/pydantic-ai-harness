"""Behavioral tests for the public Supabase capability."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.bearer import BearerAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.supabase import Supabase, SupabaseFeature

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    from pydantic_ai.toolsets import AbstractToolset

pytestmark = pytest.mark.anyio


def _mcp_toolset(toolset: AbstractToolset[None]) -> MCPToolset[None]:
    leaves: list[MCPToolset[None]] = []

    def capture(candidate: AbstractToolset[None]) -> None:
        assert isinstance(candidate, MCPToolset)
        leaves.append(candidate)

    toolset.apply(capture)
    assert len(leaves) == 1
    return leaves[0]


def _tool_names(model: TestModel) -> set[str]:
    params = model.last_model_request_parameters
    assert params is not None
    return {tool.name for tool in params.function_tools}


def _instructions(messages: list[ModelMessage]) -> str:
    return '\n'.join(
        message.instructions for message in messages if isinstance(message, ModelRequest) and message.instructions
    )


class TestSupabase:
    def test_default_remote_configuration(self):
        with pytest.warns(UserWarning, match='in-memory token storage'):
            capability = Supabase(project_ref='abcdefghijklmnopqrst')
            leaf = _mcp_toolset(capability.get_toolset())

        transport = leaf.client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == (
            'https://mcp.supabase.com/mcp?project_ref=abcdefghijklmnopqrst'
            '&features=database%2Cdebugging%2Cdevelopment%2Cdocs&read_only=true'
        )
        assert isinstance(transport.auth, OAuth)
        assert capability.id == 'supabase-abcdefghijklmnopqrst'

    def test_personal_access_token_authentication_is_hidden(self):
        capability = Supabase(project_ref='abcdefghijklmnopqrst', access_token='sbp_secret')
        leaf = _mcp_toolset(capability.get_toolset())
        transport = leaf.client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert isinstance(transport.auth, BearerAuth)
        assert transport.auth.token.get_secret_value() == 'sbp_secret'
        assert 'sbp_secret' not in repr(capability)
        assert 'sbp_secret' not in repr(leaf)

    @pytest.mark.parametrize('access_token', ['', '   ', 'oauth'])
    def test_invalid_personal_access_token_fails_closed(self, access_token: str):
        with pytest.raises(UserError, match='access_token'):
            Supabase(project_ref='abcdefghijklmnopqrst', access_token=access_token)

    def test_writable_url_uses_the_server_default(self):
        capability = Supabase(project_ref='abcdefghijklmnopqrst', access_token='token', read_only=False)
        transport = _mcp_toolset(capability.get_toolset()).client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert 'read_only' not in transport.url

    def test_agent_spec_schema_excludes_runtime_client(self):
        schema = AgentSpec.model_json_schema_with_capabilities([Supabase])
        properties = schema['$defs']['spec_params_Supabase']['properties']
        assert 'client' not in properties
        assert 'access_token' not in properties
        assert Supabase.get_serialization_name() == 'Supabase'

    def test_from_spec_preserves_safe_runtime_boundary(self):
        capability = Supabase.from_spec(
            'abcdefghijklmnopqrst',
            id='database',
            description='Development database',
            defer_loading=True,
            read_only=False,
            features=('docs',),
        )

        assert capability.project_ref == 'abcdefghijklmnopqrst'
        assert capability.id == 'database'
        assert capability.description == 'Development database'
        assert capability.defer_loading is True
        assert capability.read_only is False
        assert capability.features == ('docs',)
        assert capability.access_token is None

    @pytest.mark.parametrize('project_ref', ['', 'has spaces', 'a/b', 'a?b'])
    def test_project_ref_must_be_url_safe(self, project_ref: str):
        with pytest.raises(UserError, match='project_ref'):
            Supabase(project_ref=project_ref, access_token='token')

    @pytest.mark.parametrize('features', [(), ('database', 'database')])
    def test_feature_groups_are_deliberate(self, features: tuple[SupabaseFeature, ...]):
        with pytest.raises(UserError, match='features'):
            Supabase(project_ref='abcdefghijklmnopqrst', access_token='token', features=features)

    def test_account_feature_is_rejected(self):
        with pytest.raises(UserError, match='features'):
            Supabase(
                project_ref='abcdefghijklmnopqrst',
                access_token='token',
                features=('account',),  # pyright: ignore[reportArgumentType]
            )

    async def test_read_only_agent_tools(self, supabase_server: FastMCP):
        model = TestModel()
        agent = Agent(model, capabilities=[Supabase(project_ref='dev-project')])

        result = await agent.run('Inspect the project')

        assert _tool_names(model) == {
            'execute_sql',
            'generate_typescript_types',
            'get_advisors',
            'get_project_url',
            'get_publishable_keys',
            'list_extensions',
            'list_migrations',
            'list_tables',
            'query_logs',
            'search_docs',
        }
        instructions = _instructions(result.all_messages())
        assert 'read-only' in instructions
        assert 'Public Alpha' in instructions
        assert 'non-production' in instructions
        assert 'untrusted content' in instructions
        assert 'Inspect existing tables' in instructions
        assert 'logs and advisors before changing' in instructions
        assert 'Keep SQL and log queries narrow' in instructions
        assert 'do not poll logs' in instructions

    async def test_feature_groups_filter_the_public_agent_surface(
        self, supabase_server: FastMCP, connections: list[tuple[str, object]]
    ):
        model = TestModel()
        agent = Agent(
            model,
            capabilities=[Supabase(project_ref='dev-project', features=('docs',))],
        )

        await agent.run('Search the docs')

        assert _tool_names(model) == {'search_docs'}
        parameters = parse_qs(urlsplit(connections[-1][0]).query)
        assert parameters['features'] == ['docs']

    async def test_optional_feature_groups_remain_read_only(self, supabase_server: FastMCP):
        model = TestModel()
        agent = Agent(
            model,
            capabilities=[
                Supabase(
                    project_ref='dev-project',
                    features=('functions', 'storage', 'branching'),
                )
            ],
        )

        await agent.run('Inspect optional features')

        assert _tool_names(model) == {
            'get_edge_function',
            'get_storage_config',
            'list_branches',
            'list_edge_functions',
            'list_storage_buckets',
        }

    async def test_writable_branching_excludes_creation_without_cost_confirmation(self, supabase_server: FastMCP):
        model = TestModel(call_tools=[])
        agent = Agent(
            model,
            capabilities=[Supabase(project_ref='dev-project', read_only=False, features=('branching',))],
        )

        await agent.run('Inspect branches')

        assert _tool_names(model) == {
            'delete_branch',
            'list_branches',
            'merge_branch',
            'rebase_branch',
            'reset_branch',
        }

    async def test_unknown_remote_tools_are_not_exposed(self, supabase_server: FastMCP):
        model = TestModel(call_tools=[])
        agent = Agent(
            model,
            capabilities=[
                Supabase(
                    project_ref='dev-project',
                    read_only=False,
                    features=('database', 'debugging', 'development', 'docs', 'functions', 'storage', 'branching'),
                )
            ],
        )

        await agent.run('Inspect the project')

        assert 'future_mutation' not in _tool_names(model)

    async def test_projects_with_overlapping_tools_collide(self, supabase_server: FastMCP):
        agent = Agent(
            TestModel(call_tools=[]),
            capabilities=[Supabase(project_ref='dev-one'), Supabase(project_ref='dev-two')],
        )

        with pytest.raises(UserError, match='conflicts with existing tool'):
            await agent.run('Inspect both projects')

    async def test_projects_with_disjoint_tools_coexist(self, supabase_server: FastMCP):
        model = TestModel(call_tools=[])
        agent = Agent(
            model,
            capabilities=[
                Supabase(project_ref='dev-one', features=('docs',)),
                Supabase(project_ref='dev-two', features=('database',)),
            ],
        )

        result = await agent.run('Inspect both projects')

        assert _tool_names(model) == {
            'execute_sql',
            'list_extensions',
            'list_migrations',
            'list_tables',
            'search_docs',
        }
        instructions = _instructions(result.all_messages())
        assert 'Project `dev-one` provides these Supabase feature groups: docs.' in instructions
        assert 'Project `dev-two` provides these Supabase feature groups: database.' in instructions

    async def test_writes_require_approval(self, supabase_server: FastMCP, calls: list[str]):
        def call_sql(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolCallPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart('execute_sql', {'query': 'delete from todos'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(call_sql),
            capabilities=[Supabase(project_ref='dev-project', read_only=False)],
            output_type=[str, DeferredToolRequests],
        )
        result = await agent.run('Delete the todos')

        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['execute_sql']
        assert calls == []

        call_id = result.output.approvals[0].tool_call_id
        resumed = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call_id: True}),
        )
        assert resumed.output == 'done'
        assert calls == ['execute_sql:delete from todos']

    @pytest.mark.parametrize(
        'tool_name',
        [
            'apply_migration',
            'delete_branch',
            'deploy_edge_function',
            'execute_sql',
            'merge_branch',
            'rebase_branch',
            'reset_branch',
            'update_storage_config',
        ],
    )
    async def test_every_mutation_requires_approval(self, tool_name: str, supabase_server: FastMCP, calls: list[str]):
        model = TestModel(call_tools=[tool_name])
        agent = Agent(
            model,
            capabilities=[
                Supabase(
                    project_ref='dev-project',
                    read_only=False,
                    features=('database', 'functions', 'storage', 'branching'),
                )
            ],
            output_type=[str, DeferredToolRequests],
        )

        result = await agent.run('Change the project')

        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == [tool_name]
        assert calls == []

        call_id = result.output.approvals[0].tool_call_id
        await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call_id: True}),
        )

        assert len(calls) == 1
        assert calls[0].startswith(tool_name)

    async def test_denied_mutation_does_not_execute(self, supabase_server: FastMCP, calls: list[str]):
        model = TestModel(call_tools=['execute_sql'])
        agent = Agent(
            model,
            capabilities=[Supabase(project_ref='dev-project', read_only=False)],
            output_type=[str, DeferredToolRequests],
        )

        result = await agent.run('Delete rows')
        assert isinstance(result.output, DeferredToolRequests)
        call_id = result.output.approvals[0].tool_call_id

        await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call_id: False}),
        )

        assert calls == []

    async def test_writable_instructions_reach_the_model(self, supabase_server: FastMCP):
        agent = Agent(TestModel(call_tools=[]), capabilities=[Supabase(project_ref='dev-project', read_only=False)])

        result = await agent.run('Inspect the project')

        instructions = _instructions(result.all_messages())
        assert 'permits writes' in instructions
        assert 'require approval' in instructions

    async def test_mcp_tool_failure_is_reported(self, supabase_server: FastMCP, failures: set[str]):
        failures.add('list_tables')
        model = TestModel(call_tools=['list_tables'])
        agent = Agent(model, capabilities=[Supabase(project_ref='dev-project')])

        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries') as exc_info:
            await agent.run('List tables')

        assert exc_info.value.__cause__ is not None
        assert 'Supabase unavailable' in str(exc_info.value.__cause__)

    async def test_write_approval_composes_with_stricter_caller_policy(
        self, supabase_server: FastMCP, calls: list[str]
    ):
        capability = Supabase(project_ref='dev-project', read_only=False)
        toolset = capability.get_toolset().approval_required(lambda _ctx, tool, _args: tool.name == 'list_tables')
        model = TestModel(call_tools=['list_tables'])
        agent = Agent(model, toolsets=[toolset], output_type=[str, DeferredToolRequests])

        result = await agent.run('List tables')

        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['list_tables']
        assert calls == []
