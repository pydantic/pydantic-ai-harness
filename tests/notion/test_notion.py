"""Behavioral tests for the Notion capability and toolset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.notion import NOTION_MCP_URL, Notion, NotionToolset

from ._support import NotionState

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio


class TestNotionToolset:
    def test_official_endpoint_keeps_path_scoped_oauth_resource(self) -> None:
        assert NOTION_MCP_URL == 'https://mcp.notion.com/mcp'

    async def test_tool_discovery_manages_its_own_client_lifecycle(
        self, notion_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        client = Client(notion_server)
        toolset = NotionToolset[None](client=client)

        tools = await toolset.get_tools(run_context)

        assert 'notion-fetch' in tools
        assert 'workspace-1' in toolset.attribution
        assert client.is_connected() is False

    async def test_failed_direct_discovery_closes_client(
        self, attribution_error_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        client = Client(attribution_error_server)
        toolset = NotionToolset[None](client=client)

        with pytest.raises(UserError, match='attribution failed; no workspace tools were exposed'):
            await toolset.get_tools(run_context)

        assert client.is_connected() is False

    async def test_default_surface_is_closed_read_only_allowlist(
        self, notion_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=notion_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)

        assert set(tools) == {
            'notion-ai-search',
            'notion-fetch',
            'notion-get-users',
            'notion-query-meeting-notes',
            'notion-search',
        }
        assert all((tool.tool_def.metadata or {})['notion'] is True for tool in tools.values())
        assert all((tool.tool_def.metadata or {})['notion_mutation'] is False for tool in tools.values())
        assert all('workspace-1' in (tool.tool_def.metadata or {})['notion_attribution'] for tool in tools.values())
        assert toolset.attribution == (tools['notion-fetch'].tool_def.metadata or {})['notion_attribution']

    async def test_mutations_are_added_one_exact_name_at_a_time(
        self, notion_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=notion_server, mutations='notion-update-page')
        async with toolset:
            tools = await toolset.get_tools(run_context)

        assert 'notion-update-page' in tools
        assert 'notion-create-database' not in tools
        assert 'notion-delete-workspace' not in tools
        assert (tools['notion-update-page'].tool_def.metadata or {})['notion_mutation'] is True

    def test_unknown_mutation_is_rejected(self, notion_server: FastMCP) -> None:
        with pytest.raises(UserError, match='Unknown Notion mutation tool.*notion-delete-workspace'):
            NotionToolset(client=notion_server, mutations='notion-delete-workspace')

    def test_duplicate_mutations_are_normalized(self, notion_server: FastMCP) -> None:
        toolset = NotionToolset(
            client=notion_server,
            mutations=['notion-update-page', 'notion-update-page'],
        )
        assert toolset.mutation_tools == ('notion-update-page',)

    def test_attribution_is_unavailable_before_tool_discovery(self, notion_server: FastMCP) -> None:
        toolset = NotionToolset(client=notion_server)
        with pytest.raises(UserError, match='has not been established'):
            _ = toolset.attribution
        with pytest.raises(UserError, match='identity has not been established'):
            _ = toolset.connection_identity

    def test_expected_identity_is_validated(self, notion_server: FastMCP) -> None:
        with pytest.raises(UserError, match='must be a .* tuple'):
            NotionToolset(client=notion_server, expected_identity=('workspace-1',))  # pyright: ignore[reportArgumentType]
        with pytest.raises(UserError, match='workspace and user IDs are invalid'):
            NotionToolset(client=notion_server, expected_identity=('workspace 1', 'user-1'))

    async def test_caller_owned_prebuilt_client_is_used(
        self, notion_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        client = Client(notion_server)
        toolset = NotionToolset[None](client=client)
        assert toolset.client is client
        async with toolset:
            async with toolset:
                tools = await toolset.get_tools(run_context)
        assert 'notion-fetch' in tools

    async def test_identity_is_rechecked_when_toolset_reenters(
        self, notion_server: FastMCP, notion_state: NotionState, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=notion_server)
        async with toolset:
            await toolset.get_tools(run_context)
        async with toolset:
            await toolset.get_tools(run_context)

        notion_state['user_id'] = 'user-2'
        async with toolset:
            with pytest.raises(UserError, match='connection identity changed; no workspace tools were exposed'):
                await toolset.get_tools(run_context)

    async def test_tools_are_not_exposed_when_attribution_fails(
        self, attribution_error_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=attribution_error_server)
        async with toolset:
            with pytest.raises(UserError, match='attribution failed; no workspace tools were exposed'):
                await toolset.get_tools(run_context)

    async def test_oversized_attribution_is_rejected(
        self, oversized_attribution_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=oversized_attribution_server)
        async with toolset:
            with pytest.raises(UserError, match='exceeded the 16384-character safety limit'):
                await toolset.get_tools(run_context)

    async def test_success_without_identity_is_rejected(
        self, malformed_attribution_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=malformed_attribution_server)
        async with toolset:
            with pytest.raises(UserError, match='attribution was malformed; no workspace tools were exposed'):
                await toolset.get_tools(run_context)

    async def test_non_text_attribution_is_rejected(
        self, non_text_attribution_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=non_text_attribution_server)

        with pytest.raises(UserError, match='attribution was malformed; no workspace tools were exposed'):
            await toolset.get_tools(run_context)

    async def test_oversized_identity_field_is_rejected(
        self, oversized_identity_field_server: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=oversized_identity_field_server)
        async with toolset:
            with pytest.raises(UserError, match='attribution was malformed; no workspace tools were exposed'):
                await toolset.get_tools(run_context)

    async def test_untrusted_metadata_is_excluded_from_attribution_and_instructions(
        self, attributed_server_with_meta: FastMCP, run_context: RunContext[None]
    ) -> None:
        toolset = NotionToolset[None](client=attributed_server_with_meta)
        async with toolset:
            instructions = await toolset.get_instructions(run_context)
            tools = await toolset.get_tools(run_context)

        assert instructions is not None
        assert 'secret' not in instructions.content
        assert 'Acme' not in instructions.content
        assert 'Ada' not in instructions.content
        assert 'secret' not in toolset.attribution
        assert toolset.attribution == (tools['notion-fetch'].tool_def.metadata or {})['notion_attribution']


class TestNotion:
    def test_live_client_opts_out_of_agent_spec(self) -> None:
        assert Notion.get_serialization_name() is None

    def test_client_is_hidden_from_repr(self, notion_server: FastMCP) -> None:
        assert 'notion-fake' not in repr(Notion(client=notion_server))

    async def test_identity_is_fetched_before_search_and_returned_unchanged(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        step = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal step
            assert info.instructions is not None
            assert 'connection identity' in info.instructions
            assert 'workspace-1' in info.instructions
            assert 'user-1' in info.instructions
            assert 'unknown_block_ids' in info.instructions
            if step == 0:
                step += 1
                return ModelResponse(parts=[ToolCallPart('notion-ai-search', {'query': 'launch plan'}, 'search')])
            return ModelResponse(parts=[TextPart('Found the launch plan in Acme for Ada.')])

        agent = Agent(FunctionModel(model), capabilities=[Notion(client=notion_server)])
        result = await agent.run('Find the launch plan')

        assert result.output == 'Found the launch plan in Acme for Ada.'
        assert notion_state['calls'] == [
            ('notion-fetch', {'id': 'self'}),
            ('notion-fetch', {'id': 'self'}),
            ('notion-ai-search', {'query': 'launch plan'}),
        ]

    async def test_capability_exposes_selected_mutation_with_guidance(self, notion_server: FastMCP) -> None:
        seen_tools: set[str] = set()

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen_tools.update(tool.name for tool in info.function_tools)
            assert info.instructions is not None
            assert 'explicitly selected Notion mutation tools' in info.instructions
            assert 'IDs returned by search or fetch' in info.instructions
            assert '`notion-get-async-task`' in info.instructions
            assert '`poll_after_seconds` delay' in info.instructions
            assert 'stop on `succeeded` or `failed`' in info.instructions
            assert "caller's deadline or cancellation" in info.instructions
            assert 'do not automatically\nretry a non-idempotent mutation' in info.instructions
            assert 'request fresh approval' in info.instructions
            return ModelResponse(parts=[TextPart('done')])

        result = await Agent(
            FunctionModel(model),
            capabilities=[Notion(client=notion_server, mutations='notion-update-page')],
        ).run('hello')
        assert result.output == 'done'
        assert 'notion-update-page' in seen_tools

    async def test_search_guidance_falls_back_when_ai_search_is_unavailable(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        notion_state['ai_search_status'] = 'not_enabled'
        step = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal step
            assert info.instructions is not None
            assert 'ai_search_available=False' in info.instructions
            assert 'notion-ai-search' not in {tool.name for tool in info.function_tools}
            if step == 0:
                step += 1
                return ModelResponse(parts=[ToolCallPart('notion-search', {'query': 'launch plan'}, 'search')])
            return ModelResponse(parts=[TextPart('Found with keyword search.')])

        result = await Agent(FunctionModel(model), capabilities=[Notion(client=notion_server)]).run('Find launch plan')

        assert result.output == 'Found with keyword search.'
        assert ('notion-search', {'query': 'launch plan'}) in notion_state['calls']

    async def test_limited_ai_search_follows_provider_fallback_guidance(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        notion_state['ai_search_status'] = 'available_with_limit'

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.instructions is not None
            assert 'ai_search_available=False' in info.instructions
            tool_names = {tool.name for tool in info.function_tools}
            assert 'notion-ai-search' not in tool_names
            assert 'notion-search' in tool_names
            return ModelResponse(parts=[TextPart('Keyword search is the supported fallback.')])

        result = await Agent(FunctionModel(model), capabilities=[Notion(client=notion_server)]).run('Find launch plan')

        assert result.output == 'Keyword search is the supported fallback.'

    async def test_read_tools_with_unavailable_access_status_are_hidden(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        notion_state['unavailable_tools'].add('query_meeting_notes')

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = {tool.name for tool in info.function_tools}
            assert 'notion-query-meeting-notes' not in tool_names
            assert 'notion-search' in tool_names
            return ModelResponse(parts=[TextPart('done')])

        result = await Agent(FunctionModel(model), capabilities=[Notion(client=notion_server)]).run(
            'List meeting notes'
        )
        assert result.output == 'done'

    async def test_selected_mutation_with_unavailable_access_status_is_hidden(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        notion_state['unavailable_tools'].add('update_page')

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert 'notion-update-page' not in {tool.name for tool in info.function_tools}
            return ModelResponse(parts=[TextPart('Unavailable.')])

        result = await Agent(
            FunctionModel(model),
            capabilities=[Notion(client=notion_server, mutations='notion-update-page')],
        ).run('Update a page')
        assert result.output == 'Unavailable.'

    async def test_selected_mutation_missing_from_access_map_is_hidden(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        notion_state['missing_access_tools'].add('update_page')

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert 'notion-update-page' not in {tool.name for tool in info.function_tools}
            return ModelResponse(parts=[TextPart('Missing.')])

        result = await Agent(
            FunctionModel(model),
            capabilities=[Notion(client=notion_server, mutations='notion-update-page')],
        ).run('Update a page')
        assert result.output == 'Missing.'

    async def test_selected_mutation_composes_with_tool_approval(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        model_calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_calls
            assert info.instructions is not None
            assert 'connection identity' in info.instructions
            model_calls += 1
            if model_calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'notion-update-page',
                            {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'New launch plan'},
                            'update-1',
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart('Updated.')])

        notion = NotionToolset[None](client=notion_server, mutations='notion-update-page')
        approved = notion.approval_required(
            lambda _ctx, tool_def, _args: (tool_def.metadata or {}).get('notion_mutation') is True
        )
        agent = Agent(
            FunctionModel(model),
            deps_type=type(None),
            toolsets=[approved],
            output_type=[str, DeferredToolRequests],
        )

        deferred = await agent.run('Replace the launch plan')
        assert isinstance(deferred.output, DeferredToolRequests)
        assert [call.tool_name for call in deferred.output.approvals] == ['notion-update-page']
        assert notion_state['page_content'] == 'Old launch plan'

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'update-1': True}),
        )
        assert resumed.output == 'Updated.'
        assert notion_state['page_content'] == 'New launch plan'

    async def test_restored_approval_requires_original_connection_identity(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        def propose(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'notion-update-page',
                        {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Wrong workspace'},
                        'update-1',
                    )
                ]
            )

        original = NotionToolset[None](client=notion_server, mutations='notion-update-page')
        deferred = await Agent(
            FunctionModel(propose),
            deps_type=type(None),
            toolsets=[original.approval_required()],
            output_type=[str, DeferredToolRequests],
        ).run('Replace the launch plan')
        assert isinstance(deferred.output, DeferredToolRequests)
        expected_identity = original.connection_identity

        notion_state['workspace_id'] = 'workspace-2'
        restored = NotionToolset[None](
            client=notion_server,
            mutations='notion-update-page',
            expected_identity=expected_identity,
        )
        restored_agent = Agent(
            FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('should not run')])),
            deps_type=type(None),
            toolsets=[restored.approval_required()],
            output_type=[str, DeferredToolRequests],
        )
        with pytest.raises(UserError, match='connection identity changed'):
            await restored_agent.run(
                message_history=deferred.all_messages(),
                deferred_tool_results=DeferredToolResults(approvals={'update-1': True}),
            )
        assert notion_state['page_content'] == 'Old launch plan'

    async def test_mutation_provider_error_is_not_model_retryable(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        notion_state['mutation_error'] = True
        model_calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_calls
            model_calls += 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'notion-update-page',
                        {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Applied once'},
                        f'update-{model_calls}',
                    )
                ]
            )

        agent = Agent(
            FunctionModel(model),
            capabilities=[Notion(client=notion_server, mutations='notion-update-page')],
        )
        with pytest.raises(ToolError, match='ambiguous provider failure'):
            await agent.run('Replace the launch plan')

        update_calls = [name for name, _args in notion_state['calls'] if name == 'notion-update-page']
        assert update_calls == ['notion-update-page']
        assert model_calls == 1

    async def test_approved_mutation_is_rejected_after_connection_identity_changes(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        model_calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'notion-update-page',
                            {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Wrong workspace'},
                            'update-1',
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart('should not run')])  # pragma: no cover

        notion = NotionToolset[None](client=notion_server, mutations='notion-update-page')
        agent = Agent(
            FunctionModel(model),
            deps_type=type(None),
            toolsets=[notion.approval_required()],
            output_type=[str, DeferredToolRequests],
        )
        deferred = await agent.run('Replace the launch plan')
        assert isinstance(deferred.output, DeferredToolRequests)

        notion_state['workspace_id'] = 'workspace-2'
        with pytest.raises(UserError, match='connection identity changed'):
            await agent.run(
                message_history=deferred.all_messages(),
                deferred_tool_results=DeferredToolResults(approvals={'update-1': True}),
            )

        assert notion_state['page_content'] == 'Old launch plan'

    async def test_mutation_rechecks_identity_immediately_before_execution(
        self, rotating_identity_server: FastMCP
    ) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'notion-update-page',
                        {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Wrong workspace'},
                        'update-1',
                    )
                ]
            )

        agent = Agent(
            FunctionModel(model),
            capabilities=[Notion(client=rotating_identity_server, mutations='notion-update-page')],
        )
        with pytest.raises(
            UserError, match='connection identity changed after tool discovery; tool invocation refused'
        ):
            await agent.run('Replace the launch plan')

    async def test_mutation_rechecks_access_immediately_before_execution(
        self, notion_server: FastMCP, notion_state: NotionState
    ) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            notion_state['unavailable_tools'].add('update_page')
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'notion-update-page',
                        {'page_id': 'page-1', 'command': 'replace_content', 'new_str': 'Blocked'},
                        'update-1',
                    )
                ]
            )

        agent = Agent(
            FunctionModel(model),
            capabilities=[Notion(client=notion_server, mutations='notion-update-page')],
        )
        with pytest.raises(UserError, match='tool `notion-update-page` is no longer available'):
            await agent.run('Replace the launch plan')
        assert notion_state['page_content'] == 'Old launch plan'

    async def test_read_rechecks_identity_immediately_before_execution(self, rotating_identity_server: FastMCP) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('notion-search', {'query': 'private page'}, 'search-1')])

        agent = Agent(FunctionModel(model), capabilities=[Notion(client=rotating_identity_server)])
        with pytest.raises(
            UserError, match='connection identity changed after tool discovery; tool invocation refused'
        ):
            await agent.run('Find the private page')

    async def test_instructions_can_be_disabled(self, notion_server: FastMCP) -> None:
        seen: list[str] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions or '')
            return ModelResponse(parts=[TextPart('done')])

        await Agent(
            FunctionModel(model),
            capabilities=[Notion(client=notion_server, include_instructions=False)],
        ).run('hello')
        assert all('connection identity' not in instructions for instructions in seen)
