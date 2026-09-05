from __future__ import annotations

import pytest
import yaml
from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

import pydantic_ai_harness.slack._mcp as mcp_module
from pydantic_ai_harness.slack import (
    Slack,
    SlackContext,
    SlackContextEntity,
    SlackTool,
    SlackTools,
)
from pydantic_ai_harness.slack._context import bind_slack_context

pytestmark = pytest.mark.anyio


def fake_mcp(
    monkeypatch: pytest.MonkeyPatch, *, omit: SlackTool | None = None, extra_names: tuple[str, ...] = ()
) -> list[dict[str, str]]:
    """Replace the remote Slack boundary with the same public tool catalog."""
    authorizations: list[dict[str, str]] = []

    def build(_url: str, *, id: str, headers: dict[str, str]) -> FunctionToolset[object]:
        assert id == 'slack-mcp'
        authorizations.append(headers)
        toolset = FunctionToolset[object](id=id)
        for slack_tool in SlackTool:
            if slack_tool is omit:
                continue

            async def operation() -> str:
                return 'ok'

            toolset.add_function(operation, name=slack_tool.value, description='Slack operation')
        for name in extra_names:

            async def provider_operation() -> str:
                return 'ok'

            toolset.add_function(provider_operation, name=name, description='New Slack provider operation')
        return toolset

    monkeypatch.setattr(mcp_module, 'MCPToolset', build)
    return authorizations


def recording_model(offered: list[list[str]], instructions: list[str | None]) -> FunctionModel:
    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append([tool.name for tool in info.function_tools])
        instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


class TestSelection:
    def test_default_is_the_curated_read_only_set(self) -> None:
        assert Slack().tools == SlackTools.workspace_read()

    def test_typed_selections_compose(self) -> None:
        tools = SlackTools.read_only() | SlackTools.of(SlackTool.ADD_REACTION)
        assert tools == SlackTools.read_only() | SlackTools.of(SlackTool.ADD_REACTION)

    def test_full_read_only_adds_files_profiles_and_members(self) -> None:
        assert SlackTools.read_only().selected - SlackTools.workspace_read().selected == {
            SlackTool.READ_FILE,
            SlackTool.READ_USER_PROFILE,
            SlackTool.LIST_CHANNEL_MEMBERS,
        }

    def test_selection_requires_the_typed_constructors(self) -> None:
        with pytest.raises(TypeError):
            SlackTools(frozenset({SlackTool.READ_CHANNEL}))  # pyright: ignore[reportCallIssue]

    def test_selection_rejects_a_raw_provider_tool_name(self) -> None:
        with pytest.raises(TypeError, match='SlackTool values'):
            SlackTools.of('slack_read_channel')  # pyright: ignore[reportArgumentType]

    def test_named_selection_is_an_explicit_escape_hatch(self) -> None:
        tools = SlackTools.of(SlackTool.READ_CHANNEL) | SlackTools.named('slack_create_canvas')
        assert tools.selected == {SlackTool.READ_CHANNEL, 'slack_create_canvas'}

    @pytest.mark.parametrize('name', ['', 'create_canvas'])
    def test_named_selection_requires_a_provider_slack_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="beginning with 'slack_'"):
            SlackTools.named(name)

    def test_named_selection_requires_at_least_one_name(self) -> None:
        with pytest.raises(ValueError, match='non-empty provider names'):
            SlackTools.named()

    async def test_only_selected_tools_reach_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mcp(monkeypatch)
        offered: list[list[str]] = []
        instructions: list[str | None] = []
        capability = Slack(tools=SlackTools.of(SlackTool.SEARCH_USERS, SlackTool.READ_CHANNEL), mcp_token='xoxp-a')
        await Agent(recording_model(offered, instructions), capabilities=[capability]).run('go')
        assert offered == [['slack_search_users', 'slack_read_channel']]

    async def test_catalog_drift_is_reported_instead_of_dropping_a_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mcp(monkeypatch, omit=SlackTool.READ_CHANNEL)
        capability = Slack(tools=SlackTools.of(SlackTool.READ_CHANNEL), mcp_token='xoxp-a')
        with pytest.raises(UserError, match='slack_read_channel'):
            await Agent(TestModel(), capabilities=[capability]).run('go')

    async def test_named_tool_is_checked_against_the_discovered_catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mcp(monkeypatch)
        capability = Slack(tools=SlackTools.named('slack_create_canvas'), mcp_token='xoxp-a')
        with pytest.raises(UserError, match='slack_create_canvas'):
            await Agent(TestModel(), capabilities=[capability]).run('go')

    async def test_named_tool_can_run_when_approval_is_explicitly_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_mcp(monkeypatch, extra_names=('slack_new_read_tool',))
        capability = Slack(
            tools=SlackTools.named('slack_new_read_tool'),
            approval='none',
            mcp_token='xoxp-a',
        )
        result = await Agent(TestModel(call_tools=['slack_new_read_tool']), capabilities=[capability]).run('use it')
        assert result.output == '{"slack_new_read_tool":"ok"}'


class TestRunIdentity:
    async def test_each_run_gets_its_invoking_users_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        authorizations = fake_mcp(monkeypatch)
        agent: Agent[None, str] = Agent(TestModel(custom_output_text='done'), capabilities=[Slack()])
        first = SlackContext('C1', '1.0', '1.1', 'U1', team_id='T1', user_token='xoxp-first')
        second = SlackContext('C1', '1.0', '1.2', 'U2', team_id='T1', user_token='xoxp-second')
        with bind_slack_context(first):
            await agent.run('one')
        with bind_slack_context(second):
            await agent.run('two')
        assert authorizations == [
            {'Authorization': 'Bearer xoxp-first'},
            {'Authorization': 'Bearer xoxp-second'},
        ]

    async def test_no_user_token_fails_before_the_model_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('SLACK_MCP_TOKEN', raising=False)
        with pytest.raises(UserError, match='invoking user OAuth token'):
            await Agent(TestModel(), capabilities=[Slack()]).run('go')

    def test_tokens_are_redacted_from_repr(self) -> None:
        assert 'xoxp-secret' not in repr(Slack(mcp_token='xoxp-secret', delivery_token='xoxb-secret'))
        assert 'xoxp-secret' not in repr(SlackContext('C1', '1', '2', 'U1', user_token='xoxp-secret'))


class TestInstructions:
    async def test_names_the_current_conversation(self) -> None:
        offered: list[list[str]] = []
        instructions: list[str | None] = []
        context = SlackContext('C123', '1700.1', '1700.2', 'U456')
        with bind_slack_context(context):
            await Agent(recording_model(offered, instructions), capabilities=[Slack(tools=SlackTools.of())]).run('go')
        assert instructions[0] is not None
        assert 'Use the Slack tools' in instructions[0]
        assert 'C123' in instructions[0]
        assert 'U456' in instructions[0]

    async def test_names_the_users_active_slack_view(self) -> None:
        offered: list[list[str]] = []
        instructions: list[str | None] = []
        context = SlackContext(
            'D123',
            '1700.1',
            '1700.2',
            'U456',
            active_entities=(SlackContextEntity('slack#/types/channel_id', 'C789', 'T1'),),
        )
        with bind_slack_context(context):
            await Agent(recording_model(offered, instructions), capabilities=[Slack(tools=SlackTools.of())]).run('go')
        assert instructions[0] is not None
        assert 'relevance order' in instructions[0]
        assert 'C789' in instructions[0]

    def test_empty_custom_instructions_add_nothing(self) -> None:
        assert Slack(instructions='').get_instructions() is None


class TestApproval:
    async def test_selected_write_tools_require_approval_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mcp(monkeypatch)
        capability = Slack(tools=SlackTools.of(SlackTool.SEND_MESSAGE), mcp_token='xoxp-a')
        agent: Agent[None, str | DeferredToolRequests] = Agent(
            TestModel(call_tools=['slack_send_message']),
            output_type=[str, DeferredToolRequests],
            capabilities=[capability],
        )
        result = await agent.run('send it')
        assert isinstance(result.output, DeferredToolRequests)
        assert result.output.approvals[0].tool_name == 'slack_send_message'

    async def test_untyped_provider_tools_require_approval_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mcp(monkeypatch, extra_names=('slack_create_canvas',))
        capability = Slack(tools=SlackTools.named('slack_create_canvas'), mcp_token='xoxp-a')
        agent: Agent[None, str | DeferredToolRequests] = Agent(
            TestModel(call_tools=['slack_create_canvas']),
            output_type=[str, DeferredToolRequests],
            capabilities=[capability],
        )
        result = await agent.run('make a canvas')
        assert isinstance(result.output, DeferredToolRequests)
        assert result.output.approvals[0].tool_name == 'slack_create_canvas'

    async def test_approval_all_gates_a_read_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mcp(monkeypatch)
        capability = Slack(tools=SlackTools.of(SlackTool.READ_CHANNEL), approval='all', mcp_token='xoxp-a')
        agent: Agent[None, str | DeferredToolRequests] = Agent(
            TestModel(call_tools=['slack_read_channel']),
            output_type=[str, DeferredToolRequests],
            capabilities=[capability],
        )
        result = await agent.run('read it')
        assert isinstance(result.output, DeferredToolRequests)

    async def test_approval_none_runs_a_write_without_deferring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mcp(monkeypatch)
        capability = Slack(tools=SlackTools.of(SlackTool.SEND_MESSAGE), approval='none', mcp_token='xoxp-a')
        result = await Agent(TestModel(call_tools=['slack_send_message']), capabilities=[capability]).run('send it')
        assert not isinstance(result.output, DeferredToolRequests)

    def test_direct_construction_rejects_an_unknown_approval_policy(self) -> None:
        with pytest.raises(ValueError, match='approval must be'):
            Slack(approval='sometimes')  # pyright: ignore[reportArgumentType]


SPEC = """
name: slack-agent
model: test
capabilities:
  - Slack:
      tools: [slack_search_users, slack_read_channel]
      approval: writes
"""


class TestAgentSpec:
    def test_typed_tool_selection_loads_from_yaml(self) -> None:
        agent = Agent.from_spec(yaml.safe_load(SPEC), custom_capability_types=[Slack])
        capability = next(item for item in agent.root_capability.capabilities if isinstance(item, Slack))
        assert capability.tools == SlackTools.of(SlackTool.SEARCH_USERS, SlackTool.READ_CHANNEL)

    def test_unknown_tool_name_fails_at_load_time(self) -> None:
        spec = yaml.safe_load(SPEC)
        spec['capabilities'][0]['Slack']['tools'] = ['slack_guess_everything']
        with pytest.raises(ValueError, match='slack_guess_everything'):
            Agent.from_spec(spec, custom_capability_types=[Slack])

    def test_unknown_approval_policy_fails_at_load_time(self) -> None:
        spec = yaml.safe_load(SPEC)
        spec['capabilities'][0]['Slack']['approval'] = 'sometimes'
        with pytest.raises(ValueError, match='approval'):
            Agent.from_spec(spec, custom_capability_types=[Slack])

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('tools', 'slack_read_channel'),
            ('tools', [1]),
            ('approver_ids', 'U1'),
            ('approver_ids', [1]),
            ('instructions', 7),
        ],
    )
    def test_malformed_values_fail_at_load_time(self, field: str, value: object) -> None:
        spec = yaml.safe_load(SPEC)
        spec['capabilities'][0]['Slack'][field] = value
        with pytest.raises(ValueError):
            Agent.from_spec(spec, custom_capability_types=[Slack])
