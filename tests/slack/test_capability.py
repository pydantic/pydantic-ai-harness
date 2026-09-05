from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.slack import (
    DEFAULT_INSTRUCTIONS,
    SlackChat,
    SlackInteractions,
    SlackThread,
    default_client,
)

from .conftest import FakeSlackClient

pytestmark = pytest.mark.anyio


@dataclass
class Warehouse:
    """Deps an agent already had before anyone thought about Slack."""

    dsn: str


class TestToolset:
    def test_ask_user_and_upload_file_are_opt_in(self, slack_client: FakeSlackClient) -> None:
        capability = SlackChat(client=slack_client)
        assert set(capability.get_toolset().tools) == {'post_message', 'post_plan', 'set_status'}

    def test_configuring_both_registers_both(self, slack_client: FakeSlackClient, tmp_path: Path) -> None:
        capability = SlackChat(client=slack_client, ask_user=True, file_root=str(tmp_path))
        assert {'ask_user', 'upload_file'} <= set(capability.get_toolset().tools)

    def test_the_same_toolset_is_reused_across_calls(self, slack_client: FakeSlackClient) -> None:
        # Rebuilding it would issue a fresh plan-signing key mid-run, so a plan_id
        # the model was just given would stop working.
        capability = SlackChat(client=slack_client)
        assert capability.get_toolset() is capability.get_toolset()


class TestClient:
    def test_a_token_is_only_needed_when_something_posts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Constructing must not need credentials: capabilities are built at import
        # time in plenty of applications.
        monkeypatch.delenv('SLACK_BOT_TOKEN', raising=False)
        capability = SlackChat()
        with pytest.raises(ValueError, match='SLACK_BOT_TOKEN'):
            capability.resolve_client()

    def test_the_environment_supplies_the_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-from-the-environment')
        assert SlackChat().resolve_client() is not None

    def test_a_real_client_satisfies_the_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-from-the-environment')
        assert default_client().chat_postMessage is not None

    def test_the_client_is_built_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SLACK_BOT_TOKEN', 'xoxb-from-the-environment')
        capability = SlackChat()
        assert capability.resolve_client() is capability.resolve_client()


def instructions_of(capability: SlackChat[None]) -> str:
    """The capability's instructions, which it always states as plain text."""
    instructions = capability.get_instructions()
    assert isinstance(instructions, str)
    return instructions


class TestInstructions:
    def test_ships_guidance_so_the_agent_reports_as_it_works(self, slack_client: FakeSlackClient) -> None:
        # The point of the capability: without this every user retypes it.
        assert instructions_of(SlackChat(client=slack_client)) == DEFAULT_INSTRUCTIONS

    def test_describes_ask_user_only_when_it_is_registered(self, slack_client: FakeSlackClient) -> None:
        assert 'ask_user' not in instructions_of(SlackChat(client=slack_client))
        assert 'ask_user' in instructions_of(SlackChat(client=slack_client, ask_user=True))

    def test_names_the_channels_the_model_may_choose_between(self, slack_client: FakeSlackClient) -> None:
        instructions = instructions_of(SlackChat(client=slack_client, channels=['#alerts', '#eng']))
        assert '#alerts, #eng' in instructions

    def test_a_single_channel_needs_no_explaining(self, slack_client: FakeSlackClient) -> None:
        # There is nothing to choose, so naming it would only invite the model to
        # pass a channel it does not need to pass.
        assert instructions_of(SlackChat(client=slack_client, channels=['#alerts'])) == DEFAULT_INSTRUCTIONS

    def test_a_custom_string_is_used_verbatim(self, slack_client: FakeSlackClient) -> None:
        assert instructions_of(SlackChat(client=slack_client, instructions='Be terse.')) == 'Be terse.'

    def test_an_empty_string_adds_none(self, slack_client: FakeSlackClient) -> None:
        assert SlackChat(client=slack_client, instructions='').get_instructions() is None


class TestPromptRouting:
    def test_an_agent_that_asks_nothing_routes_nothing(self, slack_client: FakeSlackClient) -> None:
        capability = SlackChat(client=slack_client)
        assert capability.resolve_prompt(block_id='b', value='Yes', user_id='U1') is False

    def test_a_click_on_no_live_prompt_changes_nothing(self, slack_client: FakeSlackClient) -> None:
        capability = SlackChat(client=slack_client, ask_user=True)
        assert capability.resolve_prompt(block_id='b', value='Yes', user_id='U1') is False

    def test_one_registry_serves_ask_user_and_approvals(self, slack_client: FakeSlackClient) -> None:
        # Both routes have to resolve against the same object, or a click reaches
        # neither and the run waits out its timeout.
        capability = SlackChat(client=slack_client, ask_user=True, approvals=True)
        assert capability.resolve_interactions() is capability.resolve_interactions()

    def test_a_registry_you_pass_is_the_one_used(self, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        capability = SlackChat(client=slack_client, ask_user=True, interactions=interactions)
        assert capability.resolve_interactions() is interactions


class TestThroughAnAgent:
    async def test_the_tools_reach_slack_without_touching_deps(
        self, bound_thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # The agent keeps the deps it already had; Slack arrives beside them.
        agent = Agent(
            TestModel(call_tools=['post_message']),
            deps_type=Warehouse,
            capabilities=[SlackChat(client=slack_client)],
        )
        await agent.run('go', deps=Warehouse(dsn='postgres://'))
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['thread_ts'] == bound_thread.thread_ts

    async def test_the_guidance_reaches_the_model(self, slack_client: FakeSlackClient) -> None:
        agent: Agent[None, str] = Agent(
            TestModel(call_tools=[], custom_output_text='done'),
            capabilities=[SlackChat(client=slack_client)],
        )
        result = await agent.run('go')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert 'post_plan' in (request.instructions or '')

    async def test_an_agent_with_no_slack_front_door_still_reports(self, slack_client: FakeSlackClient) -> None:
        # Nothing bound, no SlackBot: the cron-job case.
        agent: Agent[None, str] = Agent(
            TestModel(call_tools=['post_message']),
            capabilities=[SlackChat(client=slack_client, channels=['#alerts'])],
        )
        await agent.run('go')
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['channel'] == '#alerts'


SPEC = """
name: alerts-bot
model: test
instructions: Watch the deploys.
capabilities:
  - SlackChat:
      channels: ['#alerts', '#eng']
      approvals: true
      token: xoxb-from-the-spec
"""


def spec_capability(agent: Agent[None, str]) -> SlackChat[None]:
    """The `SlackChat` an agent spec built, so a test can inspect what it got."""
    found = [c for c in agent.root_capability.capabilities if isinstance(c, SlackChat)]
    assert len(found) == 1
    return found[0]


class TestAgentSpec:
    def test_an_agent_defined_in_yaml_gets_the_slack_tools(self) -> None:
        agent = Agent.from_spec(yaml.safe_load(SPEC), custom_capability_types=[SlackChat])
        capability = spec_capability(agent)
        assert capability.channels == ['#alerts', '#eng']
        assert (capability.approvals, capability.token) == (True, 'xoxb-from-the-spec')

    async def test_a_spec_defined_agent_posts_where_the_spec_said(self, slack_client: FakeSlackClient) -> None:
        spec = yaml.safe_load(SPEC)
        spec['capabilities'][0]['SlackChat']['channels'] = ['#alerts']
        agent = Agent.from_spec(
            spec,
            custom_capability_types=[SlackChat],
            model=TestModel(call_tools=['post_message']),
        )
        spec_capability(agent).client = slack_client
        await agent.run('go', deps=None)
        assert slack_client.method_calls('chat_postMessage')[0].kwargs['channel'] == '#alerts'

    def test_a_thread_can_be_written_out_in_the_spec(self) -> None:
        # Pydantic leaves it a mapping, because the field also accepts a resolver
        # callable. `from_spec` is what turns it into a SlackThread.
        spec = yaml.safe_load(SPEC)
        spec['capabilities'][0]['SlackChat']['thread'] = {'channel_id': 'C777', 'thread_ts': '1.1'}
        agent = Agent.from_spec(spec, custom_capability_types=[SlackChat])
        assert spec_capability(agent).thread == SlackThread(channel_id='C777', thread_ts='1.1')

    def test_a_live_object_cannot_come_from_a_spec(self, slack_client: FakeSlackClient) -> None:
        # Silently ignoring it would leave an agent authenticating as something
        # other than the spec says.
        with pytest.raises(ValueError, match='client cannot be set from an agent spec'):
            SlackChat.from_spec(client=slack_client)
