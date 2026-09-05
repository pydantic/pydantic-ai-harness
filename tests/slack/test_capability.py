from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.slack import DEFAULT_INSTRUCTIONS, SlackChat, SlackInteractions, SlackThread

from .conftest import FakeSlackClient

pytestmark = pytest.mark.anyio


class TestToolset:
    def test_ask_user_and_upload_file_are_opt_in(self) -> None:
        assert set(SlackChat().get_toolset().tools) == {'post_message', 'post_plan', 'set_status'}

    def test_configuring_both_registers_both(self, tmp_path: Path) -> None:
        capability = SlackChat(interactions=SlackInteractions(), file_root=str(tmp_path))
        assert {'ask_user', 'upload_file'} <= set(capability.get_toolset().tools)


def instructions_of(capability: SlackChat) -> str:
    """The capability's instructions, which it always states as plain text."""
    instructions = capability.get_instructions()
    assert isinstance(instructions, str)
    return instructions


class TestInstructions:
    def test_ships_guidance_so_the_agent_reports_as_it_works(self) -> None:
        # The point of the capability: without this every user retypes it.
        assert instructions_of(SlackChat()) == DEFAULT_INSTRUCTIONS

    def test_describes_ask_user_only_when_it_is_registered(self) -> None:
        assert 'ask_user' not in instructions_of(SlackChat())
        assert 'ask_user' in instructions_of(SlackChat(interactions=SlackInteractions()))

    def test_a_custom_string_is_used_verbatim(self) -> None:
        assert instructions_of(SlackChat(instructions='Be terse.')) == 'Be terse.'

    def test_an_empty_string_adds_none(self) -> None:
        assert SlackChat(instructions='').get_instructions() is None


class TestThroughAnAgent:
    async def test_the_tools_reach_the_thread(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        agent = Agent(
            TestModel(call_tools=['post_message']),
            deps_type=SlackThread,
            capabilities=[SlackChat()],
        )
        await agent.run('go', deps=thread)
        assert slack_client.method_calls('chat_postMessage')

    async def test_the_guidance_reaches_the_model(self, thread: SlackThread) -> None:
        agent = Agent(TestModel(custom_output_text='done'), deps_type=SlackThread, capabilities=[SlackChat()])
        result = await agent.run('go', deps=thread)
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert 'post_plan' in (request.instructions or '')
