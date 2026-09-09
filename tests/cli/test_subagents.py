"""The bridge renders `sub_agents.*` events: a `>>` line as a delegation starts, a `<<` line as it ends."""

from __future__ import annotations

import io
import json
import re
from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from termflow.ansi import visible  # pyright: ignore[reportMissingTypeStubs]

from pydantic_ai_harness.cli import CliBridge
from pydantic_ai_harness.subagents import SubAgent, SubAgents

pytestmark = pytest.mark.anyio

_DURATION = r'\(\d+\.\ds\)'


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _delegate_model(**args: str) -> FunctionModel:
    """Delegate once, then answer `done`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        settled = any(isinstance(part, (RetryPromptPart, ToolReturnPart)) for m in messages for part in m.parts)
        if not settled:
            yield {0: DeltaToolCall(name='delegate_task', json_args=json.dumps(args), tool_call_id='call_0')}
        else:
            yield 'done'

    return FunctionModel(stream_function=stream)


async def _transcript(sub_agents: SubAgents[None], **args: str) -> list[str]:
    buffer = io.StringIO()
    agent = Agent(
        _delegate_model(agent_name='worker', task='do it', **args),
        deps_type=type(None),
        capabilities=[sub_agents, CliBridge(output=buffer, width=80)],
    )
    await agent.run('go')
    return visible(buffer.getvalue()).splitlines()


class TestDelegationRendering:
    async def test_start_and_end_bracket_the_delegation(self) -> None:
        worker = Agent(TestModel(custom_output_text='W'), name='worker')

        lines = await _transcript(SubAgents(agents=[SubAgent(worker)]))

        assert lines[0].startswith('> delegate_task ')
        assert lines[1] == '>> worker do it'
        assert re.fullmatch(rf'<< worker {_DURATION} W', lines[2])
        assert lines[3:] == ['done']

    async def test_the_menu_key_rides_the_start_line(self) -> None:
        worker = Agent(TestModel(custom_output_text='W'), name='worker')

        lines = await _transcript(SubAgents(agents=[SubAgent(worker)], models={'fast': TestModel()}), model='fast')

        assert lines[1] == '>> worker (fast) do it'

    async def test_a_failed_delegation_names_the_outcome_and_skips_the_retry_line(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise UnexpectedModelBehavior('kaboom')

        worker = Agent(FunctionModel(boom), name='worker')

        lines = await _transcript(SubAgents(agents=[SubAgent(worker)]))

        assert re.fullmatch(rf"<< worker failed {_DURATION} Sub-agent 'worker' failed: kaboom", lines[2])
        assert not any(line.startswith('! delegate_task') for line in lines)

    async def test_a_truncated_output_says_so(self) -> None:
        worker = Agent(TestModel(custom_output_text='o' * 5000), name='worker')

        lines = await _transcript(SubAgents(agents=[SubAgent(worker)]))

        assert re.match(r'<< worker \(\d+\.\ds, output truncated\) o+', lines[2])
        assert len(lines[2]) <= 80
