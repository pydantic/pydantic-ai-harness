"""Tests for the reference agent construction and composition."""

from __future__ import annotations

from conftest import RecordingExecutor
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import TieredCompaction

from pydantic_ai_harness_terminal_bench.agent import build_agent, build_compaction
from pydantic_ai_harness_terminal_bench.tools import CommandResult, TerminalBenchDeps


def _composed_capabilities(agent: Agent[object, object]) -> list[object]:
    """The capabilities composed onto an agent (flattened one level)."""
    root = agent.root_capability
    return list(getattr(root, 'capabilities', [root]))


def test_build_compaction_is_tiered_and_ordered() -> None:
    compaction = build_compaction(summarizer_model='anthropic:claude-opus-4-6', target_tokens=1234)
    assert isinstance(compaction, TieredCompaction)
    assert compaction.target_tokens == 1234
    # cheap-to-expensive: the zero-LLM tier before the summarizer.
    assert [type(tier).__name__ for tier in compaction.tiers] == [
        'ClearToolResults',
        'SummarizingCompaction',
    ]


def test_build_agent_has_bash_and_compaction() -> None:
    agent = build_agent(model=TestModel())
    assert any(isinstance(cap, TieredCompaction) for cap in _composed_capabilities(agent))


def test_extra_capabilities_are_composed() -> None:
    extra = build_compaction(summarizer_model=TestModel())
    agent = build_agent(model=TestModel(), extra_capabilities=[extra])
    assert sum(isinstance(cap, TieredCompaction) for cap in _composed_capabilities(agent)) == 2


async def test_agent_runs_a_bash_step_to_completion() -> None:
    executor = RecordingExecutor(
        results={'cat answer.txt': CommandResult(output='42', exit_code=0)},
    )

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        saw_tool_return = any(
            isinstance(part, ToolReturnPart) for message in messages for part in getattr(message, 'parts', [])
        )
        if not saw_tool_return:
            return ModelResponse(parts=[ToolCallPart('bash', {'command': 'cat answer.txt'})])
        return ModelResponse(parts=[TextPart('The answer is 42.')])

    agent = build_agent(model=FunctionModel(model_fn))
    result = await agent.run('what is in answer.txt', deps=TerminalBenchDeps(execute=executor))

    assert executor.calls == ['cat answer.txt']
    assert result.output == 'The answer is 42.'
    assert result.usage.tool_calls == 1
