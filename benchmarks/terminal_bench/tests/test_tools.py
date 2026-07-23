"""Tests for the bash tool and its output formatting."""

from __future__ import annotations

from conftest import RecordingExecutor
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness_terminal_bench.tools import (
    CommandResult,
    TerminalBenchDeps,
    build_bash_toolset,
    format_result,
)


def test_format_result_appends_nonzero_exit() -> None:
    out = format_result(CommandResult(output='boom', exit_code=2), max_output_chars=1000)
    assert out == 'boom\n[exit code 2]'


def test_format_result_zero_exit_no_suffix() -> None:
    assert format_result(CommandResult(output='ok', exit_code=0), max_output_chars=1000) == 'ok'


def test_format_result_empty_output() -> None:
    assert format_result(CommandResult(output='', exit_code=0), max_output_chars=1000) == '(no output)'


def test_format_result_truncates_middle() -> None:
    body = 'A' * 100 + 'B' * 100
    out = format_result(CommandResult(output=body, exit_code=0), max_output_chars=40)
    assert 'characters truncated' in out
    assert out.startswith('A')
    assert out.rstrip().endswith('B')
    assert len(out) < len(body)


async def test_bash_tool_round_trips_through_executor() -> None:
    """The agent's bash call reaches the executor and its output comes back."""
    executor = RecordingExecutor(results={'ls /app': CommandResult(output='file.txt', exit_code=0)})
    deps = TerminalBenchDeps(execute=executor)

    seen_result: list[str] = []

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in getattr(message, 'parts', []):
                if isinstance(part, ToolReturnPart) and part.tool_name == 'bash':
                    seen_result.append(str(part.content))
        if not seen_result:
            return ModelResponse(parts=[ToolCallPart('bash', {'command': 'ls /app'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), deps_type=TerminalBenchDeps, toolsets=[build_bash_toolset()])
    result = await agent.run('list the files', deps=deps)

    assert executor.calls == ['ls /app']
    assert seen_result == ['file.txt']
    assert result.output == 'done'
