"""Tests for the scripted smoke model's state machine."""

from __future__ import annotations

from typing import cast

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from pydantic_ai_harness_terminal_bench.smoke import (
    SMOKE_MARKER_PATH,
    SMOKE_TEXT,
    _scripted_turn,
)

# `_scripted_turn` ignores `info`; a cast sentinel avoids building the full model.
_INFO = cast(AgentInfo, None)


def _tool_return(content: str) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name='bash', content=content, tool_call_id='x')])


async def test_step_0_writes_marker() -> None:
    response = await _scripted_turn([], _INFO)
    call = response.parts[0]
    assert isinstance(call, ToolCallPart)
    assert call.args == {'command': f'echo {SMOKE_TEXT} > {SMOKE_MARKER_PATH}'}


async def test_step_1_reads_marker_back() -> None:
    response = await _scripted_turn([_tool_return('')], _INFO)
    call = response.parts[0]
    assert isinstance(call, ToolCallPart)
    assert call.args == {'command': f'cat {SMOKE_MARKER_PATH}'}


async def test_step_2_finishes_with_text() -> None:
    response = await _scripted_turn([_tool_return(''), _tool_return(SMOKE_TEXT)], _INFO)
    assert isinstance(response.parts[0], TextPart)


def test_scripted_turn_ignores_non_request_messages() -> None:
    # A ModelResponse in history must not be counted as a tool return.
    from pydantic_ai_harness_terminal_bench.smoke import _count_tool_returns

    history = [ModelResponse(parts=[TextPart('hi')]), _tool_return('')]
    assert _count_tool_returns(history) == 1
