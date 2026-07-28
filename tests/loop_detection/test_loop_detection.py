"""Tests for the `LoopDetection` capability.

Behavior is driven through `Agent(..., capabilities=[LoopDetection()])` with a `FunctionModel`
that emits a scripted sequence of tool calls or text, exercising each detection tier the way a
stuck agent would. The repo runs pytest with `filterwarnings=['error']`, so the tests assert
detection through the nudge message reaching the next request, a raised error, or a callback,
rather than through warnings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.loop_detection import (
    LoopDetected,
    LoopDetectedError,
    LoopDetection,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


ModelFunc = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def _scripted(responses: list[ModelResponse]) -> ModelFunc:
    """A model function that returns `responses` in order, repeating the last one forever."""
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        return responses[min(i, len(responses) - 1)]

    return fn


def _repeated_call(tool: str, n: int, *, args: dict[str, Any] | None = None) -> ModelFunc:
    """Emit `tool` (with `args`) `n` times, then a final text response so the run ends."""
    call = ModelResponse(parts=[ToolCallPart(tool, args if args is not None else {})])
    done = ModelResponse(parts=[TextPart('done')])
    return _scripted([call] * n + [done])


def _build_agent(fn: ModelFunc, cap: LoopDetection[None]) -> Agent[None, str]:
    """An agent whose tools all succeed, so tool-call tiers fire on the calls themselves."""
    agent = Agent(FunctionModel(fn), deps_type=type(None), capabilities=[cap])

    @agent.tool_plain
    def alpha() -> str:
        return 'a'

    @agent.tool_plain
    def beta() -> str:
        return 'b'

    return agent


def _nudges(result_messages: list[ModelMessage]) -> list[str]:
    """The harness-marked nudge texts injected into the conversation."""
    return [
        part.content
        for message in result_messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and '[harness:loop-detection]' in part.content
    ]


async def test_exact_repetition_nudges_reach_next_request() -> None:
    """Tier 1: the same call five times nudges the model, and the nudge lands in history."""
    agent = _build_agent(_repeated_call('alpha', 5), LoopDetection(error_cycle_threshold=99))
    result = await agent.run('go')
    nudges = _nudges(result.all_messages())
    assert len(nudges) == 1
    assert 'called 5 times with identical arguments' in nudges[0]


async def test_nudge_carries_structured_harness_marker() -> None:
    """The nudge persists as a `UserPromptPart`, so it must lead with an unambiguous harness
    marker naming the framework and capability -- not read as a real user turn."""
    agent = _build_agent(_repeated_call('alpha', 5), LoopDetection(error_cycle_threshold=99))
    result = await agent.run('go')
    nudges = _nudges(result.all_messages())
    assert len(nudges) == 1
    assert nudges[0].startswith('[harness:loop-detection] ')


async def test_error_cycle_nudges() -> None:
    """Tier 2a: a tool that returns the same result three times running nudges the model."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = sum(1 for m in messages for p in m.parts if p.part_kind == 'tool-return')
        if returns >= 3:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('stuck', {})])

    agent = Agent(FunctionModel(fn), deps_type=type(None), capabilities=[LoopDetection()])

    @agent.tool_plain
    def stuck() -> str:
        return 'ERROR: still blocked'

    result = await agent.run('go')
    nudges = _nudges(result.all_messages())
    assert any('returned the same result 3 times in a row' in n for n in nudges)


async def test_alternation_nudges() -> None:
    """Tier 2b: two tools alternating A-B-A-B-A-B (three cycles) nudge the model."""
    a = ModelResponse(parts=[ToolCallPart('alpha', {})])
    b = ModelResponse(parts=[ToolCallPart('beta', {})])
    done = ModelResponse(parts=[TextPart('done')])
    agent = _build_agent(_scripted([a, b, a, b, a, b, done]), LoopDetection())
    result = await agent.run('go')
    nudges = _nudges(result.all_messages())
    assert any('alternating between `alpha` and `beta`' in n for n in nudges)


async def test_monologue_raises_in_error_mode() -> None:
    """Tier 2c: three near-identical text-only responses are a monologue.

    A structured output type keeps the run going after each text-only response (Pydantic AI
    asks the model to call the output tool), so the monologue can accumulate.
    """

    class Answer(BaseModel):
        value: str

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('I am still thinking it over.')])

    agent = Agent(
        FunctionModel(fn),
        deps_type=type(None),
        output_type=Answer,
        retries={'output': 10},
        capabilities=[LoopDetection(on_loop='error')],
    )
    with pytest.raises(LoopDetectedError) as exc:
        await agent.run('go')
    assert exc.value.detected.tier == 'monologue'
    assert exc.value.detected.count == 3
    assert exc.value.detected.tool_name is None


async def test_below_threshold_is_silent() -> None:
    """Four identical calls (default threshold is five) never nudge."""
    agent = _build_agent(_repeated_call('alpha', 4), LoopDetection(error_cycle_threshold=99))
    result = await agent.run('go')
    assert _nudges(result.all_messages()) == []


async def test_broken_alternation_is_silent() -> None:
    """A near-alternation whose tail breaks the A-B pattern does not fire tier 2b."""
    a = ModelResponse(parts=[ToolCallPart('alpha', {})])
    b = ModelResponse(parts=[ToolCallPart('beta', {})])
    done = ModelResponse(parts=[TextPart('done')])
    # alpha beta alpha beta beta beta -- six calls, not a clean A-B-A-B-A-B.
    agent = _build_agent(_scripted([a, b, a, b, b, b, done]), LoopDetection(error_cycle_threshold=99))
    result = await agent.run('go')
    assert _nudges(result.all_messages()) == []


async def test_repeated_calls_are_not_alternation() -> None:
    """A long run of one identical call fills the alternation window but is not an A-B pattern."""
    cap = LoopDetection(repeat_threshold=99, error_cycle_threshold=99)
    agent = _build_agent(_repeated_call('alpha', 6), cap)
    result = await agent.run('go')
    assert _nudges(result.all_messages()) == []


async def test_empty_response_is_not_a_monologue() -> None:
    """A whitespace-only response normalizes to empty text and never counts as a monologue."""
    agent = _build_agent(_scripted([ModelResponse(parts=[TextPart('   ')])]), LoopDetection())
    result = await agent.run('go')
    assert _nudges(result.all_messages()) == []
    assert result.output == '   '


async def test_error_mode_raises_on_repetition() -> None:
    """Tier 1 under `on_loop='error'` aborts the run with a structured `LoopDetectedError`."""
    agent = _build_agent(_repeated_call('alpha', 5), LoopDetection(on_loop='error', error_cycle_threshold=99))
    with pytest.raises(LoopDetectedError) as exc:
        await agent.run('go')
    assert exc.value.detected.tier == 'exact_repetition'
    assert exc.value.detected.tool_name == 'alpha'
    assert exc.value.detected.count == 5


async def test_callable_receives_structured_detection() -> None:
    """A sync `on_loop` callable is handed the `LoopDetected` and can inspect every field."""
    seen: list[LoopDetected] = []

    def on_loop(detected: LoopDetected) -> None:
        seen.append(detected)

    agent = _build_agent(_repeated_call('alpha', 5), LoopDetection(on_loop=on_loop, error_cycle_threshold=99))
    await agent.run('go')
    assert [d.tier for d in seen] == ['exact_repetition']
    assert seen[0].tool_name == 'alpha'
    assert seen[0].fingerprints == ('alpha({})',)


async def test_async_callable_is_awaited() -> None:
    """An async `on_loop` callable is awaited on detection."""
    seen: list[str] = []

    async def on_loop(detected: LoopDetected) -> None:
        seen.append(detected.tier)

    agent = _build_agent(_repeated_call('alpha', 5), LoopDetection(on_loop=on_loop, error_cycle_threshold=99))
    await agent.run('go')
    assert seen == ['exact_repetition']


async def test_per_run_state_is_isolated() -> None:
    """Reusing one capability across runs judges each run independently."""
    cap = LoopDetection(error_cycle_threshold=99)

    looping = _build_agent(_repeated_call('alpha', 5), cap)
    first = await looping.run('first')
    assert len(_nudges(first.all_messages())) == 1

    # A second, short run must not inherit the first run's call window.
    short = _build_agent(_repeated_call('alpha', 2), cap)
    second = await short.run('second')
    assert _nudges(second.all_messages()) == []


async def test_argument_fingerprints_are_canonical() -> None:
    """Different argument encodings of the same call share a fingerprint; the run stays silent.

    Exercises the argument canonicalization across dict, JSON-string, `None`, and non-JSON
    string forms without reaching the repeat threshold.
    """
    calls = [
        ModelResponse(parts=[ToolCallPart('gamma', {'b': 2, 'a': 1})]),
        ModelResponse(parts=[ToolCallPart('gamma', '{"a": 1, "b": 2}')]),
        ModelResponse(parts=[ToolCallPart('gamma', None)]),
        ModelResponse(parts=[ToolCallPart('gamma', 'not-json')]),
        ModelResponse(parts=[TextPart('done')]),
    ]
    agent = Agent(
        FunctionModel(_scripted(calls)),
        deps_type=type(None),
        retries={'tools': 5},
        capabilities=[LoopDetection()],
    )

    @agent.tool_plain
    def gamma(a: int = 0, b: int = 0) -> str:
        return f'{a}-{b}'

    result = await agent.run('go')
    assert _nudges(result.all_messages()) == []
