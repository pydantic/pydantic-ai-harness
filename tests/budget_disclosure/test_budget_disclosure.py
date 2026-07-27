"""Tests for the `BudgetDisclosure` capability.

Behavior is driven through `Agent(..., capabilities=[...])` with a `FunctionModel`
that returns a preset `RequestUsage` per step, so the run's `ctx.usage` accrues
across steps. The disclosed line reaches the model as request instructions, so each
step records `AgentInfo.instructions` (the instructions the model actually received
that request) to assert on. The stored message history is inspected separately to
prove the line is never persisted as a message part.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from pydantic_ai_harness.budget_disclosure import BudgetDimension, BudgetDisclosure

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _usage(*, input_tokens: int = 0, output_tokens: int = 0) -> RequestUsage:
    return RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _build(
    usages: Sequence[RequestUsage],
    capability: BudgetDisclosure[None],
) -> tuple[Agent[None, str], list[str | None]]:
    """Agent that emits one preset-usage response per step, recording each request's instructions.

    Every response but the last returns a tool call so the run keeps stepping; the
    last returns text so the run finishes. `recorded[i]` is the instructions string
    the model received on request `i`.
    """
    recorded: list[str | None] = []
    state = {'i': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state['i']
        state['i'] += 1
        recorded.append(info.instructions)
        usage = usages[i]
        if i == len(usages) - 1:
            return ModelResponse(parts=[TextPart('done')], usage=usage)
        return ModelResponse(parts=[ToolCallPart('noop', {})], usage=usage)

    def noop() -> str:
        return 'ok'

    agent = Agent(FunctionModel(fn), deps_type=type(None), capabilities=[capability], tools=[noop])
    return agent, recorded


async def test_discloses_remaining_across_steps() -> None:
    """The line appears each request with remaining that tracks usage as it accrues."""
    limits = UsageLimits(request_limit=20, total_tokens_limit=200_000)
    usages = [
        _usage(input_tokens=1000, output_tokens=500),  # +1500 tokens after step 0
        _usage(input_tokens=2000, output_tokens=1500),  # +3500 tokens after step 1
        _usage(input_tokens=3000, output_tokens=2000),
    ]
    agent, recorded = _build(usages, BudgetDisclosure(limits=limits))
    result = await agent.run('hi', usage_limits=limits)
    assert result.output == 'done'

    # request 0: nothing consumed yet; request 1: 1 request / 1500 tokens; request 2: 2 / 5000.
    assert recorded == [
        'Budget remaining: ~200k tokens, 20 requests. Pace your work; if nearly exhausted, '
        'prioritize delivering current results over starting new work.',
        'Budget remaining: ~198k tokens, 19 requests. Pace your work; if nearly exhausted, '
        'prioritize delivering current results over starting new work.',
        'Budget remaining: ~195k tokens, 18 requests. Pace your work; if nearly exhausted, '
        'prioritize delivering current results over starting new work.',
    ]


async def test_no_limits_discloses_nothing() -> None:
    """With no `limits`, the capability contributes no instructions at all."""
    usages = [_usage(input_tokens=1000, output_tokens=500), _usage(input_tokens=1000)]
    agent, recorded = _build(usages, BudgetDisclosure())
    await agent.run('hi')
    assert recorded == [None, None]


async def test_line_is_never_a_message_part() -> None:
    """Cache safety: the line rides on request instructions only, never a stored message part."""
    limits = UsageLimits(request_limit=20, total_tokens_limit=200_000)
    usages = [_usage(input_tokens=1000, output_tokens=500), _usage(input_tokens=2000)]
    agent, recorded = _build(usages, BudgetDisclosure(limits=limits))
    result = await agent.run('hi', usage_limits=limits)

    # It did reach the model (as instructions)...
    assert all(line is not None and 'Budget remaining:' in line for line in recorded)

    # ...but no persisted message part carries it, so it never enters the cacheable prefix.
    for message in result.all_messages():
        for part in message.parts:
            content = getattr(part, 'content', '')
            assert 'Budget remaining' not in str(content)


async def test_start_at_gates_disclosure() -> None:
    """With `start_at`, disclosure stays silent until a disclosed limit is that fraction consumed."""
    limits = UsageLimits(request_limit=None, total_tokens_limit=10_000)
    usages = [
        _usage(input_tokens=3000),  # 30% consumed after step 0
        _usage(input_tokens=3000),  # 60% consumed after step 1
        _usage(),
    ]
    agent, recorded = _build(usages, BudgetDisclosure(limits=limits, start_at=0.5))
    await agent.run('hi', usage_limits=limits)

    assert recorded[0] is None  # 0% consumed
    assert recorded[1] is None  # 30% consumed, still below 50%
    assert recorded[2] is not None and '~4k tokens' in recorded[2]  # 60% consumed -> disclose


async def test_token_rounding_is_configurable() -> None:
    """Remaining tokens are rounded to `round_tokens_to` for display."""
    limits = UsageLimits(request_limit=None, total_tokens_limit=100_000)
    usages = [_usage(input_tokens=12_700), _usage()]  # 87_300 remaining at request 1

    agent_a, coarse = _build(usages, BudgetDisclosure(limits=limits))  # default 1000
    await agent_a.run('hi', usage_limits=limits)
    agent_b, coarser = _build(usages, BudgetDisclosure(limits=limits, round_tokens_to=10_000))
    await agent_b.run('hi', usage_limits=limits)

    assert coarse[1] is not None and '~87k tokens' in coarse[1]
    assert coarser[1] is not None and '~90k tokens' in coarser[1]


async def test_non_thousand_granularity_renders_raw_number() -> None:
    """A granularity that is not a multiple of 1000 renders the rounded number, not `Nk`."""
    limits = UsageLimits(request_limit=None, total_tokens_limit=10_000)
    usages = [_usage(input_tokens=8_700), _usage()]  # 1_300 remaining -> round to 1500 at step 500
    agent, recorded = _build(usages, BudgetDisclosure(limits=limits, round_tokens_to=500))
    await agent.run('hi', usage_limits=limits)
    assert recorded[1] is not None and '~1500 tokens' in recorded[1]


async def test_disclose_selects_dimensions() -> None:
    """`disclose` restricts the line to the named dimensions, from all set limits."""
    limits = UsageLimits(request_limit=20, total_tokens_limit=200_000)
    usages = [_usage(input_tokens=1000, output_tokens=500), _usage()]
    agent, recorded = _build(usages, BudgetDisclosure(limits=limits, disclose={'requests'}))
    await agent.run('hi', usage_limits=limits)
    assert recorded[0] == (
        'Budget remaining: 20 requests. Pace your work; if nearly exhausted, '
        'prioritize delivering current results over starting new work.'
    )
    assert 'tokens' not in recorded[0]


async def test_format_override() -> None:
    """A `format` callable replaces the default line and may return None to disclose nothing."""

    def fmt(ctx: object, remaining: Mapping[BudgetDimension, int]) -> str | None:
        left = remaining['total_tokens']
        return None if left > 90_000 else f'BUDGET {left}'

    limits = UsageLimits(request_limit=None, total_tokens_limit=100_000)
    usages = [_usage(input_tokens=15_000), _usage(input_tokens=10_000), _usage()]
    agent, recorded = _build(usages, BudgetDisclosure(limits=limits, format=fmt))
    await agent.run('hi', usage_limits=limits)

    assert recorded[0] is None  # 100_000 remaining > 90_000, format returns None
    assert recorded[1] == 'BUDGET 85000'  # 85_000 remaining
    assert recorded[2] == 'BUDGET 75000'  # 75_000 remaining


async def test_reused_instance_judges_each_run_independently() -> None:
    """The capability is stateless, so one instance discloses correctly across sequential runs."""
    limits = UsageLimits(request_limit=None, total_tokens_limit=100_000)
    capability = BudgetDisclosure(limits=limits)

    first_agent, first = _build([_usage(input_tokens=40_000), _usage()], capability)
    await first_agent.run('one', usage_limits=limits)
    second_agent, second = _build([_usage(input_tokens=10_000), _usage()], capability)
    await second_agent.run('two', usage_limits=limits)

    # Each run starts from a fresh budget; no high-water mark leaks between runs.
    assert first[0] is not None and '~100k tokens' in first[0]
    assert first[1] is not None and '~60k tokens' in first[1]
    assert second[0] is not None and '~100k tokens' in second[0]
    assert second[1] is not None and '~90k tokens' in second[1]


def test_invalid_start_at_rejected() -> None:
    with pytest.raises(ValueError, match='start_at must be between'):
        BudgetDisclosure(limits=UsageLimits(), start_at=1.5)


def test_invalid_round_tokens_to_rejected() -> None:
    with pytest.raises(ValueError, match='round_tokens_to must be positive'):
        BudgetDisclosure(limits=UsageLimits(), round_tokens_to=0)


def test_empty_disclose_rejected() -> None:
    with pytest.raises(ValueError, match='disclose must not be empty'):
        BudgetDisclosure(limits=UsageLimits(), disclose=set())


def test_unknown_disclose_dimension_rejected() -> None:
    with pytest.raises(ValueError, match='unknown dimensions'):
        BudgetDisclosure(limits=UsageLimits(), disclose={'cost'})  # type: ignore[arg-type]


def test_disclose_requires_configured_limit() -> None:
    limits = UsageLimits(request_limit=20, total_tokens_limit=None)
    with pytest.raises(ValueError, match='limit is not set'):
        BudgetDisclosure(limits=limits, disclose={'total_tokens'})


def test_disclose_with_no_limits_is_allowed_and_silent() -> None:
    """`disclose` names dimensions but `limits` is None: no limit to validate, discloses nothing."""
    capability = BudgetDisclosure[None](disclose={'requests'})
    assert capability.get_instructions() is None


def test_zero_limit_does_not_divide_by_zero() -> None:
    """A zero-valued limit is fully consumed by definition and must not raise `ZeroDivisionError`."""
    limits = UsageLimits(request_limit=None, total_tokens_limit=0)
    capability = BudgetDisclosure[None](limits=limits, disclose={'total_tokens'})
    remaining = capability._remaining(RunUsage(), limits)  # type: ignore[arg-type]
    assert remaining == {'total_tokens': 0}
