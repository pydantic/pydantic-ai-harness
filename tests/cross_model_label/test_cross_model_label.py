"""Tests for the `CrossModelHistoryLabel` capability.

Behavior is driven through `Agent(..., capabilities=[...])` with a `FunctionModel`. The serving
model's identity is controlled with `FunctionModel(model_name=...)`, which also stamps the run's
own responses with that name (so the model's own turns count as its own family). The foreign
history is supplied via `message_history` as scripted `ModelResponse`s carrying explicit
`model_name` (and `provider_name`) values.

The label reaches the model as request instructions, so each run records `AgentInfo.instructions`
(what the model actually received) to assert on. Runs are single-request (the model returns text):
`AgentInfo.instructions` falls back to the *previous* request's instructions on a tool-return-only
step, so a multi-step run would mask a correctly-silent later request. Multi-request behavior is
exercised instead by threading `result.all_messages()` into a second run. The stored message
history is inspected separately to prove the line is never persisted as a message part.
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.cross_model_label import (
    CrossModelHistory,
    CrossModelHistoryLabel,
    model_family,
    normalize_model_name,
)
from pydantic_ai_harness.cross_model_label._capability import _DEFAULT_FORMAT

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _response(model_name: str | None, *, provider: str | None = None, text: str = 'older') -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)], model_name=model_name, provider_name=provider)


def _history(*responses: ModelResponse) -> list[ModelMessage]:
    """A prior conversation: a user turn before each scripted response."""
    messages: list[ModelMessage] = []
    for i, response in enumerate(responses):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f'turn {i}')]))
        messages.append(response)
    return messages


def _build(
    *,
    model_name: str,
    capability: CrossModelHistoryLabel[None],
) -> tuple[Agent[None, str], list[str | None]]:
    """Agent serving as `model_name`, recording the instructions each request received.

    Each run is a single request that returns text, so `recorded[0]` is the instructions string
    the model received. Single-request runs avoid the `AgentInfo.instructions` tool-return
    fallback, which would surface a previous request's instructions on a tool-return-only step.
    Multi-request behavior is exercised by threading `result.all_messages()` into a second run.
    """
    recorded: list[str | None] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        recorded.append(info.instructions)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(fn, model_name=model_name), deps_type=type(None), capabilities=[capability])
    return agent, recorded


_EXPECTED_GPT = _DEFAULT_FORMAT.format(family='gpt-5.2')


# -- detection: fires and stays silent --


async def test_fires_when_history_is_a_different_family() -> None:
    """Continuing a gpt-written history under claude fires the label naming the gpt family."""
    agent, recorded = _build(
        model_name='claude-sonnet-4-5',
        capability=CrossModelHistoryLabel(),
    )
    await agent.run('continue', message_history=_history(_response('gpt-5.2')))
    assert recorded == [_EXPECTED_GPT]


async def test_fresh_history_is_silent() -> None:
    """No prior responses at all: nothing to compare, so no line."""
    agent, recorded = _build(model_name='claude-sonnet-4-5', capability=CrossModelHistoryLabel())
    await agent.run('start')
    assert recorded == [None]


async def test_same_family_smaller_sibling_is_silent() -> None:
    """A size-tier sibling (`gpt-5.2-mini` history under `gpt-5.2`) is the same family: silent."""
    agent, recorded = _build(model_name='gpt-5.2', capability=CrossModelHistoryLabel())
    await agent.run('continue', message_history=_history(_response('gpt-5.2-mini')))
    assert recorded == [None]


async def test_dated_snapshot_is_same_family() -> None:
    """A dated snapshot of the same model does not trigger under family granularity."""
    agent, recorded = _build(model_name='gpt-4o', capability=CrossModelHistoryLabel())
    await agent.run('continue', message_history=_history(_response('gpt-4o-2024-08-06')))
    assert recorded == [None]


async def test_missing_model_name_is_silent_not_guessed() -> None:
    """Prior responses without a `model_name` are skipped, not guessed: silent."""
    agent, recorded = _build(model_name='claude-sonnet-4-5', capability=CrossModelHistoryLabel())
    await agent.run('continue', message_history=_history(_response(None), _response(None)))
    assert recorded == [None]


async def test_provider_difference_alone_is_silent() -> None:
    """The same model family served by two providers is one family: silent."""
    agent, recorded = _build(model_name='gpt-5.2', capability=CrossModelHistoryLabel())
    await agent.run(
        'continue',
        message_history=_history(_response('gpt-5.2', provider='azure')),
    )
    assert recorded == [None]


# -- threshold semantics --


async def test_recent_goes_quiet_after_serving_model_answers() -> None:
    """`'recent'` is a one-shot handoff nudge: it fires, then goes quiet once the model's own
    turn is the most recent response.

    Driven as two sequential single-request runs (threading the history forward) rather than one
    tool-stepped run: `AgentInfo.instructions` falls back to the previous request's instructions
    on a tool-return-only request, which would mask the (correct) silent second request.
    """
    capability = CrossModelHistoryLabel()

    first_agent, first = _build(model_name='claude-sonnet-4-5', capability=capability)
    result = await first_agent.run('continue', message_history=_history(_response('gpt-5.2')))
    # First run: the most recent prior response is gpt -> fire.
    assert first == [_EXPECTED_GPT]

    second_agent, second = _build(model_name='claude-sonnet-4-5', capability=capability)
    await second_agent.run('again', message_history=result.all_messages())
    # Second run: the model's own answer (claude) from the first run is now the most recent
    # response -> quiet, even though the older gpt turn is still in history.
    assert second == [None]


async def test_recent_only_looks_at_the_immediately_preceding_response() -> None:
    """`'recent'` fires on the last response's family even if earlier ones match the current one."""
    agent, recorded = _build(model_name='claude-sonnet-4-5', capability=CrossModelHistoryLabel())
    history = _history(_response('claude-sonnet-4-5'), _response('gpt-5.2'))
    await agent.run('continue', message_history=history)
    assert recorded == [_EXPECTED_GPT]


async def test_recent_silent_when_last_response_matches_current() -> None:
    """`'recent'`: if the immediately preceding response matches the current family, stay silent
    even though an earlier response differs."""
    agent, recorded = _build(model_name='claude-sonnet-4-5', capability=CrossModelHistoryLabel())
    history = _history(_response('gpt-5.2'), _response('claude-sonnet-4-5'))
    await agent.run('continue', message_history=history)
    assert recorded == [None]


async def test_fraction_threshold_fires_when_majority_foreign() -> None:
    """A float threshold fires once that fraction of prior responses is a different family."""
    agent, recorded = _build(
        model_name='claude-sonnet-4-5',
        capability=CrossModelHistoryLabel(threshold=0.5),
    )
    # 2 of 3 prior responses are gpt (66% >= 50%), even though the most recent is claude.
    history = _history(_response('gpt-5.2'), _response('gpt-5.2'), _response('claude-sonnet-4-5'))
    await agent.run('continue', message_history=history)
    assert recorded == [_EXPECTED_GPT]


async def test_fraction_threshold_silent_below_fraction() -> None:
    """A float threshold stays silent while the foreign fraction is below it."""
    agent, recorded = _build(
        model_name='claude-sonnet-4-5',
        capability=CrossModelHistoryLabel(threshold=0.75),
    )
    # 2 of 3 prior responses are gpt (66% < 75%): silent.
    history = _history(_response('gpt-5.2'), _response('gpt-5.2'), _response('claude-sonnet-4-5'))
    await agent.run('continue', message_history=history)
    assert recorded == [None]


async def test_fraction_names_the_predominant_foreign_family() -> None:
    """With several foreign families, the fraction path names the most common one."""
    agent, recorded = _build(
        model_name='claude-sonnet-4-5',
        capability=CrossModelHistoryLabel(threshold=0.5),
    )
    history = _history(_response('gpt-5.2'), _response('gpt-5.2'), _response('gemini-2.5-pro'))
    await agent.run('continue', message_history=history)
    assert recorded == [_EXPECTED_GPT]


async def test_fraction_ties_break_to_most_recent_family() -> None:
    """When foreign families tie on count, the most recent one is named."""
    agent, recorded = _build(
        model_name='claude-sonnet-4-5',
        capability=CrossModelHistoryLabel(threshold=0.5),
    )
    history = _history(_response('gpt-5.2'), _response('gemini-2.5-pro'))
    await agent.run('continue', message_history=history)
    assert recorded == [_DEFAULT_FORMAT.format(family='gemini-2.5-pro')]


# -- granularity --


async def test_exact_granularity_fires_on_size_sibling() -> None:
    """Under `'exact'`, a size-tier sibling counts as a different model."""
    agent, recorded = _build(
        model_name='gpt-5.2',
        capability=CrossModelHistoryLabel(granularity='exact'),
    )
    await agent.run('continue', message_history=_history(_response('gpt-5.2-mini')))
    assert recorded == [_DEFAULT_FORMAT.format(family='gpt-5.2-mini')]


async def test_custom_resolver_granularity() -> None:
    """A callable `granularity` fully controls the comparison key."""

    def by_vendor(model_name: str, provider_name: str | None) -> str:
        return model_name.split('-')[0]

    agent, recorded = _build(
        model_name='gpt-5.2',
        capability=CrossModelHistoryLabel(granularity=by_vendor),
    )
    # Same vendor prefix `gpt` -> same key -> silent, despite different versions.
    await agent.run('continue', message_history=_history(_response('gpt-4o')))
    assert recorded == [None]

    other_agent, other = _build(
        model_name='gpt-5.2',
        capability=CrossModelHistoryLabel(granularity=by_vendor),
    )
    # Different vendor prefix -> fires, naming the resolver's key.
    await other_agent.run('continue', message_history=_history(_response('claude-sonnet-4-5')))
    assert other == [_DEFAULT_FORMAT.format(family='claude')]


# -- FallbackModel identity --


def _fallback_agent(
    *,
    capability: CrossModelHistoryLabel[None],
) -> tuple[Agent[None, str], list[str | None]]:
    """Agent served by a real `FallbackModel` (primary `gpt-5.2`), recording instructions.

    The primary always succeeds, so it serves every request; the point is that `ctx.model` is a
    `FallbackModel` whose composite wrapper name must not be used as the current identity.
    """
    recorded: list[str | None] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        recorded.append(info.instructions)
        return ModelResponse(parts=[TextPart('done')])

    primary = FunctionModel(fn, model_name='gpt-5.2')
    secondary = FunctionModel(fn, model_name='claude-sonnet-4-5')
    agent = Agent(
        FallbackModel(primary, secondary),
        deps_type=type(None),
        capabilities=[capability],
    )
    return agent, recorded


async def test_fallback_current_identity_from_most_recent_response() -> None:
    """Under a real `FallbackModel`, the current identity comes from the most recent response
    (core #6338), not the composite wrapper name, so a mid-history failover is labeled.

    History: an earlier `claude` turn, then a `gpt-5.2` turn (the failover target now serving).
    The earlier `claude` turn is the foreign one.
    """
    agent, recorded = _fallback_agent(capability=CrossModelHistoryLabel(threshold=0.5))
    history = _history(_response('claude-sonnet-4-5'), _response('gpt-5.2'))
    await agent.run('continue', message_history=history)
    assert recorded == [_DEFAULT_FORMAT.format(family='claude-sonnet-4-5')]


async def test_fallback_fresh_history_is_silent() -> None:
    """Under a `FallbackModel` with no prior responses, the identity falls back to the first
    candidate and there is nothing to compare: silent."""
    agent, recorded = _fallback_agent(capability=CrossModelHistoryLabel(threshold=0.5))
    await agent.run('start')
    assert recorded == [None]


# -- format override --


async def test_format_override_and_none() -> None:
    """A `format` callable replaces the line and may return None to disclose nothing."""

    def fmt(ctx: object, info: CrossModelHistory) -> str | None:
        if info.other_family == 'gpt-5.2':
            return None
        return f'PRIOR={info.other_family} n={info.differing_responses}/{info.known_responses}'

    silent_agent, silent = _build(
        model_name='claude-sonnet-4-5',
        capability=CrossModelHistoryLabel(format=fmt),
    )
    await silent_agent.run('continue', message_history=_history(_response('gpt-5.2')))
    assert silent == [None]

    named_agent, named = _build(
        model_name='claude-sonnet-4-5',
        capability=CrossModelHistoryLabel(format=fmt),
    )
    await named_agent.run('continue', message_history=_history(_response('gemini-2.5-pro')))
    assert named == ['PRIOR=gemini-2.5-pro n=1/1']


# -- cache safety --


async def test_line_is_never_a_message_part() -> None:
    """Cache/provenance safety: the line rides request instructions only, never a stored part."""
    agent, recorded = _build(model_name='claude-sonnet-4-5', capability=CrossModelHistoryLabel())
    result = await agent.run('continue', message_history=_history(_response('gpt-5.2')))

    # It did reach the model (as instructions)...
    assert recorded == [_EXPECTED_GPT]
    # ...but no persisted message part carries it, so it never enters the cacheable prefix.
    for message in result.all_messages():
        for part in message.parts:
            content = getattr(part, 'content', '')
            assert 'different model' not in str(content)


# -- statelessness --


async def test_reused_instance_judges_each_run_independently() -> None:
    """The capability is stateless, so one instance labels correctly across sequential runs."""
    capability = CrossModelHistoryLabel()

    foreign_agent, foreign = _build(model_name='claude-sonnet-4-5', capability=capability)
    await foreign_agent.run('continue', message_history=_history(_response('gpt-5.2')))

    native_agent, native = _build(model_name='gpt-5.2', capability=capability)
    await native_agent.run('continue', message_history=_history(_response('gpt-5.2')))

    assert foreign == [_EXPECTED_GPT]
    assert native == [None]


# -- validation and helpers --


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError, match='threshold must be'):
        CrossModelHistoryLabel(threshold=0.0)
    with pytest.raises(ValueError, match='threshold must be'):
        CrossModelHistoryLabel(threshold=1.5)


def test_model_family_helper() -> None:
    assert model_family('gpt-5.2-mini') == 'gpt-5.2'
    assert model_family('gpt-5.2') == 'gpt-5.2'
    assert model_family('gpt-4o-2024-08-06') == 'gpt-4o'
    assert model_family('claude-sonnet-4-5-20241022') == 'claude-sonnet-4-5'
    assert model_family('gpt-5.2-latest') == 'gpt-5.2'
    assert model_family('openai:gpt-5.2') == 'gpt-5.2'


def test_normalize_model_name_helper() -> None:
    assert normalize_model_name('OpenAI:GPT-5.2') == 'gpt-5.2'
    assert normalize_model_name('anthropic/claude-sonnet-4-5') == 'claude-sonnet-4-5'
    assert normalize_model_name('gpt-5.2') == 'gpt-5.2'
