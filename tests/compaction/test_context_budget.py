"""Tests for window-relative triggers, usage reporting, and out-of-run compaction."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.compaction import (
    ClearToolResults,
    ContextUsage,
    ContextUsageMonitor,
    DeduplicateFileReads,
    LimitWarner,
    SlidingWindow,
    SummarizingCompaction,
    SupportsFocus,
    TieredCompaction,
    compact_now,
    resolve_context_window,
)
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW, split_model_id
from pydantic_ai_harness.compaction._shared import resolve_token_trigger, validate_token_trigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeModel:
    """Stands in for a `Model`; only `model_id` is read by window resolution."""

    model_id: str = 'anthropic:claude-sonnet-4-6'


def _ctx(model: Any = None) -> Any:
    """Minimal `RunContext`-like object for driving hooks."""

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage = dataclasses.field(default_factory=RunUsage)
        model: Model = dataclasses.field(default_factory=TestModel)
        deps: None = None
        tracer: Tracer = dataclasses.field(default_factory=NoOpTracer)

    return _FakeCtx(model=model) if model is not None else _FakeCtx()


def _request_context(messages: list[ModelMessage]) -> ModelRequestContext:
    return ModelRequestContext(
        model=_FakeModel(),  # type: ignore[arg-type]
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _history(turns: int, filler: str = 'x' * 400) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for index in range(turns):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f'{index} {filler}')]))
        messages.append(ModelResponse(parts=[TextPart(content=f'reply {index} {filler}')]))
    return messages


def _fixed_window(monkeypatch: pytest.MonkeyPatch, module: str, window: int | None) -> None:
    """Pin window resolution so a test asserts behaviour, not the pricing registry."""

    def _resolve(_model: Any) -> int | None:
        return window

    monkeypatch.setattr(f'pydantic_ai_harness.compaction.{module}.resolve_context_window', _resolve)


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------


class TestSplitModelId:
    def test_provider_prefixed(self):
        assert split_model_id('anthropic:claude-sonnet-4-6') == ('anthropic', 'claude-sonnet-4-6')

    def test_bare_name_leaves_provider_unset(self):
        assert split_model_id('claude-sonnet-4-6') == (None, 'claude-sonnet-4-6')


class TestResolveContextWindow:
    def test_known_model(self):
        window = resolve_context_window('anthropic:claude-sonnet-4-6')
        assert window is not None and window >= 100_000

    def test_accepts_a_model_instance(self):
        window = resolve_context_window(_FakeModel())  # type: ignore[arg-type]
        assert window is not None and window >= 100_000

    def test_unknown_model(self):
        assert resolve_context_window('anthropic:definitely-not-a-model-xyz') is None

    def test_model_without_an_id(self):
        assert resolve_context_window(_FakeModel(model_id='')) is None  # type: ignore[arg-type]

    def test_known_model_without_a_recorded_window(self, monkeypatch: pytest.MonkeyPatch):
        _patch_snapshot(monkeypatch, context_window=None)
        assert resolve_context_window('openai:gpt-4.1') is None

    def test_non_positive_window_is_treated_as_absent(self, monkeypatch: pytest.MonkeyPatch):
        _patch_snapshot(monkeypatch, context_window=0)
        assert resolve_context_window('openai:gpt-4.1') is None


def _patch_snapshot(monkeypatch: pytest.MonkeyPatch, *, context_window: int | None) -> None:
    """Replace the pricing snapshot with one returning a fixed window."""

    class _Info:
        def __init__(self) -> None:
            self.context_window = context_window

    class _Snapshot:
        def find_provider_model(self, **_: Any) -> tuple[object, _Info]:
            return object(), _Info()

    monkeypatch.setattr('genai_prices.data_snapshot.get_snapshot', lambda: _Snapshot())


# ---------------------------------------------------------------------------
# Trigger helpers
# ---------------------------------------------------------------------------


class TestValidateTokenTrigger:
    def test_rejects_both(self):
        with pytest.raises(ValueError, match='Set at most one of max_tokens or max_fraction'):
            validate_token_trigger(1_000, 0.5)

    def test_rejects_non_positive_tokens(self):
        with pytest.raises(ValueError, match='max_tokens must be positive'):
            validate_token_trigger(0, None)

    def test_rejects_fraction_above_one(self):
        with pytest.raises(ValueError, match='max_fraction must be greater than 0 and at most 1'):
            validate_token_trigger(None, 1.5)

    def test_rejects_zero_fraction(self):
        with pytest.raises(ValueError, match='max_fraction must be greater than 0 and at most 1'):
            validate_token_trigger(None, 0.0)

    def test_accepts_neither(self):
        validate_token_trigger(None, None)

    def test_reports_the_caller_field_names(self):
        with pytest.raises(ValueError, match='Set at most one of target_tokens or target_fraction'):
            validate_token_trigger(1, 0.5, tokens_name='target_tokens', fraction_name='target_fraction')


class TestResolveTokenTrigger:
    def test_absolute_wins(self):
        assert resolve_token_trigger(1_234, None, _FakeModel()) == 1_234  # type: ignore[arg-type]

    def test_no_trigger_configured(self):
        assert resolve_token_trigger(None, None, _FakeModel()) is None  # type: ignore[arg-type]

    def test_fraction_of_the_resolved_window(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000_000)
        assert resolve_token_trigger(None, 0.9, _FakeModel()) == 900_000  # type: ignore[arg-type]

    def test_fraction_falls_back_when_unresolved(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', None)
        assert resolve_token_trigger(None, 0.5, _FakeModel()) == DEFAULT_CONTEXT_WINDOW // 2  # type: ignore[arg-type]

    def test_fraction_without_a_model(self):
        assert resolve_token_trigger(None, 0.5, None) == DEFAULT_CONTEXT_WINDOW // 2

    def test_never_resolves_below_one(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1)
        assert resolve_token_trigger(None, 0.001, _FakeModel()) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Strategies driven by a fraction
# ---------------------------------------------------------------------------


class TestFractionTriggers:
    async def test_sliding_window_trims_when_over_the_fraction(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000)
        capability: SlidingWindow[None] = SlidingWindow(max_fraction=0.1, keep_messages=2)
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) < 12

    async def test_sliding_window_leaves_a_small_history_alone(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000_000)
        capability: SlidingWindow[None] = SlidingWindow(max_fraction=0.9, keep_messages=2)
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 12

    async def test_the_same_fraction_scales_with_the_window(self, monkeypatch: pytest.MonkeyPatch):
        """One configuration, two models: the trigger follows the window."""
        capability: SlidingWindow[None] = SlidingWindow(max_fraction=0.5, keep_messages=2)

        _fixed_window(monkeypatch, '_shared', 100)
        small = _request_context(_history(6))
        await capability.before_model_request(_ctx(), small)

        _fixed_window(monkeypatch, '_shared', 10_000_000)
        large = _request_context(_history(6))
        await capability.before_model_request(_ctx(), large)

        assert len(small.messages) < len(large.messages)

    async def test_clear_tool_results_accepts_a_fraction(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000_000)
        capability: ClearToolResults[None] = ClearToolResults(max_fraction=0.9)
        request_context = _request_context(_history(2))

        assert await capability.before_model_request(_ctx(), request_context) is request_context

    async def test_deduplicate_file_reads_gates_on_a_fraction(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000_000)
        capability: DeduplicateFileReads[None] = DeduplicateFileReads(file_key=lambda call: None, max_fraction=0.9)
        request_context = _request_context(_history(2))

        assert await capability.before_model_request(_ctx(), request_context) is request_context

    def test_summarizing_compaction_accepts_a_fraction_alone(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        assert capability.max_fraction == 0.9

    def test_a_fraction_alone_satisfies_the_trigger_requirement(self):
        capability: SlidingWindow[None] = SlidingWindow(max_fraction=0.9)
        assert capability.max_tokens is None

    def test_rejects_both_triggers(self):
        with pytest.raises(ValueError, match='Set at most one of max_tokens or max_fraction'):
            SlidingWindow(max_tokens=1_000, max_fraction=0.9)


class TestTieredTargetFraction:
    async def test_escalates_against_a_resolved_target(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000)
        capability: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindow(max_tokens=1, keep_messages=2)],
            target_fraction=0.1,
        )
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) < 12

    async def test_leaves_a_history_under_target_alone(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000_000)
        capability: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindow(max_tokens=1, keep_messages=2)],
            target_fraction=0.9,
        )
        request_context = _request_context(_history(2))

        assert await capability.before_model_request(_ctx(), request_context) is request_context

    def test_requires_one_target(self):
        with pytest.raises(ValueError, match='One of target_tokens or target_fraction must be set'):
            TieredCompaction(tiers=[SlidingWindow(max_tokens=1)])

    def test_rejects_both_targets(self):
        with pytest.raises(ValueError, match='Set at most one of target_tokens or target_fraction'):
            TieredCompaction(tiers=[SlidingWindow(max_tokens=1)], target_tokens=100, target_fraction=0.5)


class TestLimitWarnerFraction:
    async def test_warns_against_the_resolved_window(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 1_000)
        capability: LimitWarner[None] = LimitWarner(max_context_fraction=0.1, warning_threshold=0.5)
        request_context = _request_context(_history(6))

        await capability.before_model_request(_ctx(), request_context)

        warning = request_context.messages[-1]
        assert isinstance(warning, ModelRequest)
        assert '[LimitWarner]' in str(warning.parts[0].content)  # type: ignore[union-attr]

    async def test_stays_quiet_below_the_threshold(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_shared', 10_000_000)
        capability: LimitWarner[None] = LimitWarner(max_context_fraction=0.9)
        request_context = _request_context(_history(2))

        await capability.before_model_request(_ctx(), request_context)

        assert len(request_context.messages) == 4

    def test_a_fraction_alone_satisfies_the_limit_requirement(self):
        capability: LimitWarner[None] = LimitWarner(max_context_fraction=0.9)
        assert 'context_window' in capability._active_kinds

    def test_rejects_both_context_limits(self):
        with pytest.raises(ValueError, match='Set at most one of max_context_tokens or max_context_fraction'):
            LimitWarner(max_context_tokens=1_000, max_context_fraction=0.9)

    def test_warn_on_accepts_a_fraction_configured_kind(self):
        capability: LimitWarner[None] = LimitWarner(max_context_fraction=0.9, warn_on=['context_window'])
        assert capability._active_kinds == ('context_window',)

    def test_requires_at_least_one_limit(self):
        with pytest.raises(ValueError, match='At least one of max_iterations, max_context_tokens'):
            LimitWarner()


# ---------------------------------------------------------------------------
# Usage reporting
# ---------------------------------------------------------------------------


class TestContextUsage:
    def test_fraction_is_used_over_window(self):
        assert ContextUsage(used_tokens=250, window_tokens=1_000, resolved=True).fraction == 0.25


class TestContextUsageMonitor:
    async def test_reports_before_each_request(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_context_usage', 1_000)
        seen: list[ContextUsage] = []
        monitor: ContextUsageMonitor[None] = ContextUsageMonitor(on_usage=seen.append)
        run = await monitor.for_run(_ctx())

        request_context = _request_context(_history(2))
        assert await run.before_model_request(_ctx(), request_context) is request_context

        assert len(seen) == 1
        assert seen[0].window_tokens == 1_000
        assert seen[0].used_tokens > 0
        assert seen[0].resolved is True

    async def test_never_edits_the_history(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_context_usage', 1_000)
        monitor: ContextUsageMonitor[None] = ContextUsageMonitor(on_usage=lambda _: None)
        run = await monitor.for_run(_ctx())
        messages = _history(3)
        request_context = _request_context(messages)

        await run.before_model_request(_ctx(), request_context)

        assert request_context.messages == messages

    async def test_override_skips_resolution(self):
        monitor: ContextUsageMonitor[None] = ContextUsageMonitor(on_usage=lambda _: None, context_window=4_242)
        run = await monitor.for_run(_ctx())
        assert run._window == 4_242
        assert run._resolved is True

    async def test_falls_back_for_an_unknown_model(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_context_usage', None)
        monitor: ContextUsageMonitor[None] = ContextUsageMonitor(on_usage=lambda _: None)
        run = await monitor.for_run(_ctx())
        assert run._window == DEFAULT_CONTEXT_WINDOW
        assert run._resolved is False

    async def test_custom_fallback(self, monkeypatch: pytest.MonkeyPatch):
        _fixed_window(monkeypatch, '_context_usage', None)
        monitor: ContextUsageMonitor[None] = ContextUsageMonitor(
            on_usage=lambda _: None, fallback_context_window=32_000
        )
        run = await monitor.for_run(_ctx())
        assert run._window == 32_000

    async def test_each_run_resolves_independently(self, monkeypatch: pytest.MonkeyPatch):
        monitor: ContextUsageMonitor[None] = ContextUsageMonitor(on_usage=lambda _: None)

        _fixed_window(monkeypatch, '_context_usage', 1_000)
        first = await monitor.for_run(_ctx())
        _fixed_window(monkeypatch, '_context_usage', 2_000)
        second = await monitor.for_run(_ctx())

        assert (first._window, second._window) == (1_000, 2_000)

    def test_rejects_a_non_positive_window(self):
        with pytest.raises(ValueError, match='context_window must be positive'):
            ContextUsageMonitor(on_usage=lambda _: None, context_window=0)

    def test_rejects_a_non_positive_fallback(self):
        with pytest.raises(ValueError, match='fallback_context_window must be positive'):
            ContextUsageMonitor(on_usage=lambda _: None, fallback_context_window=0)


# ---------------------------------------------------------------------------
# Out-of-run compaction
# ---------------------------------------------------------------------------


class _RecordingStrategy:
    """Captures what it was handed, so the throwaway context can be inspected."""

    def __init__(self) -> None:
        self.seen_model: Any = None
        self.seen_deps: Any = None
        self.seen_usage: RunUsage | None = None

    async def compact(self, messages: list[ModelMessage], ctx: Any) -> list[ModelMessage]:
        self.seen_model = ctx.model
        self.seen_deps = ctx.deps
        self.seen_usage = ctx.usage
        return messages[:1]


class TestCompactNow:
    async def test_runs_without_an_agent_run(self):
        strategy = _RecordingStrategy()
        messages = _history(3)

        result = await compact_now(strategy, messages, model=TestModel())  # type: ignore[arg-type]

        assert len(result) == 1
        assert isinstance(strategy.seen_model, TestModel)

    async def test_resolves_a_model_name(self):
        strategy = _RecordingStrategy()

        await compact_now(strategy, _history(1), model='test')  # type: ignore[arg-type]

        assert strategy.seen_model is not None

    async def test_passes_deps_and_usage_through(self):
        strategy = _RecordingStrategy()
        usage = RunUsage(requests=3)

        await compact_now(strategy, _history(1), model=TestModel(), deps='my-deps', usage=usage)  # type: ignore[arg-type]

        assert strategy.seen_deps == 'my-deps'
        assert strategy.seen_usage is usage

    async def test_focus_is_ignored_by_a_strategy_that_cannot_honour_it(self):
        strategy = _RecordingStrategy()

        result = await compact_now(strategy, _history(2), model=TestModel(), focus='auth')  # type: ignore[arg-type]

        assert len(result) == 1

    async def test_focus_steers_a_summarizing_strategy(self):
        seen: list[str] = []

        class _Focusable(_RecordingStrategy):
            def with_focus(self, focus: str) -> _Focusable:
                seen.append(focus)
                return self

        await compact_now(_Focusable(), _history(1), model=TestModel(), focus='the auth refactor')  # type: ignore[arg-type]

        assert seen == ['the auth refactor']


class TestWithFocus:
    def test_appends_the_focus_to_the_prompt(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        focused = capability.with_focus('the auth refactor')
        assert 'the auth refactor' in focused.summary_prompt

    def test_leaves_the_original_untouched(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        original = capability.summary_prompt
        capability.with_focus('anything')
        assert capability.summary_prompt == original

    def test_braces_survive_prompt_formatting(self):
        capability: SummarizingCompaction[None] = SummarizingCompaction(max_fraction=0.9)
        focused = capability.with_focus('the {config} dict')
        assert 'the {config} dict' in focused.summary_prompt.format(messages='...')


class TestNewExports:
    def test_exposed_under_the_submodule_only(self):
        import pydantic_ai_harness
        import pydantic_ai_harness.compaction as compaction

        for name in ('ContextUsage', 'ContextUsageMonitor', 'compact_now', 'resolve_context_window'):
            assert hasattr(compaction, name)
            assert not hasattr(pydantic_ai_harness, name)


class TestPositionalCompatibility:
    """The new fields are keyword-only, so they cannot shift an existing positional argument.

    Every strategy here is a plain dataclass whose own fields are positional, so inserting a
    field next to a related one would rebind every argument after it in existing call sites.
    """

    def test_sliding_window(self):
        assert SlidingWindow(None, 1_000, 40).keep_messages == 40

    def test_summarizing_compaction(self):
        assert SummarizingCompaction('openai:gpt-4o', None, 1_000, 15).keep_messages == 15

    def test_clear_tool_results(self):
        assert ClearToolResults(None, 1_000, 5).keep_pairs == 5

    def test_deduplicate_file_reads(self):
        strategy = DeduplicateFileReads(lambda call: None, '[superseded]', None, 1_000)
        assert strategy.max_tokens == 1_000

    def test_tiered_compaction(self):
        strategy = TieredCompaction([SlidingWindow(max_tokens=1)], 100, len)
        assert strategy.tokenizer is len

    def test_limit_warner(self):
        assert LimitWarner(10, 1_000, 2_000).max_total_tokens == 2_000

    def test_the_new_fields_stay_keyword_only(self):
        import dataclasses

        cases = [
            (SlidingWindow, 'max_fraction'),
            (SummarizingCompaction, 'max_fraction'),
            (ClearToolResults, 'max_fraction'),
            (DeduplicateFileReads, 'max_fraction'),
            (TieredCompaction, 'target_fraction'),
            (LimitWarner, 'max_context_fraction'),
        ]
        for cls, name in cases:
            field = next(f for f in dataclasses.fields(cls) if f.name == name)
            assert field.kw_only, f'{cls.__name__}.{name} must stay keyword-only'


class TestFocusPropagation:
    """A composing strategy has to pass the focus down to the tier that writes the summary."""

    def _tiered(self) -> TieredCompaction[None]:
        return TieredCompaction(
            tiers=[SlidingWindow(max_tokens=1, keep_messages=2), SummarizingCompaction(max_messages=1)],
            target_tokens=100,
        )

    def test_tiered_is_focusable(self):
        assert isinstance(self._tiered(), SupportsFocus)

    def test_focus_reaches_the_summarizing_tier(self):
        focused = self._tiered().with_focus('the auth flow')
        summarizer = focused.tiers[1]
        assert isinstance(summarizer, SummarizingCompaction)
        assert 'the auth flow' in summarizer.summary_prompt

    def test_tiers_that_cannot_be_focused_pass_through(self):
        tiered = self._tiered()
        assert tiered.with_focus('anything').tiers[0] is tiered.tiers[0]

    def test_the_original_is_left_alone(self):
        tiered = self._tiered()
        tiered.with_focus('the auth flow')
        summarizer = tiered.tiers[1]
        assert isinstance(summarizer, SummarizingCompaction)
        assert 'the auth flow' not in summarizer.summary_prompt

    def test_nesting_propagates_all_the_way_down(self):
        outer = TieredCompaction(tiers=[self._tiered()], target_tokens=100)
        focused = outer.with_focus('nested topic')
        inner = focused.tiers[0]
        assert isinstance(inner, TieredCompaction)
        summarizer = inner.tiers[1]
        assert isinstance(summarizer, SummarizingCompaction)
        assert 'nested topic' in summarizer.summary_prompt

    async def test_compact_now_focuses_through_a_tiered_strategy(self):
        seen: list[str] = []

        class _Tier:
            def with_focus(self, focus: str) -> _Tier:
                seen.append(focus)
                return self

            async def compact(self, messages: list[ModelMessage], ctx: Any) -> list[ModelMessage]:
                return messages[:1]

        tiered = TieredCompaction(tiers=[_Tier()], target_tokens=1)
        await compact_now(tiered, _history(4), model=TestModel(), focus='the auth flow')

        assert seen == ['the auth flow']


class TestManualCompactionSemantics:
    """`compact_now` adds no trigger; a strategy's own stop condition still applies."""

    def _short_history(self) -> list[ModelMessage]:
        return _history(3, filler='x' * 40)

    async def test_an_unconditional_strategy_runs_whatever_the_size(self):
        history = self._short_history()
        strategy: SlidingWindow[None] = SlidingWindow(max_tokens=1_000_000, keep_messages=2)

        result = await compact_now(strategy, history, model=TestModel())

        assert len(result) < len(history), 'no trigger of its own means it runs regardless'

    async def test_a_tiered_strategy_honours_its_own_target(self):
        """Already under target is not a missed compaction: there is nothing left to reclaim."""
        history = self._short_history()
        tiered: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindow(max_tokens=1, keep_messages=2)],
            target_tokens=1_000_000,
        )

        result = await compact_now(tiered, history, model=TestModel())

        assert result == history

    async def test_a_tiered_strategy_over_target_still_escalates(self):
        tiered: TieredCompaction[None] = TieredCompaction(
            tiers=[SlidingWindow(max_tokens=1, keep_messages=2)],
            target_tokens=1,
        )

        result = await compact_now(tiered, _history(6), model=TestModel())

        assert len(result) < 12
