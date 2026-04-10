"""Tests for the ToolErrorRecovery capability."""
# pyright: reportPrivateUsage=false, reportArgumentType=false

from __future__ import annotations

import pytest
from pydantic_ai.messages import ToolCallPart

from pydantic_harness.tool_error_recovery import (
    ToolErrorRecovery,
    _format_error,
    _validate_strategy,
    fallback,
    retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCtx:
    """Minimal stand-in for RunContext."""


class _FakeToolDef:
    """Minimal stand-in for ToolDefinition."""


def _make_tc(name: str, args: dict[str, object] | None = None) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args)


# ---------------------------------------------------------------------------
# Unit tests for convenience constructors
# ---------------------------------------------------------------------------


class TestRetry:
    def test_default(self):
        assert retry() == ('retry', 3)

    def test_custom(self):
        assert retry(5) == ('retry', 5)

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match='positive integer'):
            retry(0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match='positive integer'):
            retry(-1)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match='positive integer'):
            retry('three')  # type: ignore[arg-type]


class TestFallback:
    def test_default(self):
        assert fallback() == ('fallback', None)

    def test_custom_value(self):
        assert fallback('N/A') == ('fallback', 'N/A')

    def test_custom_dict(self):
        assert fallback({'error': True}) == ('fallback', {'error': True})


# ---------------------------------------------------------------------------
# Unit tests for strategy validation
# ---------------------------------------------------------------------------


class TestValidateStrategy:
    def test_inform(self):
        _validate_strategy('inform')  # should not raise

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="must be 'inform'"):
            _validate_strategy('unknown')

    def test_retry_valid(self):
        _validate_strategy(('retry', 3))  # should not raise

    def test_retry_invalid_count(self):
        with pytest.raises(ValueError, match='positive integer'):
            _validate_strategy(('retry', 0))

    def test_retry_non_int(self):
        with pytest.raises(ValueError, match='positive integer'):
            _validate_strategy(('retry', 'x'))

    def test_fallback_valid(self):
        _validate_strategy(('fallback', None))  # should not raise

    def test_unknown_tuple_kind(self):
        with pytest.raises(ValueError, match="'retry' or 'fallback'"):
            _validate_strategy(('nope', 1))

    def test_bad_shape_3_tuple(self):
        with pytest.raises(ValueError, match='2-tuple or 4-tuple'):
            _validate_strategy(('retry', 1, 2))  # type: ignore[arg-type]

    def test_non_tuple_non_string(self):
        with pytest.raises(ValueError, match='string or tuple'):
            _validate_strategy(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Unit tests for error formatting
# ---------------------------------------------------------------------------


class TestFormatError:
    def test_basic(self):
        err = RuntimeError('file not found')
        msg = _format_error('read_file', err, include_traceback=False)
        assert 'read_file' in msg
        assert 'RuntimeError' in msg
        assert 'file not found' in msg
        assert 'Traceback' not in msg

    def test_with_traceback(self):
        try:
            raise ValueError('bad value')
        except ValueError as exc:
            msg = _format_error('validate', exc, include_traceback=True)
        assert 'Traceback' in msg
        assert 'ValueError' in msg
        assert 'bad value' in msg


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        cap = ToolErrorRecovery()
        assert cap.default_strategy == 'inform'
        assert cap.tool_strategies == {}
        assert cap.include_traceback is False
        assert cap._retry_counts == {}

    def test_custom_strategies(self):
        cap = ToolErrorRecovery(
            default_strategy=('retry', 2),
            tool_strategies={'api_call': ('fallback', 'unavailable')},
        )
        assert cap.default_strategy == ('retry', 2)
        assert cap._get_strategy('api_call') == ('fallback', 'unavailable')
        assert cap._get_strategy('unknown_tool') == ('retry', 2)

    def test_invalid_default_strategy(self):
        with pytest.raises(ValueError, match="must be 'inform'"):
            ToolErrorRecovery(default_strategy='bad')

    def test_invalid_tool_strategy(self):
        with pytest.raises(ValueError, match='positive integer'):
            ToolErrorRecovery(tool_strategies={'foo': ('retry', -1)})


# ---------------------------------------------------------------------------
# for_run isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_for_run_returns_fresh_instance():
    cap = ToolErrorRecovery(
        default_strategy=('retry', 2),
        tool_strategies={'x': 'inform'},
        include_traceback=True,
    )
    cap._retry_counts['x'] = 5

    fresh = await cap.for_run(_FakeCtx())  # type: ignore[arg-type]
    assert fresh is not cap
    assert fresh._retry_counts == {}
    assert fresh.default_strategy == cap.default_strategy
    assert fresh.tool_strategies == cap.tool_strategies
    assert fresh.include_traceback == cap.include_traceback


# ---------------------------------------------------------------------------
# on_tool_execute_error — inform strategy
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_inform_returns_error_message():
    cap = ToolErrorRecovery()
    ctx: object = _FakeCtx()
    tc = _make_tc('read_file', {'path': '/missing'})
    td: object = _FakeToolDef()
    err = FileNotFoundError('/missing')

    result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={'path': '/missing'}, error=err)  # type: ignore[arg-type]
    assert 'read_file' in result
    assert 'FileNotFoundError' in result
    assert '/missing' in result


@pytest.mark.anyio()
async def test_inform_with_traceback():
    cap = ToolErrorRecovery(include_traceback=True)
    ctx: object = _FakeCtx()
    tc = _make_tc('do_thing')
    td: object = _FakeToolDef()
    try:
        raise RuntimeError('boom')
    except RuntimeError as exc:
        result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=exc)  # type: ignore[arg-type]
    assert 'Traceback' in result
    assert 'boom' in result


# ---------------------------------------------------------------------------
# on_tool_execute_error — fallback strategy
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_fallback_returns_value():
    cap = ToolErrorRecovery(tool_strategies={'api_call': ('fallback', 'unavailable')})
    ctx: object = _FakeCtx()
    tc = _make_tc('api_call')
    td: object = _FakeToolDef()
    err = ConnectionError('timeout')

    result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=err)  # type: ignore[arg-type]
    assert result == 'unavailable'


@pytest.mark.anyio()
async def test_fallback_none():
    cap = ToolErrorRecovery(tool_strategies={'x': ('fallback', None)})
    ctx: object = _FakeCtx()
    tc = _make_tc('x')
    td: object = _FakeToolDef()
    err = ValueError('oops')

    result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=err)  # type: ignore[arg-type]
    assert result is None


@pytest.mark.anyio()
async def test_fallback_dict():
    cap = ToolErrorRecovery(tool_strategies={'lookup': fallback({'found': False})})
    ctx: object = _FakeCtx()
    tc = _make_tc('lookup')
    td: object = _FakeToolDef()

    result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError('x'))  # type: ignore[arg-type]
    assert result == {'found': False}


# ---------------------------------------------------------------------------
# wrap_tool_execute — retry strategy
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_retry_succeeds_on_first_attempt():
    cap = ToolErrorRecovery(tool_strategies={'flaky': retry(3)})
    ctx: object = _FakeCtx()
    tc = _make_tc('flaky')
    td: object = _FakeToolDef()

    async def handler(args: dict[str, object]) -> str:
        return 'ok'

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert result == 'ok'


@pytest.mark.anyio()
async def test_retry_succeeds_after_failures():
    cap = ToolErrorRecovery(tool_strategies={'flaky': retry(3)})
    ctx: object = _FakeCtx()
    tc = _make_tc('flaky')
    td: object = _FakeToolDef()

    call_count = 0

    async def handler(args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError('transient')
        return 'recovered'

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert result == 'recovered'
    assert call_count == 3
    # Retry count should be cleared on success.
    assert 'flaky' not in cap._retry_counts


@pytest.mark.anyio()
async def test_retry_exhausted_falls_back_to_inform():
    cap = ToolErrorRecovery(tool_strategies={'flaky': retry(2)})
    ctx: object = _FakeCtx()
    tc = _make_tc('flaky')
    td: object = _FakeToolDef()

    async def handler(args: dict[str, object]) -> str:
        raise RuntimeError('persistent failure')

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert isinstance(result, str)
    assert 'flaky' in result
    assert 'RuntimeError' in result
    assert 'persistent failure' in result
    # Retry count should reflect total attempts.
    assert cap._retry_counts['flaky'] == 3  # 1 initial + 2 retries


@pytest.mark.anyio()
async def test_retry_with_traceback():
    cap = ToolErrorRecovery(tool_strategies={'t': retry(1)}, include_traceback=True)
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    async def handler(args: dict[str, object]) -> str:
        raise ValueError('detail')

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert 'Traceback' in result


@pytest.mark.anyio()
async def test_non_retry_strategy_passes_through_to_error_hook():
    """For inform/fallback strategies, wrap_tool_execute just calls handler directly."""
    cap = ToolErrorRecovery(default_strategy='inform')
    ctx: object = _FakeCtx()
    tc = _make_tc('tool')
    td: object = _FakeToolDef()

    async def handler(args: dict[str, object]) -> str:
        raise RuntimeError('boom')

    # wrap_tool_execute should NOT catch the error for non-retry strategies;
    # it re-raises so on_tool_execute_error can handle it.
    with pytest.raises(RuntimeError, match='boom'):
        await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------


class TestStrategyResolution:
    def test_tool_specific_overrides_default(self):
        cap = ToolErrorRecovery(
            default_strategy='inform',
            tool_strategies={'special': ('fallback', 42)},
        )
        assert cap._get_strategy('special') == ('fallback', 42)
        assert cap._get_strategy('other') == 'inform'

    def test_default_strategy_used_for_unknown(self):
        cap = ToolErrorRecovery(default_strategy=retry(5))
        assert cap._get_strategy('anything') == ('retry', 5)


# ---------------------------------------------------------------------------
# Serialization name
# ---------------------------------------------------------------------------


def test_serialization_name():
    assert ToolErrorRecovery.get_serialization_name() == 'ToolErrorRecovery'


# ---------------------------------------------------------------------------
# Retry resets on success
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_retry_count_resets_on_success():
    cap = ToolErrorRecovery(tool_strategies={'t': retry(3)})
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    # First call: fail once, then succeed.
    call_count = 0

    async def handler_fail_once(args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError('transient')
        return 'ok'

    await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler_fail_once)  # type: ignore[arg-type]
    assert 't' not in cap._retry_counts

    # Second call: should start with a fresh retry budget.
    call_count = 0
    await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler_fail_once)  # type: ignore[arg-type]
    assert 't' not in cap._retry_counts


# ---------------------------------------------------------------------------
# Default fallback strategy for all tools
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_default_fallback():
    cap = ToolErrorRecovery(default_strategy=fallback('default_value'))
    ctx: object = _FakeCtx()
    tc = _make_tc('any_tool')
    td: object = _FakeToolDef()

    result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError('x'))  # type: ignore[arg-type]
    assert result == 'default_value'


# ---------------------------------------------------------------------------
# Default retry strategy for all tools
# ---------------------------------------------------------------------------


@pytest.mark.anyio()
async def test_default_retry():
    cap = ToolErrorRecovery(default_strategy=retry(1))
    ctx: object = _FakeCtx()
    tc = _make_tc('any_tool')
    td: object = _FakeToolDef()

    async def handler(args: dict[str, object]) -> str:
        raise RuntimeError('always fails')

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert 'RuntimeError' in result
    assert 'always fails' in result


# ---------------------------------------------------------------------------
# Exponential backoff
# ---------------------------------------------------------------------------


class TestRetryDelay:
    def test_retry_with_delay_produces_4_tuple(self):
        strategy = retry(2, retry_delay=0.5)
        assert strategy == ('retry', 2, 0.5, (Exception,))

    def test_retry_defaults_produce_2_tuple(self):
        strategy = retry(3)
        assert strategy == ('retry', 3)

    def test_retry_delay_validation_negative(self):
        with pytest.raises(ValueError, match='non-negative'):
            retry(2, retry_delay=-1)

    def test_retry_delay_validation_bad_type(self):
        with pytest.raises(ValueError, match='non-negative'):
            retry(2, retry_delay='fast')  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_retry_exponential_backoff_timing(monkeypatch: pytest.MonkeyPatch):
    """Verify that exponential backoff sleeps with increasing delays."""
    from unittest.mock import AsyncMock, call

    import anyio

    import pydantic_harness.tool_error_recovery as module

    mock_sleep = AsyncMock(side_effect=anyio.sleep)
    monkeypatch.setattr(module.anyio, 'sleep', mock_sleep)

    cap = ToolErrorRecovery(tool_strategies={'t': retry(3, retry_delay=0.1)})
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    call_count = 0

    async def handler(args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ConnectionError('transient')
        return 'ok'

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert result == 'ok'
    assert call_count == 4
    # Delays: 0.1 * 2^0, 0.1 * 2^1, 0.1 * 2^2
    assert mock_sleep.call_args_list == [call(0.1), call(0.2), call(0.4)]


@pytest.mark.anyio()
async def test_retry_no_delay_by_default(monkeypatch: pytest.MonkeyPatch):
    """When retry_delay=0, anyio.sleep should not be called."""
    from unittest.mock import AsyncMock

    import anyio

    import pydantic_harness.tool_error_recovery as module

    mock_sleep = AsyncMock(side_effect=anyio.sleep)
    monkeypatch.setattr(module.anyio, 'sleep', mock_sleep)

    cap = ToolErrorRecovery(tool_strategies={'t': retry(2)})
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    call_count = 0

    async def handler(args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError('transient')
        return 'ok'

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert result == 'ok'
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Retryable exceptions filter
# ---------------------------------------------------------------------------


class TestRetryableExceptions:
    def test_retry_with_filter_produces_4_tuple(self):
        strategy = retry(2, retryable_exceptions=(ConnectionError, TimeoutError))
        assert strategy == ('retry', 2, 0, (ConnectionError, TimeoutError))

    def test_invalid_retryable_exceptions_not_tuple(self):
        with pytest.raises(ValueError, match='retryable_exceptions'):
            retry(2, retryable_exceptions=[ConnectionError])  # type: ignore[arg-type]

    def test_invalid_retryable_exceptions_not_exception(self):
        with pytest.raises(ValueError, match='retryable_exceptions'):
            retry(2, retryable_exceptions=(str,))  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_retryable_exception_match_retries():
    """When the exception matches retryable_exceptions, it should be retried."""
    cap = ToolErrorRecovery(tool_strategies={'t': retry(3, retryable_exceptions=(ConnectionError,))})
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    call_count = 0

    async def handler(args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError('transient')
        return 'recovered'

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert result == 'recovered'
    assert call_count == 3


@pytest.mark.anyio()
async def test_non_retryable_exception_skips_retry():
    """When the exception does not match retryable_exceptions, it should not be retried."""
    cap = ToolErrorRecovery(tool_strategies={'t': retry(3, retryable_exceptions=(ConnectionError,))})
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    call_count = 0

    async def handler(args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        raise ValueError('not retryable')

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    # Should return inform message immediately, no retries.
    assert call_count == 1
    assert isinstance(result, str)
    assert 'ValueError' in result
    assert 'not retryable' in result


@pytest.mark.anyio()
async def test_retryable_subclass_matches():
    """Subclasses of retryable exceptions should also be retried."""
    cap = ToolErrorRecovery(tool_strategies={'t': retry(2, retryable_exceptions=(OSError,))})
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    call_count = 0

    async def handler(args: dict[str, object]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError('subclass of OSError')  # ConnectionError is a subclass of OSError
        return 'ok'

    result = await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]
    assert result == 'ok'
    assert call_count == 2


# ---------------------------------------------------------------------------
# Error budget (max_total_errors)
# ---------------------------------------------------------------------------


class TestErrorBudget:
    def test_construction_with_budget(self):
        cap = ToolErrorRecovery(max_total_errors=5)
        assert cap.max_total_errors == 5
        assert cap._total_errors == 0

    def test_default_no_budget(self):
        cap = ToolErrorRecovery()
        assert cap.max_total_errors is None


@pytest.mark.anyio()
async def test_for_run_preserves_max_total_errors():
    cap = ToolErrorRecovery(max_total_errors=10)
    cap._total_errors = 7  # simulate errors from a previous run
    fresh = await cap.for_run(_FakeCtx())  # type: ignore[arg-type]
    assert fresh.max_total_errors == 10
    assert fresh._total_errors == 0


@pytest.mark.anyio()
async def test_error_budget_inform_strategy():
    """After max_total_errors is reached, inform strategy should let errors propagate."""
    cap = ToolErrorRecovery(default_strategy='inform', max_total_errors=2)
    ctx: object = _FakeCtx()
    tc = _make_tc('tool')
    td: object = _FakeToolDef()

    # First two errors should be recovered.
    result1 = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError('err1'))  # type: ignore[arg-type]
    assert isinstance(result1, str)
    assert 'err1' in result1

    result2 = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError('err2'))  # type: ignore[arg-type]
    assert isinstance(result2, str)
    assert 'err2' in result2

    # Third error should propagate (budget of 2 is exhausted).
    with pytest.raises(RuntimeError, match='err3'):
        await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError('err3'))  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_error_budget_fallback_strategy():
    """After budget is exhausted, fallback strategy should also let errors propagate."""
    cap = ToolErrorRecovery(default_strategy=('fallback', 'safe'), max_total_errors=1)
    ctx: object = _FakeCtx()
    tc = _make_tc('tool')
    td: object = _FakeToolDef()

    # First error: fallback works.
    result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError('err'))  # type: ignore[arg-type]
    assert result == 'safe'

    # Second error: budget exhausted, propagates.
    with pytest.raises(RuntimeError, match='err2'):
        await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError('err2'))  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_error_budget_retry_strategy():
    """After budget is exhausted during retries, the error should propagate."""
    cap = ToolErrorRecovery(tool_strategies={'t': retry(5)}, max_total_errors=3)
    ctx: object = _FakeCtx()
    tc = _make_tc('t')
    td: object = _FakeToolDef()

    async def handler(args: dict[str, object]) -> str:
        raise RuntimeError('always fails')

    # The retry loop counts each attempt. 3 errors are recovered, the 4th propagates.
    with pytest.raises(RuntimeError, match='always fails'):
        await cap.wrap_tool_execute(ctx, call=tc, tool_def=td, args={}, handler=handler)  # type: ignore[arg-type]

    assert cap._total_errors == 4


@pytest.mark.anyio()
async def test_error_budget_across_tools():
    """Error budget is shared across all tools."""
    cap = ToolErrorRecovery(default_strategy='inform', max_total_errors=2)
    ctx: object = _FakeCtx()
    td: object = _FakeToolDef()

    # Two different tools each use one error from the budget.
    tc1 = _make_tc('tool_a')
    tc2 = _make_tc('tool_b')
    tc3 = _make_tc('tool_c')

    await cap.on_tool_execute_error(ctx, call=tc1, tool_def=td, args={}, error=RuntimeError('a'))  # type: ignore[arg-type]
    await cap.on_tool_execute_error(ctx, call=tc2, tool_def=td, args={}, error=RuntimeError('b'))  # type: ignore[arg-type]

    # Third error from a different tool: budget exhausted.
    with pytest.raises(RuntimeError, match='c'):
        await cap.on_tool_execute_error(ctx, call=tc3, tool_def=td, args={}, error=RuntimeError('c'))  # type: ignore[arg-type]


@pytest.mark.anyio()
async def test_no_budget_unlimited_recovery():
    """Without max_total_errors, recovery should work indefinitely."""
    cap = ToolErrorRecovery(default_strategy='inform')
    ctx: object = _FakeCtx()
    tc = _make_tc('tool')
    td: object = _FakeToolDef()

    for i in range(100):
        result = await cap.on_tool_execute_error(ctx, call=tc, tool_def=td, args={}, error=RuntimeError(f'err{i}'))  # type: ignore[arg-type]
        assert isinstance(result, str)

    assert cap._total_errors == 100


# ---------------------------------------------------------------------------
# Validate strategy for extended retry tuples
# ---------------------------------------------------------------------------


class TestValidateStrategyExtended:
    def test_valid_4_tuple_retry(self):
        _validate_strategy(('retry', 3, 0.5, (ConnectionError, TimeoutError)))  # should not raise

    def test_too_short_tuple(self):
        with pytest.raises(ValueError, match='length 2 or 4'):
            _validate_strategy(('retry',))  # type: ignore[arg-type]

    def test_4_tuple_retry_invalid_max_retries(self):
        with pytest.raises(ValueError, match='positive integer'):
            _validate_strategy(('retry', 0, 0.5, (Exception,)))

    def test_invalid_4_tuple_bad_delay(self):
        with pytest.raises(ValueError, match='non-negative'):
            _validate_strategy(('retry', 3, -1.0, (Exception,)))

    def test_invalid_4_tuple_bad_exceptions(self):
        with pytest.raises(ValueError, match='retryable_exceptions'):
            _validate_strategy(('retry', 3, 0.0, (str,)))

    def test_invalid_4_tuple_exceptions_not_tuple(self):
        with pytest.raises(ValueError, match='retryable_exceptions'):
            _validate_strategy(('retry', 3, 0.0, [Exception]))  # type: ignore[arg-type]

    def test_fallback_3_tuple_invalid(self):
        with pytest.raises(ValueError, match='fallback strategy must be a 2-tuple'):
            _validate_strategy(('fallback', 'val', 'extra'))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public API import
# ---------------------------------------------------------------------------


def test_public_import():
    from pydantic_harness import ToolErrorRecovery as TER
    from pydantic_harness import fallback as fb
    from pydantic_harness import retry as rt

    assert TER is ToolErrorRecovery
    assert fb is fallback
    assert rt is retry
