"""Tests for the RetryPolicy capability."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.retry_policy import RetryPolicy

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend."""
    return 'asyncio'


# --- RetryPolicy validation ---


class TestRetryPolicyValidation:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_factor == 0.5
        assert policy.max_backoff == 30.0
        assert policy.retryable_status_codes == (429, 500, 502, 503, 504)
        assert policy.retryable_exceptions == (TimeoutError, ConnectionError, OSError)
        assert policy.tool_overrides == {}

    def test_custom_values(self) -> None:
        policy = RetryPolicy(
            max_retries=5,
            backoff_factor=1.0,
            max_backoff=60.0,
            retryable_status_codes=(429, 503),
            retryable_exceptions=(TimeoutError,),
        )
        assert policy.max_retries == 5
        assert policy.backoff_factor == 1.0
        assert policy.max_backoff == 60.0
        assert policy.retryable_status_codes == (429, 503)
        assert policy.retryable_exceptions == (TimeoutError,)

    def test_tool_overrides(self) -> None:
        policy = RetryPolicy(
            tool_overrides={
                'web_search': {'max_retries': 2, 'backoff_factor': 1.0},
                'shell': {'max_retries': 1},
            }
        )
        assert policy.tool_overrides['web_search']['max_retries'] == 2
        assert policy.tool_overrides['shell']['max_retries'] == 1

    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValueError, match='max_retries must be an int >= 0'):
            RetryPolicy(max_retries=-1)

    def test_float_max_retries_raises(self) -> None:
        with pytest.raises(ValueError, match='max_retries must be an int >= 0'):
            RetryPolicy(max_retries=1.5)  # type: ignore[arg-type]

    def test_zero_backoff_factor_raises(self) -> None:
        with pytest.raises(ValueError, match='backoff_factor must be a finite > 0'):
            RetryPolicy(backoff_factor=0)

    def test_negative_max_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match='max_backoff must be a finite > 0'):
            RetryPolicy(max_backoff=-1)

    def test_infinite_backoff_factor_raises(self) -> None:
        with pytest.raises(ValueError, match='backoff_factor must be a finite > 0'):
            RetryPolicy(backoff_factor=float('inf'))

    def test_infinite_max_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match='max_backoff must be a finite > 0'):
            RetryPolicy(max_backoff=float('inf'))

    def test_nan_backoff_factor_raises(self) -> None:
        with pytest.raises(ValueError, match='backoff_factor must be a finite > 0'):
            RetryPolicy(backoff_factor=float('nan'))

    def test_nan_max_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match='max_backoff must be a finite > 0'):
            RetryPolicy(max_backoff=float('nan'))

    def test_tool_override_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValueError, match=r"tool_overrides\['bad'\]\['max_retries'\] must be >= 0"):
            RetryPolicy(tool_overrides={'bad': {'max_retries': -1}})

    def test_tool_override_zero_backoff_factor_raises(self) -> None:
        with pytest.raises(ValueError, match=r"tool_overrides\['bad'\]\['backoff_factor'\] must be > 0"):
            RetryPolicy(tool_overrides={'bad': {'backoff_factor': 0}})

    @pytest.mark.parametrize('field_name', ['backoff_factor', 'max_backoff'])
    @pytest.mark.parametrize('value', [float('inf'), float('-inf'), float('nan')])
    def test_tool_override_nonfinite_backoff_raises(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError, match='must be > 0 and finite'):
            RetryPolicy(tool_overrides={'bad': {field_name: value}})

    def test_tool_override_negative_max_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match=r"tool_overrides\['bad'\]\['max_backoff'\] must be > 0"):
            RetryPolicy(tool_overrides={'bad': {'max_backoff': -5}})


# --- Retry logic ---


class TestRetryLogic:
    def test_get_tool_config_default(self) -> None:
        policy = RetryPolicy()
        config = policy.get_tool_config('unknown_tool')
        assert config == {}

    def test_get_tool_config_override(self) -> None:
        policy = RetryPolicy(tool_overrides={'web_search': {'max_retries': 2}})
        config = policy.get_tool_config('web_search')
        assert config['max_retries'] == 2

    def test_get_max_retries_default(self) -> None:
        policy = RetryPolicy(max_retries=5)
        assert policy.get_max_retries('any_tool') == 5

    def test_get_max_retries_override(self) -> None:
        policy = RetryPolicy(max_retries=5, tool_overrides={'web_search': {'max_retries': 2}})
        assert policy.get_max_retries('web_search') == 2
        assert policy.get_max_retries('other_tool') == 5

    @pytest.mark.parametrize('status', [429, 400, None])
    @pytest.mark.parametrize('nested', [False, True])
    def test_http_error_status(self, status: int | None, nested: bool) -> None:
        class Response:
            status_code = status

        class HttpError(Exception):
            status_code = None if nested else status
            response = Response() if nested else None

        assert RetryPolicy().should_retry(HttpError(), 'tool') is (status == 429)

    @pytest.mark.parametrize('error_type', ['rate_limit', 'timeout', 'server_error', 'invalid'])
    def test_provider_error_type(self, error_type: str) -> None:
        class ProviderError(Exception):
            def __init__(self) -> None:
                self.error_type = error_type

        assert RetryPolicy().should_retry(ProviderError(), 'tool') is (error_type != 'invalid')

    def test_valid_max_backoff_override(self) -> None:
        policy = RetryPolicy(tool_overrides={'tool': {'max_backoff': 1.0}})
        assert policy.calculate_delay(100, 'tool') <= 1.0

    def test_should_retry_timeout_error(self) -> None:
        policy = RetryPolicy()
        assert policy.should_retry(TimeoutError('timeout'), 'tool') is True

    def test_should_retry_connection_error(self) -> None:
        policy = RetryPolicy()
        assert policy.should_retry(ConnectionError('connection'), 'tool') is True

    def test_should_retry_non_retryable(self) -> None:
        policy = RetryPolicy()
        assert policy.should_retry(ValueError('bad value'), 'tool') is False

    def test_should_retry_custom_exception(self) -> None:
        class MyError(Exception):
            pass

        policy = RetryPolicy(retryable_exceptions=(MyError,))
        assert policy.should_retry(MyError('custom'), 'tool') is True
        assert policy.should_retry(TimeoutError('timeout'), 'tool') is False

    def test_should_retry_per_tool_override(self) -> None:
        policy = RetryPolicy(tool_overrides={'safe_tool': {'retryable_exceptions': ()}})
        assert policy.should_retry(TimeoutError('timeout'), 'tool') is True
        assert policy.should_retry(TimeoutError('timeout'), 'safe_tool') is False

    def test_calculate_delay_base(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0)
        delay = policy.calculate_delay(0, 'tool')
        assert 0.75 <= delay <= 1.25

    def test_calculate_delay_exponential(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0, max_backoff=100.0)
        for _ in range(10):
            delays = [policy.calculate_delay(i, 'tool') for i in range(3)]
            assert delays[1] > delays[0] * 0.8
            assert delays[2] > delays[1] * 0.8

    def test_calculate_delay_max_cap(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0, max_backoff=5.0)
        delay = policy.calculate_delay(10, 'tool')
        assert delay <= 6.25

    def test_calculate_delay_tool_override(self) -> None:
        policy = RetryPolicy(
            backoff_factor=0.5,
            tool_overrides={'fast_tool': {'backoff_factor': 0.1}},
        )
        delay_default = policy.calculate_delay(0, 'tool')
        delay_override = policy.calculate_delay(0, 'fast_tool')
        assert delay_override < delay_default

    def test_calculate_delay_large_attempt_no_overflow(self) -> None:
        policy = RetryPolicy(backoff_factor=0.5, max_backoff=30.0)
        delay = policy.calculate_delay(1024, 'tool')
        assert delay <= 30.0


# --- Callbacks ---


class TestCallbacks:
    def test_on_retry_callback(self) -> None:
        policy = RetryPolicy(on_retry=lambda tool, attempt, exc: None)
        assert policy.on_retry is not None

    def test_on_failure_callback(self) -> None:
        policy = RetryPolicy(on_failure=lambda tool, exc: None)
        assert policy.on_failure is not None

    def test_per_tool_on_retry_override(self) -> None:
        calls: list[str] = []

        def on_retry(tool: str, attempt: int, exc: Exception) -> None:
            calls.append('special')

        policy = RetryPolicy(
            on_retry=lambda tool, attempt, exc: calls.append('default'),
            tool_overrides={'special': {'on_retry': on_retry}},
        )
        default_callback = policy._get_on_retry('default_tool')
        special_callback = policy._get_on_retry('special')
        assert default_callback is not None
        assert special_callback is not None
        default_callback('default_tool', 1, Exception())
        special_callback('special', 1, Exception())
        assert calls == ['default', 'special']

    def test_per_tool_on_failure_override(self) -> None:
        calls: list[str] = []

        def on_failure(tool: str, exc: Exception) -> None:
            calls.append('special')

        policy = RetryPolicy(
            on_failure=lambda tool, exc: calls.append('default'),
            tool_overrides={'special': {'on_failure': on_failure}},
        )
        default_callback = policy._get_on_failure('default_tool')
        special_callback = policy._get_on_failure('special')
        assert default_callback is not None
        assert special_callback is not None
        default_callback('default_tool', Exception())
        special_callback('special', Exception())
        assert calls == ['default', 'special']


# --- Idempotency ---


class TestIdempotency:
    def test_default_not_idempotent(self) -> None:
        policy = RetryPolicy()
        assert policy._is_idempotent('any_tool') is False

    def test_allow_idempotent_retries(self) -> None:
        policy = RetryPolicy(
            allow_idempotent_retries=True,
            idempotent_tools=frozenset({'safe_read'}),
        )
        assert policy._is_idempotent('safe_read') is True
        assert policy._is_idempotent('unsafe_write') is False

    def test_per_tool_idempotent_override(self) -> None:
        policy = RetryPolicy(
            allow_idempotent_retries=True,
            tool_overrides={'custom': {'idempotent': True}},
        )
        assert policy._is_idempotent('custom') is True
        assert policy._is_idempotent('other') is False

    async def test_non_idempotent_no_retry_after_handler(self) -> None:
        """Non-idempotent tools should not retry after handler has been entered."""
        call_count = 0

        async def handler(args: object) -> str:
            nonlocal call_count
            call_count += 1
            raise TimeoutError('timeout')

        policy = RetryPolicy(max_retries=3)
        ctx = MagicMock()
        call = MagicMock()
        call.tool_name = 'write_tool'
        policy._get_on_retry = MagicMock(return_value=None)
        policy._get_on_failure = MagicMock(return_value=None)
        policy.get_max_retries = MagicMock(return_value=3)
        with pytest.raises(TimeoutError):
            await policy.wrap_tool_execute(ctx, call=call, tool_def=None, args={}, handler=handler)
        assert call_count == 1


# --- Integration with Agent (mocked) ---


class TestAgentIntegration:
    @pytest.mark.parametrize('with_callbacks', [False, True])
    @pytest.mark.parametrize('outcome', ['success', 'exhausted', 'nonretryable', 'unsafe'])
    async def test_tool_execution(self, with_callbacks: bool, outcome: str) -> None:
        retries: list[int] = []
        failures: list[Exception] = []
        attempts = 0

        def on_retry(tool: str, attempt: int, exc: Exception) -> None:
            retries.append(attempt)

        def on_failure(tool: str, exc: Exception) -> None:
            failures.append(exc)

        policy = RetryPolicy(
            max_retries=1,
            backoff_factor=0.001,
            max_backoff=0.001,
            allow_idempotent_retries=outcome != 'unsafe',
            idempotent_tools=frozenset({'lookup'}),
            on_retry=on_retry if with_callbacks else None,
            on_failure=on_failure if with_callbacks else None,
        )
        agent = Agent(TestModel(), capabilities=[policy])

        @agent.tool_plain
        def lookup() -> str:
            nonlocal attempts
            attempts += 1
            if outcome == 'nonretryable':
                raise ValueError('invalid')
            if outcome == 'success' and attempts == 2:
                return 'found'
            raise TimeoutError('timeout')

        if outcome == 'success':
            await agent.run('lookup')
        else:
            with pytest.raises(ValueError if outcome == 'nonretryable' else TimeoutError):
                await agent.run('lookup')
        assert attempts == (2 if outcome in ('success', 'exhausted') else 1)
        assert retries == ([1] if with_callbacks and attempts == 2 else [])
        assert len(failures) == int(with_callbacks and outcome in ('exhausted', 'unsafe'))

    def test_retry_policy_instantiation(self) -> None:
        policy = RetryPolicy(max_retries=2)
        assert policy.max_retries == 2

    def test_retry_policy_with_agent(self) -> None:
        policy = RetryPolicy(max_retries=1)
        agent = Agent(TestModel(), capabilities=[policy])
        assert agent is not None
