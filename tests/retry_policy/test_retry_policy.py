"""Tests for the RetryPolicy capability."""

from __future__ import annotations

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
        with pytest.raises(ValueError, match='max_retries must be >= 0'):
            RetryPolicy(max_retries=-1)

    def test_zero_backoff_factor_raises(self) -> None:
        with pytest.raises(ValueError, match='backoff_factor must be > 0'):
            RetryPolicy(backoff_factor=0)

    def test_negative_max_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match='max_backoff must be > 0'):
            RetryPolicy(max_backoff=-1)


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
        policy = RetryPolicy(
            tool_overrides={'safe_tool': {'retryable_exceptions': ()}}
        )
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
        policy = RetryPolicy(
            on_retry=lambda tool, attempt, exc: calls.append('default'),
            tool_overrides={'special': {'on_retry': lambda tool, attempt, exc: calls.append('special')}},
        )
        policy._get_on_retry('default_tool')('default_tool', 1, Exception())
        policy._get_on_retry('special')('special', 1, Exception())
        assert calls == ['default', 'special']

    def test_per_tool_on_failure_override(self) -> None:
        calls: list[str] = []
        policy = RetryPolicy(
            on_failure=lambda tool, exc: calls.append('default'),
            tool_overrides={'special': {'on_failure': lambda tool, exc: calls.append('special')}},
        )
        policy._get_on_failure('default_tool')('default_tool', Exception())
        policy._get_on_failure('special')('special', Exception())
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


# --- Integration with Agent (mocked) ---


class TestAgentIntegration:
    def test_retry_policy_instantiation(self) -> None:
        policy = RetryPolicy(max_retries=2)
        assert policy.max_retries == 2

    def test_retry_policy_with_agent(self) -> None:
        policy = RetryPolicy(max_retries=1)
        agent = Agent(TestModel(), capabilities=[policy])
        assert agent is not None
