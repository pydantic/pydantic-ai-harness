from __future__ import annotations

# pyright: reportPrivateUsage=false
from typing import Any, Literal
from unittest.mock import MagicMock

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.settings import ModelSettings

from pydantic_harness import AdaptiveReasoning
from pydantic_harness.adaptive_reasoning import (
    _has_tool_errors,
    _last_response_had_many_tool_calls,
    default_effort_fn,
)


def _make_ctx(
    *,
    run_step: int = 0,
    messages: list[Any] | None = None,
) -> RunContext[None]:
    """Build a minimal RunContext for testing."""
    model = MagicMock()
    model.system = 'test'
    ctx = RunContext[None](
        deps=None,
        model=model,
        usage=MagicMock(),
        messages=messages or [],
        run_step=run_step,
    )
    return ctx


# --- _has_tool_errors ---


class TestHasToolErrors:
    def test_no_messages(self) -> None:
        assert _has_tool_errors([]) is False

    def test_no_retry_parts(self) -> None:
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hello')])]
        assert _has_tool_errors(messages) is False

    def test_with_retry_part(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    RetryPromptPart(content='validation failed', tool_name='my_tool'),
                ]
            ),
        ]
        assert _has_tool_errors(messages) is True

    def test_checks_latest_request(self) -> None:
        """Only the most recent ModelRequest is inspected."""
        old_request = ModelRequest(parts=[RetryPromptPart(content='old error', tool_name='my_tool')])
        new_request = ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', content='ok')])
        messages: list[ModelMessage] = [old_request, new_request]
        # Most recent message is the one without errors.
        assert _has_tool_errors(messages) is False

    def test_skips_model_responses(self) -> None:
        """ModelResponse objects are skipped when searching for the latest request."""
        request_with_error = ModelRequest(parts=[RetryPromptPart(content='error', tool_name='my_tool')])
        response = ModelResponse(parts=[TextPart(content='ok')])
        messages: list[ModelMessage] = [request_with_error, response]
        assert _has_tool_errors(messages) is True


# --- _last_response_had_many_tool_calls ---


class TestLastResponseHadManyToolCalls:
    def test_no_messages(self) -> None:
        assert _last_response_had_many_tool_calls([]) is False

    def test_no_response(self) -> None:
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
        assert _last_response_had_many_tool_calls(messages) is False

    def test_below_threshold(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='t1', args='{}'),
                    ToolCallPart(tool_name='t2', args='{}'),
                ]
            ),
        ]
        assert _last_response_had_many_tool_calls(messages) is False

    def test_at_threshold(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='t1', args='{}'),
                    ToolCallPart(tool_name='t2', args='{}'),
                    ToolCallPart(tool_name='t3', args='{}'),
                ]
            ),
        ]
        assert _last_response_had_many_tool_calls(messages) is True

    def test_checks_latest_response(self) -> None:
        """Only the most recent ModelResponse is inspected."""
        old_response = ModelResponse(
            parts=[
                ToolCallPart(tool_name='t1', args='{}'),
                ToolCallPart(tool_name='t2', args='{}'),
                ToolCallPart(tool_name='t3', args='{}'),
            ]
        )
        new_response = ModelResponse(parts=[TextPart(content='done')])
        messages: list[ModelMessage] = [old_response, new_response]
        assert _last_response_had_many_tool_calls(messages) is False

    def test_skips_requests(self) -> None:
        """ModelRequest objects are skipped when searching."""
        response = ModelResponse(
            parts=[
                ToolCallPart(tool_name='t1', args='{}'),
                ToolCallPart(tool_name='t2', args='{}'),
                ToolCallPart(tool_name='t3', args='{}'),
            ]
        )
        request = ModelRequest(parts=[ToolReturnPart(tool_name='t1', content='ok')])
        messages: list[ModelMessage] = [response, request]
        assert _last_response_had_many_tool_calls(messages) is True


# --- default_effort_fn ---


class TestDefaultEffortFn:
    def test_first_step_high(self) -> None:
        ctx = _make_ctx(run_step=1)
        assert default_effort_fn(ctx) == 'high'

    def test_step_zero_high(self) -> None:
        ctx = _make_ctx(run_step=0)
        assert default_effort_fn(ctx) == 'high'

    def test_after_tool_error_high(self) -> None:
        messages = [
            ModelRequest(parts=[RetryPromptPart(content='bad args', tool_name='t')]),
        ]
        ctx = _make_ctx(run_step=3, messages=messages)
        assert default_effort_fn(ctx) == 'high'

    def test_step_two_medium(self) -> None:
        messages = [
            ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result')]),
        ]
        ctx = _make_ctx(run_step=2, messages=messages)
        assert default_effort_fn(ctx) == 'medium'

    def test_later_step_no_errors_low(self) -> None:
        ctx = _make_ctx(run_step=5, messages=[])
        assert default_effort_fn(ctx) == 'low'

    def test_step_three_low(self) -> None:
        messages = [
            ModelRequest(parts=[ToolReturnPart(tool_name='t', content='result')]),
        ]
        ctx = _make_ctx(run_step=3, messages=messages)
        assert default_effort_fn(ctx) == 'low'

    def test_many_tool_calls_medium(self) -> None:
        """A response with 3+ ToolCallParts should trigger medium effort."""
        messages: list[Any] = [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='t1', args='{}'),
                    ToolCallPart(tool_name='t2', args='{}'),
                    ToolCallPart(tool_name='t3', args='{}'),
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='t1', content='ok')]),
        ]
        ctx = _make_ctx(run_step=5, messages=messages)
        assert default_effort_fn(ctx) == 'medium'

    def test_few_tool_calls_not_medium(self) -> None:
        """A response with <3 ToolCallParts should not trigger medium effort at step 3+."""
        messages: list[Any] = [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='t1', args='{}'),
                    ToolCallPart(tool_name='t2', args='{}'),
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='t1', content='ok')]),
        ]
        ctx = _make_ctx(run_step=5, messages=messages)
        assert default_effort_fn(ctx) == 'low'


# --- AdaptiveReasoning capability ---


class TestAdaptiveReasoning:
    def test_default_construction(self) -> None:
        cap = AdaptiveReasoning()
        assert cap.effort_fn is default_effort_fn

    def test_get_model_settings_returns_callable(self) -> None:
        cap = AdaptiveReasoning()
        settings_fn = cap.get_model_settings()
        assert callable(settings_fn)

    def test_dynamic_settings_first_step(self) -> None:
        cap = AdaptiveReasoning()
        settings_fn = cap.get_model_settings()
        ctx = _make_ctx(run_step=1)
        result = settings_fn(ctx)
        assert result == ModelSettings(thinking='high')

    def test_dynamic_settings_step_two(self) -> None:
        cap = AdaptiveReasoning()
        settings_fn = cap.get_model_settings()
        messages = [
            ModelRequest(parts=[ToolReturnPart(tool_name='t', content='ok')]),
        ]
        ctx = _make_ctx(run_step=2, messages=messages)
        result = settings_fn(ctx)
        assert result == ModelSettings(thinking='medium')

    def test_dynamic_settings_later_step_low(self) -> None:
        cap = AdaptiveReasoning()
        settings_fn = cap.get_model_settings()
        messages = [
            ModelRequest(parts=[ToolReturnPart(tool_name='t', content='ok')]),
        ]
        ctx = _make_ctx(run_step=3, messages=messages)
        result = settings_fn(ctx)
        assert result == ModelSettings(thinking='low')

    def test_dynamic_settings_after_error(self) -> None:
        cap = AdaptiveReasoning()
        settings_fn = cap.get_model_settings()
        messages = [
            ModelRequest(parts=[RetryPromptPart(content='err', tool_name='t')]),
        ]
        ctx = _make_ctx(run_step=4, messages=messages)
        result = settings_fn(ctx)
        assert result == ModelSettings(thinking='high')

    def test_custom_effort_fn(self) -> None:
        def always_medium(ctx: RunContext[Any]) -> Literal['low', 'medium', 'high']:
            return 'medium'

        cap = AdaptiveReasoning(effort_fn=always_medium)
        settings_fn = cap.get_model_settings()
        ctx = _make_ctx(run_step=1)
        result = settings_fn(ctx)
        assert result == ModelSettings(thinking='medium')

    def test_custom_effort_fn_context_aware(self) -> None:
        def step_based(ctx: RunContext[Any]) -> Literal['low', 'medium', 'high']:
            if ctx.run_step > 10:
                return 'high'
            return 'low'

        cap = AdaptiveReasoning(effort_fn=step_based)
        settings_fn = cap.get_model_settings()

        ctx_early = _make_ctx(run_step=2)
        assert settings_fn(ctx_early) == ModelSettings(thinking='low')

        ctx_late = _make_ctx(run_step=11)
        assert settings_fn(ctx_late) == ModelSettings(thinking='high')

    def test_is_abstract_capability(self) -> None:
        from pydantic_ai.capabilities.abstract import AbstractCapability

        cap = AdaptiveReasoning()
        assert isinstance(cap, AbstractCapability)
