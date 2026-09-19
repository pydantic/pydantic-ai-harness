from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.model_router import ModelChoice, ModelRouter
from tests.conftest import agent_run_names  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.anyio


def _router_model(
    choice: str,
    *,
    provider_details: Mapping[str, Any] | None = None,
    inspect: Callable[[list[ModelMessage], AgentInfo], None] | None = None,
) -> FunctionModel:
    def route(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if inspect is not None:
            inspect(messages, info)
        output_tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(output_tool.name, {'response': choice})],
            provider_details=dict(provider_details) if provider_details is not None else None,
        )

    return FunctionModel(route)


def _answer_model(answer: str) -> FunctionModel:
    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(answer)])

    return FunctionModel(respond)


def _router(
    router_model: FunctionModel,
    *,
    mode: Literal['once', 'per_step'] = 'once',
    confidence_threshold: float | None = None,
    fast_model: FunctionModel | None = None,
    capable_model: FunctionModel | None = None,
) -> ModelRouter[object]:
    return ModelRouter(
        choices={
            'fast': ModelChoice(fast_model or _answer_model('fast'), 'Use for lookups.'),
            'capable': ModelChoice(capable_model or _answer_model('capable'), 'Use for difficult work.'),
        },
        router_model=router_model,
        default='capable',
        mode=mode,
        confidence_threshold=confidence_threshold,
    )


class TestModelRouter:
    @pytest.fixture
    def anyio_backend(self) -> str:
        return 'asyncio'

    async def test_routes_initial_prompt_with_literal_menu(self) -> None:
        def inspect(messages: list[ModelMessage], info: AgentInfo) -> None:
            schema = info.output_tools[0].parameters_json_schema
            assert schema['properties']['response']['enum'] == ['fast', 'capable']
            assert info.instructions is not None
            assert 'Use for lookups.' in info.instructions
            prompt = messages[-1].parts[-1]
            assert isinstance(prompt, UserPromptPart)
            assert 'Find the capital of France' in prompt.content

        agent = Agent(capabilities=[_router(_router_model('fast', inspect=inspect))])

        result = await agent.run('Find the capital of France')

        assert result.output == 'fast'
        assert result.usage.requests == 2

    async def test_reserves_a_request_for_the_pending_parent_call(self) -> None:
        router = _router(_router_model('fast'))

        result = await Agent(capabilities=[router]).run('Choose', usage_limits=UsageLimits(request_limit=1))

        assert result.output == 'capable'
        assert result.usage.requests == 1

    async def test_once_reuses_choice_and_per_step_routes_again(self) -> None:
        async def run(mode: Literal['once', 'per_step']) -> tuple[str, int, int, int]:
            router_calls = 0
            fast_calls = 0
            capable_calls = 0

            def route(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                nonlocal router_calls
                router_calls += 1
                choice = 'fast' if router_calls == 1 else 'capable'
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': choice})])

            def fast(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
                nonlocal fast_calls
                fast_calls += 1
                if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                    return ModelResponse(parts=[ToolCallPart('lookup', {})])
                return ModelResponse(parts=[TextPart('fast finished')])

            def capable(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
                nonlocal capable_calls
                capable_calls += 1
                return ModelResponse(parts=[TextPart('capable finished')])

            agent = Agent(
                capabilities=[
                    _router(
                        FunctionModel(route),
                        mode=mode,
                        fast_model=FunctionModel(fast),
                        capable_model=FunctionModel(capable),
                    )
                ]
            )

            @agent.tool_plain
            def lookup() -> str:
                return 'new information'

            result = await agent.run('Research this')
            return result.output, router_calls, fast_calls, capable_calls

        assert await run('once') == ('fast finished', 1, 2, 0)
        assert await run('per_step') == ('capable finished', 2, 1, 1)

    async def test_low_confidence_uses_default(self) -> None:
        router = _router(
            _router_model('fast', provider_details={'confidence': {'response': 0.49}}),
            confidence_threshold=0.5,
        )

        result = await Agent(capabilities=[router]).run('Choose')

        assert result.output == 'capable'

    @pytest.mark.parametrize(
        ('provider_details', 'expected'),
        [
            pytest.param(None, 'fast', id='missing'),
            pytest.param({'confidence': True}, 'fast', id='boolean'),
            pytest.param({'confidence': 0.8}, 'fast', id='scalar'),
            pytest.param({'confidence': {'other': 0.4}}, 'capable', id='mapping'),
            pytest.param({'confidence': {'response': 0.8, 'other': 0.1}}, 'fast', id='response-key'),
            pytest.param({'confidence': {'other': 'unknown'}}, 'fast', id='non-numeric'),
            pytest.param({'confidence': 'unknown'}, 'fast', id='unsupported-scalar'),
            pytest.param({'confidence': float('nan')}, 'capable', id='not-a-number'),
            pytest.param({'confidence': {'response': float('inf')}}, 'capable', id='infinite'),
            pytest.param({'confidence': -0.1}, 'capable', id='below-range'),
            pytest.param({'confidence': {'other': 1.1}}, 'capable', id='above-range'),
        ],
    )
    async def test_confidence_shapes(self, provider_details: Mapping[str, Any] | None, expected: str) -> None:
        router = _router(
            _router_model('fast', provider_details=provider_details),
            confidence_threshold=0.5,
        )

        result = await Agent(capabilities=[router]).run('Choose')

        assert result.output == expected

    async def test_router_error_uses_default(self) -> None:
        def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('router unavailable')

        result = await Agent(capabilities=[_router(FunctionModel(fail))]).run('Choose')

        assert result.output == 'capable'

    async def test_concurrent_runs_have_independent_once_cache(self) -> None:
        calls = 0

        async def route(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': 'fast'})])

        agent = Agent(capabilities=[_router(FunctionModel(route))])

        first, second = await asyncio.gather(agent.run('first'), agent.run('second'))

        assert (first.output, second.output, calls) == ('fast', 'fast', 2)

    async def test_span_records_pick_confidence_and_fallback(
        self, capfire: CaptureLogfire, instrument_all_agents: None
    ) -> None:
        router = _router(
            _router_model('fast', provider_details={'confidence': {'response': 0.4}}),
            confidence_threshold=0.5,
        )
        agent = Agent(capabilities=[router])

        await agent.run('Choose')

        spans = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'model_router.select']
        assert len(spans) == 1
        attributes = spans[0]['attributes']
        assert attributes['model_router.choice'] == 'capable'
        assert attributes['model_router.confidence'] == 0.4
        assert attributes['model_router.fallback_reason'] == 'low_confidence'
        assert attributes['model_router.mode'] == 'once'
        assert attributes['model_router.run_step'] == 1
        assert agent_run_names(capfire).count('model_router') == 1

    async def test_error_span_records_fallback(self, capfire: CaptureLogfire) -> None:
        def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('router unavailable')

        agent = Agent(capabilities=[_router(FunctionModel(fail))])
        agent.instrument = InstrumentationSettings()

        await agent.run('Choose')

        span = next(span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'model_router.select')
        assert span['attributes']['model_router.choice'] == 'capable'
        assert span['attributes']['model_router.fallback_reason'] == 'error'
        assert span['attributes']['model_router.error.type'] == 'RuntimeError'

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            pytest.param({'choices': {}}, 'choices must not be empty', id='empty-choices'),
            pytest.param(
                {'choices': {'': ModelChoice('test', 'description')}},
                'choice names must not be empty',
                id='empty-name',
            ),
            pytest.param(
                {'choices': {'fast': ModelChoice('test', '')}},
                'choice descriptions must not be empty',
                id='empty-description',
            ),
            pytest.param({'default': 'missing'}, 'default must name a configured choice', id='unknown-default'),
            pytest.param({'mode': 'sometimes'}, 'mode must be', id='mode'),
            pytest.param({'confidence_threshold': -0.1}, 'must be between 0 and 1', id='threshold-low'),
            pytest.param({'confidence_threshold': 1.1}, 'must be between 0 and 1', id='threshold-high'),
        ],
    )
    async def test_validates_configuration(self, kwargs: dict[str, Any], message: str) -> None:
        options: dict[str, Any] = {
            'choices': {'fast': ModelChoice('test', 'description')},
            'router_model': _router_model('fast'),
            'default': 'fast',
            **kwargs,
        }

        with pytest.raises(UserError, match=message):
            ModelRouter(**options)

    async def test_an_unresolvable_router_model_is_rejected_at_construction(self) -> None:
        with pytest.raises(UserError, match='Unknown model'):
            ModelRouter(
                choices={'fast': ModelChoice('test', 'description')},
                router_model='nonexistent:model',
                default='fast',
            )

    async def test_the_router_agent_is_not_rebuilt_on_every_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        builds = 0
        original = ModelRouter._instructions  # pyright: ignore[reportPrivateUsage]

        def counting(router: ModelRouter[Any]) -> str:
            nonlocal builds
            builds += 1
            return original(router)

        monkeypatch.setattr(ModelRouter, '_instructions', counting)

        router_calls = 0

        def route(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal router_calls
            router_calls += 1
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': 'fast'})])

        def fast(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart('lookup', {})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            capabilities=[
                _router(FunctionModel(route), mode='per_step', fast_model=FunctionModel(fast)),
            ]
        )

        @agent.tool_plain
        def lookup() -> str:
            return 'new information'

        await agent.run('Research this')

        assert router_calls == 2, 'per_step routes before each step'
        assert builds == 2, 'one agent at construction and one for the run-scoped copy, not one per step'

    async def test_not_agent_spec_serializable(self) -> None:
        assert ModelRouter.get_serialization_name() is None
