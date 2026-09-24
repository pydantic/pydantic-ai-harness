"""How `JevCapabilityComposer`'s picks apply to a run."""

from __future__ import annotations

import pytest
from opentelemetry.trace import NoOpTracer
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_ai_harness.jev import JevCapabilityComposer
from pydantic_ai_harness.subagents import ModelOption

from ._doubles import (
    CATALOG,
    Events,
    Guard,
    Jev,
    Notes,
    Seen,
    compose_span,
    composer,
    main_model,
    recording_tracer,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestCompose:
    async def test_the_picked_model_answers_with_the_picked_capabilities(self):
        calls: list[str] = []
        seen: list[Seen] = []
        jev = Jev(capabilities=('notes', 'clock'))
        agent = Agent(main_model(calls), capabilities=[composer(jev, seen)])

        result = await agent.run('write that down')

        assert result.output == 'strong answered'
        assert result.response.model_name == 'strong'
        assert calls == []
        assert [(s.prompt, s.tools) for s in seen] == [('write that down', ['clock_now', 'memo_take'])]
        assert jev.prompts == ['write that down']

    async def test_the_agent_keeps_its_capabilities_instructions_and_output_type(self):
        """The picks join the run, so nothing the agent was configured with is bypassed."""
        seen: list[Seen] = []
        guard = Guard()
        agent = Agent(
            main_model([]), output_type=int, instructions='Be brief.', capabilities=[guard, composer(Jev(), seen)]
        )

        result = await agent.run('count the notes')

        assert result.output == 7
        assert guard.models == ['strong']
        [request] = seen
        assert request.instructions == 'Be brief.'
        assert request.output_tools == ['final_result']

    async def test_a_capability_the_agent_already_has_is_not_added_twice(self):
        seen: list[Seen] = []
        events = Events()
        jev = Jev(capabilities=('notes', 'clock'))
        agent = Agent(main_model([]), capabilities=[Notes(prefix='memo'), composer(jev, seen)])

        await agent.run('write that down', event_stream_handler=events)

        assert [s.tools for s in seen] == [['clock_now', 'memo_take']]
        assert [e.capabilities for e in events.composed] == [('clock',)]

    async def test_the_picked_thinking_reaches_a_model_that_supports_it(self):
        thinking: list[object] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            thinking.append(info.model_request_parameters.thinking)
            return ModelResponse(parts=[TextPart(content='thought')])

        thinker = FunctionModel(respond, profile=ModelProfile(supports_thinking=True))
        jev = Jev(model='deep', thinking='medium')
        agent = Agent(
            main_model([]),
            capabilities=[JevCapabilityComposer(models={'deep': thinker}, catalog=CATALOG, jev_model=jev.model_)],
        )

        await agent.run('write that down')

        assert thinking == ['medium']

    async def test_model_option_settings_override_the_picked_thinking(self):
        requests: list[tuple[object, float | None]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            settings = info.model_settings or {}
            requests.append((info.model_request_parameters.thinking, settings.get('temperature')))
            return ModelResponse(parts=[TextPart(content='done')])

        thinker = FunctionModel(respond, profile=ModelProfile(supports_thinking=True))
        option = ModelOption(thinker, settings=ModelSettings(thinking='xhigh', temperature=0.2))
        jev = Jev(model='deep', thinking='low')
        agent = Agent(
            main_model([]),
            capabilities=[JevCapabilityComposer(models={'deep': option}, catalog=CATALOG, jev_model=jev.model_)],
        )

        await agent.run('write that down')

        assert requests == [('xhigh', 0.2)]

    async def test_a_model_passed_to_the_run_takes_precedence(self):
        calls: list[str] = []
        agent = Agent(TestModel(), capabilities=[composer(Jev(), [])])

        result = await agent.run('write that down', model=main_model(calls))

        assert result.output == 'main answered'
        assert calls == ['main']

    async def test_the_picked_capabilities_last_the_whole_run(self):
        """Jev is asked once, and every model request of the run uses the picks."""
        jev = Jev(capabilities=('notes', 'clock'))
        tools: list[list[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tools.append(sorted(tool.name for tool in info.function_tools))
            if len(tools) == 1:
                return ModelResponse(parts=[ToolCallPart('clock_now', {}), ToolCallPart('memo_take', {})])
            return ModelResponse(parts=[TextPart(content='it is noon')])

        picker = JevCapabilityComposer(models={'strong': FunctionModel(respond)}, catalog=CATALOG, jev_model=jev.model_)
        result = await Agent(main_model([]), capabilities=[picker]).run('what time is it?')

        assert result.output == 'it is noon'
        assert tools == [['clock_now', 'memo_take'], ['clock_now', 'memo_take']]
        assert len(jev.prompts) == 1

    async def test_the_event_reports_the_picks(self):
        events = Events()
        agent = Agent(main_model([]), capabilities=[composer(Jev(thinking='medium', capabilities=('clock',)), [])])

        await agent.run('what time is it?', event_stream_handler=events)

        assert [(e.model, e.thinking, e.capabilities, e.escalated) for e in events.composed] == [
            ('strong', 'medium', ('clock',), False)
        ]

    async def test_a_picker_without_confidence_is_trusted(self):
        agent = Agent(main_model([]), capabilities=[composer(Jev(model='fast', confidence={}), [])])

        result = await agent.run('write that down')

        assert result.output == 'fast answered'

    async def test_text_parts_of_a_multimodal_prompt_are_composed_and_the_whole_prompt_is_kept(self):
        jev = Jev()
        seen: list[Seen] = []
        agent = Agent(main_model([]), capabilities=[composer(jev, seen)])

        await agent.run(['look at this', BinaryContent(data=b'png', media_type='image/png'), 'and that'])

        assert jev.prompts == ['look at this\nand that']
        assert 'BinaryContent' in seen[0].prompt

    async def test_a_run_without_a_new_prompt_composes_from_the_latest_one_in_history(self):
        jev = Jev()
        history: list[ModelMessage] = [
            ModelRequest.user_text_prompt('earlier'),
            ModelResponse(parts=[TextPart(content='reply')]),
            ModelRequest.user_text_prompt('latest'),
        ]
        agent = Agent(main_model([]), capabilities=[composer(jev, [])])

        result = await agent.run(message_history=history)

        assert result.output == 'strong answered'
        assert jev.prompts == ['latest']


class TestUsage:
    async def test_jev_and_the_picked_model_count_toward_the_run(self):
        result = await Agent(main_model([]), capabilities=[composer(Jev(), [])]).run('write that down')

        assert result.usage.requests == 2

    async def test_jev_counts_toward_the_run_limits(self):
        agent = Agent(main_model([]), capabilities=[composer(Jev(), [])])

        with pytest.raises(UsageLimitExceeded):
            await agent.run('write that down', usage_limits=UsageLimits(request_limit=1))


class TestEscalation:
    async def test_an_unsure_model_pick_uses_the_last_entry(self):
        """The capabilities Jev picked are kept; only the uncertain model pick is replaced."""
        seen: list[Seen] = []
        jev = Jev(model='fast', capabilities=('notes',), confidence={'model': 0.3})
        agent = Agent(main_model([]), capabilities=[composer(jev, seen)])

        result = await agent.run('write that down')

        assert result.output == 'strong answered'
        assert [s.tools for s in seen] == [['memo_take']]

    async def test_unsure_model_can_be_chosen(self):
        jev = Jev(model='strong', confidence={'model': 0.1})
        agent = Agent(main_model([]), capabilities=[composer(jev, [], unsure_model='fast')])

        result = await agent.run('write that down')

        assert result.output == 'fast answered'

    async def test_threshold_is_inclusive(self):
        agent = Agent(main_model([]), capabilities=[composer(Jev(model='fast', confidence={'model': 0.4}), [])])

        result = await agent.run('write that down')

        assert result.output == 'fast answered'

    async def test_the_event_says_it_escalated(self):
        events = Events()
        jev = Jev(model='fast', capabilities=('clock',), confidence={'model': 0.2})
        agent = Agent(main_model([]), capabilities=[composer(jev, [])])

        await agent.run('what time is it?', event_stream_handler=events)

        assert [(e.model, e.escalated) for e in events.composed] == [('strong', True)]


class TestFallthrough:
    async def test_no_capabilities_leaves_the_run_as_configured(self):
        calls: list[str] = []
        events = Events()
        agent = Agent(main_model(calls), capabilities=[composer(Jev(capabilities=()), [])])

        result = await agent.run('hi', event_stream_handler=events)

        assert result.output == 'main answered'
        assert calls == ['main']
        assert events.composed == []

    async def test_a_prompt_without_text_is_not_composed(self):
        jev = Jev()
        agent = Agent(main_model([]), capabilities=[composer(jev, [])])

        result = await agent.run([BinaryContent(data=b'png', media_type='image/png')])

        assert result.output == 'main answered'
        assert jev.prompts == []

    async def test_a_run_without_a_prompt_is_not_composed(self):
        picker = composer(Jev(), [])

        run_capability = await picker.for_run(RunContext(deps=None, model=TestModel(), usage=RunUsage()))

        assert run_capability is picker

    @pytest.mark.parametrize(
        'history',
        [
            [ModelRequest.user_text_prompt('hi'), ModelResponse(parts=[TextPart(content='hello')])],
            [ModelRequest(parts=[ToolReturnPart(tool_name='clock_now', content='noon', tool_call_id='1')])],
        ],
        ids=['ends-with-a-response', 'pending-request-without-a-prompt'],
    )
    async def test_history_without_a_pending_prompt_is_not_composed(self, history: list[ModelMessage]):
        jev = Jev()
        picker = composer(jev, [])
        ctx = RunContext[object](deps=None, model=TestModel(), usage=RunUsage(), messages=history)

        assert await picker.for_run(ctx) is picker
        assert jev.prompts == []

    async def test_a_run_context_without_an_agent_still_composes(self):
        jev = Jev(capabilities=('clock',))
        ctx = RunContext[object](
            deps=None, model=TestModel(), usage=RunUsage(), prompt='what time is it?', tracer=NoOpTracer()
        )

        run_capability = await composer(jev, []).for_run(ctx)

        assert run_capability.get_model() is not None
        assert jev.prompts == ['what time is it?']


class TestTracing:
    async def test_the_span_records_the_picks(self):
        provider, exporter = recording_tracer()
        agent = Agent(main_model([]), capabilities=[composer(Jev(capabilities=('notes', 'clock')), [])])
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=False)

        await agent.run('write that down')

        assert dict(compose_span(exporter).attributes or {}) == {
            'jev_composer.model': 'strong',
            'jev_composer.thinking': 'high',
            'jev_composer.capabilities': ('notes', 'clock'),
            'jev_composer.action': 'compose',
            'jev_composer.run_model': 'strong',
            'jev_composer.confidence.model': 0.9,
            'jev_composer.confidence.thinking': 0.8,
        }

    async def test_the_span_records_an_escalation(self):
        provider, exporter = recording_tracer()
        jev = Jev(model='fast', capabilities=('notes',), confidence={'model': 0.2})
        agent = Agent(main_model([]), capabilities=[composer(jev, [])])
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=False)

        await agent.run('write that down')

        attributes = dict(compose_span(exporter).attributes or {})
        assert attributes['jev_composer.action'] == 'escalate'
        assert attributes['jev_composer.model'] == 'fast'
        assert attributes['jev_composer.run_model'] == 'strong'

    async def test_the_span_records_a_fallthrough_and_the_prompt_when_allowed(self):
        provider, exporter = recording_tracer()
        agent = Agent(main_model([]), capabilities=[composer(Jev(capabilities=()), [])])
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=True)

        await agent.run('hi')

        attributes = dict(compose_span(exporter).attributes or {})
        assert attributes['jev_composer.action'] == 'fallthrough'
        assert 'jev_composer.run_model' not in attributes
        assert attributes['jev_composer.prompt'] == 'hi'

    async def test_nothing_is_recorded_without_a_recording_span(self):
        agent = Agent(main_model([]), capabilities=[composer(Jev(), [])])

        result = await agent.run('write that down')

        assert result.output == 'strong answered'
