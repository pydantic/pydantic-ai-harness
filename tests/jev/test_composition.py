"""How `JevCapabilityComposer` hands a turn to the sub-agent Jev composed."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

import pytest
from pydantic_ai import Agent, AgentStreamEvent, FunctionToolCallEvent, PromptedOutput
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, Tool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.jev import ComposableCapability, JevCapabilityComposer
from pydantic_ai_harness.subagents import ModelOption

from ._doubles import (
    CATALOG,
    Clock,
    Events,
    Jev,
    Notes,
    Seen,
    compose_span,
    composer,
    main_model,
    menu_model,
    recording_tracer,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def tool_then_answer(tools: list[list[str]]) -> FunctionModel:
    """A menu model that calls every tool it is offered once, then answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools.append(sorted(tool.name for tool in info.function_tools))
        if len(tools) == 1:
            return ModelResponse(parts=[ToolCallPart(tool.name, {}) for tool in info.function_tools])
        return ModelResponse(parts=[TextPart(content='done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        tools.append(sorted(tool.name for tool in info.function_tools))
        if len(tools) == 1:
            yield {i: DeltaToolCall(name=tool.name, json_args='{}') for i, tool in enumerate(info.function_tools)}
        else:
            yield 'done'

    return FunctionModel(respond, stream_function=stream, model_name='worker')


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

    async def test_the_sub_agent_runs_its_tools_and_only_its_answer_is_recorded(self):
        tools: list[list[str]] = []
        jev = Jev(model='worker', capabilities=('notes', 'clock'))
        picker = JevCapabilityComposer[object](
            models={'worker': tool_then_answer(tools)}, catalog=CATALOG, jev_model=jev.model_
        )

        result = await Agent(main_model([]), capabilities=[picker]).run('what time is it?')

        assert tools == [['clock_now', 'memo_take'], ['clock_now', 'memo_take']]
        assert len(jev.prompts) == 1
        [request, response] = result.all_messages()
        assert isinstance(request, ModelRequest)
        assert response.parts == [TextPart(content='done')]
        assert isinstance(response, ModelResponse) and response.model_name == 'worker'

    async def test_the_sub_agent_continues_the_conversation(self):
        history_seen: list[list[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            history_seen.append(
                [str(p.content) for m in messages for p in m.parts if isinstance(p, UserPromptPart | TextPart)]
            )
            return ModelResponse(parts=[TextPart(content='fixed')])

        jev = Jev(model='deep')
        picker = JevCapabilityComposer[object](
            models={'deep': FunctionModel(respond)}, catalog=CATALOG, jev_model=jev.model_
        )
        history: list[ModelMessage] = [
            ModelRequest.user_text_prompt('the test fails'),
            ModelResponse(parts=[TextPart(content='which one?')]),
        ]

        result = await Agent(main_model([]), capabilities=[picker]).run('the first one', message_history=history)

        assert result.output == 'fixed'
        assert history_seen == [['the test fails', 'which one?', 'the first one']]
        assert jev.prompts == ['the first one']
        assert len(result.all_messages()) == 4

    async def test_the_sub_agent_gets_its_own_instructions_and_the_deps(self):
        seen_deps: list[object] = []
        seen: list[Seen] = []

        @Agent(main_model([]), deps_type=str, instructions='Be the main agent.').tool
        def unused(ctx: RunContext[str]) -> str:  # pragma: no cover - the main model is not called
            return ctx.deps

        async def remember_deps(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            seen_deps.append(ctx.deps)
            async for _ in stream:
                pass

        picker = composer(Jev(), seen, instructions='Be the sub-agent.', event_stream_handler=remember_deps)
        agent = Agent(main_model([]), deps_type=str, instructions='Be the main agent.', capabilities=[picker])

        await agent.run('write that down', deps='the deps')

        assert [s.instructions for s in seen] == ['Be the sub-agent.']
        assert set(seen_deps) == {'the deps'}

    async def test_the_sub_agent_events_reach_the_event_stream_handler(self):
        calls: list[str] = []

        async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                if isinstance(event, FunctionToolCallEvent):
                    calls.append(event.part.tool_name)

        jev = Jev(model='worker', capabilities=('clock',))
        picker = JevCapabilityComposer[object](
            models={'worker': tool_then_answer([])}, catalog=CATALOG, jev_model=jev.model_, event_stream_handler=handler
        )

        await Agent(main_model([]), capabilities=[picker]).run('what time is it?')

        assert calls == ['clock_now']

    async def test_a_streamed_run_gets_the_answer(self):
        agent = Agent(main_model([]), capabilities=[composer(Jev(), [])])

        async with agent.run_stream('write that down') as result:
            output = await result.get_output()

        assert output == 'strong answered'

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


class TestSharedCapabilities:
    async def test_shared_capabilities_join_every_sub_agent(self):
        seen: list[Seen] = []
        agent = Agent(
            main_model([]), capabilities=[composer(Jev(capabilities=('notes',)), seen, shared_capabilities=[Clock()])]
        )

        await agent.run('write that down')

        assert [s.tools for s in seen] == [['clock_now', 'memo_take']]

    async def test_a_pick_of_a_shared_class_is_left_out(self):
        seen: list[Seen] = []
        events = Events()
        shared = [Notes(prefix='mine')]
        agent = Agent(
            main_model([]),
            capabilities=[composer(Jev(capabilities=('notes', 'clock')), seen, shared_capabilities=shared)],
        )

        await agent.run('write that down', event_stream_handler=events)

        assert [s.tools for s in seen] == [['clock_now', 'mine_take']]
        assert [e.capabilities for e in events.composed] == [('clock',)]

    async def test_the_agent_may_have_a_picked_capability_too(self):
        """The sub-agent is a separate run, so the agent's own copy of a picked capability does not conflict."""
        seen: list[Seen] = []
        agent = Agent(main_model([]), capabilities=[Notes(prefix='memo'), composer(Jev(), seen)])

        result = await agent.run('write that down', capabilities=[Clock()])

        assert result.output == 'strong answered'
        assert [s.tools for s in seen] == [['memo_take']]


class TestApproval:
    async def test_deferred_approval_inside_the_sub_agent_is_an_error(self):
        """A text-only sub-agent cannot end with `DeferredToolRequests`, so a tool that defers its approval fails."""

        @dataclass
        class Guarded(AbstractCapability[object]):
            def get_toolset(self) -> FunctionToolset[object]:
                def delete() -> str:  # pragma: no cover - never approved
                    return 'deleted'

                return FunctionToolset([Tool(delete, requires_approval=True)])

        jev = Jev(model='worker', capabilities=('guarded',))
        picker = JevCapabilityComposer[object](
            models={'worker': tool_then_answer([])},
            catalog={'guarded': ComposableCapability(description='Delete things', capability=Guarded)},
            jev_model=jev.model_,
        )

        with pytest.raises(UserError, match='DeferredToolRequests'):
            await Agent(main_model([]), capabilities=[picker]).run('delete it')


class TestUsage:
    async def test_jev_and_the_sub_agent_count_toward_the_run(self):
        """Core also counts the agent's own request step, which the sub-agent's answer stands in for."""
        result = await Agent(main_model([]), capabilities=[composer(Jev(), [])]).run('write that down')

        assert result.usage.requests == 3

    @pytest.mark.parametrize('request_limit', [1, 2], ids=['jev', 'sub-agent'])
    async def test_the_run_limits_cover_jev_and_the_sub_agent(self, request_limit: int):
        tools: list[list[str]] = []
        jev = Jev(model='worker', capabilities=('clock',))
        picker = JevCapabilityComposer[object](
            models={'worker': tool_then_answer(tools)}, catalog=CATALOG, jev_model=jev.model_
        )

        with pytest.raises(UsageLimitExceeded):
            await Agent(main_model([]), capabilities=[picker]).run(
                'what time is it?', usage_limits=UsageLimits(request_limit=request_limit)
            )

        assert len(tools) == request_limit - 1


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
    async def test_no_capabilities_leaves_the_turn_to_the_agent(self):
        calls: list[str] = []
        events = Events()
        agent = Agent(main_model(calls), capabilities=[composer(Jev(capabilities=()), [])])

        result = await agent.run('hi', event_stream_handler=events)

        assert result.output == 'main answered'
        assert calls == ['main']
        assert events.composed == []

    async def test_jev_is_asked_only_on_the_first_request(self):
        jev = Jev(capabilities=())
        agent = Agent(tool_then_answer([]), toolsets=[Clock().get_toolset()], capabilities=[composer(jev, [])])

        result = await agent.run('what time is it?')

        assert result.output == 'done'
        assert len(jev.prompts) == 1

    async def test_a_run_that_needs_structured_output_is_not_composed(self):
        """The sub-agent answers in text, which an `output_type` of `int` cannot take."""
        jev = Jev()
        seen: list[Seen] = []
        agent = Agent(menu_model('main', []), output_type=int, capabilities=[composer(jev, seen)])

        result = await agent.run('count the notes')

        assert result.output == 7
        assert jev.prompts == []
        assert seen == []

    async def test_a_run_with_prompted_output_is_not_composed(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"response": 7}')])

        jev = Jev()
        agent = Agent(FunctionModel(respond), output_type=PromptedOutput(int), capabilities=[composer(jev, [])])

        result = await agent.run('count the notes')

        assert result.output == 7
        assert jev.prompts == []

    async def test_a_run_that_takes_text_or_structured_output_is_composed(self):
        agent = Agent(main_model([]), output_type=[str, int], capabilities=[composer(Jev(), [])])

        result = await agent.run('write that down')

        assert result.output == 'strong answered'

    async def test_a_prompt_without_text_is_not_composed(self):
        jev = Jev()
        agent = Agent(main_model([]), capabilities=[composer(jev, [])])

        result = await agent.run([BinaryContent(data=b'png', media_type='image/png')])

        assert result.output == 'main answered'
        assert jev.prompts == []

    async def test_a_request_without_a_prompt_is_not_composed(self):
        """A run resuming from a tool result sends no new prompt."""
        jev = Jev()
        history: list[ModelMessage] = [
            ModelRequest.user_text_prompt('what time is it?'),
            ModelResponse(parts=[ToolCallPart('clock_now', {}, tool_call_id='1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='clock_now', content='noon', tool_call_id='1')]),
        ]
        agent = Agent(main_model([]), capabilities=[composer(jev, [])])

        result = await agent.run(message_history=history)

        assert result.output == 'main answered'
        assert jev.prompts == []


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
