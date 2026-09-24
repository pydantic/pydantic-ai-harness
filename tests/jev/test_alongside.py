"""How `JevCapabilityComposer` works alongside the agent's input guardrails and its other capabilities."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings

from pydantic_ai_harness.guardrails import GuardrailResult, InputGuardrail
from pydantic_ai_harness.jev import JevCapabilityComposer

from ._doubles import CATALOG, Events, Guard, Jev, Notes, Seen, compose_span, composer, main_model, recording_tracer

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def no_secrets(prompt: str) -> GuardrailResult:
    return GuardrailResult.block('No secrets, please.') if 'secret' in prompt else GuardrailResult.allow()


def redact(prompt: str) -> GuardrailResult:
    return GuardrailResult.replace(prompt.replace('hunter2', '[redacted]'))


class TestInputGuardrails:
    async def test_a_blocked_prompt_never_reaches_jev(self):
        jev = Jev()
        events = Events()
        agent = Agent(main_model([]), capabilities=[InputGuardrail(guard=no_secrets), composer(jev, [])])

        result = await agent.run('the secret is hunter2', event_stream_handler=events)

        assert result.output == 'No secrets, please.'
        assert jev.prompts == []
        assert events.composed == []

    async def test_an_allowed_prompt_is_composed(self):
        jev = Jev()
        agent = Agent(main_model([]), capabilities=[InputGuardrail(guard=no_secrets), composer(jev, [])])

        result = await agent.run('write that down')

        assert result.output == 'strong answered'
        assert jev.prompts == ['write that down']

    async def test_jev_reads_the_prompt_as_a_guardrail_redacts_it(self):
        jev = Jev()
        seen: list[Seen] = []
        agent = Agent(main_model([]), capabilities=[Guard(), InputGuardrail(guard=redact), composer(jev, seen)])

        result = await agent.run('my password is hunter2')

        assert jev.prompts == ['my password is [redacted]']
        assert result.output == 'strong answered'
        assert [s.prompt for s in seen] == ['my password is [redacted]']

    async def test_a_replacement_that_is_not_text_is_not_composed(self):
        jev = Jev()
        agent = Agent(
            main_model([]),
            capabilities=[InputGuardrail(guard=lambda prompt: GuardrailResult.replace(7)), composer(jev, [])],
        )

        with pytest.raises(UserError):
            await agent.run('write that down')

        assert jev.prompts == []

    async def test_the_span_records_a_block_without_the_prompt(self):
        provider, exporter = recording_tracer()
        agent = Agent(main_model([]), capabilities=[InputGuardrail(guard=no_secrets), composer(Jev(), [])])
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=True)

        await agent.run('the secret is hunter2')

        assert dict(compose_span(exporter).attributes or {}) == {'jev_composer.action': 'blocked'}


def calls_then_answers(tool: str, returns: list[str], tools: list[list[str]]) -> FunctionModel:
    """A menu model that calls `tool` once, then answers; recording the tools it was offered and what came back."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools.append(sorted(t.name for t in info.function_tools))
        results = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not results:
            return ModelResponse(parts=[ToolCallPart(tool, {})])
        returns.extend(str(p.content) for p in results)
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(respond)


class TestOtherCapabilities:
    async def test_a_capability_passed_to_the_run_takes_the_place_of_the_pick(self):
        """The run's own configuration wins, so a pick cannot widen what the run was given."""
        returns: list[str] = []
        tools: list[list[str]] = []
        picker = JevCapabilityComposer(
            models={'strong': calls_then_answers('memo_take', returns, tools)}, catalog=CATALOG, jev_model=Jev().model_
        )

        result = await Agent(main_model([]), capabilities=[picker]).run(
            'write that down', capabilities=[Notes(prefix='memo', reply='the run')]
        )

        assert result.output == 'done'
        assert tools == [['memo_take'], ['memo_take']]
        assert returns == ['the run']

    async def test_two_composers_that_pick_the_same_entry_share_it(self):
        seen: list[Seen] = []
        agent = Agent(main_model([]), capabilities=[composer(Jev(), seen), composer(Jev(), seen)])

        result = await agent.run('write that down')

        assert result.output == 'strong answered'
        assert [s.tools for s in seen] == [['memo_take']]

    async def test_an_entry_whose_toolset_is_built_per_step(self):
        returns: list[str] = []
        tools: list[list[str]] = []
        picker = JevCapabilityComposer(
            models={'strong': calls_then_answers('dial', returns, tools)},
            catalog=CATALOG,
            jev_model=Jev(capabilities=('dial',)).model_,
        )

        await Agent(main_model([]), capabilities=[picker]).run('call mum')

        assert tools == [['dial'], ['dial']]
        assert returns == ['dialled']
