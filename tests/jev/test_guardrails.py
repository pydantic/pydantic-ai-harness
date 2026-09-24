"""How `JevCapabilityComposer` sits inside the run's input guardrails, wherever they are configured."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_harness.guardrails import GuardrailResult, InputGuardrail

from ._doubles import Events, Jev, Seen, composer, main_model

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def no_secrets(prompt: str) -> GuardrailResult:
    return GuardrailResult.block('No secrets, please.') if 'secret' in prompt else GuardrailResult.allow()


def redact(prompt: str) -> GuardrailResult:
    return GuardrailResult.replace(prompt.replace('hunter2', '[redacted]'))


class TestInputGuardrails:
    @pytest.mark.parametrize('on', ['agent', 'run'])
    async def test_a_blocked_prompt_never_reaches_jev(self, on: str):
        jev = Jev()
        events = Events()
        guardrail = InputGuardrail[object](guard=no_secrets)
        agent = Agent(main_model([]), capabilities=[composer(jev, []), *([guardrail] if on == 'agent' else [])])
        run_capabilities: list[AbstractCapability[object]] = [guardrail] if on == 'run' else []

        result = await agent.run('the secret is hunter2', event_stream_handler=events, capabilities=run_capabilities)

        assert result.output == 'No secrets, please.'
        assert jev.prompts == []
        assert events.composed == []

    async def test_an_allowed_prompt_is_composed(self):
        jev = Jev()
        agent = Agent(main_model([]), capabilities=[InputGuardrail(guard=no_secrets), composer(jev, [])])

        result = await agent.run('write that down')

        assert result.output == 'strong answered'
        assert jev.prompts == ['write that down']

    @pytest.mark.parametrize('guardrail_first', [True, False])
    async def test_jev_and_the_sub_agent_read_the_redacted_prompt(self, guardrail_first: bool):
        """The composer sits inside the guardrail however the two are listed."""
        jev = Jev()
        seen: list[Seen] = []
        guardrail = InputGuardrail[object](guard=redact)
        picker = composer(jev, seen)
        capabilities = [guardrail, picker] if guardrail_first else [picker, guardrail]
        agent = Agent(main_model([]), capabilities=capabilities)

        result = await agent.run('my password is hunter2')

        assert result.output == 'strong answered'
        assert jev.prompts == ['my password is [redacted]']
        assert [s.prompt for s in seen] == ['my password is [redacted]']
