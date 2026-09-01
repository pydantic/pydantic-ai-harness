"""What an `AgentConfig` written by a newer Logfire UI or Pydantic AI costs an older SDK.

The stored schema is permissive at every level so a newer writer's keys and values reach the SDK at
all, which makes this module the other half of that contract: whatever gets through has to degrade
the narrowest unit that contains it. An `AgentConfig` that fails validation is reverted *whole* by
Logfire's resolution fallback, so a value this SDK can't act on must cost one setting, one
instruction block, or one tool override -- never the instructions, the model, and every other
override along with it.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from logfire.testing import CaptureLogfire
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.logfire import (
    AgentConfig,
    AgentConfigSettings,
    AgentControl,
    InstructionBlock,
    ToolDefinitionOverride,
    _agent_control,
)

from ._helpers import published_value, variables_provider

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _forget_warned_drops() -> None:
    """Start each test with an empty once-per-process guard so every drop warns independently."""
    _agent_control._warned_drops.clear()


# Both list sections start out with more than one entry, so a dropped entry's siblings are what shows
# the drop was narrow. Kept as module constants because `assert_other_sections_survived` defaults to
# the validated form of the same content.
FULL_INSTRUCTIONS: Any = ['Be concise.', {'id': 'agent', 'instructions': 'You are a refund specialist.'}]
FULL_TOOL_DEFINITIONS: Any = [{'name': 'get_weather', 'description': 'Look it up.'}]

SURVIVING_INSTRUCTIONS = [
    InstructionBlock(instructions='Be concise.'),
    InstructionBlock(id='agent', instructions='You are a refund specialist.'),
]
SURVIVING_TOOL_DEFINITIONS = [ToolDefinitionOverride(name='get_weather', description='Look it up.')]


def full_value(
    *, instructions: Any = FULL_INSTRUCTIONS, tool_definitions: Any = FULL_TOOL_DEFINITIONS, **settings: Any
) -> dict[str, Any]:
    """A config with every section populated, so a drop's blast radius is visible in the rest.

    Keyword arguments are merged into the `settings` section; a test whose malformed value belongs to
    one of the list sections replaces that section instead, leaving the others as the control group.
    """
    return {
        'instructions': instructions,
        'model': 'openai:gpt-5',
        'settings': {'temperature': 0.4, **settings},
        'tool_definitions': tool_definitions,
    }


def assert_other_sections_survived(
    config: AgentConfig,
    *,
    instructions: list[InstructionBlock] = SURVIVING_INSTRUCTIONS,
    tool_definitions: list[ToolDefinitionOverride] = SURVIVING_TOOL_DEFINITIONS,
) -> None:
    """Assert nothing but the entry under test was lost.

    A drop inside a list section keeps the section itself, so a test that drops one of its entries
    passes what should remain of it; every other section is expected untouched.
    """
    assert config.instructions == instructions
    assert config.model == 'openai:gpt-5'
    assert config.settings is not None and config.settings.temperature == 0.4
    assert config.tool_definitions == tool_definitions


def test_unrecognized_thinking_drops_only_that_setting() -> None:
    with pytest.warns(UserWarning, match=r"sets 'thinking' to 'ultra', which this version of the SDK"):
        config = AgentConfig.model_validate(full_value(thinking='ultra', service_tier='flex'))
    assert config.settings is not None
    assert config.settings.thinking is None
    assert config.settings.service_tier == 'flex'
    # Dropped, not smuggled through as an extra key, so `_lower_settings` never emits it.
    assert config.settings.model_extra == {}
    assert_other_sections_survived(config)


def test_unrecognized_service_tier_drops_only_that_setting() -> None:
    with pytest.warns(UserWarning, match=r"sets 'service_tier' to 'realtime'"):
        config = AgentConfig.model_validate(full_value(thinking='high', service_tier='realtime'))
    assert config.settings is not None
    assert config.settings.thinking == 'high'
    assert config.settings.service_tier is None
    assert_other_sections_survived(config)


def test_recognized_values_are_untouched() -> None:
    with warnings.catch_warnings(record=True) as caught:
        settings = AgentConfigSettings.model_validate({'thinking': True, 'service_tier': 'priority'})
    assert caught == []
    assert (settings.thinking, settings.service_tier) == (True, 'priority')


def test_unknown_keys_still_flow_through_untouched() -> None:
    # The key-level tolerance this SDK already promised: only *known* fields with unrecognized values
    # are dropped, so a newer UI's keys keep reaching `extra='allow'` and the ignored top level. An
    # unknown key inside a list entry is ignored by the entry's own model, costing nothing either.
    with warnings.catch_warnings(record=True) as caught:
        config = AgentConfig.model_validate(
            {
                'future_section': {'anything': 1},
                'settings': {'future_setting': 'raw json', 'thinking': 'high'},
                'tool_definitions': [{'name': 'get_weather', 'future_override': ['x'], 'description': 'Look it up.'}],
                'instructions': [{'id': 'agent', 'instructions': 'Be concise.', 'future_field': 1}],
            }
        )
    assert caught == []
    assert config.settings is not None
    assert config.settings.model_extra == {'future_setting': 'raw json'}
    assert config.settings.thinking == 'high'
    assert config.tool_definitions == SURVIVING_TOOL_DEFINITIONS
    assert config.instructions == [InstructionBlock(id='agent', instructions='Be concise.')]


def test_invalid_override_drops_only_that_tool() -> None:
    with pytest.warns(UserWarning, match=r"override \{'name': 'get_forecast'.*\} is invalid -- new_name=''"):
        config = AgentConfig.model_validate(
            full_value(
                tool_definitions=[
                    *FULL_TOOL_DEFINITIONS,
                    {'name': 'get_forecast', 'new_name': '', 'description': 'Dropped with its entry.'},
                ]
            )
        )
    assert_other_sections_survived(config)


def test_override_that_is_not_an_object_drops_only_that_tool() -> None:
    with pytest.warns(UserWarning, match=r"override 'nope' is invalid -- override='nope'"):
        config = AgentConfig.model_validate(full_value(tool_definitions=[*FULL_TOOL_DEFINITIONS, 'nope']))
    assert_other_sections_survived(config)


def test_override_that_names_no_tool_drops_only_that_entry() -> None:
    # `name` is what an overlay is addressed by, so an entry without one cannot be applied to
    # anything. The stored schema requires it at write time; this is the same rule one SDK version
    # later, for a value that was written before the schema said so.
    with pytest.warns(UserWarning, match=r"override \{'description': 'Nothing to patch\.'\} is invalid -- name="):
        config = AgentConfig.model_validate(
            full_value(tool_definitions=[*FULL_TOOL_DEFINITIONS, {'description': 'Nothing to patch.'}])
        )
    assert_other_sections_survived(config)


def test_instruction_entry_that_is_not_a_string_or_object_drops_only_itself() -> None:
    with pytest.warns(UserWarning, match=r'Managed instruction entry 5 is invalid -- entry=5'):
        config = AgentConfig.model_validate(full_value(instructions=[*FULL_INSTRUCTIONS, 5]))
    assert_other_sections_survived(config)


def test_instruction_entries_are_bounded_in_total_not_just_individually() -> None:
    """The bound is on what this section adds to every request, so splitting the text can't evade it.

    A section written as one string is capped, so the same text written as several entries has to be
    capped too. Entries are kept in order until the budget runs out; the ones that don't fit drop
    like any other invalid entry, leaving every other section alone.
    """
    half = 'x' * (_agent_control._MAX_MODEL_FACING_TEXT_LENGTH // 2)
    with pytest.warns(UserWarning, match=r'does not fit in the \d+ remaining of the \d+-character limit'):
        config = AgentConfig.model_validate(full_value(instructions=[half, half, half]))

    # Two fit; the third is what pushes the section past the limit.
    assert_other_sections_survived(
        config, instructions=[InstructionBlock(instructions=half), InstructionBlock(instructions=half)]
    )


def test_instruction_entry_with_empty_text_drops_only_itself() -> None:
    # `''` is a half-filled field, not a way to blank a block: `instructions: null` is how a block is
    # dropped, and keeping the two distinguishable is worth losing the entry that confuses them.
    with pytest.warns(UserWarning, match=r"entry \{'id': 'agent:today', 'instructions': ''\} is invalid"):
        config = AgentConfig.model_validate(
            full_value(instructions=[*FULL_INSTRUCTIONS, {'id': 'agent:today', 'instructions': ''}])
        )
    assert_other_sections_survived(config)


def test_instruction_entry_with_neither_id_nor_text_drops_only_itself() -> None:
    # Validates as an `InstructionBlock` but says nothing: no `id` to address and no text to add. A
    # `dynamic` flag alone is exactly the shape a UI produces from a half-filled row.
    with pytest.warns(UserWarning, match=r"entry \{'dynamic': True\} has neither an `id` to address"):
        config = AgentConfig.model_validate(full_value(instructions=[*FULL_INSTRUCTIONS, {'dynamic': True}]))
    assert_other_sections_survived(config)


def test_repeated_resolutions_warn_once() -> None:
    # A managed config is resolved on every run, so the same unrecognized value must not warn per run.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(3):
            AgentConfig.model_validate(full_value(thinking='ultra'))
        AgentConfig.model_validate(full_value(thinking='hyper'))
    assert [str(warning.message).split(',')[0] for warning in caught] == [
        "Managed agent config sets 'thinking' to 'ultra'",
        "Managed agent config sets 'thinking' to 'hyper'",
    ]


def test_malformed_sections_drop_without_losing_siblings() -> None:
    # Remote values can predate the stored schema or bypass its editor. Degrading the smallest section
    # keeps unrelated managed behavior active instead of reverting the complete config to code.
    malformed: list[dict[str, Any]] = [
        {'settings': 'nope'},
        {'tool_definitions': {'get_weather': {}}},
        {'instructions': 5},
    ]
    for value in malformed:
        with pytest.warns(UserWarning):
            assert AgentConfig.model_validate({**value, 'model': 'test'}) == AgentConfig(model='test')
    with pytest.raises(ValidationError):
        AgentConfigSettings.model_validate(['nope'])


def test_a_bare_empty_instructions_string_drops_without_losing_siblings() -> None:
    with pytest.warns(UserWarning, match='instructions section is invalid'):
        assert AgentConfig.model_validate({'instructions': '', 'model': 'test'}) == AgentConfig(model='test')


async def test_agent_keeps_managed_config_around_a_dropped_setting() -> None:
    seen: list[dict[str, object]] = []

    def capture(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    with pytest.warns(UserWarning, match=r"sets 'thinking' to 'ultra'"):
        config = AgentConfig.model_validate({'settings': {'temperature': 0.4, 'thinking': 'ultra'}})
    await Agent(FunctionModel(capture), capabilities=[AgentControl('skew', default=config)]).run('hello')
    assert seen == [{'temperature': 0.4}]


async def test_an_empty_model_is_refused_by_validation_and_the_config_degrades(capfire: CaptureLogfire) -> None:
    # The one value that cannot be degraded field-by-field. Reproduced before `model` was constrained:
    # `{'model': ''}` is a perfectly well-formed `AgentConfig`, so nothing downstream caught it and
    # Pydantic AI raised `UserError: Unknown model:` on *every* run of the agent -- and the Logfire UI
    # produced it from two clicks. Refusing it in the model is what turns that into a resolution
    # fallback: the whole config reverts to code, which is the correct blast radius for a value this
    # malformed, and the sibling `instructions` published alongside it are deliberately lost with it.
    with pytest.raises(ValidationError, match='String should have at least 1 character'):
        AgentConfig.model_validate({'model': ''})

    published = {'model': '', 'instructions': 'MANAGED: never reaches the model.'}
    config = published_value('agent__empty_model', published)
    capability = AgentControl('empty_model', instructions='CODE: base prompt.', label='production')
    agent = Agent(TestModel(), instructions='code', capabilities=[capability])
    # Deliberately not asserting the warning logfire emits on a failed resolution: whether it warns, and
    # with what, is its business and varies across the range this package supports -- the floor emits
    # nothing, which failed the `lowest-versions` job while the locked version passed. What has to hold
    # is the fallback itself.
    with variables_provider(capfire, config), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = await agent.run('hello')
    instructions = [m.instructions for m in result.all_messages() if isinstance(m, ModelRequest)]
    assert instructions == ['code\n\nCODE: base prompt.']
