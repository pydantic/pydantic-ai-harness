from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any, cast

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import Rollout, Variable, VariableConfig, VariablesConfig
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import AgentControl as RootAgentControl
from pydantic_ai_harness.logfire import (
    AGENT_CONFIG_JSON_SCHEMA,
    AgentConfig,
    AgentConfigSettings,
    AgentControl,
    InstructionBlock,
    ToolDefinitionOverride,
    _agent_control,
    _managed_variable,
)

from ._helpers import advertised, capture_tools, get_weather, published_value, variables_provider

pytestmark = pytest.mark.anyio

assert RootAgentControl is AgentControl


@pytest.fixture(autouse=True)
def _forget_warned_drops() -> None:
    """Start each test with an empty once-per-process guard so every drop warns independently."""
    _agent_control._warned_drops.clear()
    _agent_control._reset_baseline_publish_guard()


def instructions_seen(messages: list[ModelMessage]) -> list[str]:
    return [m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions is not None]


async def test_empty_config_keeps_code_behavior() -> None:
    result = await Agent(TestModel(), instructions='code', capabilities=[AgentControl('empty')]).run('hello')
    assert instructions_seen(result.all_messages()) == ['code']


async def test_managed_instructions_are_appended_not_replaced() -> None:
    capability = AgentControl('instructions', default=AgentConfig(instructions='managed'))
    agent = Agent(
        TestModel(),
        instructions='code',
        toolsets=[FunctionToolset[object](instructions='toolset')],
        capabilities=[capability],
    )

    @agent.instructions
    def dynamic(ctx: RunContext[object]) -> str:
        return 'dynamic'

    result = await agent.run('hello')
    # A capability can only contribute instructions, so everything code-defined still reaches the
    # model. Static text is grouped ahead of dynamic text (for prompt-cache stability) and source
    # order is kept within each group, putting the managed value after the agent's own.
    assert instructions_seen(result.all_messages()) == ['code\n\ntoolset\n\ndynamic\n\nmanaged']


async def test_published_instructions_supersede_the_code_side_default(capfire: CaptureLogfire) -> None:
    config = published_value('agent__base_prompt', {'instructions': 'published'})
    capability = AgentControl('base_prompt', instructions='code-side base', label='production')
    with variables_provider(capfire, config):
        result = await Agent(TestModel(), instructions='code', capabilities=[capability]).run('hello')
    # The capability contributes the published value *or* its default, never both -- which is what
    # makes the capability, rather than the agent, the place for a base prompt you mean to manage.
    assert instructions_seen(result.all_messages()) == ['code\n\npublished']


async def test_instructions_shorthand_is_equivalent_to_default() -> None:
    shorthand = AgentControl('shorthand', instructions='base')
    assert shorthand.default == AgentConfig(instructions='base')

    seen: list[list[str]] = []
    for capability in (shorthand, AgentControl('long_form', default=AgentConfig(instructions='base'))):
        result = await Agent(TestModel(), instructions='code', capabilities=[capability]).run('hello')
        seen.append(instructions_seen(result.all_messages()))
    assert seen == [['code\n\nbase'], ['code\n\nbase']]


def test_instructions_and_default_together_raise() -> None:
    with pytest.raises(UserError, match='shorthand for `default=AgentConfig'):
        AgentControl('ambiguous', instructions='base', default=AgentConfig(model='test'))


def test_code_side_instructions_with_no_text_raise_user_error() -> None:
    # A published value that fails validation is remote data and degrades to code; this is a mistake in
    # the code itself, so it gets the same `UserError` treatment as passing `instructions` and `default`
    # together, rather than a Pydantic union traceback out of `__post_init__`.
    with pytest.raises(UserError, match='which has no text to contribute'):
        AgentControl('empty', instructions='')


def test_a_code_side_list_of_empty_entries_degrades_rather_than_raising() -> None:
    # Not an inconsistency with the above: an entry is the unit of degradation, so a list drops the bad
    # entries and keeps whatever else it holds. There is nowhere for the validator to learn that *this*
    # list came from code rather than from Logfire, and a warning still surfaces the drop.
    with pytest.warns(UserWarning, match=r"entry '' is invalid -- instructions=''"):
        capability = AgentControl('empty_entries', instructions=['', 'kept'])
    assert capability.default == AgentConfig(instructions=[InstructionBlock(instructions='kept')])


def weather_toolset() -> FunctionToolset[object]:
    """A toolset with an `id` and instructions of its own, so `toolset:weather` is addressable."""
    toolset = FunctionToolset[object]([get_weather], id='weather')

    @toolset.instructions
    def call_weather_first(_ctx: RunContext[object]) -> str:
        return 'TOOLSET: call get_weather first.'

    return toolset


def capture_instructions(seen: list[InstructionPart]) -> FunctionModel:
    """A model that records the instruction blocks it is shown, then ends the run.

    The parts, not the joined string, are what a block-addressing config has to be judged on: only
    they carry the `id` an override addresses and the `dynamic` flag that decides where a block sorts.
    """

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # `None` rather than `[]` is the shape an agent with no instructions at all produces.
        seen.extend(info.model_request_parameters.instruction_parts or [])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


async def run_blocks(capfire: CaptureLogfire, name: str, value: dict[str, Any]) -> list[InstructionPart]:
    """Publish `value` for `agent__<name>`, run an agent with a block of each kind, return what it sent.

    The agent covers every source a managed `id` can address: its own literal (`agent`), one declared
    `@agent.instructions` function (`agent:today`), and a toolset with an `id` (`toolset:weather`).
    """
    seen: list[InstructionPart] = []
    agent = Agent(
        capture_instructions(seen),
        instructions='AGENT: You are a concise checkout assistant.',
        toolsets=[weather_toolset()],
        capabilities=[AgentControl(name, label='production')],
    )

    @agent.instructions(name='today')
    def today(_ctx: RunContext[object]) -> str:
        return 'DYNAMIC: today is Monday.'

    with variables_provider(capfire, published_value(f'agent__{name}', value)):
        await agent.run('hello')
    return seen


def triples(parts: list[InstructionPart]) -> list[tuple[str | None, str, bool]]:
    """Each part as the `(id, text, dynamic)` the model was sent it under.

    The id is rendered to the string a managed config addresses it by, which is the form this
    capability's whole contract is written in.
    """
    return [(str(part.id) if part.id is not None else None, part.content, part.dynamic) for part in parts]


AGENT_BLOCK = ('agent', 'AGENT: You are a concise checkout assistant.', False)
TODAY_BLOCK = ('agent:today', 'DYNAMIC: today is Monday.', True)
TOOLSET_BLOCK = ('toolset:weather', 'TOOLSET: call get_weather first.', True)


@pytest.mark.parametrize(
    'name,value,expected',
    [
        pytest.param('blocks_none', {}, [AGENT_BLOCK, TODAY_BLOCK, TOOLSET_BLOCK], id='nothing-managed'),
        pytest.param(
            'blocks_bare',
            {'instructions': 'MANAGED: be brief.'},
            [AGENT_BLOCK, TODAY_BLOCK, ('capability:agent-control', 'MANAGED: be brief.', True), TOOLSET_BLOCK],
            id='bare-string-adds-a-block',
        ),
        pytest.param(
            'blocks_list',
            {'instructions': ['MANAGED: be brief.', 'MANAGED: cite sources.']},
            [
                AGENT_BLOCK,
                TODAY_BLOCK,
                ('capability:agent-control', 'MANAGED: be brief.\n\nMANAGED: cite sources.', True),
                TOOLSET_BLOCK,
            ],
            id='two-added-blocks-join-into-one-contribution',
        ),
        pytest.param(
            'blocks_mixed',
            {'instructions': ['MANAGED: be brief.', {'id': 'agent', 'instructions': 'REMOTE: refund specialist.'}]},
            [
                ('agent', 'REMOTE: refund specialist.', False),
                TODAY_BLOCK,
                ('capability:agent-control', 'MANAGED: be brief.', True),
                TOOLSET_BLOCK,
            ],
            id='one-list-can-add-and-address',
        ),
        pytest.param(
            'blocks_replace',
            {'instructions': [{'id': 'agent', 'instructions': 'REMOTE: refund specialist.'}]},
            [('agent', 'REMOTE: refund specialist.', False), TODAY_BLOCK, TOOLSET_BLOCK],
            id='replace-the-agent-literal',
        ),
        pytest.param(
            'blocks_drop',
            {'instructions': [{'id': 'agent', 'instructions': None}]},
            [TODAY_BLOCK, TOOLSET_BLOCK],
            id='drop-the-agent-literal',
        ),
        pytest.param(
            'blocks_inert',
            {'instructions': [{'id': 'toolset:nope', 'instructions': 'never sent'}]},
            [AGENT_BLOCK, TODAY_BLOCK, TOOLSET_BLOCK],
            id='an-id-nothing-matches-is-inert',
        ),
    ],
)
async def test_instruction_blocks_the_model_receives(
    capfire: CaptureLogfire, name: str, value: dict[str, Any], expected: list[tuple[str | None, str, bool]]
) -> None:
    # The whole contract in one table: an entry with no `id` adds a block, an entry with one replaces
    # or drops the block the agent assembled under that key, and an `id` nothing matches costs nothing.
    #
    # Added blocks land after the dynamic `@agent.instructions` text and before the dynamic toolset
    # text because a capability's contribution is itself dynamic: Pydantic AI sorts static blocks
    # ahead of dynamic ones so a provider can cache the stable prefix, and keeps source order within
    # each group. Every added entry becomes one part, joined by a blank line, because they are
    # contributed through `get_instructions` as a single string.
    assert triples(await run_blocks(capfire, name, value)) == expected


@pytest.mark.parametrize(
    'value',
    [
        pytest.param(
            {'instructions': [{'id': 'agent:today', 'instructions': 'PINNED: it is always Monday.'}]},
            id='replacing-it',
        ),
        pytest.param({'instructions': [{'id': 'toolset:weather', 'instructions': None}]}, id='dropping-it'),
    ],
)
async def test_a_dynamic_block_is_not_addressable(capfire: CaptureLogfire, value: dict[str, Any]) -> None:
    """A block the agent recomputes per request is not the managed config's to change.

    Replacing it pins one rendering forever; dropping it removes the computation. Either way the block
    stops doing what it was written to do, so both are refused and warned about. Note `toolset:weather`
    returns a constant and is flagged dynamic all the same: a function's shape says nothing about
    whether its text came from the run, so an author who wants fixed text addressable contributes an
    `InstructionPart` rather than a function.
    """
    with pytest.warns(UserWarning, match='which the agent recomputes per request'):
        blocks = triples(await run_blocks(capfire, 'blocks_dynamic_refused', value))
    assert blocks == [AGENT_BLOCK, TODAY_BLOCK, TOOLSET_BLOCK]


async def test_an_override_never_moves_the_static_prefix(capfire: CaptureLogfire) -> None:
    # `dynamic` decides which side of the provider's cacheable prefix a block sorts on, and an override
    # rewrites text in place without touching it. So overriding the agent's static literal leaves every
    # block's key and flag exactly where code put them: the cache boundary does not move, which is the
    # whole reason the flag is carried over rather than recomputed.
    baseline = await run_blocks(capfire, 'prefix_baseline', {})
    overridden = await run_blocks(
        capfire,
        'prefix_overridden',
        {'instructions': [{'id': 'agent', 'instructions': 'REMOTE: refund specialist.'}]},
    )
    assert [(part.id, part.dynamic) for part in overridden] == [(part.id, part.dynamic) for part in baseline]
    assert [part.content for part in overridden] == [
        'REMOTE: refund specialist.',
        'DYNAMIC: today is Monday.',
        'TOOLSET: call get_weather first.',
    ]


async def test_two_entries_addressing_the_same_block_keep_the_first(capfire: CaptureLogfire) -> None:
    # A hand-edited value (or a UI bug) can name one `id` twice. Keeping the first makes the run
    # predictable and names the entry that lost, rather than letting JSON ordering decide silently.
    with pytest.warns(UserWarning, match=r"names instruction id 'agent' more than once") as caught:
        parts = await run_blocks(
            capfire,
            'blocks_duplicate',
            {'instructions': [{'id': 'agent', 'instructions': 'FIRST'}, {'id': 'agent', 'instructions': 'SECOND'}]},
        )
    # Resolved once per run and applied on every request, but warned about once.
    assert len(caught) == 1
    assert triples(parts) == [('agent', 'FIRST', False), TODAY_BLOCK, TOOLSET_BLOCK]


async def test_overrides_on_an_agent_with_no_instructions_are_inert(capfire: CaptureLogfire) -> None:
    # One managed config is applied across deployments that need not all assemble the same blocks, so
    # an agent that sends no instructions at all is not an error -- there is simply nothing to address.
    seen: list[InstructionPart] = []
    capability = AgentControl('no_blocks', label='production')
    published = {'instructions': [{'id': 'agent', 'instructions': 'never sent'}]}
    with variables_provider(capfire, published_value('agent__no_blocks', published)):
        result = await Agent(capture_instructions(seen), capabilities=[capability]).run('hello')
    assert seen == []
    assert instructions_seen(result.all_messages()) == []


async def test_added_blocks_render_placeholders_against_deps() -> None:
    @dataclass
    class Deps:
        city: str

    # Rendering happens on the capability's joined contribution, so a `{{...}}` placeholder works the
    # same in a list entry as it does in a bare string. An addressed block is deliberately left alone:
    # its text replaces something the agent assembled, and that text was never templated either.
    capability: AgentControl[Deps] = AgentControl(
        'render',
        default=AgentConfig(
            instructions=[
                'Serve {{city}}.',
                InstructionBlock(instructions='Be brief.'),
                InstructionBlock(id='agent', instructions='Base for {{city}}.'),
            ]
        ),
        render_template=True,
    )
    agent = Agent(TestModel(), instructions='code', deps_type=Deps, capabilities=[capability])
    result = await agent.run('hello', deps=Deps(city='Paris'))
    assert instructions_seen(result.all_messages()) == ['Base for {{city}}.\n\nServe Paris.\n\nBe brief.']


def test_instructions_none_outside_run() -> None:
    # Nothing is resolved until `wrap_run` opens the run's resolution context, so the contribution
    # hook has to answer for a capability that was never entered -- as a graph built for inspection is.
    capability = AgentControl('outside_run_instructions', instructions='base')
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    assert capability.resolved is None
    assert capability.get_instructions()(ctx) is None


async def test_tool_definition_patches() -> None:
    seen: list[ToolDefinition] = []
    capability = AgentControl(
        'tools',
        default=AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(
                    name='get_weather', description='Managed.', parameter_descriptions={'city': 'Managed city.'}
                )
            ]
        ),
    )
    await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    assert advertised(seen) == {'get_weather': 'Managed.'}
    assert seen[0].parameters_json_schema['properties']['city']['description'] == 'Managed city.'


async def test_settings_schema_and_lowering() -> None:
    seen: list[dict[str, object]] = []

    def capture_settings(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    capability = AgentControl(
        'settings',
        default=AgentConfig(
            settings=AgentConfigSettings.model_validate(
                {
                    'temperature': 0.2,
                    'top_p': 0.4,
                    'provider_options': {'openai': {'temperature': 0.8}, 'custom': {'flag': True}},
                    'openai_temperature': 0.6,
                }
            )
        ),
    )
    await Agent(
        FunctionModel(capture_settings),
        model_settings={'temperature': 0.1, 'top_k': 3},
        capabilities=[capability],
    ).run('hello', model_settings={'temperature': 0.9})
    assert seen == [
        {
            'temperature': 0.9,
            'top_k': 3,
            'top_p': 0.4,
            'openai_temperature': 0.8,
            'custom_flag': True,
        }
    ]


def test_agent_config_ignores_forward_keys() -> None:
    assert AgentConfig.model_validate({'instructions': 'x', 'future': True}) == AgentConfig(instructions='x')


@pytest.mark.parametrize(
    'value,expected',
    [
        ({'instructions': '', 'model': 'test'}, AgentConfig(model='test')),
        (
            {'instructions': 'managed', 'settings': {'temperature': 'bad', 'max_tokens': 10}},
            AgentConfig(instructions='managed', settings=AgentConfigSettings(max_tokens=10)),
        ),
        ({'instructions': 'managed', 'settings': []}, AgentConfig(instructions='managed')),
        ({'instructions': 'managed', 'tool_definitions': {}}, AgentConfig(instructions='managed')),
    ],
)
def test_malformed_sections_degrade_independently(value: dict[str, Any], expected: AgentConfig) -> None:
    with pytest.warns(UserWarning):
        assert AgentConfig.model_validate(value) == expected


def test_oversized_bare_instructions_drop_the_section_and_keep_siblings() -> None:
    oversized = 'x' * 65_537
    with pytest.warns(UserWarning, match='65536-character limit') as caught:
        assert AgentConfig.model_validate({'instructions': oversized, 'model': 'test'}) == AgentConfig(model='test')
        AgentConfig.model_validate({'instructions': oversized, 'model': 'test'})
    assert len(caught) == 1


def test_oversized_instruction_entry_drops_itself_and_keeps_siblings() -> None:
    value = {'instructions': ['kept', {'id': 'agent', 'instructions': 'x' * 65_537}], 'model': 'test'}
    with pytest.warns(UserWarning, match='65536-character limit'):
        assert AgentConfig.model_validate(value) == AgentConfig(
            instructions=[InstructionBlock(instructions='kept')], model='test'
        )


def test_prebuilt_variable() -> None:
    variable = Variable(
        'agent__prebuilt',
        type=AgentConfig,
        default=AgentConfig(model='test'),
        logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE,
    )
    assert AgentControl(variable)._variable is variable


@pytest.mark.parametrize('control_first', [True, False])
async def test_managed_values_beat_another_capability(control_first: bool) -> None:
    """A published value outranks every other capability's, whichever order they were registered in.

    Both directions are asserted because that order-independence is the whole point of the innermost
    companion `for_agent` binds: contributions merge left to right, so a capability running outermost
    -- which `AgentControl` must, to keep its baggage around the run span -- always loses on its own.
    """
    seen: list[dict[str, object]] = []

    def capture_settings(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    class Opinionated(Capability[object]):
        """Any other capability contributing a model and settings of its own."""

        def get_model(self) -> TestModel:
            return TestModel(custom_output_text='the other capability')

        def get_model_settings(self) -> ModelSettings:
            return ModelSettings(temperature=0.9, top_k=5)

    def order(control: AgentControl[object]) -> list[AbstractCapability[object]]:
        return [control, Opinionated()] if control_first else [Opinionated(), control]

    # The managed `test` model answers, so the other capability's model never ran.
    model_control: AgentControl[object] = AgentControl('contested_model', default=AgentConfig(model='test'))
    result = await Agent(None, capabilities=order(model_control)).run('hello')
    assert result.output.startswith('success')

    # Settings still merge per key: the managed `temperature` wins, the uncontested `top_k` survives.
    settings_control: AgentControl[object] = AgentControl(
        'contested_settings', default=AgentConfig(settings=AgentConfigSettings(temperature=0.2))
    )
    await Agent(FunctionModel(capture_settings), capabilities=order(settings_control)).run('hello')
    assert seen == [{'temperature': 0.2, 'top_k': 5}]


def test_explicit_capability_id_is_preserved() -> None:
    assert AgentControl('explicit_id', id='custom').id == 'custom'


async def test_auto_create_uses_request_snapshot(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    _managed_variable._reset_auto_create_guard()
    created: list[VariableConfig] = []

    def create_inline(variable: Variable[object], config: VariableConfig) -> None:
        created.append(config)
        _managed_variable._create_variable(variable, config)

    monkeypatch.setattr(_managed_variable, '_spawn_create', create_inline)
    config = VariablesConfig(variables={})

    def lookup(city: str) -> str:
        """Look up a city.

        Args:
            city: City to look up.
        """
        return city

    def raw() -> str:
        return 'raw'

    raw_tool = Tool.from_schema(
        raw,
        name='raw',
        description=None,
        json_schema={
            'type': 'object',
            'properties': {'plain': 'not-a-schema', 'count': {'description': 5}, 'named': {'description': 'Named.'}},
        },
    )
    empty_tool = Tool.from_schema(raw, name='empty', description=None, json_schema={'type': 'object'})

    with variables_provider(capfire, config):
        agent = Agent(
            TestModel(),
            name='snapshot_agent',
            instructions='Code instructions.',
            model_settings={'temperature': 0.3},
            tools=[lookup, raw_tool, empty_tool],
            capabilities=[AgentControl()],
        )
        await agent.run('hello')

    assert len(created) == 1
    # The stored schema is the canonical hand-maintained one, not the payload's Pydantic-derived one.
    assert created[0].json_schema == AGENT_CONFIG_JSON_SCHEMA
    example = json.loads(created[0].example or '{}')
    assert example == {
        'instructions': [{'id': 'agent', 'instructions': 'Code instructions.', 'dynamic': False}],
        'model': 'test:test',
        'settings': {'temperature': 0.3},
        'tool_definitions': [
            {
                'name': 'lookup',
                'description': 'Look up a city.',
                'parameter_descriptions': {'city': 'City to look up.'},
            },
            {'name': 'raw', 'parameter_descriptions': {'named': 'Named.'}},
            {'name': 'empty'},
        ],
    }


async def test_auto_create_snapshots_every_instruction_block(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The baseline the Logfire UI diffs managed values against, and the reason it can offer an override
    # per block rather than one copy-the-whole-prompt button: the joined prompt telemetry records has no
    # seams, so a snapshot taken from that could only be copied wholesale -- which, since managed
    # instructions *add*, would send the agent's own text twice with a frozen date in the middle.
    _managed_variable._reset_auto_create_guard()
    created: list[VariableConfig] = []

    def create_inline(variable: Variable[object], config: VariableConfig) -> None:
        created.append(config)

    monkeypatch.setattr(_managed_variable, '_spawn_create', create_inline)

    agent = Agent(
        TestModel(),
        name='blocks_snapshot',
        instructions='AGENT: You are a concise checkout assistant.',
        toolsets=[weather_toolset()],
        capabilities=[AgentControl()],
    )

    @agent.instructions(name='today')
    def today(_ctx: RunContext[object]) -> str:
        return 'DYNAMIC: today is Monday.'

    @agent.instructions
    def unnamed(_ctx: RunContext[object]) -> str:
        return 'UNNAMED: no declared id.'

    with variables_provider(capfire, VariablesConfig(variables={})):
        await agent.run('hello')

    example = json.loads(created[0].example or '{}')
    assert example['instructions'] == [
        {'id': 'agent', 'instructions': 'AGENT: You are a concise checkout assistant.', 'dynamic': False},
        # A dynamic block contributes its seam and not its text: what it rendered to here is one
        # request's answer, built from whatever that run carried, and the baseline is published to a
        # variable every project member can read. The `id` and the flag are what the editor needs --
        # enough to show the block and that it is recomputed per request.
        {'id': 'agent:today', 'dynamic': True},
        {'id': 'toolset:weather', 'dynamic': True},
    ]
    # `UNNAMED: no declared id.` is absent entirely: the function declared no id, so there is nothing
    # to address it by, and it is dynamic, so there is no text to publish. Nothing left to say about it.
    assert 'UNNAMED' not in (created[0].example or '')
    # A toolset's instruction function is `dynamic` too even when it returns a constant, so one rule
    # covers agent-level and toolset-level blocks alike. A toolset that wants its fixed text published
    # and overridable returns an `InstructionPart` from `get_instructions()` instead, which is static
    # by default and keeps the flag it was authored with.
    assert [block['dynamic'] for block in example['instructions']] == [False, True, True]


async def test_existing_variable_publishes_changed_baseline_once_without_clobbering_config(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_agent_control, '_spawn_baseline_publish', _agent_control._publish_baseline)
    config = published_value('agent__published_baseline', {})
    original = config.variables['agent__published_baseline'].model_copy(
        update={'description': 'Kept description.', 'aliases': ['kept_alias']}
    )
    config.variables['agent__published_baseline'] = original
    today = ['Monday']

    def build(instructions: str) -> Agent[object, str]:
        agent = Agent(
            TestModel(),
            name='published_baseline',
            instructions=instructions,
            toolsets=[weather_toolset()],
            capabilities=[AgentControl(label='production')],
        )

        @agent.instructions(name='today')
        def dynamic(_ctx: RunContext[object]) -> str:
            return f'DYNAMIC: today is {today[0]}.'

        return agent

    agent = build('AGENT: code instructions.')
    with variables_provider(capfire, config):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
        updates: list[VariableConfig] = []
        original_update = provider.update_variable

        def record_update(name: str, updated: VariableConfig) -> VariableConfig:
            updates.append(updated)
            return original_update(name, updated)

        monkeypatch.setattr(provider, 'update_variable', record_update)
        await agent.run('first')
        _agent_control._reset_baseline_publish_guard()  # model a fresh process with the published baseline
        await agent.run('unchanged')
        _agent_control._reset_baseline_publish_guard()
        # A dynamic block renders differently, and the baseline does not move: its text was never in
        # there. One fewer republish, and one fewer read-modify-write racing the UI (see #565).
        today[0] = 'Tuesday'
        await agent.run('dynamic value changed')
        assert len(updates) == 1
        _agent_control._reset_baseline_publish_guard()
        await build('AGENT: rewritten instructions.').run('code changed')

    assert len(updates) == 2
    assert json.loads(updates[0].example or '{}')['instructions'] == [
        {'id': 'agent', 'instructions': 'AGENT: code instructions.', 'dynamic': False},
        {'id': 'agent:today', 'dynamic': True},
        {'id': 'toolset:weather', 'dynamic': True},
    ]
    assert json.loads(updates[1].example or '{}')['instructions'][0]['instructions'] == 'AGENT: rewritten instructions.'
    for updated in updates:
        assert updated.model_copy(update={'example': original.example}) == original


async def test_published_baseline_contains_only_code_side_behavior(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_agent_control, '_spawn_baseline_publish', _agent_control._publish_baseline)
    managed = {
        'instructions': 'MANAGED instruction.',
        'model': 'test',
        'settings': {'temperature': 0.77},
        'tool_definitions': [{'name': 'get_weather', 'new_name': 'weather_now', 'description': 'Managed description.'}],
    }
    config = published_value('agent__code_only_baseline', managed)
    with variables_provider(capfire, config):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
        updates: list[VariableConfig] = []
        original_update = provider.update_variable

        def record_update(name: str, updated: VariableConfig) -> VariableConfig:
            updates.append(updated)
            return original_update(name, updated)

        monkeypatch.setattr(provider, 'update_variable', record_update)
        await Agent(
            FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')])),
            instructions='CODE instruction.',
            model_settings={'temperature': 0.1},
            tools=[get_weather],
            capabilities=[AgentControl('code_only_baseline', label='production')],
        ).run('hello')

    assert json.loads(updates[0].example or '{}') == {
        'instructions': [{'id': 'agent', 'instructions': 'CODE instruction.', 'dynamic': False}],
        'model': 'function:function:<lambda>:',
        'settings': {'temperature': 0.1},
        'tool_definitions': [{'name': 'get_weather'}],
    }


async def test_baseline_publish_failure_does_not_affect_run_and_warns_once(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_agent_control, '_spawn_baseline_publish', _agent_control._publish_baseline)
    config = published_value('agent__publish_failure', {})
    instruction = ['first']
    agent = Agent(TestModel(), name='publish_failure', capabilities=[AgentControl(label='production')])

    @agent.instructions(name='changing')
    def changing(_ctx: RunContext[object]) -> str:
        return instruction[0]

    with variables_provider(capfire, config):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()

        def fail_update(_name: str, _config: VariableConfig) -> VariableConfig:
            raise PermissionError('missing project:write_variables')

        monkeypatch.setattr(provider, 'update_variable', fail_update)
        with pytest.warns(UserWarning, match='missing project:write_variables') as caught:
            first = await agent.run('first')
            instruction[0] = 'second'
            second = await agent.run('second')

    assert len(caught) == 1
    assert first.output.startswith('success')
    assert second.output.startswith('success')


async def test_non_json_baseline_setting_does_not_affect_run(capfire: CaptureLogfire) -> None:
    # A provider-specific setting: it reaches the baseline through `extra='allow'` (unlike
    # `extra_headers`/`extra_body`, which are withheld) and has nothing `json.dumps` can write.
    with variables_provider(capfire, published_value('agent__non_json_baseline', {})):
        with pytest.warns(UserWarning, match='Failed to publish the code baseline'):
            result = await Agent(
                TestModel(),
                model_settings=cast(ModelSettings, {'openai_custom_option': object()}),
                capabilities=[AgentControl('non_json_baseline')],
            ).run('hello')
    assert result.output.startswith('success')


async def test_oversized_code_instructions_do_not_affect_run(capfire: CaptureLogfire) -> None:
    """The length bound is about what a managed value may add, not about what the agent already says.

    `AgentConfig` describes what to apply and what exists with the same fields, so building the
    baseline validates code-side text too. An agent whose own instructions exceed the bound has a
    snapshot too big to publish -- not a broken run -- so it warns and keeps going.
    """
    with variables_provider(capfire, published_value('agent__oversized_code', {})):
        with pytest.warns(UserWarning, match='Failed to publish the code baseline'):
            result = await Agent(
                TestModel(),
                instructions='x' * (_agent_control._MAX_MODEL_FACING_TEXT_LENGTH + 1),
                capabilities=[AgentControl('oversized_code')],
            ).run('hello')
    assert result.output.startswith('success')


async def test_publish_baseline_opt_out(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_spawn(_variable: Variable[object], _example: str) -> None:
        # Never reached when the opt-out holds, which is the assertion.
        raise AssertionError('baseline publishing should be disabled')  # pragma: no cover

    monkeypatch.setattr(_agent_control, '_spawn_baseline_publish', fail_spawn)
    with variables_provider(capfire, published_value('agent__no_publish', {})):
        result = await Agent(
            TestModel(), instructions='code', capabilities=[AgentControl('no_publish', publish_baseline=False)]
        ).run('hello')
    assert result.output.startswith('success')


async def test_missing_variable_baseline_publish_warns_without_affecting_run(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_agent_control, '_spawn_baseline_publish', _agent_control._publish_baseline)
    with variables_provider(capfire, VariablesConfig(variables={})):
        with pytest.warns(UserWarning, match="variable 'agent__missing_publish' was not found"):
            result = await Agent(
                TestModel(),
                name='missing_publish',
                capabilities=[AgentControl(auto_create=False)],
            ).run('hello')
    assert result.output.startswith('success')


async def test_applied_sections_baggage() -> None:
    seen: list[object] = []

    def inspect_baggage() -> str:
        seen.append(logfire.get_baggage().get('logfire.managed.applied_sections'))
        return 'ok'

    capability = AgentControl(
        'baggage',
        default=AgentConfig(instructions='managed', settings=AgentConfigSettings(temperature=0.2)),
    )
    await Agent(TestModel(), tools=[inspect_baggage], capabilities=[capability]).run('hello')
    assert seen == ['instructions,settings']


async def test_empty_config_has_no_applied_sections_baggage() -> None:
    seen: list[object] = []

    def inspect_baggage() -> str:
        seen.append(logfire.get_baggage().get('logfire.managed.applied_sections'))
        return 'ok'

    await Agent(TestModel(), tools=[inspect_baggage], capabilities=[AgentControl('empty_baggage')]).run('hello')
    assert seen == [None]


async def test_rename_round_trip_preserves_original_context_name() -> None:
    calls = 0
    context_names: list[str | None] = []

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert [tool.name for tool in info.function_tools] == ['weather_now']
            return ModelResponse(parts=[ToolCallPart('weather_now', {'city': 'Paris'}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    def weather(ctx: RunContext[object], city: str) -> str:
        context_names.append(ctx.tool_name)
        return city

    capability = AgentControl(
        'rename',
        default=AgentConfig(tool_definitions=[ToolDefinitionOverride(name='weather', new_name='weather_now')]),
    )
    await Agent(FunctionModel(model), tools=[weather], capabilities=[capability]).run('hello')
    assert context_names == ['weather']


async def test_rename_collision_warns_and_keeps_other_patches() -> None:
    seen: list[ToolDefinition] = []
    calls = 0

    def first() -> str:
        return 'first'

    def second() -> str:  # pragma: no cover - advertised only
        return 'second'

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        seen.extend(info.function_tools)
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('first', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    capability = AgentControl(
        'collision',
        default=AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='first', new_name='second', description='Managed first.'),
            ]
        ),
    )
    with pytest.warns(UserWarning, match='already advertised'):
        await Agent(FunctionModel(model), tools=[first, second], capabilities=[capability]).run('hello')
    assert advertised(seen[:2]) == {'first': 'Managed first.', 'second': None}


async def test_unknown_tool_and_parameter_keys_are_inert() -> None:
    seen: list[ToolDefinition] = []
    capability = AgentControl(
        'unknown_tool',
        default=AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='missing', description='ignored'),
                ToolDefinitionOverride(name='get_weather', parameter_descriptions={'missing': 'ignored'}),
            ]
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    assert caught == []
    assert advertised(seen) == {'get_weather': None}
    assert get_weather('Paris') == 'sunny in Paris'


async def test_two_overrides_naming_the_same_tool_keep_the_first() -> None:
    # The same first-wins rule as a duplicated instruction `id`, and the same reason: which entry a
    # colliding pair resolves to has to be a property of the config, not of JSON key order.
    seen: list[ToolDefinition] = []
    capability = AgentControl(
        'duplicate_tool',
        default=AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='get_weather', description='First.'),
                ToolDefinitionOverride(name='get_weather', description='Second.'),
            ]
        ),
    )
    with pytest.warns(UserWarning, match=r"names tool 'get_weather' more than once") as caught:
        await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    # Read on every `get_tools`, warned about once.
    assert len(caught) == 1
    assert advertised(seen) == {'get_weather': 'First.'}


async def test_schema_without_properties_is_tolerated() -> None:
    seen: list[ToolDefinition] = []

    def raw_tool() -> str:  # pragma: no cover - advertised only
        return 'ok'

    tool = Tool.from_schema(raw_tool, name='raw_tool', description='Original.', json_schema={'type': 'object'})
    capability = AgentControl(
        'raw_schema',
        default=AgentConfig(
            tool_definitions=[ToolDefinitionOverride(name='raw_tool', parameter_descriptions={'missing': 'ignored'})]
        ),
    )
    await Agent(capture_tools(seen), tools=[tool], capabilities=[capability]).run('hello')
    assert seen[0].parameters_json_schema == {'type': 'object'}


async def test_managed_model_runs_model_less_agent_and_run_model_wins() -> None:
    managed = AgentControl('managed_model', default=AgentConfig(model='test'))
    assert (await Agent(None, capabilities=[managed]).run('hello')).output.startswith('success')

    def call_site(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('call-site')])

    result = await Agent(None, capabilities=[managed]).run('hello', model=FunctionModel(call_site))
    assert result.output == 'call-site'


async def test_unknown_managed_model_keeps_code_model() -> None:
    with pytest.warns(UserWarning, match='selects unknown model'):
        result = await Agent(
            TestModel(),
            capabilities=[AgentControl('unknown_model', default=AgentConfig(model='not-a-provider:not-a-model'))],
        ).run('hello')
    assert result.output.startswith('success')


async def test_nameless_model_selector_resolves_once_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # A nameless capability's selector is evaluated once per request step, but the managed model is a
    # run-stable value, so it memoizes and resolves the variable exactly once even across steps.
    resolves: list[str] = []
    original = AgentControl[Any]._resolve_for_selection

    def counting(self: AgentControl[Any], variable: Variable[Any], ctx: Any) -> Any:
        resolves.append(variable.name)
        return original(self, variable, ctx)

    monkeypatch.setattr(AgentControl, '_resolve_for_selection', counting)

    def a_tool() -> str:
        return 'ok'

    capability = AgentControl(default=AgentConfig(model='test'))
    # A model-less agent with one tool: `TestModel` calls the tool (step 1) then answers (step 2).
    await Agent(None, name='multi_step', tools=[a_tool], capabilities=[capability]).run('hello')
    assert resolves == ['agent__multi_step']


async def test_callable_targeting_resolution_is_reused_for_run() -> None:
    calls = 0

    def targeting(_ctx: RunContext[object]) -> str:
        nonlocal calls
        calls += 1
        return f'key-{calls}'

    capability = AgentControl(
        'callable_targeting', default=AgentConfig(model='test'), targeting_key=targeting, publish_baseline=False
    )
    result = await Agent(TestModel(), capabilities=[capability]).run('hello')
    assert result.output.startswith('success')
    assert calls == 1


async def test_known_variable_skips_snapshot_build(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    config = VariableConfig(name='agent__known', labels={}, rollout=Rollout(labels={}), overrides=[])

    def fail_to_config(self: Variable[object]) -> VariableConfig:  # pragma: no cover - failure sentinel
        raise AssertionError('snapshot should not be built')

    monkeypatch.setattr(Variable, 'to_config', fail_to_config)
    with variables_provider(capfire, VariablesConfig(variables={'agent__known': config})):
        await Agent(TestModel(), capabilities=[AgentControl('known')]).run('hello')


async def test_before_model_request_outside_run_is_inert() -> None:
    # Both halves of the split are asked: each reaches for the run's resolution, so each has to cope
    # with there not being one.
    bound = AgentControl('outside_run').for_agent(Agent(TestModel()))
    assert isinstance(bound, CombinedCapability)
    request = cast(Any, object())
    for half in bound.capabilities:
        assert await half.before_model_request(cast(Any, None), request) is request


async def test_credential_bearing_settings_stay_out_of_the_published_baseline(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`extra_headers` and `extra_body` apply to the run but never describe the agent to a reader.

    `AgentConfigSettings` allows extra keys so a provider-specific setting survives; these two arrive
    the same way and routinely carry authorization. The baseline is published to a variable every
    project member can read, so they are withheld from it -- and only from it.
    """
    monkeypatch.setattr(_agent_control, '_spawn_baseline_publish', _agent_control._publish_baseline)
    sent: list[ModelSettings | None] = []

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(info.model_settings)
        return ModelResponse(parts=[TextPart('done')])

    with variables_provider(capfire, published_value('agent__secretless_baseline', {})):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
        updates: list[VariableConfig] = []
        original_update = provider.update_variable

        def record_update(name: str, updated: VariableConfig) -> VariableConfig:
            updates.append(updated)
            return original_update(name, updated)

        monkeypatch.setattr(provider, 'update_variable', record_update)
        await Agent(
            FunctionModel(model),
            model_settings=cast(
                ModelSettings,
                {
                    'temperature': 0.1,
                    'extra_headers': {'Authorization': 'Bearer sk-secret'},
                    'extra_body': {'signature': 'sk-secret'},
                    'openai_custom_option': 'kept',
                },
            ),
            capabilities=[AgentControl('secretless_baseline')],
        ).run('hello')

    assert json.loads(updates[0].example or '{}')['settings'] == {'temperature': 0.1, 'openai_custom_option': 'kept'}
    # Withheld from the snapshot, not from the request: the run still sends them.
    assert sent[0] is not None and sent[0]['extra_headers'] == {'Authorization': 'Bearer sk-secret'}  # type: ignore[typeddict-item]


async def test_rename_routes_to_the_tool_the_model_was_handed(capfire: CaptureLogfire) -> None:
    """A dynamic toolset may put a different tool behind an advertised name between listing and call.

    Routing reads the code-side name off the tool the model was handed, so the name every downstream
    consumer sees -- a name-based `ApprovalRequiredToolset` above all -- is the name of the tool that
    actually runs. Recomputing it from a fresh listing would authorize one tool and run the other.
    """
    listings = 0

    class Flipping(WrapperToolset[object]):
        """Advertises `first` on the first listing and `second` on every later one."""

        async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
            nonlocal listings
            tools = await super().get_tools(ctx)
            listings += 1
            return {('first' if listings == 1 else 'second'): tools['first' if listings == 1 else 'second']}

    authorized: list[str] = []

    class NamePolicy(WrapperToolset[object]):
        """Stands in for a name-based policy: it records the name it is asked to authorize."""

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
        ) -> Any:
            authorized.append(name)
            return await super().call_tool(name, tool_args, ctx, tool)

    def first() -> str:
        return 'ran first'

    def second() -> str:  # pragma: no cover - advertised on the second listing only
        return 'ran second'

    calls = 0

    def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('helper', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    managed = {'tool_definitions': [{'name': 'first', 'new_name': 'helper'}, {'name': 'second', 'new_name': 'helper'}]}
    with variables_provider(capfire, published_value('agent__rename_identity', managed)):
        await Agent(
            FunctionModel(model),
            toolsets=[NamePolicy(Flipping(FunctionToolset[object]([first, second])))],
            capabilities=[AgentControl('rename_identity')],
        ).run('hello')

    assert authorized == ['first']


async def test_a_dynamic_blocks_rendered_text_never_reaches_the_baseline(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a dynamic block rendered to is one request's answer, and the baseline is not private.

    An instruction function reads the run: a tenant, a user, a retrieved document. The `example` it
    would land in is published to a Logfire variable every project member can read, and it is there to
    describe the agent, not to record a request. So a dynamic block contributes its `id` and its flag
    and nothing else -- which is also all an editor needs to show it and to not offer to change it.
    """
    monkeypatch.setattr(_agent_control, '_spawn_baseline_publish', _agent_control._publish_baseline)
    with variables_provider(capfire, published_value('agent__no_request_data', {})):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
        updates: list[VariableConfig] = []
        original_update = provider.update_variable

        def record_update(name: str, updated: VariableConfig) -> VariableConfig:
            updates.append(updated)
            return original_update(name, updated)

        monkeypatch.setattr(provider, 'update_variable', record_update)
        agent = Agent(
            FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')])),
            deps_type=str,
            instructions='You are a support agent.',
            capabilities=[AgentControl('no_request_data')],
        )

        @agent.instructions(name='tenant')
        def tenant(ctx: RunContext[str]) -> str:
            return f'You are serving tenant {ctx.deps}. Their account token is tok_SECRET_9f2.'

        await agent.run('hello', deps='ACME Health (patient records)')

    example = updates[0].example or ''
    assert 'tok_SECRET_9f2' not in example and 'ACME Health' not in example
    assert json.loads(example)['instructions'] == [
        {'id': 'agent', 'instructions': 'You are a support agent.', 'dynamic': False},
        {'id': 'agent:tenant', 'dynamic': True},
    ]
