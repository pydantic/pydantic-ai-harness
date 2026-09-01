"""Pin `AGENT_CONFIG_JSON_SCHEMA` against `AgentConfig` and against the Logfire UI's copy of it.

The schema is stored on the `agent__<name>` variable by whichever side creates it first -- this SDK
or the Logfire Agent Control UI -- and the Logfire backend validates every later version of the value
against it. So the schema has to describe everything `AgentConfig` can emit, and reject nothing
`AgentConfig` tolerates, or a value one half of the contract writes becomes unwritable or unreadable
for the other.

`jsonschema` is not a dependency of this package, so the checks below run against a small validator
covering the keywords the canonical schema actually uses.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from pydantic_ai_harness.logfire import (
    AGENT_CONFIG_JSON_SCHEMA,
    AgentConfig,
    AgentConfigSettings,
    InstructionBlock,
    ToolDefinitionOverride,
)

LOCKSTEP = (
    'AGENT_CONFIG_JSON_SCHEMA is one half of a contract with the Logfire UI: update '
    'src/services/logfire-frontend/src/app/project/managed-agents/agent-config.ts in the Logfire '
    'platform repo in lockstep, or a UI-created variable and an SDK-created one will store different '
    'schemas for the same agent.'
)

CANONICAL_SCHEMA_SHA256 = 'd5e13a30edefd2fd1f9b83bc73056156cfdd2348088b417d3034b095190b63a1'
"""SHA-256 of this schema's canonical JSON, pinned identically by the platform's `agent-config.test.ts`.

Every other assertion in this module checks a property, and properties are exactly what let the two
copies of this contract drift: they were written independently and did drift, in a description and in
whether `id` accepted `null` -- a real difference in what each side would let you save, settled by
whichever one happened to create the variable first. A digest cannot drift quietly, and there is no
package the two repos share to hold the schema once.

When this fails, decide which side is right *before* repinning, then change both. This side is the
canonical copy, so normally the platform's is what moves.
"""

INSTRUCTIONS_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['instructions']
SETTINGS_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['settings']
TOOL_OVERRIDE_SCHEMA: dict[str, Any] = AGENT_CONFIG_JSON_SCHEMA['properties']['tool_definitions']['items']

# The object branch of an `instructions` entry, the other branch being a bare string.
INSTRUCTION_BLOCK_SCHEMA: dict[str, Any] = next(
    option for option in INSTRUCTIONS_SCHEMA['anyOf'][1]['items']['anyOf'] if option['type'] == 'object'
)

# Every section populated, with every canonical setting, every override field, and every kind of
# instruction entry, so the assertions below cover the whole surface rather than the fields that
# happen to be interesting.
FULL_VALUE: dict[str, Any] = {
    'instructions': [
        'Be concise.',
        {'id': 'agent', 'instructions': 'You are a refund specialist.'},
        {'id': 'toolset:weather', 'instructions': None},
        {'id': 'agent:today', 'instructions': 'It is Monday.', 'dynamic': True},
    ],
    'model': 'openai:gpt-5',
    'settings': {
        'max_tokens': 2048,
        'temperature': 0.4,
        'top_p': 0.9,
        'top_k': 40,
        'seed': 7,
        'presence_penalty': 0.1,
        'frequency_penalty': 0.2,
        'parallel_tool_calls': True,
        'timeout': 30.0,
        'stop_sequences': ['STOP'],
        'thinking': 'high',
        'service_tier': 'flex',
        'provider_options': {'anthropic': {'thinking': {'type': 'enabled', 'budget_tokens': 16384}}},
    },
    'tool_definitions': [
        {
            'name': 'get_weather',
            'new_name': 'lookup_weather',
            'description': 'Look up the current weather for a city.',
            'parameter_descriptions': {'city': "City name, e.g. 'London'"},
        }
    ],
}

_JSON_TYPES: dict[str, type | tuple[type, ...] | None] = {
    'object': dict,
    'array': list,
    'string': str,
    'integer': int,
    'number': (int, float),
    'boolean': bool,
    'null': None,
}


def schema_errors(schema: dict[str, Any], value: Any, path: str = 'value') -> list[str]:
    """Validate `value` against the JSON Schema subset the canonical schema uses."""
    errors: list[str] = []
    if 'anyOf' in schema:
        if all(schema_errors(option, value, path) for option in schema['anyOf']):
            errors.append(f'{path}: matches none of the allowed types')
        return errors
    expected: str = schema['type']
    expected_type = _JSON_TYPES[expected]
    if expected_type is None:
        return [] if value is None else [f'{path}: expected null, got {type(value).__name__}']
    # `bool` is an `int` subclass, so a numeric schema has to reject `True` explicitly.
    if not isinstance(value, expected_type) or (expected in ('integer', 'number') and isinstance(value, bool)):
        return [f'{path}: expected {expected}, got {type(value).__name__}']
    if expected == 'string':
        if len(value) < schema.get('minLength', 0):
            errors.append(f'{path}: shorter than minLength')
        if len(value) > schema.get('maxLength', len(value)):
            errors.append(f'{path}: longer than maxLength')
    elif expected == 'object':
        members: dict[str, Any] = value
        properties: dict[str, Any] = schema.get('properties', {})
        additional: dict[str, Any] | bool = schema.get('additionalProperties', True)
        required: list[str] = schema.get('required', [])
        errors.extend(f'{path}.{key}: required' for key in required if key not in members)
        for key, item in members.items():
            item_schema = properties.get(key, additional)
            if item_schema is False:
                errors.append(f'{path}.{key}: additional properties are not allowed')
            elif item_schema is not True:
                errors.extend(schema_errors(item_schema, item, f'{path}.{key}'))
    elif expected == 'array':
        entries: list[Any] = value
        for index, item in enumerate(entries):
            errors.extend(schema_errors(schema['items'], item, f'{path}[{index}]'))
    return errors


def subschemas(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The schema and every subschema reachable from it."""
    yield schema
    properties: dict[str, Any] = schema.get('properties', {})
    for value in properties.values():
        yield from subschemas(value)
    additional: dict[str, Any] | bool | None = schema.get('additionalProperties')
    if isinstance(additional, dict):
        yield from subschemas(additional)
    items: dict[str, Any] | None = schema.get('items')
    if items is not None:
        yield from subschemas(items)
    options: list[dict[str, Any]] = schema.get('anyOf', [])
    for option in options:
        yield from subschemas(option)


def test_schema_matches_the_digest_the_logfire_ui_pins() -> None:
    digest = hashlib.sha256(json.dumps(AGENT_CONFIG_JSON_SCHEMA, sort_keys=True, separators=(',', ':')).encode())
    assert digest.hexdigest() == CANONICAL_SCHEMA_SHA256, LOCKSTEP


def test_every_model_field_is_described() -> None:
    assert set(AgentConfig.model_fields) == set(AGENT_CONFIG_JSON_SCHEMA['properties']), LOCKSTEP
    assert set(AgentConfigSettings.model_fields) == set(SETTINGS_SCHEMA['properties']), LOCKSTEP
    assert set(ToolDefinitionOverride.model_fields) == set(TOOL_OVERRIDE_SCHEMA['properties']), LOCKSTEP
    assert set(InstructionBlock.model_fields) == set(INSTRUCTION_BLOCK_SCHEMA['properties']), LOCKSTEP


def test_schema_is_permissive_at_every_level() -> None:
    # A closed schema anywhere would make the Logfire backend reject a write that `AgentConfig` is
    # built to tolerate, so the forward-compatibility contract lives in the stored schema too.
    for subschema in subschemas(AGENT_CONFIG_JSON_SCHEMA):
        assert subschema.get('additionalProperties') is not False, LOCKSTEP
        assert 'enum' not in subschema, LOCKSTEP
        # Flat: no `$defs`/`$ref` indirection and none of Pydantic's `title`/`default` noise, which
        # render badly in a form editor.
        assert not {'$defs', '$ref', 'title', 'default'} & set(subschema), LOCKSTEP


def test_the_only_required_field_is_the_one_an_entry_is_useless_without() -> None:
    # `required` is a write-time rejection, so it is only ever justified when the value it would reject
    # cannot mean anything -- not merely when this release has no use for it. A tool overlay that names
    # no tool patches nothing, and letting the UI save one would only produce a row that silently does
    # nothing. Everything else stays optional by omission.
    required = {
        tuple(subschema['required']) for subschema in subschemas(AGENT_CONFIG_JSON_SCHEMA) if 'required' in subschema
    }
    assert required == {('name',)}, LOCKSTEP
    assert TOOL_OVERRIDE_SCHEMA['required'] == ['name'], LOCKSTEP


def test_null_is_only_offered_where_it_means_something() -> None:
    # Optional-by-omission everywhere, so a `null` branch never exists just to spell out "unset". The
    # one exception carries meaning: an instruction entry's `null` text is how a block gets dropped, so
    # it has to be distinguishable from an entry that simply has no text yet.
    nullable = [subschema for subschema in subschemas(AGENT_CONFIG_JSON_SCHEMA) if subschema.get('type') == 'null']
    assert len(nullable) == 1, LOCKSTEP
    assert {'type': 'null'} in INSTRUCTION_BLOCK_SCHEMA['properties']['instructions']['anyOf'], LOCKSTEP


def test_empty_strings_are_rejected_wherever_they_would_be_meaningless() -> None:
    # `''` is a half-filled field, never a value: `None`/omission already means "leave this to code".
    # `model: ''` is the one that bites hardest -- Pydantic AI raises `Unknown model:` on every request
    # -- and the config around it is otherwise valid, so nothing downstream would catch it.
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'model': ''}) == ['value.model: shorter than minLength']
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': ''}) == [
        'value.instructions: matches none of the allowed types'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': [{'id': 'agent', 'instructions': ''}]}) == [
        'value.instructions: matches none of the allowed types'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'tool_definitions': [{'name': ''}]}) == [
        'value.tool_definitions[0].name: shorter than minLength'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'tool_definitions': [{'name': 't', 'new_name': ''}]}) == [
        'value.tool_definitions[0].new_name: shorter than minLength'
    ]


def test_oversized_instruction_text_is_rejected_at_write_time() -> None:
    oversized = 'x' * 65_537
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': oversized}) == [
        'value.instructions: matches none of the allowed types'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': ['kept', oversized]}) == [
        'value.instructions: matches none of the allowed types'
    ]


def test_everything_agent_config_emits_validates() -> None:
    dumped = AgentConfig.model_validate(FULL_VALUE).model_dump(exclude_none=True)
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, dumped) == [], LOCKSTEP
    # A round trip normalizes each list entry to its object form and drops the fields it left unset,
    # so the dumped value is the same config rather than the same bytes.
    assert dumped['instructions'] == [
        {'instructions': 'Be concise.'},
        {'id': 'agent', 'instructions': 'You are a refund specialist.'},
        {'id': 'toolset:weather'},
        {'id': 'agent:today', 'instructions': 'It is Monday.', 'dynamic': True},
    ]
    assert {key: value for key, value in dumped.items() if key != 'instructions'} == {
        key: value for key, value in FULL_VALUE.items() if key != 'instructions'
    }


def test_schema_shaped_value_round_trips() -> None:
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, FULL_VALUE) == [], LOCKSTEP
    config = AgentConfig.model_validate(FULL_VALUE)
    assert config.model == FULL_VALUE['model']
    assert config.settings == AgentConfigSettings.model_validate(FULL_VALUE['settings'])
    assert config.tool_definitions == [ToolDefinitionOverride.model_validate(FULL_VALUE['tool_definitions'][0])]
    assert config.instructions == [
        InstructionBlock(instructions='Be concise.'),
        InstructionBlock(id='agent', instructions='You are a refund specialist.'),
        InstructionBlock(id='toolset:weather'),
        InstructionBlock(id='agent:today', instructions='It is Monday.', dynamic=True),
    ]


def test_a_bare_instructions_string_stays_a_bare_string() -> None:
    # The shape a value was written in survives the round trip, so a config someone wrote (or the UI
    # wrote before the list form existed) reads back the way they left it.
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': 'Be concise.'}) == []
    assert AgentConfig.model_validate({'instructions': 'Be concise.'}).instructions == 'Be concise.'


def test_boolean_thinking_and_empty_sections_validate() -> None:
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'thinking': True}, 'tool_definitions': []}) == []
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': []}) == []


def test_unknown_keys_are_accepted_everywhere() -> None:
    # `AgentConfig` ignores keys it doesn't know so a value written by a newer UI degrades to the
    # sections this SDK understands. The stored schema has to accept them for that value to be
    # writable at all.
    value = {
        'future_section': {'anything': 1},
        'settings': {'future_setting': 'raw json'},
        'tool_definitions': [{'name': 'get_weather', 'future_override': ['x']}],
        'instructions': [{'id': 'agent', 'instructions': 'x', 'future_field': 1}],
    }
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, value) == [], LOCKSTEP
    assert AgentConfig.model_validate(value).settings is not None


def test_wrong_types_are_rejected() -> None:
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'instructions': 5}) == [
        'value.instructions: matches none of the allowed types'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'tool_definitions': {'get_weather': {}}}) == [
        'value.tool_definitions: expected array, got dict'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'max_tokens': True}}) == [
        'value.settings.max_tokens: expected integer, got bool'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'thinking': 5}}) == [
        'value.settings.thinking: matches none of the allowed types'
    ]
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'settings': {'stop_sequences': [1]}}) == [
        'value.settings.stop_sequences[0]: expected string, got int'
    ]


def test_a_tool_overlay_must_name_its_tool() -> None:
    assert schema_errors(AGENT_CONFIG_JSON_SCHEMA, {'tool_definitions': [{'description': 'x'}]}) == [
        'value.tool_definitions[0].name: required'
    ]


def test_closed_schema_would_reject_a_newer_uis_key() -> None:
    # What an `additionalProperties: false` root (the schema the UI used to write) does to the same
    # value, and why neither side stores one.
    closed = {**AGENT_CONFIG_JSON_SCHEMA, 'additionalProperties': False}
    assert schema_errors(closed, {'future_section': {}}) == [
        'value.future_section: additional properties are not allowed'
    ]
