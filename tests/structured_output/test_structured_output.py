from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import Any

import pytest
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import pydantic_ai_harness
from pydantic_ai_harness import SchemaOutput
from pydantic_ai_harness.structured_output import SchemaOutput as SchemaOutputFromSubmodule

REPORT: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': ['pass', 'fail']},
        'findings': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['verdict'],
    'additionalProperties': False,
}


class _Replay:
    """A model that emits the given tool-call payloads in order, recording what it was sent."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        self.payloads = list(payloads)
        self.info: AgentInfo | None = None
        self.retries: list[RetryPromptPart] = []
        self.calls = 0

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if self.info is None:
            self.info = info
        for message in messages:
            for part in message.parts:
                if isinstance(part, RetryPromptPart) and part not in self.retries:
                    self.retries.append(part)
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        name = (info.output_tools or [])[0].name
        return ModelResponse(parts=[ToolCallPart(name, payload)])

    @property
    def wire_schema(self) -> dict[str, Any]:
        assert self.info is not None
        return (self.info.output_tools or [])[0].parameters_json_schema


def _run(schema: dict[str, Any], *payloads: dict[str, Any], **kwargs: Any) -> tuple[_Replay, Any]:
    model = _Replay(*payloads)
    agent = Agent(FunctionModel(model), output_type=SchemaOutput(schema, **kwargs))
    return model, agent.run_sync('go')


def _first_retry(schema: dict[str, Any], bad: dict[str, Any]) -> str:
    """The message the model is sent back after one rejected payload it never corrects."""
    model = _Replay(bad)
    agent = Agent(FunctionModel(model), output_type=SchemaOutput(schema, max_retries=1))
    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync('go')
    return model.retries[0].model_response()


def _lint_warnings(schema: dict[str, Any]) -> list[str]:
    """The satisfiability warnings only, with the unsupported-keyword warning filtered out."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        SchemaOutput(schema)
    return [str(w.message) for w in caught if 'no value can satisfy' in str(w.message)]


class TestWiring:
    def test_returns_a_tool_output(self) -> None:
        assert isinstance(SchemaOutput(REPORT), ToolOutput)

    def test_defaults(self) -> None:
        output = SchemaOutput(REPORT)

        assert output.name == 'structured_output'
        assert output.description == 'Return your final response in the requested structured format'
        assert output.max_retries == 3

    def test_a_schema_description_joins_the_tool_description_cleanly(self) -> None:
        # Core pops the schema's top-level `description` and appends it after `. `.
        model, _ = _run({**REPORT, 'description': 'A code review report.'}, {'verdict': 'pass'})

        assert model.info is not None
        description = (model.info.output_tools or [])[0].description
        assert description == 'Return your final response in the requested structured format. A code review report.'

    def test_overrides_forward_to_the_output_tool(self) -> None:
        output = SchemaOutput(REPORT, name='verdict', description='Report the verdict.', max_retries=1)

        assert (output.name, output.description, output.max_retries) == ('verdict', 'Report the verdict.', 1)

    def test_schema_reaches_the_model_unwrapped(self) -> None:
        model, _ = _run(REPORT, {'verdict': 'pass'})

        assert model.wire_schema == REPORT
        assert model.info is not None
        assert model.info.function_tools == []

    def test_is_the_only_output_tool_and_ends_the_run(self) -> None:
        model, result = _run(REPORT, {'verdict': 'pass'})

        assert model.info is not None
        assert len(model.info.output_tools or []) == 1
        assert model.calls == 1
        assert result.output == {'verdict': 'pass'}

    def test_does_not_mutate_the_callers_schema(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'string'}}}
        before = deepcopy(schema)

        SchemaOutput(schema, name='named')

        assert schema == before

    def test_the_wire_schema_is_frozen_at_construction(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'string'}}}
        model = _Replay({'a': 'x'})
        agent = Agent(FunctionModel(model), output_type=SchemaOutput(schema))
        schema['properties']['b'] = {'type': 'integer'}
        agent.run_sync('go')

        assert model.wire_schema == {'type': 'object', 'properties': {'a': {'type': 'string'}}}

    def test_composes_in_a_list_with_other_output_types(self) -> None:
        model = _Replay({'verdict': 'pass'})
        agent = Agent(FunctionModel(model), output_type=[str, SchemaOutput(REPORT)])

        assert agent.run_sync('go').output == {'verdict': 'pass'}

    def test_is_exported_from_the_package_root(self) -> None:
        assert pydantic_ai_harness.SchemaOutput is SchemaOutputFromSubmodule
        assert 'SchemaOutput' in pydantic_ai_harness.__all__


class TestAcceptsValidOutput:
    @pytest.mark.parametrize(
        'schema,value',
        [
            ({'type': 'object', 'properties': {'a': {'type': 'string'}}}, {'a': 'x'}),
            ({'type': 'object', 'properties': {'a': {'type': 'integer'}}}, {'a': 1}),
            ({'type': 'object', 'properties': {'a': {'type': 'number'}}}, {'a': 1}),
            ({'type': 'object', 'properties': {'a': {'type': 'number'}}}, {'a': 1.5}),
            ({'type': 'object', 'properties': {'a': {'type': 'boolean'}}}, {'a': True}),
            ({'type': 'object', 'properties': {'a': {'type': 'null'}}}, {'a': None}),
            ({'type': 'object', 'properties': {'a': {'type': 'object'}}}, {'a': {}}),
            ({'type': 'object', 'properties': {'a': {'type': ['string', 'null']}}}, {'a': None}),
            ({'type': 'object', 'properties': {'a': {'enum': ['x', 'y']}}}, {'a': 'y'}),
            ({'type': 'object', 'properties': {'a': {'const': 7}}}, {'a': 7}),
            ({'type': 'object', 'properties': {'a': {'items': {'type': 'string'}}}}, {'a': ['x']}),
            ({'type': 'object', 'required': ['a'], 'properties': {'a': {'type': 'string'}}}, {'a': 'x'}),
            ({'type': 'object', 'additionalProperties': False, 'properties': {}}, {}),
            ({'type': 'object', 'additionalProperties': {'type': 'string'}}, {'extra': 'x'}),
            (
                {'type': 'object', 'properties': {'a': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}},
                {'a': 3},
            ),
            ({'type': 'object', 'properties': {'a': {'type': 'string'}}}, {}),
        ],
    )
    def test_valid_payload_is_returned_untouched(self, schema: dict[str, Any], value: dict[str, Any]) -> None:
        model, result = _run(schema, value)

        assert result.output == value
        assert model.calls == 1

    def test_output_is_a_plain_dict(self) -> None:
        _, result = _run(REPORT, {'verdict': 'pass'})

        assert type(result.output) is dict


class TestRejectsInvalidOutput:
    @pytest.mark.parametrize(
        'schema,value,expected',
        [
            (
                {'type': 'object', 'required': ['a'], 'properties': {'a': {'type': 'string'}}},
                {},
                '$.a: required property missing',
            ),
            (
                {'type': 'object', 'additionalProperties': False, 'properties': {}},
                {'nope': 1},
                '$.nope: additional property not allowed',
            ),
            (
                {'type': 'object', 'properties': {'a': {'type': 'string'}}},
                {'a': 1},
                '$.a: expected string, got integer',
            ),
            (
                {'type': 'object', 'properties': {'a': {'type': ['string', 'null']}}},
                {'a': 1},
                '$.a: expected string or null, got integer',
            ),
            (
                {'type': 'object', 'properties': {'a': {'type': 'integer'}}},
                {'a': True},
                '$.a: expected integer, got boolean',
            ),
            (
                {'type': 'object', 'properties': {'a': {'type': 'integer'}}},
                {'a': 1.5},
                '$.a: expected integer, got number',
            ),
            (
                {'type': 'object', 'properties': {'a': {'enum': ['x', 'y']}}},
                {'a': 'z'},
                "$.a: 'z' is not one of the allowed values (allowedValues: x, y)",
            ),
            (
                {'type': 'object', 'properties': {'a': {'const': 'only'}}},
                {'a': 'other'},
                "$.a: 'other' is not the allowed value (allowedValues: only)",
            ),
            (
                {'type': 'object', 'properties': {'a': {'items': {'type': 'string'}}}},
                {'a': ['ok', 3]},
                '$.a[1]: expected string, got integer',
            ),
            (
                {'type': 'object', 'properties': {'a': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}},
                {'a': 1.5},
                '$.a: 1.5 does not match any of the allowed schemas',
            ),
            (
                {'type': 'object', 'additionalProperties': {'type': 'string'}},
                {'extra': 4},
                '$.extra: expected string, got integer',
            ),
            (
                {
                    'type': 'object',
                    'properties': {'a': {'type': 'object', 'properties': {'b': {'type': 'string'}}}},
                },
                {'a': {'b': 2}},
                '$.a.b: expected string, got integer',
            ),
        ],
    )
    def test_violation_is_reported_to_the_model(
        self, schema: dict[str, Any], value: dict[str, Any], expected: str
    ) -> None:
        assert expected in _first_retry(schema, value)

    def test_enum_matching_distinguishes_true_from_one(self) -> None:
        message = _first_retry({'type': 'object', 'properties': {'a': {'enum': [1]}}}, {'a': True})

        assert 'is not one of the allowed values' in message

    def test_every_violation_is_collected_in_one_retry(self) -> None:
        model, _ = _run(REPORT, {'totally': 'wrong', 'verdict': 'NOT-IN-ENUM'}, {'verdict': 'pass'})

        message = model.retries[0].model_response()
        assert '$.totally: additional property not allowed' in message
        assert "$.verdict: 'NOT-IN-ENUM' is not one of the allowed values (allowedValues: pass, fail)" in message

    def test_the_retry_names_the_output_tool(self) -> None:
        model, _ = _run(REPORT, {'verdict': 'nope'}, {'verdict': 'pass'}, name='verdict_tool')

        assert model.retries[0].tool_name == 'verdict_tool'

    def test_the_model_can_correct_itself(self) -> None:
        model, result = _run(REPORT, {'verdict': 'NOT-IN-ENUM'}, {'verdict': 'pass'})

        assert model.calls == 2
        assert result.output == {'verdict': 'pass'}

    def test_the_retry_budget_is_finite(self) -> None:
        with pytest.raises(UnexpectedModelBehavior):
            _run(REPORT, {'verdict': 'never-valid'}, max_retries=1)


class TestErrorText:
    def test_long_values_are_truncated(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'enum': ['ok']}}}
        message = _first_retry(schema, {'a': 'x' * 500})

        assert '...' in message
        assert 'x' * 299 in message
        assert 'x' * 300 not in message

    def test_long_enum_labels_are_truncated(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'enum': ['y' * 200, 'ok']}}}
        message = _first_retry(schema, {'a': 'z'})

        assert 'ok' in message
        assert 'y' * 80 in message
        assert 'y' * 81 not in message

    def test_non_string_enum_labels_are_rendered(self) -> None:
        message = _first_retry({'type': 'object', 'properties': {'a': {'enum': [1, 2]}}}, {'a': 9})

        assert 'allowedValues: 1, 2' in message

    def test_message_names_the_schema(self) -> None:
        message = _first_retry(REPORT, {'verdict': 'nope'})

        assert message.startswith('Output does not match required schema:')


class TestSatisfiabilityLint:
    def test_root_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(UserError, match='root_not_object'):
            SchemaOutput({'type': 'array', 'items': {'type': 'string'}})

    def test_root_without_a_type_is_refused(self) -> None:
        with pytest.raises(UserError, match='declares no type'):
            SchemaOutput({'properties': {'a': {'type': 'string'}}})

    def test_required_property_that_is_forbidden_is_refused(self) -> None:
        with pytest.raises(UserError, match='required_property_forbidden'):
            SchemaOutput({'type': 'object', 'properties': {}, 'required': ['a'], 'additionalProperties': False})

    def test_empty_enum_on_a_required_property_is_refused(self) -> None:
        with pytest.raises(UserError, match='enum_type_mismatch'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'enum': []}}})

    def test_enum_conflicting_with_its_type_is_refused(self) -> None:
        with pytest.raises(UserError, match='no enum value is of the declared type'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'type': 'string', 'enum': [1, 2]}}})

    def test_const_conflicting_with_its_type_is_refused(self) -> None:
        with pytest.raises(UserError, match='const_mismatch'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'type': 'string', 'const': 5}}})

    def test_const_outside_its_enum_is_refused(self) -> None:
        with pytest.raises(UserError, match='is not among the enum values'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'enum': ['x'], 'const': 'y'}}})

    @pytest.mark.parametrize(
        'constraint',
        [
            {'type': 'integer', 'minimum': 10, 'maximum': 1},
            {'type': 'integer', 'exclusiveMinimum': 10, 'exclusiveMaximum': 1},
            {'type': 'string', 'minLength': 5, 'maxLength': 2},
            {'type': 'array', 'minItems': 5, 'maxItems': 2},
            {'type': 'object', 'minProperties': 5, 'maxProperties': 2},
        ],
    )
    def test_crossed_bounds_are_refused(self, constraint: dict[str, Any]) -> None:
        with pytest.raises(UserError, match='crossed_bounds'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': constraint}})

    def test_bounds_that_are_not_numbers_are_ignored(self) -> None:
        with pytest.warns(UserWarning, match='does not enforce'):
            SchemaOutput({'type': 'object', 'properties': {'a': {'type': 'integer', 'minimum': True, 'maximum': 'x'}}})

    def test_bounds_that_touch_are_satisfiable(self) -> None:
        with pytest.warns(UserWarning, match='does not enforce'):
            SchemaOutput({'type': 'object', 'properties': {'a': {'type': 'integer', 'minimum': 2, 'maximum': 2}}})

    def test_an_optional_property_only_warns(self) -> None:
        with pytest.warns(UserWarning, match='can still succeed'):
            output = SchemaOutput({'type': 'object', 'properties': {'a': {'enum': []}}})

        assert isinstance(output, ToolOutput)

    def test_an_array_element_only_warns(self) -> None:
        with pytest.warns(UserWarning, match='can still succeed'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'items': {'enum': []}}}})

    def test_an_anyof_branch_only_warns(self) -> None:
        with pytest.warns(UserWarning, match='can still succeed'):
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['a'],
                    'properties': {'a': {'anyOf': [{'type': 'string'}, {'enum': []}]}},
                }
            )

    def test_a_nested_required_chain_is_refused(self) -> None:
        with pytest.raises(UserError, match='enum_type_mismatch'):
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['a'],
                    'properties': {'a': {'type': 'object', 'required': ['b'], 'properties': {'b': {'enum': []}}}},
                }
            )

    def test_a_satisfiable_schema_is_silent(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SchemaOutput(REPORT)


class TestUnsupportedKeywords:
    def test_unenforced_keywords_are_named(self) -> None:
        with pytest.warns(UserWarning, match=r'does not enforce.*minLength'):
            SchemaOutput({'type': 'object', 'properties': {'a': {'type': 'string', 'minLength': 3}}})

    def test_the_warning_reports_the_path(self) -> None:
        with pytest.warns(UserWarning, match=r'at \$\.a'):
            SchemaOutput({'type': 'object', 'properties': {'a': {'type': 'string', 'pattern': '^x'}}})

    def test_unenforced_keywords_still_reach_the_model(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'string', 'minLength': 3}}}
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        model = _Replay({'a': 'xy'})
        Agent(FunctionModel(model), output_type=output).run_sync('go')

        assert model.wire_schema['properties']['a']['minLength'] == 3

    def test_an_unenforced_constraint_does_not_fail_the_run(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'string', 'minLength': 3}}}
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'a': 'xy'})), output_type=output).run_sync('go')

        assert result.output == {'a': 'xy'}

    def test_tuple_form_items_is_reported(self) -> None:
        with pytest.warns(UserWarning, match=r'does not enforce.*items'):
            SchemaOutput({'type': 'object', 'properties': {'a': {'items': [{'type': 'string'}, {'type': 'integer'}]}}})

    def test_keywords_inside_anyof_and_additional_properties_are_reported(self) -> None:
        with pytest.warns(UserWarning, match=r'does not enforce.*format'):
            SchemaOutput(
                {
                    'type': 'object',
                    'properties': {'a': {'anyOf': [{'type': 'string', 'format': 'email'}]}},
                    'additionalProperties': {'type': 'string', 'format': 'uri'},
                }
            )

    def test_annotations_are_not_reported(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SchemaOutput(
                {
                    'type': 'object',
                    'title': 'Report',
                    'description': 'a report',
                    'properties': {'a': {'type': 'string', 'description': 'the a', 'default': 'x'}},
                }
            )


class TestRefsAndDefs:
    def test_non_recursive_refs_are_inlined_and_enforced(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['leaf'],
            'properties': {'leaf': {'$ref': '#/$defs/Leaf'}},
            '$defs': {'Leaf': {'type': 'object', 'required': ['v'], 'properties': {'v': {'type': 'string'}}}},
        }
        model, result = _run(schema, {'leaf': {'v': 1}}, {'leaf': {'v': 'ok'}})

        assert '$.leaf.v: expected string, got integer' in model.retries[0].model_response()
        assert result.output == {'leaf': {'v': 'ok'}}

    def test_recursive_refs_are_refused_by_core(self) -> None:
        with pytest.raises(UserError, match='recursive'):
            SchemaOutput(
                {
                    'type': 'object',
                    'properties': {'child': {'$ref': '#/$defs/Node'}},
                    '$defs': {'Node': {'type': 'object', 'properties': {'child': {'$ref': '#/$defs/Node'}}}},
                }
            )


class TestMalformedSchemaFragments:
    """A schema is caller data: a fragment in the wrong shape is ignored, not crashed on."""

    def test_a_property_whose_schema_is_not_an_object_is_ignored(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': 'not-a-schema'}}
        _, result = _run(schema, {'a': 1})

        assert result.output == {'a': 1}

    def test_an_anyof_branch_that_is_not_an_object_is_ignored(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {'a': {'anyOf': ['not-a-schema', {'type': 'string'}]}},
        }
        _, result = _run(schema, {'a': 'ok'})

        assert result.output == {'a': 'ok'}

    def test_an_anyof_with_no_usable_branch_rejects(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'anyOf': ['not-a-schema']}}}

        assert 'does not match any of the allowed schemas' in _first_retry(schema, {'a': 'x'})

    def test_an_empty_type_list_enforces_nothing(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': []}}}
        _, result = _run(schema, {'a': 'anything'})

        assert result.output == {'a': 'anything'}

    def test_required_entries_that_are_not_strings_are_ignored(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'required': [1], 'properties': {}}
        _, result = _run(schema, {})

        assert result.output == {}

    def test_an_undeclared_key_is_allowed_when_additional_properties_is_unset(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'string'}}}
        _, result = _run(schema, {'a': 'x', 'extra': 1})

        assert result.output == {'a': 'x', 'extra': 1}

    def test_an_array_without_items_is_not_inspected(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'array'}}}
        _, result = _run(schema, {'a': [1, 'mixed', None]})

        assert result.output == {'a': [1, 'mixed', None]}

    def test_an_array_where_a_scalar_was_declared_is_reported(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'string'}}}

        assert '$.a: expected string, got array' in _first_retry(schema, {'a': []})


class TestDraft07NumberSemantics:
    """`integer` matches any number with a zero fractional part, and JSON `1.0` parses to a float."""

    @pytest.mark.parametrize('value', [1, 1.0, -3.0])
    def test_an_integral_float_satisfies_integer(self, value: float) -> None:
        _, result = _run({'type': 'object', 'properties': {'a': {'type': 'integer'}}}, {'a': value})

        assert result.output == {'a': value}

    def test_a_fractional_number_still_fails_integer(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'type': 'integer'}}}

        assert '$.a: expected integer, got number' in _first_retry(schema, {'a': 1.5})

    def test_an_integral_float_enum_is_not_refused_as_unsatisfiable(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'type': 'integer', 'enum': [1.0, 2.0]}},
        }

        assert _lint_warnings(schema) == []
        _, result = _run(schema, {'a': 2})
        assert result.output == {'a': 2}

    def test_an_integral_float_const_is_not_refused_as_unsatisfiable(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'type': 'integer', 'const': 1.0}},
        }

        assert _lint_warnings(schema) == []


class TestJsonEqualityIsStructural:
    """`True == 1` in Python, but `true` and `1` are distinct JSON values at every depth."""

    @pytest.mark.parametrize(
        'const,value',
        [
            ({'k': 1}, {'k': True}),
            ([1], [True]),
            ([1, 2], [1]),
            ({'k': 1}, {'other': 1}),
            ({'k': 1}, [1]),
            ([{'k': 1}], [{'k': False}]),
        ],
    )
    def test_a_lookalike_container_is_rejected(self, const: object, value: object) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'const': const}}}

        assert 'is not the allowed value' in _first_retry(schema, {'a': value})

    @pytest.mark.parametrize('value', [{'k': 1}, {'k': 1.0}])
    def test_an_equal_container_is_accepted(self, value: object) -> None:
        _, result = _run({'type': 'object', 'properties': {'a': {'const': {'k': 1}}}}, {'a': value})

        assert result.output == {'a': value}

    def test_an_equal_array_is_accepted(self) -> None:
        _, result = _run({'type': 'object', 'properties': {'a': {'const': [1, 'x']}}}, {'a': [1, 'x']})

        assert result.output == {'a': [1, 'x']}


class TestAdditionalPropertiesScope:
    """`additionalProperties` applies to names in neither `properties` nor `patternProperties`."""

    def test_a_declared_property_is_never_additional(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {'a': 'not-a-schema'},
            'additionalProperties': False,
        }
        _, result = _run(schema, {'a': 1})

        assert result.output == {'a': 1}

    def test_pattern_properties_suppresses_the_rejection(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'^x': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'x1': 'ok'})), output_type=output).run_sync('go')

        assert result.output == {'x1': 'ok'}

    def test_an_undeclared_name_is_still_rejected_without_pattern_properties(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {}, 'additionalProperties': False}

        assert '$.nope: additional property not allowed' in _first_retry(schema, {'nope': 1})


class TestLintSeesThroughRefs:
    def test_an_unsatisfiable_subschema_behind_a_ref_is_refused(self) -> None:
        with pytest.raises(UserError, match='enum_type_mismatch'):
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['leaf'],
                    'properties': {'leaf': {'$ref': '#/$defs/Leaf'}},
                    '$defs': {
                        'Leaf': {
                            'type': 'object',
                            'required': ['v'],
                            'properties': {'v': {'type': 'string', 'enum': [1, 2]}},
                        }
                    },
                }
            )

    def test_a_finding_visible_to_both_passes_is_reported_once(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'enum': []}}}

        assert len(_lint_warnings(schema)) == 1

    def test_a_ref_with_siblings_is_refused_without_first_warning(self) -> None:
        # The first pass cannot type `$.a` (its `type` is behind the `$ref`), so only the
        # second pass can tell that `b` is on a required path.
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'$ref': '#/$defs/R', 'required': ['b'], 'properties': {'b': {'enum': []}}}},
            '$defs': {'R': {'type': 'object'}},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(UserError, match='enum_type_mismatch'):
                SchemaOutput(schema)

        assert not [w for w in caught if 'can still succeed' in str(w.message)]


class TestMessagesRenderJson:
    """The reply is JSON, so a correction must not tell the model to emit `True` or `None`."""

    def test_boolean_and_null_allowed_values(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'enum': [True, None]}}}

        assert 'allowedValues: true, null' in _first_retry(schema, {'a': 'x'})

    def test_the_offending_value_renders_as_json(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'const': 'only'}}}

        assert '{"k": true}' in _first_retry(schema, {'a': {'k': True}})

    def test_strings_keep_their_quoted_form(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'enum': ['pass', 'fail']}}}
        message = _first_retry(schema, {'a': 'nope'})

        assert "'nope' is not one of the allowed values (allowedValues: pass, fail)" in message

    def test_a_value_json_cannot_serialize_falls_back(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'enum': [object()]}}}
        message = _first_retry(schema, {'a': 'x'})

        assert 'is not one of the allowed values' in message
        assert 'object object at' in message


class TestSchemaIsRuntimeData:
    """A schema can arrive from a file or a workflow argument, so a bad one must not crash."""

    @pytest.mark.parametrize('schema', [True, [], 'x', None, 7])
    def test_a_non_object_schema_raises_the_documented_error(self, schema: object) -> None:
        with pytest.raises(UserError, match='requires a JSON Schema object'):
            SchemaOutput(schema)  # pyright: ignore[reportArgumentType]


class TestPatternPropertiesDecidesCoverage:
    def test_a_matching_name_is_exempt_from_additional_properties(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'^x': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'x1': 'ok'})), output_type=output).run_sync('go')

        assert result.output == {'x1': 'ok'}

    def test_a_name_matching_no_pattern_is_still_rejected(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'^x': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        model = _Replay({'zzz': 'sneaky'})
        with pytest.raises(UnexpectedModelBehavior):
            Agent(FunctionModel(model), output_type=output).run_sync('go')

        assert '$.zzz: additional property not allowed' in model.retries[0].model_response()

    def test_an_uncompilable_pattern_exempts_nothing(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'[unclosed': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        model = _Replay({'anything': 1})
        with pytest.raises(UnexpectedModelBehavior):
            Agent(FunctionModel(model), output_type=output).run_sync('go')

        assert 'additional property not allowed' in model.retries[0].model_response()

    def test_a_required_name_matching_a_pattern_is_satisfiable(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['x1'],
            'properties': {},
            'patternProperties': {'^x': {'type': 'string'}},
            'additionalProperties': False,
        }

        assert _lint_warnings(schema) == []

    def test_a_required_name_matching_no_pattern_is_refused(self) -> None:
        with pytest.raises(UserError, match='required_property_forbidden'):
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['zzz'],
                    'properties': {},
                    'patternProperties': {'^x': {'type': 'string'}},
                    'additionalProperties': False,
                }
            )

    def test_a_matching_name_is_not_checked_against_a_schema_valued_additional_properties(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'patternProperties': {'^x_': {'type': 'integer'}},
            'additionalProperties': {'type': 'string'},
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'x_1': 1})), output_type=output).run_sync('go')

        assert result.output == {'x_1': 1}

    def test_a_name_matching_no_pattern_is_checked_against_it(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'patternProperties': {'^x_': {'type': 'integer'}},
            'additionalProperties': {'type': 'string'},
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        model = _Replay({'other': 1})
        with pytest.raises(UnexpectedModelBehavior):
            Agent(FunctionModel(model), output_type=output).run_sync('go')

        assert '$.other: expected string, got integer' in model.retries[0].model_response()


class TestPatternsRunInLinearTime:
    """The name a pattern is matched against is the model's, so matching must not backtrack."""

    def test_a_backtracking_pattern_cannot_stall_validation(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'^(a+)+$': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema, max_retries=1)
        # 28 bytes: `re` needs about ten seconds here and doubles per byte, so a regression
        # to a backtracking engine fails the bound below instead of hanging the suite.
        model = _Replay({'a' * 28 + 'b': 'x'})
        started = time.perf_counter()
        with pytest.raises(UnexpectedModelBehavior):
            Agent(FunctionModel(model), output_type=output).run_sync('go')

        assert time.perf_counter() - started < 2
        assert 'additional property not allowed' in model.retries[0].model_response()

    def test_a_pattern_the_engine_rejects_exempts_nothing(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'^(?=x)': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema, max_retries=1)
        model = _Replay({'x1': 'looks covered'})
        with pytest.raises(UnexpectedModelBehavior):
            Agent(FunctionModel(model), output_type=output).run_sync('go')

        assert '$.x1: additional property not allowed' in model.retries[0].model_response()

    def test_matching_is_unanchored(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'_id$': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'user_id': 'u1'})), output_type=output).run_sync('go')

        assert result.output == {'user_id': 'u1'}

    def test_character_classes_are_unicode_aware(self) -> None:
        # The same dialect as pydantic's own `pattern`: `\\d` is any Unicode digit, not `[0-9]`.
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {},
            'patternProperties': {'^\\d+$': {'type': 'string'}},
            'additionalProperties': False,
        }
        with pytest.warns(UserWarning, match='does not enforce'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'\u0661': 'one'})), output_type=output).run_sync('go')

        assert result.output == {'\u0661': 'one'}


class TestLintReadsEveryDeclaredType:
    def test_an_array_valued_type_still_catches_an_impossible_enum(self) -> None:
        with pytest.raises(UserError, match='enum_type_mismatch'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'type': ['string'], 'enum': [1]}}})

    def test_an_array_valued_type_still_catches_an_impossible_const(self) -> None:
        with pytest.raises(UserError, match='const_mismatch'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'type': ['string'], 'const': 1}}})

    def test_a_value_matching_either_declared_type_is_satisfiable(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'type': ['string', 'null'], 'enum': ['x', None]}},
        }

        assert _lint_warnings(schema) == []

    def test_bounds_beyond_float_precision_are_still_crossed(self) -> None:
        with pytest.raises(UserError, match='crossed_bounds'):
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['a'],
                    'properties': {'a': {'type': 'integer', 'minimum': 9007199254740993, 'maximum': 9007199254740992}},
                }
            )


class TestNestedUnenforcedKeywords:
    @pytest.mark.parametrize(
        'schema',
        [
            {'type': 'object', 'properties': {'a': {'patternProperties': {'^x': {'minLength': 3}}}}},
            {'type': 'object', 'properties': {'a': {'oneOf': [{'type': 'string', 'minLength': 3}]}}},
            {'type': 'object', 'properties': {'a': {'allOf': [{'type': 'string', 'minLength': 3}]}}},
        ],
    )
    def test_a_keyword_inside_an_unenforced_container_is_named(self, schema: dict[str, Any]) -> None:
        with pytest.warns(UserWarning, match=r'does not enforce.*minLength'):
            SchemaOutput(schema)


class TestAnyOfSatisfiability:
    def test_an_anyof_with_no_satisfiable_branch_is_refused(self) -> None:
        with pytest.raises(UserError, match='no_satisfiable_branch'):
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['a'],
                    'properties': {'a': {'anyOf': [{'enum': []}, {'type': 'string', 'const': 5}]}},
                }
            )

    def test_an_empty_anyof_is_refused(self) -> None:
        with pytest.raises(UserError, match='no_satisfiable_branch'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'anyOf': []}}})

    def test_one_satisfiable_branch_is_enough(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'anyOf': [{'enum': []}, {'type': 'string'}]}},
        }

        # The dead branch still warns on its own; what must not happen is a refusal.
        assert not any('no_satisfiable_branch' in w for w in _lint_warnings(schema))

    def test_a_branch_that_is_not_a_schema_blocks_the_claim(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'anyOf': ['not-a-schema', {'enum': []}]}},
        }

        assert not any('no_satisfiable_branch' in w for w in _lint_warnings(schema))

    def test_an_optional_anyof_only_warns(self) -> None:
        with pytest.warns(UserWarning, match='can still succeed'):
            SchemaOutput({'type': 'object', 'properties': {'a': {'anyOf': [{'enum': []}]}}})


class TestBooleanSubschemas:
    """Since draft-06 a bare `true` or `false` may stand wherever a subschema may."""

    def test_a_true_anyof_branch_accepts_everything(self) -> None:
        _, result = _run({'type': 'object', 'properties': {'a': {'anyOf': [True]}}}, {'a': 1})

        assert result.output == {'a': 1}

    def test_a_false_property_schema_rejects_the_value(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': False}}

        assert '$.a: no value is allowed here' in _first_retry(schema, {'a': 1})

    def test_a_false_property_schema_permits_the_absence(self) -> None:
        _, result = _run({'type': 'object', 'properties': {'a': False}}, {})

        assert result.output == {}

    def test_false_items_rejects_every_element(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'items': False}}}

        assert '$.a[0]: no value is allowed here' in _first_retry(schema, {'a': [1]})

    def test_false_items_permits_the_empty_array(self) -> None:
        _, result = _run({'type': 'object', 'properties': {'a': {'items': False}}}, {'a': []})

        assert result.output == {'a': []}

    def test_true_additional_properties_admits_any_key(self) -> None:
        _, result = _run({'type': 'object', 'properties': {}, 'additionalProperties': True}, {'extra': 1})

        assert result.output == {'extra': 1}

    def test_a_false_anyof_branch_is_reported_with_the_others(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'anyOf': [False, {'type': 'string'}]}}}
        message = _first_retry(schema, {'a': 1})

        assert 'branch 0: $.a: no value is allowed here; branch 1: $.a: expected string, got integer' in message

    def test_a_required_property_with_a_false_schema_is_refused(self) -> None:
        with pytest.raises(UserError, match='false_schema'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': False}})

    def test_an_optional_property_with_a_false_schema_is_silent(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SchemaOutput({'type': 'object', 'properties': {'a': False}})

    def test_an_anyof_of_only_false_branches_is_refused(self) -> None:
        with pytest.raises(UserError, match='no_satisfiable_branch'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': {'anyOf': [False]}}})

    def test_a_true_branch_makes_an_anyof_satisfiable(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'required': ['a'], 'properties': {'a': {'anyOf': [False, True]}}}

        assert _lint_warnings(schema) == []


class TestBoundsBindOnlyToTheirType:
    @pytest.mark.parametrize(
        'constraint',
        [
            {'type': 'integer', 'minimum': 2, 'exclusiveMaximum': 2},
            {'type': 'number', 'exclusiveMinimum': 2, 'maximum': 2},
            {'type': 'number', 'exclusiveMinimum': 2, 'exclusiveMaximum': 2},
        ],
    )
    def test_an_exclusive_bound_meeting_its_partner_is_refused(self, constraint: dict[str, Any]) -> None:
        with pytest.raises(UserError, match='crossed_bounds'):
            SchemaOutput({'type': 'object', 'required': ['a'], 'properties': {'a': constraint}})

    def test_an_exclusive_bound_that_leaves_room_is_satisfiable(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'type': 'number', 'exclusiveMinimum': 1, 'maximum': 2}},
        }

        assert _lint_warnings(schema) == []

    @pytest.mark.parametrize(
        'constraint',
        [
            {'type': 'integer', 'minLength': 5, 'maxLength': 2},
            {'type': 'string', 'minimum': 10, 'maximum': 1},
            {'type': ['integer', 'string'], 'minimum': 10, 'maximum': 1},
            {'minimum': 10, 'maximum': 1},
        ],
    )
    def test_bounds_that_do_not_bind_the_declared_type_are_not_refused(self, constraint: dict[str, Any]) -> None:
        schema: dict[str, Any] = {'type': 'object', 'required': ['a'], 'properties': {'a': constraint}}

        assert _lint_warnings(schema) == []

    def test_a_crossed_bound_beyond_float_range_is_a_user_error(self) -> None:
        with pytest.raises(UserError, match='minimum 1000') as info:
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['a'],
                    'properties': {'a': {'type': 'integer', 'minimum': 10**400, 'maximum': 1}},
                }
            )

        assert 'maximum 1 ' in str(info.value)


class TestObjectKeywordsBindOnlyToObjects:
    """`required` and `properties` say nothing about a non-object, and the validator accepts one."""

    def test_a_dead_required_chain_under_a_string_typed_node_only_warns(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'type': 'string', 'required': ['b'], 'properties': {'b': {'enum': []}}}},
        }
        with pytest.warns(UserWarning, match='can still succeed'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'a': 'text'})), output_type=output).run_sync('go')

        assert result.output == {'a': 'text'}

    def test_a_dead_required_chain_under_an_untyped_node_only_warns(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'required': ['b'], 'properties': {'b': {'enum': []}}}},
        }

        assert any('enum_type_mismatch' in w for w in _lint_warnings(schema))

    def test_a_forbidden_required_name_under_a_non_object_node_only_warns(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'type': ['object', 'null'], 'required': ['b'], 'additionalProperties': False}},
        }

        assert any('required_property_forbidden' in w for w in _lint_warnings(schema))


class TestEnumAndConstAgainstTheRestOfTheNode:
    def test_a_const_object_missing_a_required_property_is_refused(self) -> None:
        with pytest.raises(UserError, match='does not satisfy the rest of the schema'):
            SchemaOutput({'type': 'object', 'const': {}, 'required': ['x']})

    def test_a_const_failing_a_property_schema_is_refused(self) -> None:
        with pytest.raises(UserError, match='const_mismatch'):
            SchemaOutput({'type': 'object', 'const': {'x': 'text'}, 'properties': {'x': {'type': 'integer'}}})

    def test_an_enum_whose_every_member_fails_the_node_is_refused(self) -> None:
        with pytest.raises(UserError, match='enum_mismatch'):
            SchemaOutput({'type': 'object', 'enum': [{}], 'required': ['x']})

    def test_one_conforming_enum_member_is_enough(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'enum': [{}, {'x': 1}], 'required': ['x']}

        assert _lint_warnings(schema) == []
        _, result = _run(schema, {'x': 1})
        assert result.output == {'x': 1}


class TestAnyOfBranchesDieOnlyOnTheirRequiredPath:
    def test_a_branch_with_an_optional_dead_property_is_satisfiable(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'anyOf': [{'type': 'object', 'properties': {'x': {'type': 'string', 'enum': [1]}}}],
        }
        # The dead `x` still warns on its own; what must not happen is a refusal.
        with pytest.warns(UserWarning, match='can still succeed'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({})), output_type=output).run_sync('go')

        assert result.output == {}

    def test_a_branch_with_a_dead_array_element_is_satisfiable(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {'a': {'anyOf': [{'type': 'array', 'items': {'enum': []}}]}},
        }
        with pytest.warns(UserWarning, match='can still succeed'):
            output = SchemaOutput(schema)
        result = Agent(FunctionModel(_Replay({'a': []})), output_type=output).run_sync('go')

        assert result.output == {'a': []}

    def test_a_branch_without_a_type_takes_its_parents(self) -> None:
        # The branch applies to the same instance as its object-typed parent, so `required` binds.
        with pytest.raises(UserError, match='no_satisfiable_branch'):
            SchemaOutput({'type': 'object', 'anyOf': [{'required': ['x'], 'properties': {'x': {'enum': []}}}]})

    def test_a_branch_under_a_parent_admitting_null_is_not_bound(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'required': ['a'],
            'properties': {
                'a': {'type': ['object', 'null'], 'anyOf': [{'required': ['x'], 'properties': {'x': {'enum': []}}}]}
            },
        }

        assert any('enum_type_mismatch' in w for w in _lint_warnings(schema))

    def test_a_branch_dead_on_its_own_required_path_still_counts(self) -> None:
        with pytest.raises(UserError, match='no_satisfiable_branch'):
            SchemaOutput(
                {
                    'type': 'object',
                    'required': ['a'],
                    'properties': {
                        'a': {'anyOf': [{'type': 'object', 'required': ['x'], 'properties': {'x': {'enum': []}}}]}
                    },
                }
            )


class TestRetryMessageBounds:
    def test_the_error_list_is_capped(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'items': {'type': 'string'}}}}
        message = _first_retry(schema, {'a': list(range(60))})

        assert '$.a[49]: expected string, got integer' in message
        assert '$.a[50]' not in message
        assert '$.a[49]: expected string, got integer; ... and 10 more' in message

    def test_errors_are_separated_by_semicolons(self) -> None:
        message = _first_retry(REPORT, {'totally': 'wrong', 'verdict': 'NOT-IN-ENUM'})

        assert '$.totally: additional property not allowed; $.verdict:' in message

    def test_an_anyof_failure_names_each_branch_reason(self) -> None:
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {'a': {'anyOf': [{'type': 'string'}, {'type': 'object', 'required': ['id']}]}},
        }
        message = _first_retry(schema, {'a': {}})

        assert '(branch 0: $.a: expected string, got object; branch 1: $.a.id: required property missing)' in message

    def test_the_whole_message_is_bounded_in_characters(self) -> None:
        # Two branches each carry 50 reasons quoting a 300-character value: the count cap
        # alone would let this one `anyOf` error run past 30,000 characters.
        schema: dict[str, Any] = {
            'type': 'object',
            'properties': {'a': {'anyOf': [{'items': {'enum': ['a']}}, {'items': {'enum': ['b']}}]}},
        }
        message = _first_retry(schema, {'a': [{'k': 'x' * 300}] * 60})

        assert 10_000 < len(message) < 10_200
        assert '... (message truncated)' in message

    def test_branch_reasons_are_capped_too(self) -> None:
        schema: dict[str, Any] = {'type': 'object', 'properties': {'a': {'anyOf': [{'items': {'type': 'string'}}]}}}
        message = _first_retry(schema, {'a': list(range(60))})

        assert '$.a[49]: expected string, got integer; ... and 10 more)' in message
        assert '$.a[50]' not in message
