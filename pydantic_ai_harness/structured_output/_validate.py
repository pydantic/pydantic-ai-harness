"""Instance validation for `SchemaOutput`.

Hand-written over a small keyword set rather than delegating to `jsonschema`. The
dependency is not in the base install -- it reaches `uv.lock` only transitively via
the `mcp` extra -- and the part that has to be right here is the error text the model
reads on a retry, not draft coverage.

Error shape follows the Claude Code reference, which engineers these messages for a
model rather than a human: every violation is collected so one retry can fix all of
them instead of converging a field per round trip, each carries the instance path it
failed at, and `enum`/`const` name their allowed values outright.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TypeGuard

# Checked against the model's reply. A subset of draft-07, matching the dialect the
# reference tells callers to stay inside.
ENFORCED_KEYWORDS = frozenset(
    {'type', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const', 'anyOf'}
)

# Carried to the model as documentation and never checked against the reply.
ANNOTATION_KEYWORDS = frozenset({'$schema', 'title', 'description', 'default', 'examples'})

# The offending value, and each label inside `allowedValues`, are truncated separately:
# a single oversized enum member should not crowd out the rest of the list.
_VALUE_CHARS = 300
_LABEL_CHARS = 80


def is_schema(value: object) -> TypeGuard[Mapping[str, object]]:
    """Whether a JSON value is a subschema. JSON object keys are always strings."""
    return isinstance(value, dict)


def is_array(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, list)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + '...'


def _render(value: object) -> str:
    """The offending value, quoted, for the body of a message."""
    return _truncate(_json_literal(value), _VALUE_CHARS)


def _label(value: object) -> str:
    """One `allowedValues` entry. Strings render bare so the list reads as a vocabulary."""
    text = value if isinstance(value, str) else _json_literal(value)
    return _truncate(text, _LABEL_CHARS)


def _json_literal(value: object) -> str:
    """Render as JSON, so a message never tells the model to emit `True` or `None`."""
    if isinstance(value, str):
        return repr(value)
    return json.dumps(value, default=repr)


def _allowed_values(values: Sequence[object]) -> str:
    return ', '.join(_label(value) for value in values)


def json_type(value: object) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int):
        return 'integer'
    if isinstance(value, float):
        return 'integer' if value.is_integer() else 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return 'unknown'  # pragma: no cover - the model's reply is parsed JSON, which has no other types


def _type_matches(declared: str, actual: str) -> bool:
    # JSON Schema's `number` admits integers; nothing else widens.
    if declared == 'number':
        return actual in ('integer', 'number')
    return declared == actual


def json_equal(left: object, right: object) -> bool:
    """JSON equality, structurally. `True == 1` in Python, but `true` and `1` are distinct JSON values.

    Numbers still compare mathematically, so `1` equals `1.0` as draft-07 requires.
    """
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if is_array(left) and is_array(right):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right))
    if is_schema(left) and is_schema(right):
        return left.keys() == right.keys() and all(json_equal(left[k], right[k]) for k in left)
    return left == right


def _child(path: str, key: str) -> str:
    return f'{path}.{key}'


def validate_instance(schema: Mapping[str, object], value: object, path: str = '$') -> list[str]:
    """Collect every way `value` violates `schema`, deepest-first within each node."""
    errors: list[str] = []
    _walk(schema, value, path, errors)
    return errors


def _walk(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    _check_any_of(schema, value, path, errors)
    _check_type(schema, value, path, errors)
    _check_enum(schema, value, path, errors)
    _check_const(schema, value, path, errors)
    _check_object(schema, value, path, errors)
    _check_array(schema, value, path, errors)


def _check_any_of(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    branches = schema.get('anyOf')
    if not is_array(branches):
        return
    for branch in branches:
        if is_schema(branch) and not validate_instance(branch, value, path):
            return
    errors.append(f'{path}: {_render(value)} does not match any of the allowed schemas')


def _check_type(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    declared = schema.get('type')
    if isinstance(declared, str):
        wanted = [declared]
    elif is_array(declared):
        wanted = [name for name in declared if isinstance(name, str)]
    else:
        return
    if not wanted:
        return
    actual = json_type(value)
    if any(_type_matches(name, actual) for name in wanted):
        return
    errors.append(f'{path}: expected {" or ".join(wanted)}, got {actual}')


def _check_enum(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    allowed = schema.get('enum')
    if not is_array(allowed):
        return
    if any(json_equal(value, candidate) for candidate in allowed):
        return
    errors.append(
        f'{path}: {_render(value)} is not one of the allowed values (allowedValues: {_allowed_values(allowed)})'
    )


def _check_const(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    if 'const' not in schema:
        return
    expected = schema['const']
    if json_equal(value, expected):
        return
    errors.append(f'{path}: {_render(value)} is not the allowed value (allowedValues: {_label(expected)})')


def _check_object(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    if not is_schema(value):
        return
    raw_properties = schema.get('properties')
    properties: Mapping[str, object] = raw_properties if is_schema(raw_properties) else {}
    _check_required(schema, value, path, errors)

    additional = schema.get('additionalProperties')
    for key, item in value.items():
        if key in properties:
            subschema = properties[key]
            if is_schema(subschema):
                _walk(subschema, item, _child(path, key), errors)
        elif additional is False and not covered_by_pattern(key, schema):
            errors.append(f'{_child(path, key)}: additional property not allowed')
        elif is_schema(additional):
            _walk(additional, item, _child(path, key), errors)


def covered_by_pattern(key: str, schema: Mapping[str, object]) -> bool:
    """Whether `patternProperties` declares this name, which exempts it from `additionalProperties`.

    The patterns decide membership even though the subschemas they point at are not enforced:
    ignoring them entirely would let any key through wherever `patternProperties` appears.
    """
    patterns = schema.get('patternProperties')
    if not is_schema(patterns):
        return False
    return any(_pattern_matches(pattern, key) for pattern in patterns)


def _pattern_matches(pattern: str, key: str) -> bool:
    try:
        return re.search(pattern, key) is not None
    except re.error:
        # A pattern Python cannot compile declares nothing, so it exempts nothing.
        return False


def _check_required(schema: Mapping[str, object], value: Mapping[str, object], path: str, errors: list[str]) -> None:
    required = schema.get('required')
    if not is_array(required):
        return
    for name in required:
        if isinstance(name, str) and name not in value:
            errors.append(f'{_child(path, name)}: required property missing')


def _check_array(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    if not is_array(value):
        return
    items = schema.get('items')
    # Tuple-form `items` (a list of positional schemas) is outside the enforced dialect
    # and is reported by the unsupported-keyword scan instead.
    if not is_schema(items):
        return
    for index, item in enumerate(value):
        _walk(items, item, f'{path}[{index}]', errors)


def render_retry_message(errors: Sequence[str]) -> str:
    return 'Output does not match required schema: ' + ', '.join(errors)
