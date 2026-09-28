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
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import TypeGuard

from pydantic_core import SchemaError, SchemaValidator, ValidationError, core_schema

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

# A message names at most this many violations, at the top level and again inside each
# `anyOf` branch. A systematic mistake across a long array would otherwise produce a line
# per element, and the whole message re-enters the model's context on every retry; one
# example of the mistake is what a correction needs. The count cap multiplies through
# nested `anyOf` branches, so the finished message is also bounded in characters.
_MAX_ERRORS = 50
_MAX_CHARS = 10_000


def is_schema(value: object) -> TypeGuard[Mapping[str, object]]:
    """Whether a JSON value is an object-form subschema. JSON object keys are always strings."""
    return isinstance(value, dict)


def is_array(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, list)


def _is_subschema(value: object) -> bool:
    """A subschema is an object or, since draft-06, a bare boolean."""
    return isinstance(value, bool) or is_schema(value)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + '...'


def _render(value: object) -> str:
    """The offending value, quoted, for the body of a message."""
    return _truncate(_json_literal(value), _VALUE_CHARS)


def _label(value: object) -> str:
    """One `allowedValues` entry.

    Strings render bare so the list reads as a vocabulary, unless bare rendering would be
    ambiguous: an empty string, surrounding whitespace, or the list separator inside the
    value would otherwise read as a different set of members.
    """
    if isinstance(value, str) and value and value == value.strip() and ', ' not in value:
        text = value
    else:
        text = _json_literal(value)
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


def child_path(path: str, key: str) -> str:
    """The path of a property, dotted for an identifier-like name and bracketed otherwise.

    `$.a.b` would name both the key `a.b` and a `b` under `a`, and a retry that points the
    model at the wrong field costs a round trip; `$["a.b"]` cannot be misread.
    """
    return f'{path}.{key}' if key.isidentifier() else f'{path}[{json.dumps(key)}]'


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


def _apply(subschema: object, value: object, path: str, errors: list[str]) -> None:
    """Validate `value` against a subschema, honouring the boolean forms.

    `true` admits every value and `false` none. A fragment in any other shape is caller
    data in the wrong form, and is ignored rather than crashed on.
    """
    if subschema is False:
        errors.append(f'{path}: no value is allowed here')
    elif is_schema(subschema):
        _walk(subschema, value, path, errors)


def _check_any_of(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    branches = schema.get('anyOf')
    if not is_array(branches):
        return
    # Each branch's own reasons are kept: told only that nothing matched, the model cannot
    # tell which branch it was one field away from satisfying.
    reasons: list[str] = []
    for index, branch in enumerate(branches):
        if not _is_subschema(branch):
            continue
        found: list[str] = []
        _apply(branch, value, path, found)
        if not found:
            return
        reasons.append(f'branch {index}: {"; ".join(bounded(found))}')
    detail = f' ({"; ".join(reasons)})' if reasons else ''
    errors.append(f'{path}: {_render(value)} does not match any of the allowed schemas{detail}')


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

    additional = schema.get('additionalProperties', True)
    for key, item in value.items():
        child = child_path(path, key)
        if key in properties:
            _apply(properties[key], item, child, errors)
        elif additional is True or covered_by_pattern(key, schema):
            # Draft-07 scopes `additionalProperties` to names matched by neither `properties`
            # nor `patternProperties`, whatever form it takes. The pattern's own subschema is
            # outside the enforced dialect.
            continue
        elif additional is False:
            errors.append(f'{child}: additional property not allowed')
        else:
            _apply(additional, item, child, errors)


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
    matcher = _compile_pattern(pattern)
    return matcher is not None and matcher(key)


@lru_cache(maxsize=256)
def _compile_pattern(pattern: str) -> Callable[[str], bool] | None:
    """A matcher on pydantic's linear-time regex engine, or `None` for a pattern it rejects.

    The name being matched is the model's, and `re` backtracks: a pattern such as `^(a+)+$`
    against a crafted name holds the event loop for a time exponential in the name's length.
    The Rust engine pydantic uses for its own `pattern` constraint runs in linear time by
    construction, at the price of lookaround and backreferences, which it refuses when
    compiling. A pattern it cannot compile declares nothing, so it exempts nothing.

    Matching is unanchored, as draft-07 specifies for `patternProperties`.
    """
    try:
        validator = SchemaValidator(core_schema.str_schema(pattern=pattern, regex_engine='rust-regex'))
    except SchemaError:
        return None

    def matches(key: str) -> bool:
        try:
            validator.validate_python(key)
        except ValidationError:
            return False
        return True

    return matches


def _check_required(schema: Mapping[str, object], value: Mapping[str, object], path: str, errors: list[str]) -> None:
    required = schema.get('required')
    if not is_array(required):
        return
    for name in required:
        if isinstance(name, str) and name not in value:
            errors.append(f'{child_path(path, name)}: required property missing')


def _check_array(schema: Mapping[str, object], value: object, path: str, errors: list[str]) -> None:
    if not is_array(value):
        return
    items = schema.get('items')
    # Tuple-form `items` (a list of positional schemas) is outside the enforced dialect
    # and is reported by the unsupported-keyword scan instead.
    if not _is_subschema(items):
        return
    for index, item in enumerate(value):
        _apply(items, item, f'{path}[{index}]', errors)


def bounded(errors: Sequence[str]) -> list[str]:
    """At most `_MAX_ERRORS` of them, with a count of what was left out."""
    if len(errors) <= _MAX_ERRORS:
        return list(errors)
    return [*errors[:_MAX_ERRORS], f'... and {len(errors) - _MAX_ERRORS} more']


def render_retry_message(errors: Sequence[str]) -> str:
    # Individual errors already contain `, ` inside `allowedValues`, so they are separated
    # by something else.
    body = '; '.join(bounded(errors))
    if len(body) > _MAX_CHARS:
        body = f'{body[:_MAX_CHARS]}... (message truncated)'
    return 'Output does not match required schema: ' + body
