"""Construction-time inspection of a caller's schema.

Two passes, both run once when `SchemaOutput` is built rather than per validation.

`lint_schema` looks for a schema that admits no value at all. Without it, an
impossible schema is discovered by spending the whole retry budget on a model that
cannot win, and the run fails with a validation error rather than the real cause.

`unsupported_keywords` reports what reaches the model but is never checked, so the
warning can name the gap. An unenforced constraint that reads as enforced is the
outcome worth avoiding.

Scope separates "can never work" from "works until someone fills that field". A
finding is `whole` when the failing subschema sits on a path every valid instance
must have: the root, or a chain of required properties. It is `subschema` when the
path is optional -- an unrequired property, an array element (the empty array still
satisfies), or one `anyOf` branch among several.

A keyword constrains only the instance type it applies to, so crossed `minimum` and
`maximum` on a node that admits strings are no contradiction: every string satisfies
the node, and `validate_instance` accepts one. Bounds, `required` and `properties`
are therefore held against a node only when its declared `type` leaves no other kind
of value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic_ai_harness.structured_output._validate import (
    ANNOTATION_KEYWORDS,
    ENFORCED_KEYWORDS,
    covered_by_pattern,
    is_array,
    is_schema,
    json_equal,
    json_type,
    validate_instance,
)

LintScope = Literal['whole', 'subschema']

_NUMERIC = frozenset({'integer', 'number'})
_OBJECT = frozenset({'object'})

# Lower and upper bound keywords, grouped by the instance type they constrain. Every
# lower bound is compared with every upper bound of its group, so an inclusive bound
# is also checked against the exclusive form of its partner.
_BOUNDS: tuple[tuple[frozenset[str], tuple[str, ...], tuple[str, ...]], ...] = (
    (_NUMERIC, ('minimum', 'exclusiveMinimum'), ('maximum', 'exclusiveMaximum')),
    (frozenset({'string'}), ('minLength',), ('maxLength',)),
    (frozenset({'array'}), ('minItems',), ('maxItems',)),
    (_OBJECT, ('minProperties',), ('maxProperties',)),
)


@dataclass(frozen=True)
class LintFinding:
    """One reason a schema, or part of it, can never be satisfied."""

    code: str
    path: str
    message: str
    scope: LintScope

    def describe(self) -> str:
        return f'{self.path}: {self.message} ({self.code})'


def lint_schema(schema: Mapping[str, object]) -> list[LintFinding]:
    findings: list[LintFinding] = []
    # Core requires the root to be exactly `type: 'object'`, treating a missing `type`
    # the same as a wrong one, and dereferences a root `$ref` before it looks.
    if '$ref' not in schema and schema.get('type') != 'object':
        declared = schema.get('type')
        saw = f'declares type {declared!r}' if declared is not None else 'declares no type'
        findings.append(
            LintFinding(
                code='root_not_object',
                path='$',
                message=f'the root schema {saw}, but structured output must be an object',
                scope='whole',
            )
        )
    _lint_node(schema, '$', True, findings)
    return findings


def _declared_types(schema: Mapping[str, object]) -> list[str]:
    """Every type name the schema declares. `type` may be a single name or a list of them."""
    declared = schema.get('type')
    if isinstance(declared, str):
        return [declared]
    if is_array(declared):
        return [name for name in declared if isinstance(name, str)]
    return []


def _binds(schema: Mapping[str, object], applies_to: frozenset[str]) -> bool:
    """Whether a keyword that constrains only `applies_to` types constrains every value this node admits."""
    declared = _declared_types(schema)
    return bool(declared) and all(name in applies_to for name in declared)


def _lint_node(schema: Mapping[str, object], path: str, required_path: bool, findings: list[LintFinding]) -> None:
    scope: LintScope = 'whole' if required_path else 'subschema'
    _lint_enum(schema, path, scope, findings)
    _lint_const(schema, path, scope, findings)
    _lint_bounds(schema, path, scope, findings)
    _lint_properties(schema, path, required_path, findings)
    _lint_any_of(schema, path, scope, findings)
    _lint_branches(schema, path, findings)


def _lint_enum(schema: Mapping[str, object], path: str, scope: LintScope, findings: list[LintFinding]) -> None:
    allowed = schema.get('enum')
    if not is_array(allowed):
        return
    if not allowed:
        findings.append(LintFinding('enum_type_mismatch', path, 'enum is empty, so no value can satisfy it', scope))
        return
    declared = _declared_types(schema)
    if declared and not any(_matches(declared, candidate) for candidate in allowed):
        findings.append(
            LintFinding(
                'enum_type_mismatch',
                path,
                f'no enum value is of the declared type {" or ".join(declared)!r}',
                scope,
            )
        )
        return
    if not any(_satisfies_rest(schema, candidate) for candidate in allowed):
        findings.append(LintFinding('enum_mismatch', path, 'no enum value satisfies the rest of the schema', scope))


def _lint_const(schema: Mapping[str, object], path: str, scope: LintScope, findings: list[LintFinding]) -> None:
    if 'const' not in schema:
        return
    expected = schema['const']
    declared = _declared_types(schema)
    if declared and not _matches(declared, expected):
        findings.append(
            LintFinding(
                'const_mismatch',
                path,
                f'const {expected!r} is not of the declared type {" or ".join(declared)!r}',
                scope,
            )
        )
        return
    allowed = schema.get('enum')
    if is_array(allowed) and allowed and not any(json_equal(expected, c) for c in allowed):
        findings.append(LintFinding('const_mismatch', path, f'const {expected!r} is not among the enum values', scope))
        return
    if not _satisfies_rest(schema, expected):
        findings.append(
            LintFinding('const_mismatch', path, f'const {expected!r} does not satisfy the rest of the schema', scope)
        )


def _satisfies_rest(schema: Mapping[str, object], candidate: object) -> bool:
    """Whether an `enum` or `const` candidate passes the node's other enforced keywords.

    The candidate is a concrete value, so the validator itself is the check: `required`,
    `properties`, `items` and `anyOf` apply to it exactly as they would to a reply.
    """
    rest = {keyword: value for keyword, value in schema.items() if keyword not in ('enum', 'const')}
    return not validate_instance(rest, candidate)


def _lint_bounds(schema: Mapping[str, object], path: str, scope: LintScope, findings: list[LintFinding]) -> None:
    for applies_to, low_keys, high_keys in _BOUNDS:
        if not _binds(schema, applies_to):
            continue
        for low_key in low_keys:
            for high_key in high_keys:
                message = _crossed(schema, low_key, high_key)
                if message is not None:
                    findings.append(LintFinding('crossed_bounds', path, message, scope))


def _crossed(schema: Mapping[str, object], low_key: str, high_key: str) -> str | None:
    low = _as_number(schema.get(low_key))
    high = _as_number(schema.get(high_key))
    if low is None or high is None:
        return None
    exclusive = low_key.startswith('exclusive') or high_key.startswith('exclusive')
    if low > high or (exclusive and low == high):
        # `repr` keeps an integer bound exact; formatting through `float` overflows past
        # 1.8e308 and loses precision long before that.
        return f'{low_key} {low!r} and {high_key} {high!r} leave no value between them'
    return None


def _lint_properties(schema: Mapping[str, object], path: str, required_path: bool, findings: list[LintFinding]) -> None:
    raw_properties = schema.get('properties')
    properties: Mapping[str, object] = raw_properties if is_schema(raw_properties) else {}
    raw_required = schema.get('required')
    required = [name for name in raw_required if isinstance(name, str)] if is_array(raw_required) else []
    # `required` and `properties` say nothing about a non-object, so they bind every valid
    # instance only when the node declares itself an object.
    binding = required_path and _binds(schema, _OBJECT)
    scope: LintScope = 'whole' if binding else 'subschema'

    if schema.get('additionalProperties') is False:
        for name in required:
            if name not in properties and not covered_by_pattern(name, schema):
                findings.append(
                    LintFinding(
                        'required_property_forbidden',
                        f'{path}.{name}',
                        'is required but is not declared in properties, and additionalProperties is false',
                        scope,
                    )
                )

    for name, subschema in properties.items():
        if subschema is False and name in required:
            findings.append(
                LintFinding(
                    'false_schema',
                    f'{path}.{name}',
                    'is required but its schema is false, which admits no value',
                    scope,
                )
            )
        elif is_schema(subschema):
            _lint_node(subschema, f'{path}.{name}', binding and name in required, findings)


def _lint_any_of(schema: Mapping[str, object], path: str, scope: LintScope, findings: list[LintFinding]) -> None:
    """`anyOf` is satisfiable when one branch is, so all of them failing is the node failing."""
    branches = schema.get('anyOf')
    if not is_array(branches):
        return
    for index, branch in enumerate(branches):
        if branch is True:
            return
        if branch is False:
            continue
        if not is_schema(branch):
            # A branch this validator cannot read may still admit a value.
            return
        reasons: list[LintFinding] = []
        _lint_node(_inherit_type(schema, branch), f'{path}.anyOf[{index}]', True, reasons)
        # Only a finding on the branch's own required path kills it. One under an optional
        # property or an array element leaves the branch satisfiable with that part unset.
        if not any(reason.scope == 'whole' for reason in reasons):
            return
    findings.append(LintFinding('no_satisfiable_branch', path, 'no anyOf branch can be satisfied', scope))


def _lint_branches(schema: Mapping[str, object], path: str, findings: list[LintFinding]) -> None:
    items = schema.get('items')
    # An empty array satisfies any `items`, so an impossible element schema never
    # makes the whole schema unsatisfiable.
    if is_schema(items):
        _lint_node(items, f'{path}[]', False, findings)

    branches = schema.get('anyOf')
    if not is_array(branches):
        return
    for index, branch in enumerate(branches):
        if is_schema(branch):
            _lint_node(_inherit_type(schema, branch), f'{path}.anyOf[{index}]', False, findings)


def _inherit_type(parent: Mapping[str, object], branch: Mapping[str, object]) -> Mapping[str, object]:
    """An `anyOf` branch applies to the same instance as its parent, so the parent's `type` binds it too.

    Only a branch without its own `type` takes the parent's, and it takes the whole list, so a
    parent admitting `object` or `null` still leaves the branch's object keywords unbound.
    """
    parent_types = _declared_types(parent)
    if 'type' in branch or not parent_types:
        return branch
    return {**branch, 'type': parent_types}


def _matches(declared: list[str], value: object) -> bool:
    actual = json_type(value)
    return any(actual in ('integer', 'number') if name == 'number' else name == actual for name in declared)


def _as_number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def unsupported_keywords(schema: Mapping[str, object]) -> dict[str, list[str]]:
    """Map each keyword that reaches the model unenforced to the paths it appears at."""
    found: dict[str, list[str]] = {}
    _scan(schema, '$', found)
    return {keyword: found[keyword] for keyword in sorted(found)}


def _scan(schema: Mapping[str, object], path: str, found: dict[str, list[str]]) -> None:
    for keyword in schema:
        if keyword in ENFORCED_KEYWORDS or keyword in ANNOTATION_KEYWORDS:
            continue
        found.setdefault(keyword, []).append(path)

    for keyword in ('properties', 'patternProperties'):
        container = schema.get(keyword)
        if is_schema(container):
            for name, subschema in container.items():
                if is_schema(subschema):
                    _scan(subschema, f'{path}.{name}', found)

    items = schema.get('items')
    if is_schema(items):
        _scan(items, f'{path}[]', found)
    elif is_array(items):
        # Tuple-form `items`: the positional schemas are not enforced either.
        found.setdefault('items', []).append(path)

    for keyword in ('anyOf', 'oneOf', 'allOf'):
        branches = schema.get(keyword)
        if is_array(branches):
            for index, branch in enumerate(branches):
                if is_schema(branch):
                    _scan(branch, f'{path}.{keyword}[{index}]', found)

    additional = schema.get('additionalProperties')
    if is_schema(additional):
        _scan(additional, f'{path}.*', found)
