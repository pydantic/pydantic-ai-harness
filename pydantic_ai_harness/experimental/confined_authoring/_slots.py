"""Data models and schema machinery for confined authoring.

A confined-authoring `slot` is a typed, sandboxed extension an agent authors for
itself. This module holds the persisted slot shape (`AuthoredSlot`), the typed
parameter model the model fills in (`SlotParameter`), the host-provided pool
entry a slot may call (`InjectedFunction`), and the deterministic mapping from
the small value-type subset to a JSON schema, a `pydantic_core` validator, and a
Monty type annotation. Every schema and stub a slot needs is derived from one
`_TypeSpec` table, so the tool-call schema, the runtime arg validator, and the
static type-check stubs cannot drift.
"""

from __future__ import annotations

import keyword
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal

from pydantic import BaseModel
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_core import SchemaValidator, core_schema

SlotKind = Literal['tool']
"""The kinds of slot an agent can author. Only `tool` slots exist today; the
manifest carries the field so `hook` and `instruction` kinds can be added
without a migration."""

SlotStatus = Literal['draft', 'validated', 'active', 'disabled']
"""Lifecycle of an authored slot.

- `draft`: written but not passing validation. `last_error` says why.
- `validated`: passed static validation at authoring time; will be served next run.
- `active`: re-validated against the current function pool and served this run.
- `disabled`: turned off by `disable_tool_slot`; never served.
"""

SlotValueType = Literal['string', 'integer', 'number', 'boolean', 'array', 'object']
"""The value types a slot parameter or return may take -- the JSON-schema-subset
confined authoring supports. Kept small on purpose: every type maps cleanly to a
JSON schema, a validator, and a Monty annotation."""


@dataclass(frozen=True)
class _TypeSpec:
    """How one `SlotValueType` renders across the three surfaces a slot needs."""

    json_type: str
    monty_annotation: str
    core_schema_builder: Callable[[], core_schema.CoreSchema]


_TYPE_SPECS: dict[SlotValueType, _TypeSpec] = {
    'string': _TypeSpec('string', 'str', core_schema.str_schema),
    'integer': _TypeSpec('integer', 'int', core_schema.int_schema),
    'number': _TypeSpec('number', 'float', core_schema.float_schema),
    'boolean': _TypeSpec('boolean', 'bool', core_schema.bool_schema),
    'array': _TypeSpec('array', 'list', core_schema.list_schema),
    'object': _TypeSpec('object', 'dict', core_schema.dict_schema),
}


class SlotParameter(BaseModel):
    """One typed parameter of an authored tool slot.

    The authored code reads the parameter by `name` as a bound variable; the
    calling model supplies it as a normal, schema-validated tool argument.
    """

    name: str
    """Identifier the slot code reads the value from. A Python identifier, unique within the slot."""

    type: SlotValueType = 'string'
    """The parameter's value type, from the supported subset."""

    required: bool = True
    """Whether the calling model must supply this argument."""

    description: str | None = None
    """Optional description shown to the calling model in the tool schema."""


class AuthoredSlot(BaseModel):
    """A persisted slot: the full record the manifest stores and a UI can read."""

    name: str
    kind: SlotKind = 'tool'
    description: str
    parameters: list[SlotParameter] = []
    uses: list[str] = []
    returns: SlotValueType | None = None
    code: str
    status: SlotStatus = 'draft'
    last_error: str | None = None


@dataclass(frozen=True)
class InjectedFunction(Generic[AgentDepsT]):
    """A host-provided function that authored slots may be granted access to.

    The capability's `functions` list is the capability-scoped pool. A slot
    reaches the host only through the subset of this pool it declares in `uses`
    (default-deny): no other function, and no ambient import, filesystem,
    environment, clock, subprocess, or network, is reachable from inside a slot's
    sandbox.

    `call` receives the run's `RunContext` (for deps, usage, model) and the
    keyword arguments the slot passed, and returns a JSON-serializable result.
    """

    name: str
    """Identifier the slot code calls this function by. A Python identifier, unique in the pool."""

    call: Callable[[RunContext[AgentDepsT], dict[str, object]], Awaitable[object]]
    """Async host implementation: `(ctx, kwargs) -> result`. The result is made JSON-safe before it re-enters the sandbox."""

    parameters: dict[str, object]
    """Object JSON schema for the function's keyword arguments, shown to the authoring model and used for the slot's static type-check."""

    returns: dict[str, object] | None = None
    """Optional JSON schema for the function's return value, rendered into the signature the authoring model sees."""

    description: str | None = None
    """Optional description rendered as the function's docstring in the author-facing catalog."""


def is_valid_identifier(name: str) -> bool:
    """Whether `name` is a Python identifier that is not a reserved keyword."""
    return name.isidentifier() and not keyword.iskeyword(name)


def index_functions(functions: Sequence[InjectedFunction[AgentDepsT]]) -> dict[str, InjectedFunction[AgentDepsT]]:
    """Index the injected-function pool by name, rejecting invalid or duplicate names."""
    by_name: dict[str, InjectedFunction[AgentDepsT]] = {}
    for function in functions:
        if not is_valid_identifier(function.name):
            raise ValueError(f'injected function name {function.name!r} is not a valid Python identifier; rename it')
        if function.name in by_name:
            raise ValueError(f'two injected functions are named {function.name!r}; names must be unique')
        by_name[function.name] = function
    return by_name


def build_args_json_schema(parameters: Sequence[SlotParameter]) -> dict[str, object]:
    """Build the tool-call JSON schema the calling model fills in for a slot."""
    properties: dict[str, object] = {}
    required: list[str] = []
    for parameter in parameters:
        prop: dict[str, object] = {'type': _TYPE_SPECS[parameter.type].json_type}
        if parameter.description:
            prop['description'] = parameter.description
        properties[parameter.name] = prop
        if parameter.required:
            required.append(parameter.name)
    return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}


def build_args_validator(parameters: Sequence[SlotParameter]) -> SchemaValidator:
    """Build a validator that turns a slot's raw tool arguments into a checked dict.

    Uses `extra_behavior='forbid'` so an argument the slot did not declare is
    rejected rather than passed silently into the sandbox.
    """
    fields = {
        parameter.name: core_schema.typed_dict_field(
            _TYPE_SPECS[parameter.type].core_schema_builder(), required=parameter.required
        )
        for parameter in parameters
    }
    return SchemaValidator(core_schema.typed_dict_schema(fields, extra_behavior='forbid'))


def build_return_validator(return_type: SlotValueType) -> SchemaValidator:
    """Build a validator for a slot's declared return value."""
    return SchemaValidator(_TYPE_SPECS[return_type].core_schema_builder())


def monty_annotation(value_type: SlotValueType) -> str:
    """The Monty type annotation for a slot value type (e.g. `object` -> `dict`)."""
    return _TYPE_SPECS[value_type].monty_annotation


def _parameter_declaration(parameter: SlotParameter) -> str:
    """A parameter's type-check declaration. An optional parameter is `T | None` (it may be absent)."""
    annotation = monty_annotation(parameter.type)
    return f'{parameter.name}: {annotation}' if parameter.required else f'{parameter.name}: {annotation} | None'


def _function_signature(function: InjectedFunction[AgentDepsT]) -> FunctionSignature:
    """Render one injected function as a keyword-only async signature."""
    return FunctionSignature.from_schema(
        name=function.name,
        parameters_schema=dict(function.parameters),
        return_schema=dict(function.returns) if function.returns is not None else None,
    )


def render_function_stubs(
    functions: Sequence[InjectedFunction[AgentDepsT]],
    parameters: Sequence[SlotParameter],
) -> str:
    """Build the Monty type-check stubs: typed parameter inputs plus the async signatures a slot may call."""
    signatures = [_function_signature(function) for function in functions]
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    type_blocks = FunctionSignature.render_type_definitions(signatures, conflicting)
    function_blocks = [
        signature.render('raise NotImplementedError()', is_async=True, conflicting_type_names=conflicting)
        for signature in signatures
    ]
    parts = ['import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal']
    if parameters:
        parts.append('\n'.join(_parameter_declaration(parameter) for parameter in parameters))
    parts.extend(type_blocks)
    parts.extend(function_blocks)
    return '\n\n'.join(parts)


def render_function_catalog(functions: Sequence[InjectedFunction[AgentDepsT]]) -> str:
    """Render the injected-function pool as async signatures for the authoring guidance."""
    if not functions:
        return 'No injected functions are available; authored slots can only compute over their parameters.'
    signatures = [_function_signature(function) for function in functions]
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    type_blocks = FunctionSignature.render_type_definitions(signatures, conflicting)
    function_blocks = [
        signature.render('...', description=function.description, is_async=True, conflicting_type_names=conflicting)
        for signature, function in zip(signatures, functions)
    ]
    blocks = [*type_blocks, *function_blocks]
    return '```python\n' + '\n\n'.join(blocks) + '\n```'
