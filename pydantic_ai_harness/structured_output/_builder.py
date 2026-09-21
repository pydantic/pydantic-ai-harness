"""The `SchemaOutput` factory."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter
from pydantic.json_schema import JsonSchemaValue
from pydantic_ai import ModelRetry, StructuredDict, ToolOutput
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.structured_output._lint import LintFinding, lint_schema, unsupported_keywords
from pydantic_ai_harness.structured_output._validate import (
    bounded,
    is_schema,
    render_retry_message,
    validate_instance,
)

__all__ = ['SchemaOutput']

# The reference is Claude Code's structured-output tool, the feature the Agent SDK documents at
# https://code.claude.com/docs/en/agent-sdk/structured-outputs. The wire text and the retry
# budget below mirror it.

# The reference's tool description, minus its second sentence ("You MUST call this tool
# exactly once at the end of your response"). An output tool ends the run natively and there
# is no competing text channel to foreclose, so that instruction would describe machinery the
# model cannot observe. No trailing period: core appends the schema's own top-level
# `description` after `. `, and a period here would put `..` on the wire.
DEFAULT_DESCRIPTION = 'Return your final response in the requested structured format'

# Core's own output budget defaults to 1, which is tight for a shape the model has to get
# right in one attempt. The reference allows 5; 3 keeps a run that cannot converge from
# spending five more model requests to find out.
DEFAULT_MAX_RETRIES = 3


def SchemaOutput(
    schema: JsonSchemaValue,
    *,
    name: str = 'structured_output',
    description: str = DEFAULT_DESCRIPTION,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ToolOutput[dict[str, Any]]:
    """Build an `output_type` that validates the model's reply against `schema`.

    `StructuredDict` already puts a JSON Schema on the wire as an output tool, but it
    compiles to `dict[str, Any]` and checks nothing that comes back. This wraps it in an
    output function that validates and raises `ModelRetry`, which core turns into a real
    retry carrying the violations, so the model can correct itself.

    The schema reaches the model unwrapped -- no parameter name leaks into it -- because
    the validator's single argument is annotated with a model-like type.

    Args:
        schema: A JSON Schema of `type: 'object'`. Enforced keywords are `type`,
            `properties`, `required`, `additionalProperties`, `items`, `enum`, `const` and
            `anyOf`, plus the boolean subschema forms `true` and `false`; anything else
            reaches the model as documentation but is not checked, and is named in a
            `UserWarning` at construction.
        name: The output tool's name, as the model and your traces see it.
        description: The output tool's description.
        max_retries: How many times the model may be asked to correct its output. Forwarded
            to `ToolOutput.max_retries`, so this is core's budget rather than a second counter.

    Returns:
        A `ToolOutput` accepted anywhere `output_type` is, including inside a list
        alongside other output types.

    Raises:
        UserError: If `schema` is not a JSON object, if it admits no value at all, or if
            core rejects it -- a root that is not an object, or recursive `$ref`s.

    Example:
    ```python {title="schema_output.py"}
    from pydantic_ai import Agent

    from pydantic_ai_harness import SchemaOutput

    REPORT = {
        'type': 'object',
        'properties': {
            'verdict': {'type': 'string', 'enum': ['pass', 'fail']},
            'findings': {'type': 'array', 'items': {'type': 'string'}},
        },
        'required': ['verdict'],
        'additionalProperties': False,
    }

    agent = Agent('anthropic:claude-fable-5', output_type=SchemaOutput(REPORT))
    ```
    """
    _require_object_schema(schema)
    # Core returns the caller's own dict for an object schema, so without a copy the output
    # tool would keep reading a schema the caller can still edit, while the validator holds
    # the snapshot taken here -- the two would silently disagree.
    schema = deepcopy(schema)

    findings = lint_schema(schema)
    _refuse(findings)

    structured = StructuredDict(schema)
    # Core inlines `$defs` and raises on refs that survive, so the validator never has to
    # resolve a pointer. `TypeAdapter` hands that inlined copy back.
    resolved: JsonSchemaValue = TypeAdapter(structured).json_schema()

    # Lint again now that `$ref`s are inlined: the first pass cannot see through a pointer,
    # and a subschema reached only by one would otherwise go unchecked. A node the first
    # pass could not type (a `$ref` with siblings) may turn an optional-path finding into a
    # refusal here, so warnings wait until both passes have had their say.
    already = {(f.code, f.path, f.scope) for f in findings}
    later = [f for f in lint_schema(resolved) if (f.code, f.path, f.scope) not in already]
    _refuse(later)
    _warn(findings + later)

    unsupported = unsupported_keywords(resolved)
    if unsupported:
        listed = ', '.join(f'{keyword} (at {", ".join(paths)})' for keyword, paths in unsupported.items())
        warnings.warn(
            f'SchemaOutput does not enforce {listed}. '
            'These reach the model as documentation only. Move the constraint into the '
            "field's description if the model needs to honour it.",
            UserWarning,
            stacklevel=2,
        )

    def _validate_structured_output(data: Any) -> dict[str, Any]:
        errors = validate_instance(resolved, data)
        if errors:
            raise ModelRetry(render_retry_message(errors))
        return dict(data)

    # `from __future__ import annotations` stringifies the signature, and the unwrapping in
    # core reads the annotation object. Assigning it directly is what keeps the schema flat
    # on the wire.
    _validate_structured_output.__annotations__['data'] = structured

    return ToolOutput(_validate_structured_output, name=name, description=description, max_retries=max_retries)


def _require_object_schema(schema: object) -> None:
    """A schema read from a file or a workflow argument is runtime data, not a literal."""
    if not is_schema(schema):
        raise UserError(
            f'SchemaOutput requires a JSON Schema object, got {type(schema).__name__}. '
            "Structured output must be declared as {'type': 'object', ...}."
        )


def _refuse(findings: list[LintFinding]) -> None:
    blocking = [finding for finding in findings if finding.scope == 'whole']
    if blocking:
        raise UserError(f'SchemaOutput was given a schema that no value can satisfy: {_listed(blocking)}')


def _listed(findings: list[LintFinding]) -> str:
    # One dead subschema reached through many required names is many findings, and nesting
    # multiplies them. The first handful say what is wrong; the rest only lengthen the message.
    return '; '.join(bounded([finding.describe() for finding in findings]))


def _warn(findings: list[LintFinding]) -> None:
    if findings:
        listed = _listed(findings)
        warnings.warn(
            f'SchemaOutput was given a schema with a part that no value can satisfy: {listed}. '
            'The run can still succeed while those fields are left unset.',
            UserWarning,
            stacklevel=3,
        )
