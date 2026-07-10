"""Static validation for authored tool slots.

A slot is validated before it can be served, so a broken slot never reaches the
calling model as a live tool. Validation covers four things:

- the declared `uses` allowlist is a subset of the capability's function pool;
- parameter names are usable identifiers that do not shadow a used function;
- Monty's pre-exec type-check passes against the parameter and function stubs,
  which catches wrong argument types, calls to undeclared names, and an async
  result used without `await`;
- the declared return type, when set, matches the slot's final expression
  (checked statically by annotating that expression before the type-check).

A discarded async call -- an injected function invoked as a bare statement whose
coroutine is thrown away -- is not caught by the type-check, so a small AST pass
flags it explicitly.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence

from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.experimental.confined_authoring._slots import (
    InjectedFunction,
    SlotParameter,
    SlotValueType,
    is_valid_identifier,
    monty_annotation,
    render_function_stubs,
)

try:
    from pydantic_monty import Monty, MontySyntaxError, MontyTypingError
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for confined authoring. Install it with: uv add "pydantic-ai-harness[code-mode]"'
    ) from _import_error

_RESULT_SINK = '_confined_result'


class SlotValidationError(Exception):
    """Raised when an authored slot fails static validation."""


def validate_tool_slot(
    *,
    parameters: Sequence[SlotParameter],
    uses: Sequence[str],
    code: str,
    returns: SlotValueType | None,
    functions: Mapping[str, InjectedFunction[AgentDepsT]],
) -> None:
    """Validate one tool slot against the capability's function pool.

    Raises `SlotValidationError` on the first failure. Returns `None` when the
    slot is safe to serve. Does not check the slot name -- the store owns that,
    because an invalid name must be rejected before anything is persisted.
    """
    _check_parameters(parameters, uses)
    _check_uses(uses, functions)

    used_functions = [functions[name] for name in uses]
    stubs = render_function_stubs(used_functions, parameters)
    checked_code = _code_for_type_check(code, returns)
    _type_check(checked_code, [parameter.name for parameter in parameters], stubs)
    _check_no_discarded_calls(code, set(uses))


def _check_parameters(parameters: Sequence[SlotParameter], uses: Sequence[str]) -> None:
    """Reject parameter names that are unusable, duplicated, or shadow a used function."""
    seen: set[str] = set()
    used = set(uses)
    for parameter in parameters:
        if not is_valid_identifier(parameter.name):
            raise SlotValidationError(f'parameter name {parameter.name!r} is not a valid Python identifier')
        if parameter.name in seen:
            raise SlotValidationError(f'duplicate parameter name {parameter.name!r}')
        if parameter.name in used:
            raise SlotValidationError(
                f'parameter {parameter.name!r} shadows the injected function of the same name; rename one'
            )
        seen.add(parameter.name)


def _check_uses(uses: Sequence[str], functions: Mapping[str, InjectedFunction[AgentDepsT]]) -> None:
    """Reject a `uses` entry that is not in the capability's function pool (default-deny)."""
    unknown = [name for name in uses if name not in functions]
    if unknown:
        available = sorted(functions) or ['(none)']
        raise SlotValidationError(
            f'slot uses function(s) not in the capability pool: {sorted(unknown)}. Available: {available}'
        )


def _code_for_type_check(code: str, returns: SlotValueType | None) -> str:
    """Return the source to type-check, annotating the final expression when a return type is declared.

    With a declared return type, the slot's final expression is rewritten to an
    annotated assignment so Monty verifies the produced value matches the
    declared type. A slot that declares a return type but does not end in an
    expression cannot produce that value, which is itself a validation error. If
    the code does not parse, the original source is type-checked so Monty reports
    the syntax error.
    """
    if returns is None:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        raise SlotValidationError(
            f'slot declares a {returns!r} return but its code does not end with a result expression to return'
        )
    final = tree.body[-1]
    tree.body[-1] = ast.AnnAssign(
        target=ast.Name(id=_RESULT_SINK, ctx=ast.Store()),
        annotation=ast.Name(id=monty_annotation(returns), ctx=ast.Load()),
        value=final.value,
        simple=1,
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _type_check(code: str, input_names: list[str], stubs: str) -> None:
    """Run Monty's static type-check, surfacing syntax and type errors as validation errors."""
    try:
        Monty(code, inputs=input_names, type_check=True, type_check_stubs=stubs)
    except MontySyntaxError as exc:
        raise SlotValidationError(f'syntax error:\n{exc.display()}') from exc
    except MontyTypingError as exc:
        raise SlotValidationError(f'type error:\n{exc.display("concise")}') from exc


def _check_no_discarded_calls(code: str, used: set[str]) -> None:
    """Flag an injected function called as a bare statement whose coroutine is discarded.

    Monty's type-check catches an un-awaited coroutine only when its value is
    used incompatibly; a result thrown away entirely slips through, so it is
    caught here. A call inside `await`, an assignment, or an argument to another
    call (such as `asyncio.gather`) is not a discarded statement and is left
    alone.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:  # pragma: no cover -- the type-check already reported this
        return
    discarded: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Name) and func.id in used and func.id not in discarded:
            discarded.append(func.id)
    if discarded:
        raise SlotValidationError(
            f'injected function(s) called without `await`, discarding the result: {discarded}. '
            f'Assign and await the call, e.g. `x = await {discarded[0]}(...)`.'
        )
