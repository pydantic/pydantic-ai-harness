"""Toolsets for confined authoring.

Two toolsets, composed by the capability:

- `_AuthoringControlToolset` exposes the fixed authoring tools -- `author_tool_slot`,
  `list_tool_slots`, `disable_tool_slot` -- that write to the slot store.
- `ConfinedAuthoringToolset` serves the store's active slots as real tools and
  runs each one in a Monty sandbox, dispatching only the injected functions the
  slot declared.

The serving toolset freezes the servable slots at run start (`for_run`), so a
slot authored mid-run becomes callable on the next `agent.run`, matching how
Pydantic AI resolves a run's tools once at the start.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import AbstractToolset, RunContext, ToolDefinition
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_core import SchemaValidator, ValidationError, to_jsonable_python
from typing_extensions import Self

from pydantic_ai_harness.experimental.confined_authoring._slots import (
    AuthoredSlot,
    SlotParameter,
    SlotValueType,
    build_args_json_schema,
    build_args_validator,
    build_return_validator,
)
from pydantic_ai_harness.experimental.confined_authoring._store import SlotStore

try:
    from pydantic_monty import (
        MontyRepl,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
        ResourceLimits,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for confined authoring. Install it with: uv add "pydantic-ai-harness[code-mode]"'
    ) from _import_error

from pydantic_ai_harness._monty_exec import MontyExecutor, PrintCapture, is_sandbox_panic

# `max_duration_secs` bounds pure sandbox compute: Monty checks it per bytecode step, so it caps a
# runaway like `while True: pass` that would otherwise block the host event loop until an external
# kill. It does NOT cover time spent awaiting host-side injected functions, which run suspended on
# the host rather than in the sandbox. 30s is generous for a typed tool-slot's pure compute; a host
# can raise any of these via `resource_limits=`.
_DEFAULT_LIMITS: ResourceLimits = {
    'max_memory': 256 * 1024 * 1024,
    'max_allocations': 50_000_000,
    'max_duration_secs': 30.0,
}


class _AuthoringControlToolset(FunctionToolset[AgentDepsT]):
    """The `author_tool_slot`, `list_tool_slots`, and `disable_tool_slot` tools over one store."""

    def __init__(self, store: SlotStore[AgentDepsT]) -> None:
        super().__init__(id='confined_authoring_control')
        self._store = store
        self.add_function(self.author_tool_slot, name='author_tool_slot')
        self.add_function(self.list_tool_slots, name='list_tool_slots')
        self.add_function(self.disable_tool_slot, name='disable_tool_slot')

    async def author_tool_slot(
        self,
        name: str,
        description: str,
        code: str,
        parameters: list[SlotParameter] | None = None,
        uses: list[str] | None = None,
        returns: SlotValueType | None = None,
    ) -> str:
        """Author a sandboxed tool the agent can call on its next run.

        `code` is a Monty script (a subset of Python). It reads each declared
        parameter as a bound variable, calls the injected functions it listed in
        `uses` (each is async -- `await` it and use its result), and its final
        expression is the tool's return value. Only the functions in `uses` are
        reachable; there is no import, filesystem, environment, clock,
        subprocess, or network access. The slot is validated immediately and,
        on success, becomes callable on the next agent run, not the current one.

        Args:
            name: Tool name for the slot. Lowercase letters, digits, and
                underscores, starting with a letter. Reusing a name replaces the
                previous slot.
            description: What the tool does, shown to the model that later calls it.
            code: The Monty script implementing the tool.
            parameters: The tool's typed parameters. Each is read by name inside `code`.
            uses: Names of injected functions the slot may call. Anything not
                listed is unavailable inside the sandbox.
            returns: Optional declared return type, checked against the code's
                final expression during validation.
        """
        try:
            record = self._store.author_tool(
                name=name,
                description=description,
                code=code,
                parameters=parameters or [],
                uses=uses or [],
                returns=returns,
            )
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        if record.last_error is not None:
            return (
                f'Tool slot {name!r} was written but failed validation: {record.last_error}\n'
                f'Fix the code and call author_tool_slot again with the same name.'
            )
        return f'Tool slot {name!r} authored and validated. It becomes callable on the next agent run, not this one.'

    async def list_tool_slots(self) -> str:
        """List the tool slots authored so far, with their status and any validation error."""
        records = self._store.list_all()
        if not records:
            return 'No tool slots authored yet.'
        lines: list[str] = []
        for record in records:
            suffix = f' -- ERROR: {record.last_error}' if record.last_error is not None else ''
            lines.append(f'- {record.name} [{record.status}] {record.description}{suffix}')
        return '\n'.join(lines)

    async def disable_tool_slot(self, name: str) -> str:
        """Disable a tool slot so it is no longer served on the next run.

        Args:
            name: Name of the slot to disable.
        """
        if self._store.disable(name):
            return f'Tool slot {name!r} disabled; it will not be served on the next run.'
        return f'No tool slot named {name!r}.'


@dataclass(frozen=True)
class _ServedTool:
    """One servable slot with its precomputed tool definition and argument validator."""

    slot: AuthoredSlot
    tool_def: ToolDefinition
    validator: SchemaValidator


def _prepare_served(slot: AuthoredSlot) -> _ServedTool:
    """Build the tool definition and argument validator for a servable slot."""
    return_note = f'\n\nReturns a `{slot.returns}` value.' if slot.returns is not None else ''
    tool_def = ToolDefinition(
        name=slot.name,
        description=f'{slot.description}{return_note}',
        parameters_json_schema=build_args_json_schema(slot.parameters),
        metadata={'confined_authoring': True, 'slot_kind': slot.kind},
    )
    return _ServedTool(slot=slot, tool_def=tool_def, validator=build_args_validator(slot.parameters))


@dataclass
class ConfinedAuthoringToolset(AbstractToolset[AgentDepsT]):
    """Serves a slot store's active slots as tools, each executed in a Monty sandbox.

    Attach this on its own to serve slots to an agent that cannot author them --
    for example a least-privilege agent that runs what a separate authoring agent
    produced, over a shared `SlotStore`.
    """

    store: SlotStore[AgentDepsT]
    """The slot store whose active slots are served."""

    max_retries: int = 3
    """Maximum retries for a served slot tool (a sandbox runtime error counts as a retry)."""

    resource_limits: ResourceLimits | None = None
    """Sandbox limits for slot execution. `None` uses a memory/allocation backstop plus a default
    30s cap on in-sandbox compute. A partial mapping merges onto that backstop, overriding only the
    caps it names."""

    toolset_id: str | None = None
    """Stable toolset id; defaults to `confined_authoring_slots`."""

    _served: dict[str, _ServedTool] = field(default_factory=dict[str, _ServedTool], init=False, repr=False)

    @property
    def id(self) -> str | None:
        return self.toolset_id or 'confined_authoring_slots'

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Self:
        """Freeze the servable slots for this run so the served tool set is stable within it."""
        clone = copy.copy(self)
        clone._served = {slot.name: _prepare_served(slot) for slot in self.store.load_servable()}
        return clone

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=served.tool_def,
                max_retries=self.max_retries,
                args_validator=served.validator,
            )
            for name, served in self._served.items()
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        served = self._served[name]
        slot = served.slot
        allowed = set(slot.uses)
        functions = self.store.pool

        async def dispatch(function_name: str, kwargs: dict[str, Any]) -> Any:
            # Only names in `allowed` reach here: MontyExecutor rejects any other call with a NameError
            # before dispatch, which is the default-deny boundary. Results are made JSON-safe so only
            # plain data re-enters the sandbox.
            result = await functions[function_name].call(ctx, dict(kwargs))
            return to_jsonable_python(result)

        # Bind every declared parameter: an omitted optional argument becomes None, matching the
        # `T | None` type the slot was validated against, rather than an unbound-name error.
        inputs: dict[str, object] = {parameter.name: None for parameter in slot.parameters if not parameter.required}
        inputs.update(tool_args)

        capture = PrintCapture()
        limits: ResourceLimits = {**_DEFAULT_LIMITS, **(self.resource_limits or {})}
        try:
            repl = MontyRepl(limits=limits)
            monty_state = repl.feed_start(slot.code, inputs=inputs, print_callback=capture)
            completed = await MontyExecutor(dispatch=dispatch, valid_names=allowed).run(monty_state)
        except (MontySyntaxError, MontyTypingError) as exc:  # pragma: no cover -- validated before serving
            raise ModelRetry(f'Tool slot {name!r} failed to run:\n{capture.prepend_to(exc.display())}') from exc
        except MontyRuntimeError as exc:
            raise ModelRetry(f'Tool slot {name!r} raised at runtime:\n{capture.prepend_to(exc.display())}') from exc
        except BaseException as exc:
            if not is_sandbox_panic(exc):
                raise
            raise ModelRetry(
                f'Tool slot {name!r} aborted inside the sandbox. This can happen when the same injected '
                'call is awaited more than once in one asyncio.gather -- give each gathered call its own '
                'invocation.'
            ) from exc

        result = to_jsonable_python(completed.output)
        return _shape_result(slot, result, capture.joined)


def _shape_result(slot: AuthoredSlot, result: object, printed: str) -> object:
    """Apply the declared return-type guard and fold in any captured print output."""
    if slot.returns is not None:
        try:
            build_return_validator(slot.returns).validate_python(result)
        except ValidationError:
            return {
                'error': (f'Tool slot {slot.name!r} returned a value that is not the declared {slot.returns!r} type.'),
                'value': result,
            }
    if not printed:
        return result if result is not None else {}
    if result is None:
        return {'output': printed}
    return {'output': printed, 'result': result}


def authoring_toolsets(
    store: SlotStore[AgentDepsT], *, max_retries: int, resource_limits: ResourceLimits | None
) -> list[AbstractToolset[AgentDepsT]]:
    """The authoring control toolset and the slot-serving toolset over one store."""
    return [
        _AuthoringControlToolset[AgentDepsT](store),
        ConfinedAuthoringToolset[AgentDepsT](store=store, max_retries=max_retries, resource_limits=resource_limits),
    ]
