"""SQLMode toolset: exposes selected tools as DuckDB functions behind a `run_sql` tool."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import uuid4

from anyio.from_thread import BlockingPortal
from anyio.to_thread import run_sync
from pydantic import Field, TypeAdapter
from pydantic_ai import RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import SchemaValidatorProt, ToolsetTool
from typing_extensions import TypedDict

from pydantic_ai_harness.experimental.sql_mode._duckdb import run_query

_RUN_SQL_TOOL_NAME = 'run_sql'

# JSON-Schema scalar types that map onto a native DuckDB column type. Everything
# else (objects, arrays, unions, refs, untyped) is carried as DuckDB `JSON`.
_SCALAR_DUCK_TYPES: dict[str, str] = {
    'string': 'VARCHAR',
    'integer': 'BIGINT',
    'number': 'DOUBLE',
    'boolean': 'BOOLEAN',
}

_INVALID_IDENTIFIER_CHARS = re.compile(r'[^a-zA-Z0-9_]')


class _RunSqlArguments(TypedDict):
    query: Annotated[str, Field(description='The DuckDB SQL query to run.')]


_RUN_SQL_ADAPTER = TypeAdapter(_RunSqlArguments)
_RUN_SQL_JSON_SCHEMA = _RUN_SQL_ADAPTER.json_schema()
_RUN_SQL_ARGS_VALIDATOR: SchemaValidatorProt = _RUN_SQL_ADAPTER.validator  # pyright: ignore[reportAssignmentType]

_BASE_DESCRIPTION = """\
Run a SQL query against a fresh, locked-down, in-memory DuckDB database.

Write SQL to orchestrate tool calls: invoke the tool functions listed below from
inside the query, navigate the JSON they return, pipe values between them, and
join, filter, and aggregate the results -- all in a single query and round-trip.

Every call gets a brand-new in-memory database; nothing persists between calls.
The database is sandboxed -- no filesystem, no network, no extension loading -- \
but every built-in DuckDB function is available, including the JSON functions \
(`json_extract`, `->`, `->>`, `unnest`, `array_agg`, `struct_pack`, and more).

A tool whose result is shown as `JSON` returns a JSON value: read fields from it \
with `->`/`->>`, or feed it into another tool that accepts a `JSON` argument. To \
call a tool once per item, pair it with `unnest`:

    SELECT my_tool(item) AS result FROM (SELECT unnest(['a', 'b', 'c']) AS item)

The result set is returned as `columns` (each with a name and type) and `rows`. \
Keep result sets small -- aggregate or `LIMIT` in SQL rather than returning raw rows.\
"""


@dataclass
class _CallableParam:
    """A single parameter of a SQL-callable tool, resolved to its DuckDB type."""

    name: str
    duck_type: str
    is_json: bool


@dataclass
class _CallableTool:
    """A tool exposed as a DuckDB function inside `run_sql`."""

    sql_name: str
    original_name: str
    description: str | None
    params: tuple[_CallableParam, ...]
    return_is_json: bool
    return_duck_type: str
    args_schema: dict[str, Any]
    return_schema: dict[str, Any] | None


@dataclass(kw_only=True)
class _RunSqlTool(ToolsetTool[AgentDepsT]):
    """`ToolsetTool` subclass that caches data computed during `get_tools`.

    Stores the SQL-callable tool definitions (for the description) and the
    wrapped toolset's tools (to build the dispatch `ToolManager`), avoiding a
    redundant `get_tools` call in `call_tool`.
    """

    callable_tools: tuple[_CallableTool, ...]
    wrapped_tools: dict[str, ToolsetTool[AgentDepsT]]


def _sanitize(name: str) -> str:
    """Turn a tool name into a valid SQL identifier (hyphens/dots become underscores)."""
    sanitized = _INVALID_IDENTIFIER_CHARS.sub('_', name)
    if sanitized[:1].isdigit():
        return f'_{sanitized}'
    return sanitized


def _duck_type(schema: dict[str, Any] | None) -> tuple[str, bool]:
    """Map a JSON-Schema fragment to `(duck_type, is_json)`.

    Scalars map to a native DuckDB type; `Optional[scalar]` is unwrapped; objects,
    arrays, multi-member unions, `$ref`s, and untyped schemas become `JSON`.
    """
    if schema is None:
        return 'JSON', True
    members = schema.get('anyOf') or schema.get('oneOf')
    if members is not None:
        non_null = [member for member in members if member.get('type') != 'null']
        if len(non_null) == 1:
            return _duck_type(non_null[0])
        return 'JSON', True
    schema_type = schema.get('type')
    if isinstance(schema_type, str):
        duck_type = _SCALAR_DUCK_TYPES.get(schema_type)
        if duck_type is not None:
            return duck_type, False
    return 'JSON', True


def _build_callable_tools(sandboxed_tools: dict[str, ToolsetTool[Any]]) -> tuple[_CallableTool, ...]:
    """Resolve each sandboxed tool's `ToolDefinition` into a `_CallableTool`."""
    callable_tools: list[_CallableTool] = []
    seen: dict[str, str] = {}
    for name, tool in sandboxed_tools.items():
        tool_def = tool.tool_def
        sql_name = _sanitize(name)
        if sql_name in seen:
            raise UserError(
                f'SQLMode: tools {seen[sql_name]!r} and {name!r} both map to the SQL function '
                f'name {sql_name!r}. Rename one of them.'
            )
        if sql_name == _RUN_SQL_TOOL_NAME:
            raise UserError(
                f'Tool {name!r} maps to the SQL function name {sql_name!r}, which is reserved for '
                f"SQLMode's run_sql meta-tool. Rename your tool to avoid conflicts."
            )
        seen[sql_name] = name
        properties: dict[str, Any] = tool_def.parameters_json_schema.get('properties', {})
        params = [
            _CallableParam(name=param_name, duck_type=duck_type, is_json=is_json)
            for param_name, param_schema in properties.items()
            for duck_type, is_json in [_duck_type(param_schema)]
        ]
        return_duck_type, return_is_json = _duck_type(tool_def.return_schema)
        callable_tools.append(
            _CallableTool(
                sql_name=sql_name,
                original_name=name,
                description=tool_def.description,
                params=tuple(params),
                return_is_json=return_is_json,
                return_duck_type=return_duck_type,
                args_schema=dict(tool_def.parameters_json_schema),
                return_schema=tool_def.return_schema,
            )
        )
    return tuple(callable_tools)


def _render_tool(tool: _CallableTool) -> str:
    """Render one SQL-callable tool as a signature, docstring, and JSON schemas."""
    signature = ', '.join(f'{param.name} {param.duck_type}' for param in tool.params)
    lines = [f'{tool.sql_name}({signature}) -> {tool.return_duck_type}']
    if tool.description:
        lines.extend(f'    {line}' for line in tool.description.splitlines())
    if any(param.is_json for param in tool.params):
        lines.append(f'    arguments match this JSON Schema: {json.dumps(tool.args_schema)}')
    if tool.return_is_json and tool.return_schema is not None:
        lines.append(f'    returns JSON matching: {json.dumps(tool.return_schema)}')
    return '\n'.join(lines)


def _build_description(callable_tools: tuple[_CallableTool, ...]) -> str:
    """Build the `run_sql` description: base guidance plus the rendered tool listing."""
    if not callable_tools:
        return _BASE_DESCRIPTION
    listing = '\n\n'.join(_render_tool(tool) for tool in callable_tools)
    return f'{_BASE_DESCRIPTION}\n\nTool functions available inside the query:\n\n{listing}'


@dataclass
class SQLModeToolset(WrapperToolset[AgentDepsT]):
    """Implementation toolset for the `SQLMode` capability.

    Exposes a single `run_sql` tool alongside any native (non-sandboxed) tools.
    Tools selected by `tool_selector` are registered as DuckDB functions and
    become callable from the SQL the model writes; the rest stay visible to the
    model as normal tool calls.
    """

    tool_selector: ToolSelector[AgentDepsT] = 'all'
    """Which wrapped tools to expose as DuckDB functions. Non-matching tools stay native."""

    max_rows: int = 1000
    """Maximum number of result rows returned to the model before truncation."""

    max_retries: int = 3
    """Maximum number of retries for the `run_sql` tool."""

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the `run_sql` tool plus any native (non-sandboxed) tools."""
        wrapped_tools = await self.wrapped.get_tools(ctx)

        sandboxed: dict[str, ToolsetTool[AgentDepsT]] = {}
        native: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in wrapped_tools.items():
            if tool.tool_def.tool_kind is not None:
                # Framework control tools (tool search, capability loading) drive protocol-level
                # flows and must stay native.
                native[name] = tool
            elif tool.tool_def.defer_loading:
                # Tool Search keeps these out of the model's context until discovered.
                native[name] = tool
            elif tool.tool_def.unless_native:
                # Keep the local fallback native so `Model.prepare_request` can drop it.
                native[name] = tool
            elif await matches_tool_selector(self.tool_selector, ctx, tool.tool_def):
                sandboxed[name] = tool
            else:
                native[name] = tool

        if _RUN_SQL_TOOL_NAME in native:
            raise UserError(
                f"Tool name '{_RUN_SQL_TOOL_NAME}' is reserved for SQLMode. Rename your tool to avoid conflicts."
            )

        callable_tools = _build_callable_tools(sandboxed)
        result: dict[str, ToolsetTool[AgentDepsT]] = dict(native)
        result[_RUN_SQL_TOOL_NAME] = _RunSqlTool(
            toolset=self,
            tool_def=ToolDefinition(
                name=_RUN_SQL_TOOL_NAME,
                description=_build_description(callable_tools),
                parameters_json_schema=_RUN_SQL_JSON_SCHEMA,
                metadata={'code_arg_name': 'query', 'code_arg_language': 'sql'},
                sequential=True,
            ),
            max_retries=self.max_retries,
            args_validator=_RUN_SQL_ARGS_VALIDATOR,
            callable_tools=callable_tools,
            wrapped_tools=wrapped_tools,
        )
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Run the model's SQL query, or pass through to a native (non-sandboxed) tool."""
        if not isinstance(tool, _RunSqlTool):
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        query: str = tool_args['query']
        parent_tm = ctx.tool_manager
        assert parent_tm is not None, 'SQLModeToolset requires ctx.tool_manager to be set'
        tool_manager = ToolManager[AgentDepsT](
            toolset=self.wrapped,
            root_capability=parent_tm.root_capability,
            ctx=ctx,
            tools=tool.wrapped_tools,
        )

        async def dispatch(original_name: str, kwargs: dict[str, Any]) -> Any:
            """Run one SQL-invoked tool call through the run step's tool manager."""
            call = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=uuid4().hex)
            result = await tool_manager.handle_call(call, wrap_validation_errors=False)
            if isinstance(result, ToolReturn):
                return result.return_value
            return result

        # The error is captured inside the portal scope and re-raised afterwards
        # so a `ModelRetry` reaches the caller directly, rather than wrapped in
        # the portal task group's `ExceptionGroup`.
        outcome: dict[str, Any] = {}
        error: BaseException | None = None
        async with BlockingPortal() as portal:
            try:
                outcome = await run_sync(run_query, tool.callable_tools, query, portal, self.max_rows, dispatch)
            except BaseException as exc:
                error = exc
        if error is not None:
            raise error
        return outcome
