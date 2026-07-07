"""DuckDB-backed execution for SQLMode.

Each `run_sql` call gets a fresh, locked-down, in-memory DuckDB connection: the
selected tools are registered as DuckDB user-defined functions, the query runs,
and the connection is thrown away -- nothing persists between calls.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from collections.abc import Awaitable, Callable
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_core import to_jsonable_python

try:
    import duckdb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'duckdb is required for SQLMode. Install it with: pip install "pydantic-ai-harness[sql-mode]"'
    ) from e

if importlib.util.find_spec('numpy') is None:  # pragma: no cover
    raise ImportError(
        'numpy is required for SQLMode -- DuckDB needs it to register Python functions. '
        'Install it with: pip install "pydantic-ai-harness[sql-mode]"'
    )

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

    from pydantic_ai_harness.experimental.sql_mode._toolset import _CallableTool  # pyright: ignore[reportPrivateUsage]

# A coroutine that runs a tool call: `(original_tool_name, kwargs) -> result`.
DispatchFn = Callable[[str, dict[str, Any]], Awaitable[Any]]

# Applied in order on every connection. `lock_configuration` MUST come last: it
# freezes every preceding setting so the model's SQL cannot turn them back on.
# See https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview
_LOCKDOWN_STATEMENTS: tuple[str, ...] = (
    'SET autoload_known_extensions = false',
    'SET autoinstall_known_extensions = false',
    'SET allow_community_extensions = false',
    'SET enable_external_access = false',
    'SET lock_configuration = true',
)


def _build_udf(
    tool: _CallableTool,
    portal: BlockingPortal,
    dispatch: DispatchFn,
    original_errors: list[BaseException],
) -> Callable[..., Any]:
    """Build the synchronous callable DuckDB invokes for `tool`.

    DuckDB calls this once per row, from a worker thread. JSON arguments are
    parsed; the tool is dispatched on the event loop through `portal`; the return
    value is serialized back to JSON when the tool's return type is non-scalar.
    """

    def udf(*raw_args: Any) -> Any:
        """Parse arguments, dispatch the tool call, and serialize its result for DuckDB."""
        kwargs = {param.name: (json.loads(raw) if param.is_json else raw) for param, raw in zip(tool.params, raw_args)}
        try:
            result = portal.call(functools.partial(dispatch, tool.original_name, kwargs))
        except BaseException as exc:
            original_errors.append(exc)
            raise
        if tool.return_is_json:
            return json.dumps(to_jsonable_python(result, serialize_unknown=True))
        return result

    # DuckDB matches a UDF's arity via `inspect.signature`, so give the variadic
    # wrapper a fixed-arity signature matching the registered parameter types.
    udf.__signature__ = Signature(  # pyright: ignore[reportFunctionMemberAccess]
        [Parameter(param.name, Parameter.POSITIONAL_OR_KEYWORD) for param in tool.params]
    )
    return udf


def _format_result(cursor: duckdb.DuckDBPyConnection, max_rows: int) -> dict[str, Any]:
    """Shape a finished DuckDB result into a JSON-friendly dict of columns and rows."""
    columns = [{'name': str(col[0]), 'type': str(col[1])} for col in cursor.description]
    json_columns = {index for index, col in enumerate(columns) if col['type'].upper() == 'JSON'}

    visible_plus_one = cursor.fetchmany(max_rows + 1)
    truncated = len(visible_plus_one) > max_rows
    visible_rows = visible_plus_one[:max_rows]

    records: list[dict[str, Any]] = []
    for row in visible_rows:
        record: dict[str, Any] = {}
        for index, col in enumerate(columns):
            value = row[index]
            if index in json_columns and isinstance(value, str):
                value = json.loads(value)
            record[col['name']] = value
        records.append(record)

    result: dict[str, Any] = {
        'columns': columns,
        'rows': to_jsonable_python(records, serialize_unknown=True),
        'row_count': len(records),
    }
    if truncated:
        result['truncated'] = True
        result['note'] = f'Showing the first {max_rows} rows; the result set was larger and has been truncated.'
    return result


def run_query(
    callable_tools: tuple[_CallableTool, ...],
    query: str,
    portal: BlockingPortal,
    max_rows: int,
    dispatch: DispatchFn,
) -> dict[str, Any]:
    """Run `query` against a fresh, locked-down, in-memory DuckDB database.

    Called in a worker thread. The connection is created, has `callable_tools`
    registered as user-defined functions, is locked down, runs the query, and is
    closed. Raises `ModelRetry` on a SQL error so the model can try again.
    """
    con = duckdb.connect(':memory:')
    original_errors: list[BaseException] = []
    try:
        con.execute('LOAD json')
        for tool in callable_tools:
            try:
                con.create_function(  # pyright: ignore[reportUnknownMemberType]
                    tool.sql_name,
                    _build_udf(tool, portal, dispatch, original_errors),
                    [duckdb.dtype(param.duck_type) for param in tool.params],
                    duckdb.dtype(tool.return_duck_type),
                    side_effects=True,
                )
            except duckdb.Error as e:
                raise UserError(
                    f'SQLMode could not register tool {tool.original_name!r} as the DuckDB function '
                    f'{tool.sql_name!r}: {e}. Rename the tool so it does not clash with a DuckDB built-in.'
                ) from e
        for statement in _LOCKDOWN_STATEMENTS:
            con.execute(statement)
        try:
            cursor = con.execute(query)
            return _format_result(cursor, max_rows)
        except duckdb.Error as e:
            if original_errors:
                raise original_errors[0]
            raise ModelRetry(f'SQL error: {e}') from e
    finally:
        con.close()
