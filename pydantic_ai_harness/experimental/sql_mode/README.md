# SQL Mode

Let the model orchestrate tool calls by writing **SQL** instead of issuing one tool call per round-trip.

> **Experimental.** Import from `pydantic_ai_harness.experimental.sql_mode`; importing emits `HarnessExperimentalWarning`.

## The idea

SQL is already a contained, well-defined language for expressing a wide range of logic -- filtering, joining, aggregating, transforming -- and DuckDB is a mature execution engine that [can lock itself down](https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview). SQL Mode turns that into an orchestration layer: your tools become DuckDB functions, and the model writes one SQL query that calls them, navigates the JSON they return, pipes values between them, and shapes the result -- in a single round-trip.

It is the SQL counterpart to [Code Mode](../../code_mode/README.md): where Code Mode sandboxes generated *Python*, SQL Mode runs generated *SQL* against a sandboxed, in-memory DuckDB.

## Usage

`SQLMode` is a [capability](https://ai.pydantic.dev/capabilities/) -- add it to the agent and it wraps the agent's tools automatically. No manual registration.

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.sql_mode import SQLMode


class Coordinates(BaseModel):
    lat: float
    lon: float


agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SQLMode()])


@agent.tool_plain
def geocode(city: str) -> Coordinates:
    """Look up a city's coordinates."""
    ...


@agent.tool_plain
async def get_forecast(lat: float, lon: float) -> dict[str, float]:
    """Get today's forecast for a latitude/longitude."""
    ...


result = agent.run_sync('Compare the temperature in Paris and Tokyo.')
```

The agent's tools (`geocode`, `get_forecast`) are removed from the model's tool list and exposed as DuckDB functions inside a single `run_sql` tool. The model writes one query that fans them over a list and chains them:

```sql
SELECT city,
       get_forecast((geocode(city)->>'lat')::DOUBLE, (geocode(city)->>'lon')::DOUBLE)->>'temp_c' AS temp
FROM (SELECT unnest(['Paris', 'Tokyo']) AS city)
ORDER BY temp DESC;
```

## How it works

Each `run_sql` call:

1. Opens a fresh, in-memory DuckDB database.
2. Registers every selected tool as a DuckDB user-defined function.
3. Locks the database down (see below).
4. Runs the model's query off the event loop, in a worker thread, dispatching tool calls back through the agent's tool manager -- so tools keep their `RunContext`, dependencies, validation, and capability hooks.
5. Returns the result set as `columns` (name + type) and `rows`.

Nothing persists between calls -- every `run_sql` gets a brand-new database.

### Types and JSON

Tool signatures come from each tool's `ToolDefinition`. Scalar parameters and return values (`str`, `int`, `float`, `bool`) map to native DuckDB column types; everything else -- pydantic models, `dict`s, `list`s -- is carried as DuckDB's `JSON` type. Each tool's pydantic JSON Schema is rendered into the `run_sql` description, so the model knows the shape of what it is navigating. Read JSON fields with DuckDB's [JSON functions](https://duckdb.org/docs/stable/data/json/json_functions) -- `->`, `->>`, `json_extract`, and friends.

### Selecting which tools to expose

By default `SQLMode()` exposes every tool. Pass `tools=` to expose only some -- the rest stay available as normal tool calls:

```python
SQLMode(tools=['geocode', 'get_forecast'])     # by name
SQLMode(tools=lambda ctx, td: td.name != 'send_email')  # by predicate
```

## Security

Before the model's SQL runs, the connection applies DuckDB's [hardening settings](https://duckdb.org/docs/stable/operations_manual/securing_duckdb/overview):

```sql
SET autoload_known_extensions = false;
SET autoinstall_known_extensions = false;
SET allow_community_extensions = false;
SET enable_external_access = false;
SET lock_configuration = true;          -- last: freezes the settings above
```

This blocks all filesystem and network access (`read_csv`, `ATTACH`, `COPY ... TO`, `httpfs`, ...) and extension loading, and prevents the model's SQL from turning any of it back on. The model can only run pure SQL and call the tools you registered.

## Installation

```bash
uv add "pydantic-ai-harness[sql-mode]"
```

The `sqlmode` extra is also supported as an alias.

This pulls in `duckdb`, `numpy` (DuckDB needs it to register Python functions), and `anyio`. Importing the feature without these raises `ImportError`.

## API

```python
SQLMode(
    tools: ToolSelector = 'all',  # which tools to expose as DuckDB functions
    max_retries: int = 3,         # retries for run_sql on a query/tool error
    max_rows: int = 1000,         # result rows returned before truncation
)
```

## Limitations

- **One row at a time.** A tool used in `SELECT my_tool(x) FROM big_table` is called once per row, sequentially. Shape data with `unnest`/`array_agg` deliberately; this is orchestration, not bulk compute.
- **No state between calls.** Each `run_sql` gets a fresh database.
- **Tool names** that are SQL keywords (`describe`, `select`, ...) or that collide with a DuckDB built-in's exact signature can't be registered or called. Rename the tool.
- **Optional parameters** become required in the SQL call -- DuckDB functions are fixed-arity.
- **A sole pydantic-model parameter** is flattened into its fields by Pydantic AI's tool schema; give a tool a second parameter to keep a model argument as a single `JSON` value.
