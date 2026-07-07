"""Tests for the `SQLMode` capability and the `SQLModeToolset` it wraps.

Style follows `tests/code_mode/test_code_mode.py`: module-level
`pytestmark = pytest.mark.anyio`, an `anyio_backend` fixture, and `build_ctx`
for invoking the toolset directly with a prepared `ToolManager`.
"""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from pydantic import BaseModel
from pydantic_ai import AbstractToolset, Agent, RunContext, Tool, ToolDefinition
from pydantic_ai.capabilities import CapabilityOrdering, ToolSearch
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturn
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai_harness.experimental.sql_mode import SQLMode, SQLModeToolset
from pydantic_ai_harness.experimental.sql_mode._toolset import (  # pyright: ignore[reportPrivateUsage]
    _duck_type,
    _sanitize,
)

pytestmark = pytest.mark.anyio

T = TypeVar('T')


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def build_run_context(deps: T) -> RunContext[T]:
    """Build a `RunContext` for invoking toolsets directly in tests."""
    return RunContext[T](deps=deps, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)


async def build_ctx(toolset: AbstractToolset[None]) -> RunContext[None]:
    """Build a `RunContext` with a prepared `ToolManager` (required by `call_tool`)."""
    ctx = build_run_context(None)
    ctx.tool_manager = await ToolManager[None](toolset=toolset).for_run_step(ctx)
    return ctx


async def run_sql(wrapper: SQLModeToolset[None], query: str) -> Any:
    """Run `query` through the wrapper's `run_sql` tool and return the result."""
    ctx = await build_ctx(wrapper)
    tools = await wrapper.get_tools(ctx)
    return await wrapper.call_tool('run_sql', {'query': query}, ctx, tools['run_sql'])


# ---------------------------------------------------------------------------
# Sample tools
# ---------------------------------------------------------------------------


class Location(BaseModel):
    """A latitude/longitude pair."""

    lat: float
    lon: float


def geocode(city: str) -> Location:
    """Look up a city's coordinates."""
    return Location(lat=48.85, lon=2.35)


async def place_label(place: Location, prefix: str) -> str:
    """Describe a location, with a prefix.

    Takes two arguments, so `place` stays a single JSON object rather than being
    flattened into its fields the way a sole model parameter would be.
    """
    return f'{prefix}{place.lat}'


def shout(text: str) -> str:
    return text.upper()


def explode(x: int) -> int:
    """A tool that always fails."""
    raise RuntimeError('tool exploded')


def retry_tool(x: int) -> int:
    """A tool that asks the model to retry."""
    raise ModelRetry('try again')


def make_toolset(*fns: Any) -> FunctionToolset[None]:
    """Wrap plain functions in a `FunctionToolset`."""
    return FunctionToolset[None](tools=[Tool(fn) for fn in fns])


_ANY_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())


class _StaticToolset(AbstractToolset[None]):
    """Minimal toolset returning hand-built `ToolDefinition`s (mirrors code_mode's stub)."""

    def __init__(self, tool_defs: list[ToolDefinition], results: dict[str, Any] | None = None) -> None:
        self._tool_defs = tool_defs
        self._results = results or {}

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover - required by AbstractToolset, never read

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        return {
            td.name: ToolsetTool(toolset=self, tool_def=td, max_retries=1, args_validator=_ANY_VALIDATOR)
            for td in self._tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        return self._results[name]


# ---------------------------------------------------------------------------
# Unit tests: name sanitizing and JSON-Schema -> DuckDB type mapping
# ---------------------------------------------------------------------------


def test_sanitize() -> None:
    """`_sanitize` produces valid SQL identifiers."""
    assert _sanitize('get_weather') == 'get_weather'
    assert _sanitize('get-weather.v2') == 'get_weather_v2'
    assert _sanitize('3d_scan') == '_3d_scan'


def test_duck_type() -> None:
    """`_duck_type` maps JSON-Schema fragments to DuckDB types."""
    assert _duck_type(None) == ('JSON', True)
    assert _duck_type({'type': 'string'}) == ('VARCHAR', False)
    assert _duck_type({'type': 'integer'}) == ('BIGINT', False)
    assert _duck_type({'type': 'number'}) == ('DOUBLE', False)
    assert _duck_type({'type': 'boolean'}) == ('BOOLEAN', False)
    assert _duck_type({'type': 'object'}) == ('JSON', True)
    assert _duck_type({}) == ('JSON', True)
    assert _duck_type({'type': ['string', 'null']}) == ('JSON', True)
    assert _duck_type({'anyOf': [{'type': 'string'}, {'type': 'null'}]}) == ('VARCHAR', False)
    assert _duck_type({'anyOf': [{'type': 'string'}, {'type': 'integer'}]}) == ('JSON', True)


# ---------------------------------------------------------------------------
# Capability wiring
# ---------------------------------------------------------------------------


def test_capability_ordering_and_wrapper() -> None:
    """`SQLMode` orders itself outermost and produces a `SQLModeToolset`."""
    capability = SQLMode[None]()
    ordering = capability.get_ordering()
    assert isinstance(ordering, CapabilityOrdering)
    assert ordering.position == 'outermost'
    assert ToolSearch in ordering.wraps
    wrapper = capability.get_wrapper_toolset(make_toolset(geocode))
    assert isinstance(wrapper, SQLModeToolset)


async def test_default_wraps_all_tools() -> None:
    """`SQLMode()` exposes only `run_sql`, rendering every tool's signature."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(geocode, place_label))
    assert isinstance(wrapper, SQLModeToolset)
    tools = await wrapper.get_tools(build_run_context(None))
    assert list(tools) == ['run_sql']
    description = tools['run_sql'].tool_def.description
    assert description is not None
    assert 'geocode(city VARCHAR) -> JSON' in description
    assert "Look up a city's coordinates." in description
    assert 'place_label(place JSON, prefix VARCHAR) -> VARCHAR' in description
    assert tools['run_sql'].tool_def.metadata == {'code_arg_name': 'query', 'code_arg_language': 'sql'}
    assert tools['run_sql'].tool_def.sequential is True


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


async def test_scalar_arg_tool() -> None:
    """A tool with a scalar argument runs from SQL."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(shout))
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(wrapper, "SELECT shout('hi') AS loud")
    assert result['rows'] == [{'loud': 'HI'}]


async def test_json_chain_and_columns() -> None:
    """A JSON result pipes into another tool; JSON columns are parsed, NULL JSON kept."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(geocode, place_label))
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(
        wrapper,
        "SELECT place_label(geocode('Paris'), 'at ') AS d, geocode('Paris')->>'lat' AS lat, NULL::JSON AS missing",
    )
    assert result['rows'] == [{'d': 'at 48.85', 'lat': '48.85', 'missing': None}]


async def test_fan_out_with_unnest() -> None:
    """A tool can be called once per row via `unnest`."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(geocode))
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(
        wrapper,
        "SELECT geocode(city)->>'lat' AS lat FROM (SELECT unnest(['Paris', 'Tokyo']) AS city)",
    )
    assert [row['lat'] for row in result['rows']] == ['48.85', '48.85']


async def test_per_row_tool_side_effects() -> None:
    """DuckDB invokes SQL-callable tools once per row."""
    calls = 0

    def counter_tool(n: int) -> int:
        nonlocal calls
        calls += 1
        return n

    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(counter_tool))
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(wrapper, 'SELECT counter_tool(n) AS value FROM (SELECT unnest([1, 2, 3]) AS n)')
    assert result['rows'] == [{'value': 1}, {'value': 2}, {'value': 3}]
    assert calls == 3


async def test_truncation() -> None:
    """Result sets larger than `max_rows` are truncated and flagged."""
    wrapper = SQLMode[None](max_rows=2).get_wrapper_toolset(make_toolset(geocode))
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(wrapper, 'SELECT unnest([1, 2, 3, 4]) AS n')
    assert result['row_count'] == 2
    assert result['truncated'] is True
    assert 'first 2 rows' in result['note']
    assert 'truncated' in result['note']


async def test_max_rows_default_returns_larger_result() -> None:
    """The default `max_rows` allows ordinary multi-row SQL results."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset())
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(wrapper, 'SELECT i AS n FROM range(12) AS t(i)')
    assert result['row_count'] == 12
    assert result['rows'][0] == {'n': 0}
    assert result['rows'][-1] == {'n': 11}
    assert 'truncated' not in result


async def test_statement_with_no_rows() -> None:
    """A statement that produces no rows returns an empty result set."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(geocode))
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(wrapper, 'CREATE TABLE t (x INTEGER)')
    assert result['rows'] == []


async def test_scalar_and_json_returns_via_static_tools() -> None:
    """A scalar `return_schema` maps to a native type; an object one stays JSON."""
    double = ToolDefinition(
        name='double',
        description='Double a number.',
        parameters_json_schema={'type': 'object', 'properties': {'n': {'type': 'integer'}}},
        return_schema={'type': 'integer'},
    )
    box = ToolDefinition(
        name='box',
        parameters_json_schema={'type': 'object', 'properties': {}},
        return_schema={'type': 'object', 'properties': {'v': {'type': 'integer'}}},
    )
    toolset = _StaticToolset([double, box], results={'double': 42, 'box': {'v': 7}})
    wrapper = SQLMode[None]().get_wrapper_toolset(toolset)
    assert isinstance(wrapper, SQLModeToolset)

    description = (await wrapper.get_tools(build_run_context(None)))['run_sql'].tool_def.description or ''
    assert 'double(n BIGINT) -> BIGINT' in description
    assert 'box() -> JSON' in description
    assert 'returns JSON matching:' in description

    result = await run_sql(wrapper, 'SELECT double(21) AS d, box() AS b')
    assert result['rows'] == [{'d': 42, 'b': {'v': 7}}]


async def test_tool_returning_tool_return() -> None:
    """A tool that returns a `ToolReturn` is unwrapped to its return value."""
    wrapped = ToolDefinition(name='wrapped', parameters_json_schema={'type': 'object', 'properties': {}})
    toolset = _StaticToolset([wrapped], results={'wrapped': ToolReturn(return_value={'ok': True})})
    wrapper = SQLMode[None]().get_wrapper_toolset(toolset)
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(wrapper, "SELECT wrapped()->>'ok' AS ok")
    assert result['rows'] == [{'ok': 'true'}]


# ---------------------------------------------------------------------------
# Tool selection and pass-through
# ---------------------------------------------------------------------------


async def test_selector_keeps_unmatched_tools_native() -> None:
    """Tools not matched by the selector stay available as normal tool calls."""
    wrapper = SQLMode[None](tools=['geocode']).get_wrapper_toolset(make_toolset(geocode, shout))
    assert isinstance(wrapper, SQLModeToolset)
    ctx = await build_ctx(wrapper)
    tools = await wrapper.get_tools(ctx)
    assert set(tools) == {'run_sql', 'shout'}
    # The native tool passes straight through to the wrapped toolset.
    assert await wrapper.call_tool('shout', {'text': 'hi'}, ctx, tools['shout']) == 'HI'


async def test_search_tools_and_deferred_stay_native() -> None:
    """Framework-control, deferred-loading, and `unless_native` tools are never sandboxed."""
    defs = [
        ToolDefinition(
            name='search_tools', parameters_json_schema={'type': 'object', 'properties': {}}, tool_kind='tool-search'
        ),
        ToolDefinition(
            name='deferred', parameters_json_schema={'type': 'object', 'properties': {}}, defer_loading=True
        ),
        ToolDefinition(
            name='fallback', parameters_json_schema={'type': 'object', 'properties': {}}, unless_native='web_search'
        ),
    ]
    wrapper = SQLMode[None]().get_wrapper_toolset(_StaticToolset(defs))
    assert isinstance(wrapper, SQLModeToolset)
    tools = await wrapper.get_tools(build_run_context(None))
    assert set(tools) == {'run_sql', 'search_tools', 'deferred', 'fallback'}


async def test_capability_load_stays_native() -> None:
    """Framework capability-loading stays native instead of becoming SQL-callable."""
    load_capability = ToolDefinition(
        name='load_capability',
        parameters_json_schema={'type': 'object', 'properties': {'id': {'type': 'string'}}},
        tool_kind='capability-load',
    )
    wrapper = SQLMode[None]().get_wrapper_toolset(_StaticToolset([load_capability]))
    assert isinstance(wrapper, SQLModeToolset)
    tools = await wrapper.get_tools(build_run_context(None))
    assert set(tools) == {'run_sql', 'load_capability'}
    assert 'load_capability(' not in (tools['run_sql'].tool_def.description or '')


async def test_reserved_run_sql_name() -> None:
    """A native tool named `run_sql` conflicts with the SQLMode meta-tool."""
    clash = ToolDefinition(name='run_sql', parameters_json_schema={'type': 'object', 'properties': {}})
    wrapper = SQLMode[None](tools=[]).get_wrapper_toolset(_StaticToolset([clash]))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(UserError, match='reserved for SQLMode'):
        await wrapper.get_tools(build_run_context(None))


async def test_sandboxed_run_sql_name_is_reserved() -> None:
    """A sandboxed tool cannot sanitize to the `run_sql` meta-tool name."""
    clash = ToolDefinition(name='run-sql', parameters_json_schema={'type': 'object', 'properties': {}})
    wrapper = SQLMode[None]().get_wrapper_toolset(_StaticToolset([clash]))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(UserError, match="SQLMode's run_sql meta-tool"):
        await wrapper.get_tools(build_run_context(None))


async def test_sanitized_name_collision_raises_user_error() -> None:
    """Two tools cannot map to the same SQL function name."""
    hyphen = ToolDefinition(name='get-weather', parameters_json_schema={'type': 'object', 'properties': {}})
    underscore = ToolDefinition(name='get_weather', parameters_json_schema={'type': 'object', 'properties': {}})
    wrapper = SQLMode[None]().get_wrapper_toolset(_StaticToolset([hyphen, underscore]))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(UserError) as exc_info:
        await wrapper.get_tools(build_run_context(None))
    message = str(exc_info.value)
    assert "'get-weather'" in message
    assert "'get_weather'" in message
    assert "SQL function name 'get_weather'" in message


async def test_no_tools() -> None:
    """`SQLMode` over an empty toolset still exposes `run_sql` for plain SQL."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset())
    assert isinstance(wrapper, SQLModeToolset)
    result = await run_sql(wrapper, 'SELECT 1 + 1 AS n')
    assert result['rows'] == [{'n': 2}]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


async def test_sql_error_raises_model_retry() -> None:
    """A malformed query surfaces as `ModelRetry`."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(geocode))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(ModelRetry, match='SQL error'):
        await run_sql(wrapper, 'SELCT nonsense')


async def test_lockdown_blocks_filesystem() -> None:
    """The locked-down database refuses to read local files."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(geocode))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(ModelRetry, match='SQL error'):
        await run_sql(wrapper, "SELECT * FROM read_csv('/etc/passwd')")


async def test_lockdown_freezes_external_access_setting() -> None:
    """The model cannot re-enable external access after lockdown."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(geocode))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(ModelRetry, match='SQL error'):
        await run_sql(wrapper, 'SET enable_external_access=true')


async def test_tool_exception_preserves_original_error() -> None:
    """An exception raised inside a tool is not laundered into `ModelRetry`."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(explode))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(RuntimeError, match='tool exploded'):
        await run_sql(wrapper, 'SELECT explode(1) AS x')


async def test_tool_model_retry_preserves_original_error() -> None:
    """A tool-raised `ModelRetry` keeps its original message."""
    wrapper = SQLMode[None]().get_wrapper_toolset(make_toolset(retry_tool))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(ModelRetry, match='try again'):
        await run_sql(wrapper, 'SELECT retry_tool(1) AS x')


async def test_builtin_name_collision_raises_user_error() -> None:
    """A tool whose name clashes with a DuckDB built-in fails with a clear error."""
    collide = ToolDefinition(
        name='add',
        parameters_json_schema={'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}},
        return_schema={'type': 'integer'},
    )
    wrapper = SQLMode[None]().get_wrapper_toolset(_StaticToolset([collide], results={'add': 1}))
    assert isinstance(wrapper, SQLModeToolset)
    with pytest.raises(UserError, match='clash with a DuckDB built-in'):
        await run_sql(wrapper, 'SELECT 1')


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


async def test_agent_runs_sql_tool() -> None:
    """An agent with the `SQLMode` capability can call `run_sql` end to end."""
    responses = iter(
        [
            ModelResponse(parts=[ToolCallPart('run_sql', {'query': "SELECT geocode('Paris')->>'lat' AS lat"})]),
            ModelResponse(parts=[TextPart('done')]),
        ]
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return next(responses)

    agent = Agent(FunctionModel(model_fn), capabilities=[SQLMode()])
    agent.tool_plain(geocode)

    result = await agent.run('where is paris')
    assert result.output == 'done'


async def test_agent_run_sql_exhausts_max_retries() -> None:
    """`SQLMode.max_retries` is wired to the `run_sql` tool retry budget."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('run_sql', {'query': 'SELCT nope'})])

    agent = Agent(FunctionModel(model_fn), capabilities=[SQLMode(max_retries=1)])

    with pytest.raises(UnexpectedModelBehavior, match='max retries'):
        await agent.run('run invalid sql')
