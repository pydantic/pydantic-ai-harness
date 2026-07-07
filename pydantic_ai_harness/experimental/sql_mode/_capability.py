"""SQLMode capability: route selected tools through a DuckDB SQL sandbox."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._tool_search import ToolSearch as _ToolSearch
from pydantic_ai.tools import AgentDepsT, ToolSelector

from pydantic_ai_harness.experimental.sql_mode._toolset import SQLModeToolset


@dataclass
class SQLMode(AbstractCapability[AgentDepsT]):
    """Capability that lets the model orchestrate tool calls by writing SQL.

    The agent's tools are registered as DuckDB functions and exposed behind a
    single `run_sql` tool -- the model writes one SQL query that calls them,
    navigates the JSON they return, and joins, filters, and aggregates the
    results, against a locked-down in-memory DuckDB.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.experimental.sql_mode import SQLMode

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SQLMode()])

    @agent.tool_plain
    def get_weather(city: str) -> dict[str, float]:
        \"\"\"Get the current weather for a city.\"\"\"
        ...
    ```
    """

    tools: ToolSelector[AgentDepsT] = field(default='all')
    """Which tools to expose as DuckDB functions inside `run_sql`.

    - `'all'` (default): every tool the agent has.
    - `Sequence[str]`: only tools whose names are listed.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: tools where it returns `True`.

    Tools that do not match stay available to the model as normal tool calls.
    """

    max_retries: int = 3
    """Maximum number of retries for the `run_sql` tool (a failed query counts as a retry)."""

    _: KW_ONLY

    max_rows: int = 1000
    """Maximum number of result rows returned to the model before truncation."""

    def get_ordering(self) -> CapabilityOrdering:
        """SQLMode wraps around ToolSearch so `search_tools` stays a native tool."""
        return CapabilityOrdering(position='outermost', wraps=[_ToolSearch])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, splitting it into native + SQL-callable subsets."""
        return SQLModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_rows=self.max_rows,
            max_retries=self.max_retries,
        )
