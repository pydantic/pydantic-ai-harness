"""SQLMode: let the model orchestrate tool calls by writing SQL against locked-down DuckDB."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.sql_mode._capability import SQLMode
from pydantic_ai_harness.experimental.sql_mode._toolset import SQLModeToolset

warn_experimental('sql_mode')

__all__ = ['SQLMode', 'SQLModeToolset']
