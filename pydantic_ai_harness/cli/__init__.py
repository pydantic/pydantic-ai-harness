"""Terminal client for the coding harness. See `README.md` next to this package."""

from pydantic_ai_harness.cli._bridge import CliBridge
from pydantic_ai_harness.cli._config import DEFAULT_MODEL, Config, Palette, Theme
from pydantic_ai_harness.cli._main import cli_agent, main
from pydantic_ai_harness.cli._repl import Lines, Repl

__all__ = ['DEFAULT_MODEL', 'CliBridge', 'Config', 'Lines', 'Palette', 'Repl', 'Theme', 'cli_agent', 'main']
