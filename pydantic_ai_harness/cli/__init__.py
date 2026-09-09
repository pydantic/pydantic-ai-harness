"""Terminal client for the coding harness. See `README.md` next to this package."""

from pydantic_ai_harness.cli._approve import (
    Answers,
    Approver,
    CliDeps,
    DeclineAll,
    TerminalApprover,
    Verdict,
    allow_all,
)
from pydantic_ai_harness.cli._bridge import CliBridge
from pydantic_ai_harness.cli._config import DEFAULT_MODEL, Config, Palette, Theme
from pydantic_ai_harness.cli._main import cli_agent, main
from pydantic_ai_harness.cli._repl import Lines, Repl

__all__ = [
    'DEFAULT_MODEL',
    'Answers',
    'Approver',
    'CliBridge',
    'CliDeps',
    'Config',
    'DeclineAll',
    'Lines',
    'Palette',
    'Repl',
    'TerminalApprover',
    'Theme',
    'Verdict',
    'allow_all',
    'cli_agent',
    'main',
]
