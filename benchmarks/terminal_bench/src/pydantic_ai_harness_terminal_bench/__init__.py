"""Terminal-Bench reference agent for Pydantic AI on Harbor.

A minimal Pydantic AI agent whose weight lives in harness capabilities, wrapped
in a Harbor `BaseAgent` adapter so it can be evaluated on Terminal-Bench 2.x.

See the README for how to run it. The Harbor adapter
(`PydanticAITerminalBenchAgent`) imports `harbor`, so it is exported lazily:
importing this package does not require Harbor unless you touch the adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai_harness_terminal_bench.agent import build_agent, build_compaction
from pydantic_ai_harness_terminal_bench.config import (
    build_usage_limits,
    convert_model_name,
    parse_trial_ids,
    resolve_model_name,
)
from pydantic_ai_harness_terminal_bench.tools import (
    CommandExecutor,
    CommandResult,
    TerminalBenchDeps,
    bash,
    build_bash_toolset,
)

if TYPE_CHECKING:
    from pydantic_ai_harness_terminal_bench.harbor_agent import PydanticAITerminalBenchAgent
    from pydantic_ai_harness_terminal_bench.live import LiveBenchAgent

__all__ = [
    'CommandExecutor',
    'CommandResult',
    'LiveBenchAgent',
    'PydanticAITerminalBenchAgent',
    'TerminalBenchDeps',
    'bash',
    'build_agent',
    'build_bash_toolset',
    'build_compaction',
    'build_usage_limits',
    'convert_model_name',
    'parse_trial_ids',
    'resolve_model_name',
]

# The Harbor adapter and the live agent import `harbor`, so they are exported
# lazily: importing this package stays Harbor-free unless you touch them.
_LAZY_HARBOR_EXPORTS = {
    'PydanticAITerminalBenchAgent': 'harbor_agent',
    'LiveBenchAgent': 'live',
}


def __getattr__(name: str) -> object:
    module = _LAZY_HARBOR_EXPORTS.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(f'{__name__}.{module}'), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
