"""Complete agents built from harness capabilities.

Each submodule here bundles capabilities into a ready-to-use agent. The capability
classes stay importable on their own, so an agent you start from comes apart into the
same blocks you would compose by hand.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness.agents.coder import DEFAULT_ALLOWED_COMMANDS, Coder
from pydantic_ai_harness.agents.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher

if TYPE_CHECKING:
    from pydantic_ai_harness.agents.coder import coder_agent
    from pydantic_ai_harness.agents.researcher import researcher_agent

__all__ = [
    'DEFAULT_ALLOWED_COMMANDS',
    'DEFAULT_RESEARCHER_INSTRUCTIONS',
    'Coder',
    'Researcher',
    'coder_agent',
    'researcher_agent',
]

_AGENT_EXPORTS = {
    'coder_agent': 'coder',
    'researcher_agent': 'researcher',
}


def __getattr__(name: str) -> object:
    module_name = _AGENT_EXPORTS.get(name)
    if module_name is not None:
        from importlib import import_module

        return getattr(import_module(f'.{module_name}', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
