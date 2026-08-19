"""Previous import location for `pydantic_ai_harness.agents.coder`.

Complete agents moved under `pydantic_ai_harness.agents`; importing from here still
works but emits a `HarnessDeprecationWarning`. Import from `pydantic_ai_harness.agents.coder` instead.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness._warn import warn_moved
from pydantic_ai_harness.agents.coder import DEFAULT_ALLOWED_COMMANDS, Coder

if TYPE_CHECKING:
    from pydantic_ai_harness.agents.coder import coder_agent

warn_moved('coder', 'agents.coder')

__all__ = ['DEFAULT_ALLOWED_COMMANDS', 'Coder', 'coder_agent']


def __getattr__(name: str) -> object:
    if name == 'coder_agent':
        from pydantic_ai_harness.agents.coder import coder_agent

        return coder_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
