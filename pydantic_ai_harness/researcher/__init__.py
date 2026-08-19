"""Previous import location for `pydantic_ai_harness.agents.researcher`.

Complete agents moved under `pydantic_ai_harness.agents`; importing from here still
works but emits a `HarnessDeprecationWarning`. Import from `pydantic_ai_harness.agents.researcher` instead.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness._warn import warn_moved
from pydantic_ai_harness.agents.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS, Researcher

if TYPE_CHECKING:
    from pydantic_ai_harness.agents.researcher import researcher_agent

warn_moved('researcher', 'agents.researcher')

__all__ = ['DEFAULT_RESEARCHER_INSTRUCTIONS', 'Researcher', 'researcher_agent']


def __getattr__(name: str) -> object:
    if name == 'researcher_agent':
        from pydantic_ai_harness.agents.researcher import researcher_agent

        return researcher_agent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
