"""Toolset for exposing home automation backends to agents."""

from pydantic_ai import FunctionToolset
from pydantic_ai._run_context import AgentDepsT

from pydantic_ai_harness.home_automation.backends import HomeBackend


class HomeAutomationToolset(FunctionToolset[AgentDepsT]):
    """Function toolset backed by a `HomeBackend` implementation."""

    backend: HomeBackend

    def __init__(self, backend: HomeBackend) -> None:
        self.backend = backend
        super().__init__(
            tools=[
                backend.list_services,
                backend.list_entities,
                backend.get_state,
                backend.list_states,
                backend.call_service,
            ]
        )
