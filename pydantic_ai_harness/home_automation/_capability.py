from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_harness.home_automation._toolset import HomeAutomationToolset
from pydantic_ai_harness.home_automation.backends import HomeBackend


@dataclass
class HomeAutomation(AbstractCapability[AgentDepsT]):
    """Capability exposing Home Assistant-style service discovery."""

    backend: HomeBackend

    def get_instructions(self) -> str:
        """Return guidance for agents using the home automation capability."""
        return (
            'You have access to tools that let you inspect and control smart home '
            'entities such as lights, switches, and climate devices through Home '
            'Assistant. Use `list_entities` or `list_states` to discover available '
            'entities, and `get_state` when you need the current state of one entity. '
            'Use `list_services` to discover valid services and their arguments before '
            'calling them. When using `call_service`, pass the exact `domain`, '
            '`service_name`, and `entity_id` returned by Home Assistant. After a '
            'service call, prefer `verified_state` as the strongest confirmation of '
            'what happened, then `changed_states`, and finally `service_response`. '
            '`call_service` may perform follow-up state reads when Home Assistant does '
            'not return changed states or response data.'
        )

    def get_toolset(self) -> HomeAutomationToolset[AgentDepsT]:
        """Expose the home automation tools to the agent runtime."""
        return HomeAutomationToolset(self.backend)
