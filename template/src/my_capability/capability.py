"""Example capability implementation.

Replace this with your own capability. See the hooks reference for all
available lifecycle methods: https://ai.pydantic.dev/hooks/
"""

from __future__ import annotations

from pydantic_ai.capabilities.abstract import AbstractCapability


class MyCapability(AbstractCapability[object]):
    """A capability that does X.

    Usage:
        ```python
        from pydantic_ai import Agent
        from my_capability import MyCapability

        agent = Agent('openai:gpt-4o', capabilities=[MyCapability()])
        ```
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    # -- Provide instructions to the agent --

    def get_instructions(self) -> str | None:
        if not self._enabled:
            return None
        return 'You have MyCapability enabled.'

    # -- Hook into model requests --
    # Uncomment and implement the hooks you need.
    # See AbstractCapability for all available hooks.

    # def before_model_request(
    #     self,
    #     ctx: RunContext[object],
    #     request_context: ModelRequestContext,
    # ) -> None:
    #     """Called before each model request. Modify messages or settings."""

    # def after_model_request(
    #     self,
    #     ctx: RunContext[object],
    #     request_context: ModelRequestContext,
    #     response: ModelResponse,
    # ) -> ModelResponse | None:
    #     """Called after each model request. Modify or replace the response."""

    # -- Spec serialization (for YAML/JSON agent specs) --

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'MyCapability'

    @classmethod
    def from_spec(cls, *, enabled: bool = True) -> MyCapability:
        return cls(enabled=enabled)
