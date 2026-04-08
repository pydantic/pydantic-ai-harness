"""Tests for MyCapability."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from my_capability import MyCapability


class TestMyCapability:
    def test_enabled(self, test_model: TestModel) -> None:
        cap = MyCapability(enabled=True)
        agent = Agent(test_model, capabilities=[cap])
        result = agent.run_sync('hello')
        assert result.output is not None

    def test_disabled(self, test_model: TestModel) -> None:
        cap = MyCapability(enabled=False)
        agent = Agent(test_model, capabilities=[cap])
        result = agent.run_sync('hello')
        assert result.output is not None

    def test_instructions_when_enabled(self) -> None:
        cap = MyCapability(enabled=True)
        assert cap.get_instructions() == 'You have MyCapability enabled.'

    def test_instructions_when_disabled(self) -> None:
        cap = MyCapability(enabled=False)
        assert cap.get_instructions() is None

    def test_serialization_name(self) -> None:
        assert MyCapability.get_serialization_name() == 'MyCapability'

    def test_from_spec(self) -> None:
        cap = MyCapability.from_spec(enabled=False)
        assert not cap._enabled
