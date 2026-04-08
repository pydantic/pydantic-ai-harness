"""Shared test fixtures."""

from __future__ import annotations

import pydantic_ai.models
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Prevent accidental real API calls
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def test_model() -> TestModel:
    return TestModel()


@pytest.fixture
def test_agent(test_model: TestModel) -> Agent[None, str]:
    return Agent(test_model)
