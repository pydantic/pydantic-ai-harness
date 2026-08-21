"""Tests for the upstream `pydantic-ai-absurd` integration."""

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

pytest.importorskip('pydantic_ai_absurd')

from pydantic_ai_absurd import AbsurdDurability as UpstreamAbsurdDurability
from pydantic_ai_absurd import AbsurdParallelExecutionMode as UpstreamAbsurdParallelExecutionMode

import pydantic_ai_harness
from pydantic_ai_harness.absurd import AbsurdDurability, AbsurdParallelExecutionMode

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestAbsurdIntegration:
    def test_reexports_upstream_objects(self) -> None:
        assert AbsurdDurability is UpstreamAbsurdDurability
        assert AbsurdParallelExecutionMode is UpstreamAbsurdParallelExecutionMode

    def test_root_lazy_export_is_the_upstream_object(self) -> None:
        assert pydantic_ai_harness.AbsurdDurability is UpstreamAbsurdDurability

    async def test_composes_with_named_test_model_agent(self) -> None:
        agent = Agent(TestModel(), name='harness-test', capabilities=[AbsurdDurability()])

        result = await agent.run('hello')

        assert agent.name == 'harness-test'
        assert result.output == 'success (no tool calls)'
