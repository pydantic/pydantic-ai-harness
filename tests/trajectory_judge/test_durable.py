from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import TrajectoryJudge
from pydantic_ai_harness.trajectory_judge import AllGood, TrajectoryVerdict

from .._recording_durability import RecordingDurability

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestTrajectoryJudgeDurability:
    async def test_waits_and_records_operation(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        verdicts: list[TrajectoryVerdict] = []

        async def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            started.set()
            await release.wait()
            return ModelResponse(parts=[ToolCallPart('final_result_AllGood', {})])

        durability = RecordingDurability()
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='trajectory',
            capabilities=[
                durability,
                TrajectoryJudge(id='review', model=FunctionModel(judge), every=1, on_verdict=verdicts.append),
            ],
        )
        usage = RunUsage()
        task = asyncio.create_task(agent.run('hello', usage=usage))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            assert not task.done()
            assert verdicts == []
            release.set()
            result = await asyncio.wait_for(task, timeout=5)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert result.output == 'done'
        assert verdicts == [AllGood()]
        assert usage.requests == 2
        assert [name for name, _ in durability.calls] == [
            'trajectory__model.request',
            'trajectory__capability__review.judge',
        ]

    async def test_requires_explicit_id(self) -> None:
        with pytest.raises(UserError, match='explicit `id`'):
            Agent(
                TestModel(), name='trajectory', capabilities=[RecordingDurability(), TrajectoryJudge(model=TestModel())]
            )
