"""Exercise trajectory judging through DBOS's persisted step results."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai_harness import TrajectoryJudge

_calls: list[str] = []


def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _calls.append('judge')
    return ModelResponse(parts=[ToolCallPart('final_result_AllGood', {})])


agent = Agent(
    TestModel(custom_output_text='done'),
    name='trajectory_dbos',
    capabilities=[DBOSDurability(), TrajectoryJudge(id='review', model=FunctionModel(judge), every=1)],
)


@DBOS.workflow()
def trajectory_workflow() -> int:
    return agent.run_sync('hello').usage.requests


@pytest.fixture
def dbos_instance(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    config: DBOSConfig = {
        'name': 'trajectory_judge_tests',
        'system_database_url': f'sqlite:///{tmp_path_factory.mktemp("trajectory-dbos") / "db.sqlite"}',
        'run_admin_server': False,
        'enable_otlp': False,
    }
    DBOS(config=config)
    DBOS.launch()
    try:
        yield
    finally:
        DBOS.destroy()


class TestTrajectoryJudgeDBOS:
    def test_reuses_judgement(self, dbos_instance: None) -> None:
        _calls.clear()
        workflow_id = str(uuid.uuid4())
        with SetWorkflowID(workflow_id):
            assert trajectory_workflow() == 2
        with SetWorkflowID(workflow_id):
            assert trajectory_workflow() == 2
        assert _calls == ['judge']
