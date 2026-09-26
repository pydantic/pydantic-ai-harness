"""Durable execution of file-change approvals on a local workspace."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileSystem


@workflow.defn
class FileWorkflow(PydanticAIWorkflow):
    agent: Agent[None, str]

    @workflow.run
    async def run(self) -> str:
        result = await self.agent.run('try to change a file')
        return result.output


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'  # Temporal's test server requires an asyncio event loop.


@pytest.mark.parametrize('capability', [FileSystem(), Coder()], ids=['filesystem', 'coder'])
@pytest.mark.parametrize(
    'tool_name,args',
    [
        ('write_file', {'path': 'note.txt', 'content': 'changed'}),
        ('edit_file', {'path': 'note.txt', 'old_text': 'original', 'new_text': 'changed'}),
        ('create_directory', {'path': 'newdir'}),
    ],
)
@pytest.mark.anyio
async def test_temporal_default_runner_vetoes_before_mutation(
    tmp_path: Path, capability: FileSystem | Coder, tool_name: str, args: dict[str, str]
) -> None:
    if isinstance(capability, Coder) and tool_name == 'create_directory':
        pytest.skip('Coder does not expose create_directory')
    (tmp_path / 'note.txt').write_text('original')
    requests: list[str] = []

    def model(messages: object, info: object) -> ModelResponse:
        if not any(p.part_kind == 'tool-return' for m in messages for p in m.parts):  # type: ignore[attr-defined]
            return ModelResponse(parts=[ToolCallPart(tool_name, args)])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: object, info: object):
        for index, part in enumerate(model(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {index: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args))}

    agent = Agent(
        FunctionModel(model, stream_function=stream),
        name='durable_file_veto',
        capabilities=[LocalWorkspace(tmp_path), capability, TemporalDurability()],
    )

    @agent.on_event(FileChangeRequestEvent)
    async def veto(ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        requests.append(event.operation)
        event.cancel('denied')

    FileWorkflow.agent = agent
    FileWorkflow.__pydantic_ai_agents__ = [agent]
    async with await WorkflowEnvironment.start_local() as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        # The test's dynamically configured agent lives in this test module; production
        # agents are normally constructed at module scope instead of injected by the test.
        runner = SandboxedWorkflowRunner(
            restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types')
        )
        async with Worker(client, task_queue='file-veto', workflows=[FileWorkflow], workflow_runner=runner):
            assert (
                await client.execute_workflow(
                    FileWorkflow.run, id=uuid4().hex, task_queue='file-veto', execution_timeout=timedelta(seconds=20)
                )
                == 'done'
            )
    assert requests == [tool_name.removesuffix('_file') if tool_name != 'create_directory' else 'create_directory']
    assert (tmp_path / 'note.txt').read_text() == 'original'
    assert not (tmp_path / 'newdir').exists()
