"""Subprocess runner for the DBOS background-step recovery test."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dbos import DBOS, DBOSConfig, SetWorkflowID, WorkflowHandleAsync
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.background_tools import BackgroundTools

database = Path(sys.argv[1])
markers = Path(sys.argv[2])
workflow_id = sys.argv[3]
mode = sys.argv[4]
step_finished = markers / 'step-finished'
release_delivery = markers / 'release-delivery'
step_calls = markers / 'step-calls'
result_file = markers / 'result'


def _has_ack(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, ToolReturnPart) and 'running in background' in str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def _completion_count(messages: list[ModelMessage]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and "Background tool 'durable_research'" in part.content
    )


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _completion_count(messages):
        return ModelResponse(parts=[TextPart(content='done')])
    if _has_ack(messages):
        return ModelResponse(parts=[TextPart(content='waiting')])
    return ModelResponse(parts=[ToolCallPart(tool_name='durable_research', args={}, tool_call_id='research-call')])


agent = Agent(
    FunctionModel(_model),
    name='background_tools_dbos_agent',
    capabilities=[BackgroundTools(), DBOSDurability()],
)


@DBOS.step(name='background_tools_dbos_agent__research_step')
async def research_step() -> str:
    """Record and return the externally durable work."""
    with step_calls.open('a') as file:
        file.write(f'{os.getpid()}\n')
    return 'durable result'


@agent.tool_plain(metadata={'background': True})
async def durable_research() -> str:
    """Wait after the durable step so the first process can be killed before delivery."""
    result = await research_step()
    step_finished.touch()
    while not release_delivery.exists():
        await asyncio.sleep(0.02)
    return result


@DBOS.workflow()
async def run_agent(prompt: str) -> tuple[str, int]:
    """Run the agent as a DBOS workflow."""
    result = await agent.run(prompt)
    return str(result.output), _completion_count(result.all_messages())


async def main() -> None:
    """Start or recover the workflow selected on the command line."""
    config: DBOSConfig = {
        'name': 'background_tools_dbos_recovery',
        'system_database_url': f'sqlite:///{database}',
        'run_admin_server': False,
        'enable_otlp': False,
    }
    DBOS(config=config)
    DBOS.launch()
    try:
        if mode == 'first':
            with SetWorkflowID(workflow_id):
                output, completions = await run_agent('go')
        else:
            handle: WorkflowHandleAsync[tuple[str, int]] = await DBOS.retrieve_workflow_async(workflow_id)
            output, completions = await handle.get_result(polling_interval_sec=0.05)
        result_file.write_text(f'{output}\n{completions}\n')
    finally:
        DBOS.destroy()


asyncio.run(main())
