"""Foreground deadlines include immediate start listeners."""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.shell import Shell, ShellCommandStartEvent


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass(kw_only=True)
class BlockingStart(AbstractCapability[None]):
    pid: int | None = None
    cancelled: bool = False

    @on_event(ShellCommandStartEvent)
    async def on_start(self, ctx: RunContext[None], event: ShellCommandStartEvent) -> None:
        self.pid = event.pid
        try:
            await anyio.sleep_forever()
        finally:
            self.cancelled = True


@pytest.mark.anyio
async def test_start_listener_is_cancelled_at_command_deadline(tmp_path: Path) -> None:
    listener = BlockingStart()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='run_command', json_args='{"command":"sleep 30"}')}
        else:
            yield 'done'

    agent = Agent(
        FunctionModel(stream_function=model),
        deps_type=type(None),
        capabilities=[Shell(cwd=tmp_path, denied_commands=[], default_timeout=0.1), listener],
    )
    with anyio.fail_after(10):
        result = await agent.run('go')

    assert listener.cancelled
    assert listener.pid is not None
    assert [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ] == ['[Command timed out after 0.1s]']
    with pytest.raises(ProcessLookupError):
        os.kill(listener.pid, 0)
