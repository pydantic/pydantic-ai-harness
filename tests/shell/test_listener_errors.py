"""Host listener failures must not become model retries."""

from __future__ import annotations

import errno
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import CapabilityEvent, ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.shell import (
    Shell,
    ShellCommandEndEvent,
    ShellCommandRequestEvent,
    ShellCommandStartEvent,
    ShellOutputLineEvent,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass(kw_only=True)
class RaisingListener(AbstractCapability[None]):
    event_type: type[CapabilityEvent]
    error: OSError

    @on_event(ShellCommandRequestEvent, ShellCommandStartEvent, ShellOutputLineEvent, ShellCommandEndEvent)
    async def on_shell_event(self, ctx: RunContext[None], event: CapabilityEvent) -> None:
        if isinstance(event, self.event_type):
            raise self.error


def calls_model(tool_names: Sequence[str]) -> FunctionModel:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        results = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        assert len(results) < len(tool_names), 'The listener failure must abort before another model request'
        name = tool_names[len(results)]
        if name in ('run_command', 'start_command'):
            args = {'command': 'echo hello'}
        else:
            command_id = str(results[0].content).split('ID: ')[1].strip()
            args = {'command_id': command_id}
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=f'call_{len(results)}')}

    return FunctionModel(stream_function=respond)


@pytest.mark.parametrize('code', [errno.EACCES, errno.ENOENT, errno.ENOTDIR])
@pytest.mark.parametrize(
    ('event_type', 'tool_names'),
    [
        (ShellCommandRequestEvent, ['run_command']),
        (ShellCommandRequestEvent, ['start_command']),
        (ShellCommandStartEvent, ['run_command']),
        (ShellCommandStartEvent, ['start_command']),
        (ShellOutputLineEvent, ['run_command']),
        (ShellCommandEndEvent, ['run_command']),
        (ShellCommandEndEvent, ['start_command', 'stop_command']),
    ],
)
async def test_listener_oserror_propagates(
    tmp_path: Path, code: int, event_type: type[CapabilityEvent], tool_names: list[str]
) -> None:
    error = OSError(code, 'host listener failed')
    agent = Agent(
        calls_model(tool_names),
        deps_type=type(None),
        capabilities=[Shell(cwd=tmp_path), RaisingListener(event_type=event_type, error=error)],
    )

    with pytest.raises(OSError) as exc_info:
        await agent.run('go')

    assert exc_info.value is error
