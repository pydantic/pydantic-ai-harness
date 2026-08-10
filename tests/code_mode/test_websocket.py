"""End-to-end tests for CodeMode's remote Monty transport."""

from __future__ import annotations

import asyncio
import shutil
import socket
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_monty import MountDir

from pydantic_ai_harness import CodeMode
from tests.code_mode import websocket_relay  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio

_RELAY_SCRIPT = Path(websocket_relay.__file__)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
async def websocket_relay_url() -> AsyncIterator[str]:
    """Start the protocol relay on an ephemeral loopback port."""
    monty_bin = shutil.which('monty')
    if monty_bin is None:  # pragma: no cover
        pytest.fail('The `monty` binary is required for the WebSocket transport test.')

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_RELAY_SCRIPT),
        '--monty-bin',
        monty_bin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        url_line = await asyncio.wait_for(process.stdout.readline(), timeout=5)
        if not url_line:  # pragma: no cover
            stderr = (await process.stderr.read()).decode()
            pytest.fail(f'WebSocket relay exited before startup: {stderr}')
        yield url_line.decode().strip()
    finally:
        if process.returncode is None:  # pragma: no branch
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:  # pragma: no cover
            process.kill()
            await process.wait()


async def test_code_mode_runs_over_websocket(
    websocket_relay_url: str,
    tmp_path: Path,
) -> None:
    """Remote feeds retain state while host tools, prints, and mounts round-trip."""
    (tmp_path / 'input.txt').write_text('mounted data')
    observed_returns: list[Any] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
        ]
        observed_returns[:] = [part.content for part in returns]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'run_code',
                        {'code': 'value = await add(a=2, b=3)\nprint("remote tool result", value)'},
                    )
                ]
            )
        if len(returns) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'run_code',
                        {
                            'code': (
                                'from pathlib import Path\n'
                                'text = Path("/workspace/input.txt").read_text()\n'
                                'print(text)\n'
                                'value * 10'
                            )
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[None, str] = Agent(
        FunctionModel(model),
        capabilities=[
            CodeMode(
                monty_sandbox_url=websocket_relay_url,
                mount=MountDir(virtual_path='/workspace', host_path=tmp_path),
            )
        ],
    )

    @agent.tool_plain
    async def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
        return a + b

    result = await agent.run('exercise the remote sandbox')

    assert result.output == 'done'
    assert observed_returns == [
        {'output': 'remote tool result 5\n'},
        {'output': 'mounted data\n', 'result': 50},
    ]


def _text_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('done')])


@pytest.mark.parametrize(
    'url',
    ['ws://sandbox.example.com:8799/monty', 'ws://192.0.2.1:8799', 'ws:///monty'],
)
async def test_plaintext_remote_sandbox_url_rejected(url: str) -> None:
    """`ws://` to a non-loopback host fails when the toolset enters, before any connection is dialed."""
    agent: Agent[None, str] = Agent(
        FunctionModel(_text_model),
        capabilities=[CodeMode(monty_sandbox_url=url)],
    )
    with pytest.raises(UserError, match='wss://'):
        await agent.run('never dials')


@pytest.mark.parametrize(
    'url',
    ['ws://127.0.0.1:8799', 'ws://localhost:8799', 'ws://[::1]:8799', 'wss://sandbox.example.com/monty'],
)
async def test_loopback_or_tls_sandbox_url_accepted(url: str) -> None:
    """Loopback `ws://` and any `wss://` URL pass validation; workers dial lazily, not at enter."""
    agent: Agent[None, str] = Agent(FunctionModel(_text_model), capabilities=[CodeMode(monty_sandbox_url=url)])
    result = await agent.run('no run_code call, so nothing dials')
    assert result.output == 'done'


async def test_dial_failure_redacts_sandbox_url() -> None:
    """A failed dial's retry message must not leak the URL, which may carry credentials."""
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        free_port = probe.getsockname()[1]
    secret_url = f'ws://127.0.0.1:{free_port}/?token=hunter2'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        retries = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        ]
        if not retries:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': '1 + 1'})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[None, str] = Agent(FunctionModel(model), capabilities=[CodeMode(monty_sandbox_url=secret_url)])
    result = await agent.run('dial a dead worker')

    assert result.output == 'done'
    retry_parts = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ]
    assert retry_parts, 'expected the dial failure to surface as a retry'
    content = str(retry_parts[0].content)
    assert 'token=hunter2' not in content
    assert secret_url not in content
    assert '<monty_sandbox_url>' in content
