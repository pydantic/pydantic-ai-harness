"""Core-only reproducer: an outer AnyIO cancel must close a running MCP subprocess."""

import os
import sys
from pathlib import Path

import anyio
import pytest
from anyio.abc import SocketAttribute, SocketStream
from anyio.streams.buffered import BufferedByteReceiveStream
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel

READY_TIMEOUT = 30


class MCPProcessLeak(AssertionError):
    """Core returned from a cancelled run without closing its stdio subprocess."""


@pytest.mark.xfail(
    strict=True,
    raises=MCPProcessLeak,
    reason='Core leaves stdio alive under outer AnyIO cancellation: https://github.com/pydantic/pydantic-ai/issues/8548',
)
async def test_outer_anyio_cancel_closes_stdio(tmp_path: Path) -> None:
    ready = anyio.Event()
    pid_path = tmp_path / 'server.pid'

    async def receive(stream: SocketStream) -> None:
        async with stream:
            expected = pid_path.read_bytes()
            assert await BufferedByteReceiveStream(stream).receive_exactly(len(expected)) == expected
            ready.set()

    with anyio.fail_after(READY_TIMEOUT):
        async with await anyio.create_tcp_listener(local_host='127.0.0.1') as listener:
            port = listener.extra(SocketAttribute.local_address)[1]
            toolset = MCPToolset(
                StdioTransport(
                    command=sys.executable,
                    args=[str(Path(__file__).with_name('mcp_server.py'))],
                    env={'CLAI_MCP_PID_FILE': str(pid_path), 'CLAI_MCP_READY_PORT': str(port)},
                    keep_alive=False,
                )
            )
            agent = Agent(TestModel(call_tools=['wait']), toolsets=[toolset])

            async def run() -> None:
                await agent.run('Wait')

            try:
                async with anyio.create_task_group() as group:
                    group.start_soon(listener.serve, receive)
                    group.start_soon(run)
                    await ready.wait()
                    group.cancel_scope.cancel()
                try:
                    os.kill(int(pid_path.read_text()), 0)
                except ProcessLookupError:
                    pass
                else:
                    raise MCPProcessLeak('Stdio process still alive after outer AnyIO cancellation')
            finally:
                # Explicit client cleanup prevents a leak when core skips teardown.
                with anyio.fail_after(READY_TIMEOUT, shield=True):
                    await toolset.client.close()
                with pytest.raises(ProcessLookupError):
                    os.kill(int(pid_path.read_text()), 0)
