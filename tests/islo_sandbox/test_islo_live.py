"""Opt-in smoke tests against a real Islo sandbox.

The fake-backed suite covers harness-owned branching. This tier checks the
provider contract the fake cannot prove: create configuration reaches a real
process, exec and file endpoints share a filesystem, and explicit deletion
completes.

Run locally:
`PYDANTIC_AI_HARNESS_ISLO_LIVE=1 ISLO_API_KEY=... uv run pytest -m islo_live tests/islo_sandbox/test_islo_live.py`
"""

from __future__ import annotations

import os
import uuid

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.islo_sandbox import IsloSandbox, IsloSandboxSession

pytestmark = [
    pytest.mark.islo_live,
    pytest.mark.skipif(
        os.getenv('PYDANTIC_AI_HARNESS_ISLO_LIVE') != '1' or os.getenv('ISLO_API_KEY') is None,
        reason='requires PYDANTIC_AI_HARNESS_ISLO_LIVE=1 and ISLO_API_KEY',
    ),
]


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


async def test_real_exec_files_and_create_configuration() -> None:
    """Validate the Islo SDK shapes and one filesystem shared by shell and file APIs."""
    probe = f'islo-live-{uuid.uuid4().hex}'
    async with IsloSandboxSession(
        sandbox_timeout=180,
        workdir='/tmp',
        env={'HARNESS_ISLO_PROBE': probe},
        poll_interval=0.25,
    ) as session:
        assert session.sandbox_name is not None
        assert session.sandbox_id is not None

        process = await session.exec(
            ['sh', '-c', 'printf %s "$HARNESS_ISLO_PROBE"; printf warning >&2; exit 3'],
            timeout=30,
        )
        assert process.stdout == probe
        assert process.stderr == 'warning'
        assert process.returncode == 3

        path = f'{probe}/roundtrip.bin'
        payload = b'from-file-api\x00'
        made_directory = await session.exec(['mkdir', '-p', probe], timeout=30)
        assert made_directory.returncode == 0
        await session.write_bytes(path, payload)
        shell_read = await session.exec(['sh', '-c', f'wc -c < {path}'], timeout=30)
        assert shell_read.stdout.strip() == str(len(payload))

        shell_path = f'{probe}/from-shell.txt'
        shell_write = await session.exec(['sh', '-c', f'printf from-shell > {shell_path}'], timeout=30)
        assert shell_write.returncode == 0
        assert await session.read_bytes(shell_path, max_bytes=1024) == b'from-shell'

        entries = await session.list_files(probe)
        assert set(entries) == {('from-shell.txt', False), ('roundtrip.bin', False)}

        sandbox_name = session.sandbox_name
        async with IsloSandboxSession(sandbox_name=sandbox_name, poll_interval=0.25) as attached:
            assert attached.sandbox_id == session.sandbox_id
            assert await attached.read_bytes(shell_path, max_bytes=1024) == b'from-shell'

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'printf agent-e2e'}, 'run-1')])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(FunctionModel(call_then_finish), capabilities=[IsloSandbox(session=session)])
        agent_result = await agent.run('run the sandbox command')
        assert agent_result.output == 'done'
        assert any(
            part.content == '[stdout]\nagent-e2e'
            for message in agent_result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        )

        after_reuse = await session.exec(['printf', 'still-running'], timeout=30)
        assert after_reuse.stdout == 'still-running'

    assert session.sandbox_name is None
    assert session.sandbox_id is None
