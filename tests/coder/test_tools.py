import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.coder import Coder

pytestmark = pytest.mark.anyio


async def call(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
    *,
    capabilities: Sequence[AbstractCapability[None]] = (),
) -> str:
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart(name, arguments)])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for part in respond(messages, info).parts:
            if isinstance(part, ToolCallPart):
                yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}
            elif isinstance(part, TextPart):
                yield part.content

    result = await Agent(
        FunctionModel(respond, stream_function=stream),
        deps_type=type(None),
        capabilities=[Coder(tmp_path), *capabilities],
    ).run('Use the tool')
    return '\n'.join(
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    )


class TestCoder:
    async def test_schema(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(tmp_path)]).run('Inspect tools')
        assert model.last_model_request_parameters is not None
        tools = {tool.name: tool for tool in model.last_model_request_parameters.function_tools}
        assert set(tools) == {'read_file', 'write_file', 'edit_file', 'list_files', 'grep', 'shell'}
        assert 'expected_hash' not in str(tools)

    async def test_read_write(self, tmp_path: Path) -> None:
        assert 'hash:' not in await call(tmp_path, 'write_file', {'path': 'test.txt', 'content': 'one\ntwo\n'})
        output = await call(tmp_path, 'read_file', {'path': 'test.txt', 'offset': 1, 'limit': 1})
        assert 'two' in output and 'one' not in output and 'hash:' not in output
        assert '2' in output

    async def test_batch(self, tmp_path: Path) -> None:
        path = tmp_path / 'test.txt'
        path.write_bytes(b'one\r\ntwo\r\n')
        output = await call(
            tmp_path,
            'edit_file',
            {
                'path': 'test.txt',
                'replacements': [
                    {'old_text': 'one', 'new_text': 'three'},
                    {'old_text': 'three', 'new_text': 'four'},
                ],
            },
        )
        assert 'Edited' in output and 'hash:' not in output
        assert path.read_bytes() == b'four\r\ntwo\r\n'

    @pytest.mark.parametrize(
        'arguments',
        [
            {'replacements': []},
            {},
            {'old_text': 'one'},
            {'old_text': 'one', 'new_text': 'two', 'replacements': [{'old_text': 'one', 'new_text': 'two'}]},
            {'replacements': [{'old_text': 'one', 'new_text': 'two'}, {'old_text': 'missing', 'new_text': 'three'}]},
            {'old_text': '', 'new_text': 'two'},
            {'old_text': 'x', 'new_text': 'two'},
        ],
    )
    async def test_invalid_edit_is_atomic(self, tmp_path: Path, arguments: dict[str, object]) -> None:
        path = tmp_path / 'test.txt'
        path.write_text('one x x')
        await call(tmp_path, 'edit_file', {'path': 'test.txt', **arguments})
        assert path.read_text() == 'one x x'

    async def test_search(self, tmp_path: Path) -> None:
        (tmp_path / 'test.py').write_text('One\ntwo\nthree\n')
        (tmp_path / 'other.txt').write_text('One\n')
        assert await call(tmp_path, 'list_files', {'glob': '*.py'}) == 'test.py'
        output = await call(tmp_path, 'grep', {'pattern': 'one', 'ignore_case': True, 'file_type': 'py', 'context': 1})
        assert 'One' in output and 'two' in output and 'other.txt' not in output
        assert 'truncated' in await call(tmp_path, 'grep', {'pattern': '.', 'limit': 1})
        assert await call(tmp_path, 'grep', {'pattern': '-not-found', 'literal': True, 'glob': '*.py'}) == ''

    @pytest.mark.parametrize(
        'name,arguments',
        [
            ('grep', {'pattern': '(', 'context': 0}),
            ('grep', {'pattern': '.', 'context': 21}),
            ('list_files', {'limit': 0}),
            ('list_files', {'path': '..'}),
        ],
    )
    async def test_search_errors(self, tmp_path: Path, name: str, arguments: dict[str, object]) -> None:
        assert await call(tmp_path, name, arguments)

    async def test_shell(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'do-not-expose')
        output = await call(tmp_path, 'shell', {'command': 'mkdir child; printf hello; exit 7'})
        assert 'hello' in output and '"exit_code": 7' in output
        assert (tmp_path / 'child').is_dir()
        output = await call(tmp_path, 'shell', {'command': 'printf "${OPENAI_API_KEY-unset}"'})
        assert 'unset' in output and 'do-not-expose' not in output

    @pytest.mark.parametrize('timeout', [0, 271])
    async def test_shell_timeout_validation(self, tmp_path: Path, timeout: int) -> None:
        assert 'timeout must' in await call(tmp_path, 'shell', {'command': 'echo hi', 'timeout': timeout})

    @pytest.mark.parametrize('path', ['../outside', '.env', 'missing.txt'])
    async def test_edit_path_errors_retry(self, tmp_path: Path, path: str) -> None:
        output = await call(tmp_path, 'edit_file', {'path': path, 'old_text': 'one', 'new_text': 'two'})
        assert 'Cannot read' in output

    @pytest.mark.skipif(os.name == 'nt', reason='POSIX FIFO')
    async def test_edit_fifo_retries(self, tmp_path: Path) -> None:
        os.mkfifo(tmp_path / 'pipe')
        output = await call(tmp_path, 'edit_file', {'path': 'pipe', 'old_text': 'one', 'new_text': 'two'})
        assert 'regular file' in output
