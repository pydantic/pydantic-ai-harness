"""Behavioral tests for the verification completion guard."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.require_verification import RequireVerification

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _verification_nudges(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, (SystemPromptPart, UserPromptPart))
        if 'no fresh passing verification evidence' in str(part.content)
    ]


def _agent(
    model_fn: Callable[[list[ModelMessage], AgentInfo], ModelResponse],
    guard: RequireVerification[object] | None = None,
    *,
    command_result: object | Callable[[str], object] = '(no output)',
) -> Agent[object, str]:
    agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[guard or RequireVerification[object]()])

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:  # pyright: ignore[reportUnusedFunction]
        """Write a file."""
        return f'wrote {path}: {len(content)}'

    @agent.tool_plain
    def edit_file(path: str, old_text: str, new_text: str) -> str:  # pyright: ignore[reportUnusedFunction]
        """Edit a file."""
        return f'edited {path}: {old_text!r} -> {new_text!r}'

    @agent.tool_plain
    def run_command(command: str) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Run a command."""
        if callable(command_result):
            return command_result(command)
        return command_result

    return agent


async def test_redirects_completion_until_fresh_verification() -> None:
    calls = 0
    nudges: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        if calls == 2:
            return ModelResponse(parts=[TextPart('done')])
        if calls == 3:
            nudges.extend(_verification_nudges(messages))
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'pytest tests/test_app.py'})])
        return ModelResponse(parts=[TextPart('verified')])

    result = await _agent(model_fn).run('change the code')

    assert result.output == 'verified'
    assert calls == 4
    assert len(nudges) == 1
    assert 'no fresh passing verification evidence' in nudges[0]


@pytest.mark.parametrize(
    'command',
    [
        'pytest -q',
        'uv run pytest -q',
        'python -m pytest -q',
        'env CI=1 pytest -q',
        'pnpm run test',
        'ruff check .',
        'make lint',
        'pyright',
        'tsc --noEmit',
        'uv build',
        'cargo check',
    ],
)
async def test_recognizes_verification_command_families(command: str) -> None:
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': command})])
        return ModelResponse(parts=[TextPart('done')])

    result = await _agent(model_fn, command_result={'ok': True}).run('change and verify')

    assert result.output == 'done'
    assert calls == 3


@pytest.mark.parametrize('failure', ['failure\n[exit code: 1]', '[Command timed out after 30s]'])
async def test_failed_verification_is_reported_and_can_be_retried(failure: str) -> None:
    calls = 0
    command_results = iter([failure, '(no output)'])
    nudges: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        if calls in (2, 4):
            if calls == 4:
                nudges.extend(_verification_nudges(messages))
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'pytest -q'})])
        if calls == 3:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[TextPart('verified')])

    def sequenced_command(command: str) -> str:
        assert command == 'pytest -q'
        return next(command_results)

    result = await _agent(model_fn, command_result=sequenced_command).run('change and verify')

    assert result.output == 'verified'
    assert calls == 5
    assert len(nudges) == 1
    assert 'latest recognized test command failed' in nudges[0]


async def test_verification_becomes_stale_after_later_edit() -> None:
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls in (1, 4):
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'pytest -q'})])
        if calls == 2:
            return ModelResponse(
                parts=[ToolCallPart('edit_file', {'path': 'app.py', 'old_text': '1', 'new_text': '2'})]
            )
        if calls == 3:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[TextPart('verified')])

    result = await _agent(model_fn).run('verify, edit, and finish')

    assert result.output == 'verified'
    assert calls == 5


async def test_mutating_shell_command_stales_prior_evidence() -> None:
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'pytest -q'})])
        if calls == 3:
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'ruff format .'})])
        if calls == 4:
            return ModelResponse(parts=[TextPart('done')])
        if calls == 5:
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'ruff check .'})])
        return ModelResponse(parts=[TextPart('verified')])

    result = await _agent(model_fn).run('format and verify')

    assert result.output == 'verified'
    assert calls == 6


async def test_documentation_only_edit_is_exempt() -> None:
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'docs/guide.md', 'content': 'text'})])
        return ModelResponse(parts=[TextPart('done')])

    result = await _agent(model_fn).run('edit docs')

    assert result.output == 'done'
    assert calls == 2


async def test_custom_mutation_and_verification_tools() -> None:
    calls = 0
    guard = RequireVerification[object](mutating_tools=('save_code',), verification_tools=('check_code',))

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('save_code', {'payload': 'x = 1'})])
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart('check_code', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[guard])

    @agent.tool_plain
    def save_code(payload: str) -> str:  # pyright: ignore[reportUnusedFunction]
        """Save code."""
        return payload

    @agent.tool_plain
    def check_code() -> str:  # pyright: ignore[reportUnusedFunction]
        """Check code."""
        return 'ok'

    result = await agent.run('save and verify')

    assert result.output == 'done'
    assert calls == 3


async def test_redirect_attempts_are_bounded() -> None:
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        return ModelResponse(parts=[TextPart(f'done-{calls}')])

    result = await _agent(model_fn, RequireVerification[object](max_attempts=2)).run('change without checks')

    assert result.output == 'done-4'
    assert calls == 4


@pytest.mark.parametrize(
    'command',
    [
        'echo looks good',
        'echo pytest',
        'uv run echo pytest',
        'python -c "print(\'pytest\')"',
        'pytest || true',
        'pytest; true',
        'pytest | tee test.log',
    ],
)
async def test_non_verification_shell_forms_do_not_count(command: str) -> None:
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': command})])
        return ModelResponse(parts=[TextPart(f'done-{calls}')])

    result = await _agent(model_fn, RequireVerification[object](max_attempts=1)).run('change and run command')

    assert result.output == 'done-4'
    assert calls == 4


async def test_shell_tool_ignores_non_string_command_argument() -> None:
    calls = 0
    guard = RequireVerification[object](max_attempts=1, shell_tools=('odd_shell',))

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart('odd_shell', {'command': 7})])
        return ModelResponse(parts=[TextPart(f'done-{calls}')])

    agent = _agent(model_fn, guard)

    @agent.tool_plain
    def odd_shell(command: int) -> int:  # pyright: ignore[reportUnusedFunction]
        """Return a non-string command value."""
        return command

    result = await agent.run('change and run odd shell')

    assert result.output == 'done-4'
    assert calls == 4


async def test_evidence_state_is_isolated_between_runs() -> None:
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1'})])
        return ModelResponse(parts=[TextPart(f'done-{calls}')])

    agent = _agent(model_fn, RequireVerification[object](max_attempts=1))
    first = await agent.run('edit code')
    second = await agent.run('answer without editing')

    assert first.output == 'done-3'
    assert second.output == 'done-4'
    assert calls == 4


def test_rejects_negative_attempt_limit() -> None:
    with pytest.raises(ValueError, match='max_attempts must be at least 0'):
        RequireVerification[object](max_attempts=-1)
