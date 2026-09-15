"""Exercise the documented agent lifecycle without model or Sprites requests."""

from __future__ import annotations

import re
from pathlib import Path

import pydantic_ai.models
import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .fake_sprites import FakeExecResult, FakeSprites, RunCall


@pytest.mark.parametrize('document', ['pydantic_ai_harness/sprites/README.md', 'docs/sprite-sandbox.md'])
@pytest.mark.parametrize(('block', 'expected_commands'), [(0, 1), (2, 2)])
def test_documented_agent_lifecycle(
    document: str, block: int, expected_commands: int, fake_sprites: FakeSprites, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[RunCall] = []

    def respond(call: RunCall) -> FakeExecResult:
        # The caller-owned session must remain open until both agent runs finish.
        assert fake_sprites.destroy_calls == []
        assert len(fake_sprites.sprites) == 1
        commands.append(call)
        return FakeExecResult(stdout=b'example output\n')

    def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert {tool.name for tool in info.function_tools} == {
            'run_command',
            'read_file',
            'write_file',
            'list_directory',
        }
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart('Done')])
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'echo example'})])

    model = FunctionModel(model_response)

    def infer_model(*args: object, **kwargs: object) -> FunctionModel:
        return model

    fake_sprites.responder = respond
    monkeypatch.setenv('SPRITE_TOKEN', 'test-token')
    monkeypatch.setattr(pydantic_ai.models, 'infer_model', infer_model)
    path = Path(__file__).parents[2] / document
    snippets = re.findall(r'```python\n(.*?)```', path.read_text(), re.DOTALL)
    exec(compile(snippets[block], str(path), 'exec'), {})

    assert len(commands) == expected_commands
    assert len(fake_sprites.create_calls) == 1
    assert fake_sprites.destroy_calls == [fake_sprites.create_calls[0]['name']]
    assert fake_sprites.sprites == {}
    assert fake_sprites.close_calls == 1
