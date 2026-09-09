from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from termflow.ansi import visible  # pyright: ignore[reportMissingTypeStubs]

from pydantic_ai_harness.cli import Config, cli_agent, main


def _echo_model() -> FunctionModel:
    """Call `run_command` with `ls -d .` once, then answer `ran`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        returned = any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts)
        if returned:
            yield 'ran'
        else:
            yield {0: DeltaToolCall(name='run_command', json_args=json.dumps({'command': 'ls -d .'}))}

    return FunctionModel(stream_function=stream)


class TestMain:
    def test_prompt_prints_response(self, capsys: pytest.CaptureFixture[str]) -> None:
        with cli_agent.override(model=TestModel(call_tools=[], custom_output_text='Hello from the harness.')):
            main(['-p', 'hi'])
        assert capsys.readouterr().out == 'Hello from the harness.\n'

    def test_without_prompt_reads_a_session_from_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr('sys.stdin', io.StringIO('hi\n'))
        with cli_agent.override(model=TestModel(call_tools=[], custom_output_text='Hello.')):
            main([])
        assert capsys.readouterr().out == 'harness> Hello.\nharness> \n'

    def test_model_flag_is_forwarded(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        with pytest.raises(SystemExit) as exc:
            main(['-p', 'hi', '--model', 'anthropic:claude-fable-5'])
        assert exc.value.code == 2
        assert 'ANTHROPIC_API_KEY' in capsys.readouterr().err

    def test_model_comes_from_the_config_file(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        Config(model='openai:gpt-5.6-sol').save()
        with pytest.raises(SystemExit) as exc:
            main(['-p', 'hi'])
        assert exc.value.code == 2
        assert 'OPENAI_API_KEY' in capsys.readouterr().err

    def test_model_flag_beats_the_config_file(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        Config(model='openai:gpt-5.6-sol').save()
        with pytest.raises(SystemExit):
            main(['-p', 'hi', '--model', 'anthropic:claude-fable-5'])
        assert 'ANTHROPIC_API_KEY' in capsys.readouterr().err

    def test_one_shot_mode_declines_commands_unless_yolo(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(Config.default_path().parent.parent)

        with cli_agent.override(model=_echo_model()):
            main(['-p', 'list'])
        declined = visible(capsys.readouterr().out)
        assert '< run_command [Command was not run: one-shot mode has no terminal' in declined
        assert '$ ls -d .' not in declined

        with cli_agent.override(model=_echo_model()):
            main(['-p', 'list', '--yolo'])
        assert '$ ls -d .\n.\nexit 0 (' in visible(capsys.readouterr().out)

    def test_yolo_comes_from_the_config_file(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(Config.default_path().parent.parent)
        Config(yolo=True).save()
        with cli_agent.override(model=_echo_model()):
            main(['-p', 'list'])
        assert '$ ls -d .\n.\nexit 0 (' in visible(capsys.readouterr().out)

    def test_invalid_config_file_exits_with_its_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        path = Config.default_path()
        path.parent.mkdir()
        path.write_text('{"show_thinking": "maybe"}')
        with pytest.raises(SystemExit) as exc:
            main(['-p', 'hi'])
        assert exc.value.code == 2
        assert f'Invalid config file {path}' in capsys.readouterr().err
