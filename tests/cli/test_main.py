from __future__ import annotations

import io

import pytest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.cli import cli_agent, main


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
