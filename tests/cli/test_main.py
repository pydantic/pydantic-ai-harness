from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.cli import cli_agent, main


class TestMain:
    def test_prompt_prints_response(self, capsys: pytest.CaptureFixture[str]) -> None:
        with cli_agent.override(model=TestModel(call_tools=[], custom_output_text='Hello from the harness.')):
            main(['-p', 'hi'])
        assert capsys.readouterr().out == 'Hello from the harness.\n'

    def test_model_flag_is_forwarded(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        with pytest.raises(SystemExit) as exc:
            main(['-p', 'hi', '--model', 'anthropic:claude-fable-5'])
        assert exc.value.code == 2
        assert 'ANTHROPIC_API_KEY' in capsys.readouterr().err

    def test_prompt_is_required(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2
        assert '--prompt' in capsys.readouterr().err
