from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import UserError
from termflow import RenderStyle  # pyright: ignore[reportMissingTypeStubs]

from pydantic_ai_harness.cli import DEFAULT_MODEL, Config, Theme


class TestConfig:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / 'nested' / 'config.json'
        config = Config(model='openai:gpt-5.6-sol', theme=Theme(palette='nord', code_style='vim'), show_thinking=True)

        config.save(path)

        assert Config.load(path) == config
        assert json.loads(path.read_text()) == {
            'model': 'openai:gpt-5.6-sol',
            'theme': {'palette': 'nord', 'code_style': 'vim'},
            'show_thinking': True,
        }

    def test_default_path_is_under_home(self, home: Path) -> None:
        assert Config.default_path() == home / '.pydantic-ai-harness' / 'config.json'
        assert Config.load() == Config(model=DEFAULT_MODEL, theme=Theme(), show_thinking=False)

        Config(model='test').save()

        assert Config.load() == Config(model='test')

    def test_partial_file_keeps_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / 'config.json'
        path.write_text('{"theme": {"palette": "dracula"}}')

        assert Config.load(path) == Config(theme=Theme(palette='dracula'))

    def test_invalid_file_names_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / 'config.json'
        path.write_text('{"theme": {"palette": "neon"}}')

        with pytest.raises(UserError, match=rf'Invalid config file {path}') as exc:
            Config.load(path)
        assert "'default', 'dracula', 'gruvbox' or 'nord'" in str(exc.value)


class TestTheme:
    def test_palette_and_code_style_map_to_termflow(self) -> None:
        theme = Theme(palette='gruvbox', code_style='native')

        assert theme.render_style() == RenderStyle.gruvbox()
        assert theme.highlighter().style_name == 'native'
        assert Theme().render_style() == RenderStyle.default()
