"""The CLI's config file: model, theme, and render policy, persisted as JSON under the user's home."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import UserError
from termflow import Highlighter, RenderStyle  # pyright: ignore[reportMissingTypeStubs]

DEFAULT_MODEL = 'anthropic:claude-fable-5'

Palette = Literal['default', 'dracula', 'gruvbox', 'nord']
"""The named Termflow palettes a `Theme` can pick."""

_PALETTES: dict[Palette, Callable[[], RenderStyle]] = {
    'default': RenderStyle.default,
    'dracula': RenderStyle.dracula,
    'gruvbox': RenderStyle.gruvbox,
    'nord': RenderStyle.nord,
}


@dataclass(kw_only=True)
class Theme:
    """Colors for everything the CLI renders."""

    palette: Palette = 'default'
    """Termflow palette for Markdown, tool lines, and status notes."""
    code_style: str = 'monokai'
    """Pygments style for code blocks. An unknown name falls back to `monokai`."""

    def render_style(self) -> RenderStyle:
        return _PALETTES[self.palette]()

    def highlighter(self) -> Highlighter:
        return Highlighter(style=self.code_style)


class Config(BaseModel):
    """The CLI's persisted settings.

    `load` and `save` read and write `default_path()` unless given a path; a missing file means
    the defaults, and an invalid one raises `UserError` naming the path.
    """

    model: str = DEFAULT_MODEL
    """Model in Pydantic AI `provider:name` form, resolved by core's `infer_model` when a run starts."""
    theme: Theme = Theme()
    show_thinking: bool = False
    """Render the model's thinking parts, dimmed, as they stream."""
    yolo: bool = False
    """Approve every shell command and file change without asking."""

    @staticmethod
    def default_path() -> Path:
        return Path.home() / '.pydantic-ai-harness' / 'config.json'

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = cls.default_path() if path is None else path
        try:
            text = path.read_text()
        except FileNotFoundError:
            return cls()
        try:
            return cls.model_validate_json(text)
        except ValidationError as exc:
            raise UserError(f'Invalid config file {path}:\n{exc}') from exc

    def save(self, path: Path | None = None) -> None:
        path = self.default_path() if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + '\n')
