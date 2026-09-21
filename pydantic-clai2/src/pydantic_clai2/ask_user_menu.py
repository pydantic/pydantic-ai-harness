"""The built-in `ask_user` plugin: inline questions that keep the transcript visible."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, TypeAlias

from pydantic_ai_harness.ask_user import (
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
    Question,
    QuestionOption,
)
from rich.console import Console, RenderableType
from rich.text import Text
from termflow.tui.layout import truncate  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.terminal import raw_mode  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from .menu_worker import menu_key, run_worker
from .plugins import FullScreen, PluginHost
from .prompt_surface import PromptSurface

MenuResult: TypeAlias = tuple[str, ...] | Literal['previous', 'next'] | None


@dataclass(kw_only=True)
class QuestionMenu:
    """Inline picker state, independent of terminal input and rendering."""

    question: Question
    position: int
    total: int
    prompt: str | None = None
    cursor: int = 0
    selected: set[int] = field(default_factory=set[int])

    @property
    def title(self) -> str:
        """Include progress when the request contains several questions."""
        if self.total == 1:
            return self.question.header
        return f'{self.question.header} (question {self.position} of {self.total})'

    @property
    def hint(self) -> str:
        """Show the available actions without a separate Space-key convention."""
        action = 'toggle; Done continues' if self.question.multi_select else 'select'
        navigation = 'Left/Right questions - ' if self.total > 1 else ''
        return f'{navigation}Up/Down move - number/Enter {action} - Esc decline'

    def choose(self, key: str) -> MenuResult:
        """Apply a key; return selections only when a nonempty answer is submitted."""
        if self.total > 1 and key in ('left', 'right'):
            return 'previous' if key == 'left' else 'next'
        count = len(self.question.options)
        rows = count + int(self.question.multi_select)
        if key in ('up', 'down', 'tab'):
            self.cursor = (self.cursor + (-1 if key == 'up' else 1)) % rows
        elif key == 'enter' or key in tuple(str(i) for i in range(1, rows + 1)):
            if key != 'enter':
                self.cursor = int(key) - 1
            if not self.question.multi_select:
                self.selected = {self.cursor}
                return (self.question.options[self.cursor].label,)
            if self.cursor == count:
                if self.selected:
                    return tuple(option.label for i, option in enumerate(self.question.options) if i in self.selected)
            elif self.cursor in self.selected:
                self.selected.remove(self.cursor)
            else:
                self.selected.add(self.cursor)
        return None

    def frame(self, *, width: int, height: int) -> tuple[str, ...]:
        """Bound the picker to half the viewport, scrolling choices around the cursor."""
        budget = max(3, height // 2)
        choices: list[str] = []
        for index, option in enumerate(self.question.options):
            marker = (
                ('[x] ' if index in self.selected else '[ ] ') if self.question.multi_select or self.selected else ''
            )
            description = f' - {option.description}' if option.description else ''
            choices.append(f'{index + 1}. {marker}{option.label}{description}')
        if self.question.multi_select:
            choices.append(f'{len(choices) + 1}. Done' + ('' if self.selected else ' (select at least one)'))
        lines: list[str] = []
        focus = 0
        console = Console()
        for index, choice in enumerate(choices):
            if index == self.cursor:
                focus = len(lines)
            wrapped = Text(choice).wrap(console, width=max(1, width - 2), overflow='fold')
            for line_index, line in enumerate(wrapped):
                prefix = '> ' if index == self.cursor and line_index == 0 else '  '
                role = theme.ACCENT if index == self.cursor else theme.INFO
                lines.append(theme.sgr(role) + truncate(prefix + line.plain, width) + '\x1b[0m')
        prompt = Text(self.prompt if self.prompt is not None else self.question.question)
        prompt_lines = prompt.wrap(console, width=max(1, width), overflow='fold')
        prompt_budget = max(0, budget - 3)
        prompt_rows = [line.plain for line in prompt_lines[:prompt_budget]]
        if len(prompt_lines) > prompt_budget and prompt_rows:
            prompt_rows[-1] = truncate(truncate(prompt_rows[-1], max(0, width - 4)) + ' ...', width)
        visible = max(1, budget - 2 - len(prompt_rows))
        start = min(focus, max(0, len(lines) - visible))
        title = truncate(self.title, width)
        return (
            theme.sgr(theme.ACCENT, bold=True) + title + '\x1b[0m',
            *(theme.sgr(theme.ACCENT) + row + '\x1b[0m' for row in prompt_rows),
            *lines[start : start + visible],
            theme.sgr(theme.MUTED) + truncate(self.hint, width) + '\x1b[0m',
        )

    def run(self, *, console: Console, key_source: Callable[[], str] = menu_key) -> MenuResult:
        """Borrow the released editor surface, never entering the alternate screen."""
        surface = console.file
        if not isinstance(surface, PromptSurface):
            surface = PromptSurface(output=surface, size=lambda: console.size)
        try:
            with raw_mode():
                return self.read(surface=surface, console=console, key_source=key_source)
        finally:
            surface.release()

    def read(self, *, surface: PromptSurface, console: Console, key_source: Callable[[], str] = menu_key) -> MenuResult:
        """Update an already-owned surface without writing to the transcript."""
        while True:
            surface.paint(self.frame(width=console.width, height=console.height))
            key = key_source()
            if key in ('escape', 'ctrl-c'):
                return None
            result = self.choose(key)
            if result is not None:
                return result


class TerminalAnswerer:
    """Serialize inline question requests while the shell's input reader is suspended."""

    def __init__(
        self,
        *,
        full_screen: FullScreen,
        console: Console | None = None,
        runner: Callable[[QuestionMenu], MenuResult] | None = None,
    ) -> None:
        """Use the shell handoff for exclusive input ownership, not an alternate screen."""
        self._full_screen = full_screen
        self._console = console if console is not None else Console()
        self._runner = runner
        self._terminal = asyncio.Lock()

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        """Answer every question or decline the entire request."""
        menus = [
            QuestionMenu(question=question, position=position, total=len(request.questions))
            for position, question in enumerate(request.questions, start=1)
        ]
        position = 0
        async with self._terminal, self._full_screen():
            surface = self._console.file
            if not isinstance(surface, PromptSurface):
                surface = PromptSurface(output=surface, size=lambda: self._console.size)
            try:
                with raw_mode():
                    while True:
                        reviewing = position == len(menus)
                        menu = self.review_menu(menus) if reviewing else menus[position]
                        operation = (
                            partial(self._runner, menu)
                            if self._runner
                            else partial(menu.read, surface=surface, console=self._console)
                        )
                        result = await run_worker(operation)
                        if result is None:
                            return AskUserResponse(cancelled=True)
                        if result == 'previous':
                            position = max(0, position - 1)
                        elif result == 'next':
                            position = min(len(menus), position + 1)
                        elif reviewing:
                            if result == ('Submit answers',) and all(menu.selected for menu in menus):
                                break
                            position = next((i for i, menu in enumerate(menus) if not menu.selected), 0)
                        else:
                            menu.selected = {
                                i for i, option in enumerate(menu.question.options) if option.label in result
                            }
                            if len(menus) == 1:
                                break
                            position += 1
            finally:
                surface.release()
        return AskUserResponse(
            answers=tuple(
                AskUserAnswer(
                    header=menu.question.header,
                    selected=tuple(
                        option.label for i, option in enumerate(menu.question.options) if i in menu.selected
                    ),
                )
                for menu in menus
            )
        )

    @staticmethod
    def review_menu(menus: list[QuestionMenu]) -> QuestionMenu:
        """Summarize drafts before submitting the whole batch."""
        summary = '\n'.join(
            f'{menu.question.header}: '
            + (
                ', '.join(option.label for i, option in enumerate(menu.question.options) if i in menu.selected)
                or '(unanswered)'
            )
            for menu in menus
        )
        return QuestionMenu(
            question=Question(
                header='Review answers',
                question='Review your answers before submitting.',
                options=(QuestionOption(label='Submit answers'), QuestionOption(label='Review answers')),
            ),
            prompt=summary,
            position=len(menus) + 1,
            total=len(menus) + 1,
        )


def render_answer(event: AskUserAnsweredEvent) -> RenderableType:
    """Leave a record of what was picked in the transcript, since the menu itself is gone."""
    text = Text()
    if event.response.cancelled:
        text.append('● You declined to answer', style=theme.color(theme.MUTED))
        return text
    for index, answer in enumerate(event.response.answers):
        if index:
            text.append('\n')
        text.append('● ', style=theme.color(theme.MUTED))
        text.append(answer.header, style=theme.color(theme.ACCENT))
        text.append(f': {", ".join(answer.selected)}', style=theme.color(theme.MUTED))
    return text


def activate(host: PluginHost[None]) -> None:
    """Register `AskUser` with the terminal answerer and a transcript line per answer."""
    host.add(AskUser(answerer=TerminalAnswerer(full_screen=host.full_screen, console=host.console)))
    host.render(AskUserAnsweredEvent)(render_answer)
