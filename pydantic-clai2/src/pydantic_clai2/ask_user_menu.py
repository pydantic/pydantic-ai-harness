"""Inline questions that leave the conversation in the terminal scrollback."""

import asyncio
from dataclasses import dataclass, field

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from pydantic_ai_harness.ask_user import (
    AskUser,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserResponse,
    Question,
)
from rich.console import RenderableType
from rich.text import Text

from . import theme
from .plugins import FullScreen, PluginHost


@dataclass(kw_only=True)
class QuestionMenu:
    """A compact question: Enter selects or toggles, and multi-select has a Done row."""

    question: Question
    position: int
    total: int
    cursor: int = 0
    selected: set[int] = field(default_factory=set[int])

    @property
    def title(self) -> str:
        """Name the question and its position in the request."""
        if self.total == 1:
            return self.question.header
        return f'{self.question.header} (question {self.position} of {self.total})'

    @property
    def hint(self) -> str:
        """Keep the selection keys visible without requiring Space."""
        action = 'Enter toggle - Done submits' if self.question.multi_select else 'Enter select'
        return f'Up/Down move - 1-{len(self.question.options)} pick - {action} - Esc decline'

    def build(self) -> Application[tuple[str, ...] | None]:
        """Draw beneath the transcript, without entering the alternate screen."""
        keys = KeyBindings()
        count = len(self.question.options) + int(self.question.multi_select)

        @keys.add('up')
        def up(event: KeyPressEvent) -> None:
            self.cursor = (self.cursor - 1) % count

        @keys.add('down')
        def down(event: KeyPressEvent) -> None:
            self.cursor = (self.cursor + 1) % count

        @keys.add('enter')
        def choose(event: KeyPressEvent) -> None:
            if self.cursor == len(self.question.options):
                if self.selected:
                    event.app.exit(result=tuple(self.question.options[i].label for i in sorted(self.selected)))
            elif self.question.multi_select:
                self.selected.symmetric_difference_update({self.cursor})
            else:
                event.app.exit(result=(self.question.options[self.cursor].label,))

        def number(event: KeyPressEvent) -> None:
            self.cursor = int(event.data) - 1
            choose(event)

        for index in range(len(self.question.options)):
            keys.add(str(index + 1))(number)

        @keys.add('escape')
        @keys.add('c-c')
        @keys.add('c-d')
        def decline(event: KeyPressEvent) -> None:
            event.app.exit(result=None)

        def rows() -> FormattedText:
            fragments: list[tuple[str, str]] = []
            labels = [option.label for option in self.question.options]
            if self.question.multi_select:
                labels.append('Done' if self.selected else 'Done (select at least one option)')
            for index, label in enumerate(labels):
                if index == self.cursor:
                    fragments.append(('[SetCursorPosition]', ''))
                style = theme.current().accent if index == self.cursor else theme.current().muted
                marker = '[x]' if index in self.selected else '[ ]'
                prefix = f'{marker} ' if self.question.multi_select and index < len(self.question.options) else ''
                number = f'{index + 1}. ' if index < len(self.question.options) else ''
                ending = '\n' if index < len(labels) - 1 else ''
                fragments.append((style, f'{">" if index == self.cursor else " "} {number}{prefix}{label}{ending}'))
            return FormattedText(fragments)

        def description() -> FormattedText:
            text = '' if self.cursor == len(self.question.options) else self.question.options[self.cursor].description
            return FormattedText([(theme.current().muted, text or '')])

        options = Window(
            FormattedTextControl(rows, focusable=True),
            wrap_lines=True,
            dont_extend_height=True,
            height=lambda: Dimension(max=max(2, app.output.get_size().rows // 2)),
        )
        app: Application[tuple[str, ...] | None] = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(FormattedTextControl(FormattedText([(theme.current().accent, self.title)])), height=1),
                        Window(
                            FormattedTextControl(self.question.question),
                            wrap_lines=True,
                            dont_extend_height=True,
                        ),
                        options,
                        Window(FormattedTextControl(description), wrap_lines=True, dont_extend_height=True),
                        Window(FormattedTextControl(self.hint), wrap_lines=True, dont_extend_height=True),
                    ]
                ),
                focused_element=options,
            ),
            key_bindings=keys,
            full_screen=False,
            erase_when_done=True,
        )
        app.ttimeoutlen = 0.05
        return app


class TerminalAnswerer:
    """Serialize question requests while the shell's editor and output are suspended."""

    def __init__(self, *, full_screen: FullScreen) -> None:
        """Borrow terminal ownership without clearing the conversation."""
        self._full_screen = full_screen
        self._terminal = asyncio.Lock()

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        """Answer every question or decline the request with Esc or Ctrl-C."""
        answers: list[AskUserAnswer] = []
        async with self._terminal, self._full_screen():
            for position, question in enumerate(request.questions, start=1):
                app = QuestionMenu(question=question, position=position, total=len(request.questions)).build()
                selected = await app.run_async(handle_sigint=False)
                if selected is None:
                    return AskUserResponse(cancelled=True)
                answers.append(AskUserAnswer(header=question.header, selected=selected))
        return AskUserResponse(answers=tuple(answers))


def render_answer(event: AskUserAnsweredEvent) -> RenderableType:
    """Leave a record of the selected answers in the transcript."""
    text = Text()
    if event.response.cancelled:
        text.append('● You declined to answer', style=theme.current().muted)
        return text
    for index, answer in enumerate(event.response.answers):
        if index:
            text.append('\n')
        text.append('● ', style=theme.current().muted)
        text.append(answer.header, style=theme.current().accent)
        text.append(f': {", ".join(answer.selected)}', style=theme.current().muted)
    return text


def activate(host: PluginHost[None]) -> None:
    """Register `AskUser` with inline questions and a transcript line per answer."""
    host.add(AskUser(answerer=TerminalAnswerer(full_screen=host.full_screen)))
    host.render(AskUserAnsweredEvent)(render_answer)
