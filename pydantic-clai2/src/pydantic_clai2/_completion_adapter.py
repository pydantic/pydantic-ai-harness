"""Bridge Termflow completion declarations to the current prompt widget."""

from collections.abc import Iterable
from functools import lru_cache

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.styles import DynamicStyle, Style
from termflow.tui.completion import CompleteEvent as TermflowEvent  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.completion import Document as TermflowDocument  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from .commands import Commands


@lru_cache(maxsize=3)
def completion_style(colors: theme.Theme) -> Style:
    """Build each palette once; the live editor selects it on redraw."""
    return Style.from_dict(
        {
            'frame.border': colors.muted,
            'bottom-toolbar': f'noreverse bg:default {colors.thinking}',
            'bottom-toolbar.text': f'noreverse bg:default {colors.thinking}',
            'completion-menu': 'bg:default',
            'completion-menu.completion': f'bg:default {colors.muted}',
            'completion-menu.completion.current': f'bg:default {colors.accent}',
            'completion-menu.meta.completion': f'bg:default {colors.muted}',
            'completion-menu.meta.completion.current': f'bg:default {colors.primary}',
            'scrollbar.background': 'bg:default',
            'scrollbar.button': f'bg:default {colors.panel}',
        }
    )


COMPLETION_STYLE = DynamicStyle(lambda: completion_style(theme.current()))


class PromptCompleter(Completer):
    """Keep prompt-toolkit types out of the command and plugin interfaces."""

    def __init__(self, commands: Commands) -> None:
        """Adapt a conversation-local Termflow registry."""
        self.commands = commands

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        """Translate only at the current line editor boundary."""
        for item in self.commands.get_completions(
            TermflowDocument(document.text, document.cursor_position),
            TermflowEvent(complete_event.text_inserted, complete_event.completion_requested),
        ):
            yield Completion(
                item.text,
                start_position=item.start_position,
                display=f'{item.display or item.text}  {item.display_meta}' if item.display_meta else item.display,
                display_meta='',
            )
