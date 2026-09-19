"""Pydantic brand palette and the roles CLAI paints with. Stdlib only, so the splash can import it.

Source: pydantic.dev `.agents/skills/pydantic-visual-identity/references/brand-identity.md`.
"""

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

LITHIUM = '#E520E9'
"""Primary brand accent. Logo, interactive highlights."""
CALCIUM = '#FF6550'
"""Secondary accent. Warm counterpoint; errors."""
PURPLE = '#9B77FF'
"""Tertiary accent. Code syntax, decorative elements."""
AQUA = '#77FFD8'
"""Dark-mode accent. Code syntax, links."""
SUGAR = '#FBFFEA'
"""Headline text on dark surfaces."""
LIGHT_PURPLE = '#F0E0FD'
"""Highlight."""
DARK_PURPLE = '#36182D'
"""Dark background surface."""
ELEMENT_PURPLE = '#49353F'
"""Icons, outlines, secondary elements on dark surfaces."""
GREY = '#8F888E'
"""Muted text and code on dark surfaces."""
AI_CYAN = '#00FFEB'
"""Pydantic AI sub-brand accent."""
AI_YELLOW = '#D0FF71'
"""Pydantic AI sub-brand accent."""

ACCENT = f'bold {LITHIUM}'
"""Names, arguments, the thing to look at."""
INFO = AI_CYAN
"""Guidance the user asked for or needs next."""
WARNING = AI_YELLOW
"""Something stopped early but nothing broke."""
ERROR = CALCIUM
"""Something broke."""
MUTED = GREY
"""Housekeeping: tool markers, previews, hints."""
THINKING = PURPLE
"""The model's reasoning heading."""
BANNER = (LITHIUM, PURPLE, AI_CYAN)
"""Top-to-bottom gradient for the CLAI banner."""
DIFF_ADDITION = '#465258'
"""Aqua over Dark Purple, one quarter strength."""
DIFF_DELETION = '#682B36'
"""Calcium over Dark Purple, one quarter strength."""

ThemeName = Literal['pydantic', 'light', 'system']


@dataclass(frozen=True, kw_only=True)
class Theme:
    """Colours for one terminal appearance; brand constants above stay unchanged."""

    description: str
    primary: str = LITHIUM
    info: str = INFO
    warning: str = WARNING
    error: str = ERROR
    muted: str = MUTED
    thinking: str = THINKING
    surface: str = DARK_PURPLE
    panel: str = ELEMENT_PURPLE
    link: str = AQUA
    text: str = SUGAR
    highlight: str = LIGHT_PURPLE
    diff_addition: str = DIFF_ADDITION
    diff_deletion: str = DIFF_DELETION
    diff_marker_brighten: float = 2.0
    syntax: str = 'monokai'
    """Rich Syntax theme name; read at render time, not import time."""

    @property
    def accent(self) -> str:
        """The primary colour with emphasis, suitable for Rich and prompt-toolkit."""
        return f'bold {self.primary}'


THEMES: dict[ThemeName, Theme] = {
    'pydantic': Theme(description='Pydantic colours for dark terminals (default).'),
    'light': Theme(
        description='Darker accents for light terminals.',
        primary='#980F9C',
        info='#006B63',
        warning='#755800',
        error='#B52B1C',
        muted='#696168',
        thinking='#6940B5',
        surface='#F6F2F5',
        panel='#E8DFE7',
        link='#006B63',
        text='#36182D',
        highlight='#6940B5',
        diff_addition='#DDEEE5',
        diff_deletion='#F8E1DD',
        diff_marker_brighten=-0.65,
        syntax='friendly',
    ),
    'system': Theme(
        description='Use terminal foreground and background.\nNo fixed UI colours; works on light or dark.',
        primary='default',
        info='default',
        warning='default',
        error='default',
        muted='default',
        thinking='default',
        surface='default',
        panel='default',
        link='default',
        text='default',
        highlight='default',
        diff_addition='default',
        diff_deletion='default',
        syntax='ansi_dark',
    ),
}
_ACTIVE: ContextVar[Callable[[], ThemeName]] = ContextVar('clai_theme', default=lambda: 'pydantic')


def current() -> Theme:
    """Read the session's active roles, including changes made in command workers."""
    return THEMES[_ACTIVE.get()()]


@contextmanager
def use(get_name: Callable[[], ThemeName]) -> Generator[None]:
    """Scope colours to a shell; child tasks and menu workers share its settings reader."""
    active = _ACTIVE
    token = active.set(get_name)
    try:
        yield
    finally:
        active.reset(token)


_BASIC = {
    LITHIUM: 95,
    CALCIUM: 91,
    PURPLE: 35,
    AQUA: 96,
    SUGAR: 97,
    LIGHT_PURPLE: 97,
    GREY: 90,
    AI_CYAN: 96,
    AI_YELLOW: 93,
    '#980F9C': 35,
    '#006B63': 36,
    '#755800': 33,
    '#B52B1C': 31,
    '#696168': 90,
    '#6940B5': 35,
    DARK_PURPLE: 30,
    ELEMENT_PURPLE: 30,
    DIFF_ADDITION: 90,
    DIFF_DELETION: 31,
    '#F6F2F5': 97,
    '#E8DFE7': 97,
    '#DDEEE5': 97,
    '#F8E1DD': 97,
}


def truecolor() -> bool:
    """Whether the terminal advertises 24-bit colour."""
    return os.getenv('COLORTERM', '').lower() in ('truecolor', '24bit')


def sgr(color: str, *, bold: bool = False) -> str:
    """Raw escape for surfaces that bypass Rich, with a 16-colour fallback."""
    prefix = '1;' if bold else ''
    if color == 'default':
        return f'\x1b[{prefix}39m'
    if truecolor():
        red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
        return f'\x1b[{prefix}38;2;{red};{green};{blue}m'
    return f'\x1b[{prefix}{_BASIC[color]}m'
