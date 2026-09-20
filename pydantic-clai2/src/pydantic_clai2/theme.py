"""Terminal colour roles and Termflow palettes. Heavy imports stay out of the startup splash."""

# ruff: noqa: PLC0415 -- the splash must not import Termflow at startup.

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from termflow.themes import TerminalPalette  # pyright: ignore[reportMissingTypeStubs]

_ACTIVE: ContextVar[Callable[[], str]] = ContextVar('clai_theme', default=lambda: 'catppuccin_mocha')


def current() -> TerminalPalette:
    """Read the session's palette, including settings changed in menu workers."""
    from termflow.themes import PALETTES  # pyright: ignore[reportMissingTypeStubs]

    return PALETTES[_ACTIVE.get()()]


@contextmanager
def use(get_name: Callable[[], str], *, output: IO[str] | None = None) -> Generator[None]:
    """Scope a palette to a shell and reset terminal colours when it exits."""
    from termflow.themes import (  # pyright: ignore[reportMissingTypeStubs]
        apply_palette,  # pyright: ignore[reportUnknownVariableType] -- upstream also accepts an untyped dict.
        reset_palette,
    )

    active = _ACTIVE
    token = active.set(get_name)
    try:
        if output is not None:
            apply_palette(current(), output=output, register_reset=False)
        yield
    finally:
        if output is not None:
            reset_palette(output=output)
        active.reset(token)


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

ACCENT = 'bold bright_blue'
"""Names, arguments, the thing to look at."""
INFO = 'cyan'
"""Guidance the user asked for or needs next."""
WARNING = 'yellow'
"""Something stopped early but nothing broke."""
ERROR = 'red'
"""Something broke."""
MUTED = 'bright_black'
"""Housekeeping: tool markers, previews, hints."""
THINKING = 'magenta'
"""The model's reasoning heading."""
BANNER = ('bright_blue', 'magenta', 'cyan')
"""Top-to-bottom gradient for the CLAI banner."""
_ANSI = {'bright_blue': 94, 'cyan': 36, 'yellow': 33, 'red': 31, 'bright_black': 90, 'magenta': 35}

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
}


def truecolor() -> bool:
    """Whether the terminal advertises 24-bit colour."""
    return os.getenv('COLORTERM', '').lower() in ('truecolor', '24bit')


def sgr(color: str, *, bold: bool = False) -> str:
    """Raw escape for surfaces that bypass Rich, with a 16-colour fallback."""
    if color.startswith('bold '):
        color = color.removeprefix('bold ')
        bold = True
    prefix = '1;' if bold else ''
    if color in _ANSI:
        return f'\x1b[{prefix}{_ANSI[color]}m'
    if truecolor():
        red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
        return f'\x1b[{prefix}38;2;{red};{green};{blue}m'
    return f'\x1b[{prefix}{_BASIC[color]}m'
