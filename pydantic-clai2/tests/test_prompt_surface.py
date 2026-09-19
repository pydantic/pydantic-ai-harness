"""Raw terminal transactions, independent of terminal synchronization support."""

import io

import pytest

from pydantic_clai2.prompt_surface import PromptSurface


@pytest.mark.parametrize('tty', [False, True])
async def test_streaming_does_not_touch_reserved_rows_or_cursor_visibility(tty: bool) -> None:
    class Output(io.StringIO):
        def isatty(self) -> bool:
            return tty

    output = Output()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    surface.paint(('top border', 'draft cursor', 'bottom border', 'status'))
    start = len(output.getvalue())
    for text in ('one', ' two', '\n', 'next'):
        surface.write(text)
        surface.flush()
        surface.paint(('top border', 'draft cursor', 'bottom border', 'status'))
    await surface.drain()
    assert output.getvalue()[start:] == ('one two\r\nnext\r\n' if tty else 'one two\nnext\n')
    assert surface.isatty() is tty
    surface.release()
    start = len(output.getvalue())
    surface.release()
    assert output.getvalue()[start:] == ''


def test_typing_updates_only_draft_row_and_never_shows_hardware_cursor() -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (80, 24))
    surface.paint(('top', 'draft', 'bottom', 'status'))
    start = len(output.getvalue())
    surface.paint(('top', 'draft!', 'bottom', 'status'))
    update = output.getvalue()[start:]
    assert '\x1b[22;1H' in update
    assert 'draft!' in update
    assert 'top' not in update and 'bottom' not in update and 'status' not in update
    assert '\x1b[?25' not in update
    assert '\x1b[J' not in update and '\x1b[2J' not in update and '\x1b[2K' not in update


def test_growth_shrink_resize_and_menu_reentry_restore_margins() -> None:
    output = io.StringIO()
    size = (80, 24)
    surface = PromptSurface(output=output, size=lambda: size)
    surface.paint(('prompt', 'footer'))
    assert '\x1b[1;22r' in output.getvalue()
    surface.paint(('first', 'second', 'footer'))
    assert '\x1b[1;21r' in output.getvalue()
    surface.paint(('prompt', 'footer'))
    assert '\x1b[22;1H\x1b[2K' in output.getvalue()
    size = (40, 12)
    surface.write('resized output')
    surface.paint(('prompt', 'footer'))
    assert '\x1b[1;10r' in output.getvalue()
    surface.release()
    assert output.getvalue().endswith('\x1b[0m\x1b[?2004l\x1b[?25h\x1b[?2026l')
    surface.write('menu\n')
    surface.paint(('prompt', 'footer'))
    assert output.getvalue().count('\x1b[?25l') == 2


async def test_empty_writes_and_already_complete_drain() -> None:
    output = io.StringIO()
    surface = PromptSurface(output=output, size=lambda: (1, 2))
    assert surface.write('') == 0
    await surface.drain()
    assert output.getvalue() == ''
    surface.paint(('one', 'two', 'three'))
    assert '\x1b[1;2r' in output.getvalue()
