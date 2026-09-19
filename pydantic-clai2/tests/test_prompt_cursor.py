"""Track the writer position used to locate the editor after terminal reflow."""

import pytest

from pydantic_clai2.prompt_cursor import TranscriptCursor


def test_cells_delayed_wrap_newlines_and_split_styles() -> None:
    cursor = TranscriptCursor()
    for text in ('abcd', '\x1b[38;', '2;229;32;233m', 'e', '界', '\n', 'x\r', 'y\b', '\t'):
        cursor.feed(text, width=4, bottom=3)
    assert (cursor.row, cursor.column) == (3, 4)
    cursor.feed('\x1b]title\x07', width=4, bottom=3)
    cursor.feed('\x1b]title\x1b\\', width=4, bottom=3)
    assert (cursor.row, cursor.column) == (3, 4)
    cursor.feed('\x1bD', width=4, bottom=3)
    assert cursor.row == 3


@pytest.mark.parametrize(
    ('control', 'position'),
    [
        ('\x1b[2A', (3, 5)),
        ('\x1b[2B', (7, 5)),
        ('\x1b[2C', (5, 7)),
        ('\x1b[2D', (5, 3)),
        ('\x1b[2G', (5, 2)),
        ('\x1b[2d', (2, 5)),
        ('\x1b[2;3H', (2, 3)),
        ('\x1b[H', (1, 1)),
        ('\x1b[;f', (1, 1)),
        ('\x1b[?1A', (5, 5)),
        ('\x1b[0A', (4, 5)),
        ('\x1b[99C', (5, 20)),
    ],
)
def test_cursor_controls(control: str, position: tuple[int, int]) -> None:
    cursor = TranscriptCursor(row=5, column=5)
    cursor.feed(control, width=20, bottom=10)
    assert (cursor.row, cursor.column) == position


@pytest.mark.parametrize(('save', 'restore'), [('\x1b7', '\x1b8'), ('\x1b[s', '\x1b[u')])
def test_saved_cursor_and_zero_width_text(save: str, restore: str) -> None:
    cursor = TranscriptCursor(row=5, column=5)
    cursor.feed(save + '\x1b[1;1H' + restore, width=20, bottom=10)
    cursor.feed('\u0301\x00', width=20, bottom=10)
    assert (cursor.row, cursor.column) == (5, 5)


def test_wide_character_wraps_before_right_margin() -> None:
    cursor = TranscriptCursor(row=1, column=4)
    cursor.feed('界', width=4, bottom=3)
    assert (cursor.row, cursor.column) == (2, 3)
