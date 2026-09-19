"""Raw terminal transactions, independent of terminal synchronization support."""

import io

import pytest
from surface_terminal import SurfaceTerminal

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
    assert '\x1b[>4;1m' in output.getvalue()
    surface.paint(('first', 'second', 'footer'))
    assert '\x1b[1;21r' in output.getvalue()
    surface.paint(('prompt', 'footer'))
    assert '\x1b[22;1H\x1b[2K' in output.getvalue()
    size = (40, 12)
    surface.write('resized output')
    surface.cursor_position(row=10, column=1)
    surface.paint(('prompt', 'footer'))
    assert '\x1b[1;10r' in output.getvalue()
    surface.release()
    assert output.getvalue().endswith('\x1b[>4;0m\x1b[0m\x1b[?2004l\x1b[?25h\x1b[?2026l')
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


@pytest.mark.parametrize('stream_first', [False, True])
@pytest.mark.parametrize('bottom_anchored', [False, True])
def test_repeated_resize_erases_old_input_without_scrolling_it_into_history(
    stream_first: bool, bottom_anchored: bool
) -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    surface = surface_for(terminal)
    rows = ('TOP BORDER', 'UNSUBMITTED DRAFT', 'BOTTOM BORDER', 'FOOTER')
    surface.paint(rows)
    surface.write('transcript line\npartial')
    for width, height in ((100, 40), (60, 28), (120, 45), (40, 12), (80, 24)):
        terminal.resize(width=width, height=height, bottom_anchored=bottom_anchored)
        if stream_first:
            surface.write(' continuation')
        surface.paint(rows)
        lines = terminal.lines()
        assert lines[-4:] == list(rows)
        assert all(not any(label in line for label in rows) for line in lines[:-4])
        assert all(not any(label in line for label in rows) for line in terminal.history)
        assert terminal.row <= height - len(rows) - 1
        assert terminal.wrap
    surface.release()
    assert not any('UNSUBMITTED DRAFT' in line for line in terminal.lines())


def test_resize_does_not_teleport_partial_output_to_first_column() -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    surface = surface_for(terminal)
    rows = ('TOP', 'DRAFT', 'BOTTOM', 'FOOTER')
    surface.paint(rows)
    surface.write('partial')
    terminal.resize(width=100, height=40)
    surface.paint(rows)
    surface.write(' continuation')
    assert 'partial continuation' in terminal.lines()


def test_resizing_to_tiny_terminal_then_growing_removes_hidden_editor_rows() -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    surface = surface_for(terminal)
    surface.paint(('TOP', 'DRAFT', 'BOTTOM', 'FOOTER'))
    terminal.resize(width=20, height=2)
    surface.paint(('DRAFT',))
    terminal.resize(width=80, height=24)
    surface.paint(('TOP', 'DRAFT', 'BOTTOM', 'FOOTER'))
    assert terminal.lines().count('DRAFT') == 1
    assert 'DRAFT' not in terminal.history


def surface_for(terminal: SurfaceTerminal) -> PromptSurface:
    surface = PromptSurface(output=terminal, size=lambda: (terminal.width, terminal.height))
    terminal.report = lambda row, column: surface.cursor_position(row=row, column=column)
    return surface


def test_bottom_anchored_resize_preserves_every_transcript_line() -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    surface = surface_for(terminal)
    rows = ('TOP', 'DRAFT', 'BOTTOM', 'FOOTER')
    surface.paint(rows)
    for index in range(60):
        surface.write(f'TRANSCRIPT_{index:03d}\n')
    for height in (45, 20, 50, 24):
        terminal.resize(width=80, height=height, bottom_anchored=True)
        surface.paint(rows)
        all_lines = terminal.history + terminal.lines()
        assert all(f'TRANSCRIPT_{index:03d}' in all_lines for index in range(60))
        assert all_lines.count('DRAFT') == 1


def test_resize_waits_for_cursor_report_and_ignores_stale_geometry() -> None:
    output = io.StringIO()
    size = (80, 24)
    surface = PromptSurface(output=output, size=lambda: size)
    rows = ('TOP', 'DRAFT', 'BOTTOM', 'FOOTER')
    surface.paint(rows)
    size = (100, 40)
    start = len(output.getvalue())
    surface.write('queued text')
    assert output.getvalue()[start:] == '\x1b[6n'
    size = (80, 24)
    surface.paint(rows)
    surface.cursor_position(row=36, column=1)
    assert output.getvalue()[start:] == '\x1b[6n\x1b[6n'
    surface.cursor_position(row=20, column=1)
    assert output.getvalue().endswith('queued text')
    start = len(output.getvalue())
    surface.cursor_position(row=20, column=12)
    assert output.getvalue()[start:] == ''


def test_no_report_timeout_flushes_output_without_erasing_guessed_old_rows() -> None:
    output = io.StringIO()
    size = (80, 24)
    now = 0.0
    surface = PromptSurface(output=output, size=lambda: size, clock=lambda: now)
    rows = ('TOP', 'DRAFT', 'BOTTOM', 'FOOTER')
    surface.paint(rows)
    size = (100, 40)
    surface.write('queued')
    now = 0.3
    start = len(output.getvalue())
    surface.paint(rows)
    assert '\x1b[2K' not in output.getvalue()[start:]
    assert output.getvalue().endswith('queued')
    # The next successful report reacquires position, without guessing the
    # footprint relative to a cursor that was unknown after the timeout.
    size = (80, 24)
    surface.paint(rows)
    start = len(output.getvalue())
    surface.cursor_position(row=20, column=7)
    assert '\x1b[2K' not in output.getvalue()[start:]


@pytest.mark.parametrize('pending', [False, True])
def test_release_at_resize_flushes_deferred_output_and_restores_modes(pending: bool) -> None:
    output = io.StringIO()
    size = (80, 24)
    surface = PromptSurface(output=output, size=lambda: size)
    surface.paint(('DRAFT', 'FOOTER'))
    size = (60, 20)
    if pending:
        surface.write('queued before exit\n')
    surface.release()
    assert output.getvalue().endswith('\x1b[?25h\x1b[?2026l')
    if pending:
        assert 'queued before exit\n' in output.getvalue()


def test_width_reflow_clears_the_expanded_editor_footprint() -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    surface = surface_for(terminal)
    surface.paint(('┌' + '─' * 78 + '┐', '│DRAFT' + ' ' * 73 + '│', '└' + '─' * 78 + '┘', 'FOOTER'))
    surface.write('partial')
    terminal.reflow(width=30, height=30)
    surface.paint(('┌' + '─' * 28 + '┐', '│DRAFT' + ' ' * 23 + '│', '└' + '─' * 28 + '┘', 'FOOTER'))
    lines = terminal.lines()
    assert not any(any(char in line for char in '┌┐│└┘─') for line in lines[:-4])
    assert sum('DRAFT' in line for line in lines) == 1
    surface.write(' continuation')
    assert 'partial continuation' in terminal.lines()
