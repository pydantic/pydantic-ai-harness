"""A pinned prompt below the transcript's terminal scroll region."""

import io
import time
from collections.abc import Callable
from threading import RLock
from typing import IO

from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.layout import truncate  # pyright: ignore[reportMissingTypeStubs]

from .prompt_cursor import TranscriptCursor


class PromptSurface(io.StringIO):
    """Serialize transcript and editor writes while keeping the hardware cursor hidden."""

    def __init__(
        self, *, output: IO[str], size: Callable[[], tuple[int, int]], clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Bind terminal IO and injectable geometry/time sources."""
        super().__init__()
        self.output = output
        self.size = size
        self.clock = clock
        self._lock = RLock()
        self._geometry = (0, 0)
        self._rows: tuple[str, ...] = ()
        self._active = False
        self._partial = False
        self._cursor = TranscriptCursor()
        self._cursor_known = True
        self._resize_started: float | None = None
        self._resize_size = (0, 0)
        self._resize_rows: tuple[str, ...] = ()
        self._deferred = ''

    def isatty(self) -> bool:
        """Preserve Rich and Termflow terminal detection."""
        return self.output.isatty()

    def write(self, text: str) -> int:
        """Stream without repainting the editor; briefly queue during a resize query."""
        with self._lock:
            if self._active and self.size() != self._geometry:
                self.paint(self._resize_rows if self._resize_started is not None else self._rows)
            if self._resize_started is not None:
                self._deferred += text
            else:
                self.output.write(text.replace('\n', '\r\n') if self._active and self.output.isatty() else text)
                self.output.flush()
                if self._active:
                    self._cursor.feed(text, width=self._geometry[0], bottom=self._geometry[1] - len(self._rows))
            if text:
                self._partial = not text.endswith('\n')
        return len(text)

    def flush(self) -> None:
        """Flush without repainting or ending an incomplete line."""
        with self._lock:
            self.output.flush()

    async def drain(self) -> None:
        """Settle an incomplete transcript line before a turn/menu boundary."""
        if self._partial:
            self.write('\n')

    def paint(self, rows: tuple[str, ...]) -> None:
        """Paint changed rows, or request the actual post-resize cursor position."""
        with self._lock:
            width, height = self.size()
            width, height = max(1, width), max(2, height)
            rows = tuple(truncate(row, width) for row in rows[-(height - 2) :]) if height > 2 else ()
            if self._active and ((width, height) != self._geometry or self._resize_started is not None):
                self._resize_rows = rows
                if self._resize_started is None:
                    self._resize_started = self.clock()
                    self._resize_size = (width, height)
                    self.output.write('\x1b[6n')
                    self.output.flush()
                elif self.clock() - self._resize_started >= 0.25:
                    # No CPR support: repaint the new band, but do not erase
                    # guessed old coordinates that might now contain transcript.
                    self._finish_resize(position=None)
                return
            self._paint(rows=rows, width=width, height=height)

    def _paint(self, *, rows: tuple[str, ...], width: int, height: int) -> None:
        bottom = height - len(rows)
        changed_geometry = self._geometry != (width, height) or len(rows) != len(self._rows)
        parts: list[str] = []
        if not self._active:
            parts.extend(
                ['\x1b[?25l\x1b[?2004h\x1b[>4;1m', '\r\n' * len(rows), f'\x1b[1;{bottom}r', f'\x1b[{bottom};1H']
            )
            self._active = True
            self._cursor = TranscriptCursor(row=bottom)
            self._cursor_known = True
        elif changed_geometry:
            old_bottom = self._geometry[1] - len(self._rows)
            growth = max(0, old_bottom - bottom)
            parts.append('\x1b7')
            if growth:
                parts.extend([f'\x1b[{old_bottom};1H', '\r\n' * growth, '\x1b8', f'\x1b[{growth}A', '\x1b7'])
                self._cursor.row = max(1, self._cursor.row - growth)
            parts.extend([f'\x1b[1;{bottom}r', '\x1b8'])
            if bottom > old_bottom:
                parts.append('\x1b7')
                for row in range(old_bottom + 1, bottom + 1):
                    parts.append(f'\x1b[{row};1H\x1b[2K')
                parts.append('\x1b8')
        parts.append(self._row_changes(rows=rows, bottom=bottom, width=width, force=changed_geometry))
        if any(parts):
            self._transaction(''.join(parts))
        self._rows, self._geometry = rows, (width, height)

    def _row_changes(self, *, rows: tuple[str, ...], bottom: int, width: int, force: bool) -> str:
        parts: list[str] = []
        for index, row in enumerate(rows):
            if force or index >= len(self._rows) or row != self._rows[index]:
                clear = '\x1b[K' if visible_length(row) < width else ''
                parts.append(f'\x1b[{bottom + index + 1};1H\x1b[0m{row}\x1b[0m{clear}')
        # Disable autowrap only for editor paint, never for the transcript.
        return '\x1b7\x1b[?7l' + ''.join(parts) + '\x1b[?7h\x1b8' if parts else ''

    def cursor_position(self, *, row: int, column: int) -> None:
        """Consume a resize CPR without letting its bytes reach the draft editor."""
        with self._lock:
            if self._resize_started is None:
                return
            if self.size() != self._resize_size:
                # The window changed again while the report was in flight.
                self._resize_started = self.clock()
                self._resize_size = self.size()
                self.output.write('\x1b[6n')
                self.output.flush()
                return
            self._finish_resize(position=(row, column))

    def _finish_resize(self, *, position: tuple[int, int] | None) -> None:
        width, height = self.size()
        width, height = max(1, width), max(2, height)
        rows = self._resize_rows[-(height - 2) :] if height > 2 else ()
        bottom = height - len(rows)
        parts = ['\x1b7', '\x1b[r']
        if position is not None:
            # Locate the old editor relative to the reported transcript cursor.
            # This works whether resize kept absolute rows or shifted the whole
            # viewport. Erasing the old absolute band loses output under tmux.
            if self._cursor_known:
                gap = self._geometry[1] - len(self._rows) + 1 - self._cursor.row
                first = position[0] + gap
                footprint = sum(max(1, (visible_length(row) + width - 1) // width) for row in self._rows)
                for line in range(max(1, first), min(height, first + footprint - 1) + 1):
                    parts.append(f'\x1b[{line};1H\x1b[2K')
            self._cursor.row, self._cursor.column = position
        parts.append('\x1b8')
        if rows:
            # Preserve the column, clamping the writer inside the new region.
            parts.extend(['\x1bD' * len(rows), f'\x1b[{len(rows)}A'])
        parts.extend(['\x1b7', f'\x1b[1;{bottom}r', '\x1b8'])
        parts.append(self._row_changes(rows=rows, bottom=bottom, width=width, force=True))
        self._transaction(''.join(parts))
        self._cursor_known = position is not None
        self._cursor.row = min(bottom, self._cursor.row)
        self._cursor.column = min(width, self._cursor.column)
        self._rows, self._geometry = rows, (width, height)
        self._resize_started = None
        self._flush_deferred()

    def _flush_deferred(self) -> None:
        deferred, self._deferred = self._deferred, ''
        if deferred:
            self.write(deferred)

    def _transaction(self, text: str) -> None:
        self.output.write('\x1b[?2026h' + text + '\x1b[?2026l')
        self.output.flush()

    def release(self) -> None:
        """Restore terminal modes before another owner takes input and output."""
        with self._lock:
            if not self._active:
                return
            if self._resize_started is not None or self.size() != self._geometry:
                if self._resize_started is None:
                    self._resize_rows = self._rows
                self._finish_resize(position=None)
            bottom = self._geometry[1] - len(self._rows)
            parts = ['\x1b[r']
            for row in range(bottom + 1, self._geometry[1] + 1):
                parts.append(f'\x1b[{row};1H\x1b[2K')
            parts.extend([f'\x1b[{bottom};1H', '\x1b[>4;0m\x1b[0m\x1b[?2004l\x1b[?25h'])
            self._transaction(''.join(parts))
            self._rows = ()
            self._active = False
