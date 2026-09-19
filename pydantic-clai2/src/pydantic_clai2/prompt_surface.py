"""A pinned prompt below the transcript's terminal scroll region."""

import io
from collections.abc import Callable
from threading import RLock
from typing import IO

from termflow.tui.layout import truncate  # pyright: ignore[reportMissingTypeStubs]


class PromptSurface(io.StringIO):
    """Own transcript writes and reserved rows, without a second screen renderer.

    The hardware cursor stays in the transcript. The editor paints its own
    cursor cell, so typing does not toggle terminal cursor visibility.
    """

    def __init__(self, *, output: IO[str], size: Callable[[], tuple[int, int]]) -> None:
        """Bind the actual terminal and an injectable geometry source."""
        super().__init__()
        self.output = output
        self.size = size
        self._lock = RLock()
        self._geometry = (0, 0)
        self._rows: tuple[str, ...] = ()
        self._active = False
        self._partial = False

    def isatty(self) -> bool:
        """Preserve Rich and Termflow terminal detection."""
        return self.output.isatty()

    def write(self, text: str) -> int:
        """Stream immediately inside the margins without touching the editor."""
        with self._lock:
            if self._active and self.size() != self._geometry:
                self.paint(self._rows)
            # Raw input disables ONLCR on POSIX; make line endings explicit.
            self.output.write(text.replace('\n', '\r\n') if self._active and self.output.isatty() else text)
            self.output.flush()
            if text:
                self._partial = not text.endswith('\n')
        return len(text)

    def flush(self) -> None:
        """Forward output flushing without repainting the input."""
        with self._lock:
            self.output.flush()

    async def drain(self) -> None:
        """There is no deferred output; settle an incomplete line at a boundary."""
        if self._partial:
            self.write('\n')

    def paint(self, rows: tuple[str, ...]) -> None:
        """Paint changed reserved rows, retaining the transcript cursor position."""
        with self._lock:
            width, height = self.size()
            width, height = max(1, width), max(2, height)
            # DECSTBM needs at least two transcript rows. On a two-row
            # terminal keep input state but hide the editor until it grows.
            rows = tuple(truncate(row, width) for row in rows[-(height - 2) :]) if height > 2 else ()
            bottom = height - len(rows)
            resized = self._geometry != (width, height)
            geometry_changed = resized or len(rows) != len(self._rows)
            parts: list[str] = []
            if not self._active:
                # Reserve space by scrolling existing contents, not clearing them.
                parts.extend(
                    ['\x1b[?25l\x1b[?2004h\x1b[>4;1m', '\r\n' * len(rows), f'\x1b[1;{bottom}r', f'\x1b[{bottom};1H']
                )
                self._active = True
            elif geometry_changed:
                old_bottom = self._geometry[1] - len(self._rows)
                growth = max(0, old_bottom - bottom)
                parts.append('\x1b7')
                if growth:
                    # Scroll the old region before assigning its bottom rows to
                    # the editor; preserve the in-progress transcript line.
                    parts.extend([f'\x1b[{old_bottom};1H', '\r\n' * growth, '\x1b8', f'\x1b[{growth}A', '\x1b7'])
                parts.extend([f'\x1b[1;{bottom}r', '\x1b8'])
                if not resized and bottom > old_bottom:
                    parts.append('\x1b7')
                    for row in range(old_bottom + 1, bottom + 1):
                        parts.append(f'\x1b[{row};1H\x1b[2K')
                    parts.append('\x1b8')
            parts.append('\x1b7')
            changed = False
            for index, row in enumerate(rows):
                if geometry_changed or index >= len(self._rows) or row != self._rows[index]:
                    parts.append(f'\x1b[{bottom + index + 1};1H\x1b[0m{row}\x1b[0m\x1b[K')
                    changed = True
            parts.append('\x1b8')
            if changed or geometry_changed:
                self.output.write('\x1b[?2026h' + ''.join(parts) + '\x1b[?2026l')
                self.output.flush()
            self._rows = rows
            self._geometry = (width, height)

    def release(self) -> None:
        """Restore margins, cursor and paste mode before another terminal owner."""
        with self._lock:
            if not self._active:
                return
            bottom = self._geometry[1] - len(self._rows)
            parts = ['\x1b[?2026h', '\x1b[r']
            for row in range(bottom + 1, self._geometry[1] + 1):
                parts.append(f'\x1b[{row};1H\x1b[2K')
            parts.extend([f'\x1b[{bottom};1H', '\x1b[>4;0m\x1b[0m\x1b[?2004l\x1b[?25h\x1b[?2026l'])
            self.output.write(''.join(parts))
            self.output.flush()
            self._rows = ()
            self._active = False
