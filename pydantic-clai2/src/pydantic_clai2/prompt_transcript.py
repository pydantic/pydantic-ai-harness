"""Bounded, styled transcript tail for repainting the viewport after resize."""

import io
from collections import deque
from dataclasses import dataclass

from rich.ansi import AnsiDecoder
from rich.color import ColorSystem
from rich.console import Console
from rich.style import Style
from rich.text import Text
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]


@dataclass(frozen=True, kw_only=True)
class TranscriptFrame:
    """Visible rows and the styling needed by the next streaming write."""

    rows: tuple[str, ...]
    continuation_style: str


def style_prefix(style: Style) -> str:
    """Render an SGR prefix without replaying hyperlinks or visible text."""
    return style.update_link(None).render(' ', color_system=ColorSystem.TRUECOLOR).split(' ', 1)[0]


class TranscriptBuffer:
    """Retain recent output, never editor paint or terminal-control transactions."""

    def __init__(self, *, max_lines: int = 2000, max_chars: int = 1_000_000) -> None:
        """Bound both completed lines and an unterminated streaming line."""
        if max_lines < 1 or max_chars < 1:
            raise ValueError('Transcript limits must be positive.')
        self.max_lines = max_lines
        self.max_chars = max_chars
        self._lines: deque[Text] = deque()
        self._chars = 0
        self._pending = ''
        self._decoder = AnsiDecoder()
        self._console = Console(file=io.StringIO(), force_terminal=True, color_system='truecolor')

    def write(self, text: str) -> None:
        """Decode completed ANSI lines, retaining partial sequences between writes."""
        self._pending += text
        lines = self._pending.split('\n')
        self._pending = lines.pop()
        for line in lines:
            # CRLF is a line ending, not a progress-line overwrite.
            decoded = self._decoder.decode_line(line.removesuffix('\r'))
            decoded = decoded[-self.max_chars :]
            self._lines.append(decoded)
            self._chars += len(decoded)
            while len(self._lines) > self.max_lines or self._chars > self.max_chars:
                self._chars -= len(self._lines.popleft())
        if len(self._pending) > self.max_chars:
            prefix, self._pending = self._pending[: -self.max_chars], self._pending[-self.max_chars :]
            self._decoder.decode_line(prefix)

    def frame(self, *, width: int, height: int) -> TranscriptFrame:
        """Rewrap recent styled text without performing any terminal IO."""
        decoder = AnsiDecoder()
        decoder.style = self._decoder.style
        pending = decoder.decode_line(self._pending)
        rows: deque[str] = deque(maxlen=max(1, height))
        for line in (*self._lines, pending):
            text = line.copy()
            text.plain = ''.join(char if char.isprintable() or char == '\t' else '?' for char in text.plain)
            text.expand_tabs(8)
            offsets: list[int] = []
            cells = 0
            for index, char in enumerate(text.plain):
                size = visible_length(char)
                if cells and cells + size > width:
                    offsets.append(index)
                    cells = 0
                cells += size
            for piece in text.divide(offsets):
                piece.truncate(width, overflow='crop')
                # Rich only decodes recorded ANSI styling here. It owns neither
                # the editor nor a live renderer. Non-SGR controls are not replayed.
                rows.append(
                    ''.join(
                        segment.style.update_link(None).render(segment.text, color_system=ColorSystem.TRUECOLOR)
                        if segment.style
                        else segment.text
                        for segment in piece.render(self._console)
                    )
                )
        return TranscriptFrame(rows=tuple(rows), continuation_style=style_prefix(decoder.style))
