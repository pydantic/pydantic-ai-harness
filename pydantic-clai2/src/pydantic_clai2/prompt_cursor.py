"""Track the transcript cursor so resize replies can locate the old editor band."""

from dataclasses import dataclass

from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]


@dataclass(kw_only=True)
class TranscriptCursor:
    """Track text, SGR and common cursor controls in the top-anchored region."""

    row: int = 1
    column: int = 1
    pending_wrap: bool = False
    escape: str = ''
    saved: tuple[int, int] = (1, 1)

    def feed(self, text: str, *, width: int, bottom: int) -> None:
        """Advance through transcript bytes, including escapes split across writes."""
        for char in text:
            if self.escape:
                self.escape += char
                if self.escape.startswith('\x1b]'):
                    complete = char == '\x07' or self.escape.endswith('\x1b\\')
                elif self.escape.startswith('\x1b['):
                    complete = len(self.escape) > 2 and '@' <= char <= '~'
                else:
                    complete = char != '[' and char != ']'
                if complete:
                    self.control(width=width, bottom=bottom)
                    self.escape = ''
            elif char == '\x1b':
                self.escape = char
            else:
                self.character(char, width=width, bottom=bottom)

    def character(self, char: str, *, width: int, bottom: int) -> None:
        """Track terminal cells and delayed autowrap, not Python string length."""
        if char == '\n':
            self.row, self.column, self.pending_wrap = min(bottom, self.row + 1), 1, False
        elif char == '\r':
            self.column, self.pending_wrap = 1, False
        elif char == '\b':
            self.column, self.pending_wrap = max(1, self.column - 1), False
        elif char == '\t':
            self.column = min(width, ((self.column - 1) // 8 + 1) * 8 + 1)
        elif char.isprintable():
            cells = visible_length(char)
            if cells:
                if self.pending_wrap or self.column + cells - 1 > width:
                    self.row, self.column = min(bottom, self.row + 1), 1
                self.pending_wrap = self.column + cells > width
                self.column = min(width, self.column + cells)

    def control(self, *, width: int, bottom: int) -> None:
        """Ignore styling; account for cursor movement in console output."""
        if self.escape in ('\x1b7', '\x1b[s'):
            self.saved = (self.row, self.column)
        elif self.escape in ('\x1b8', '\x1b[u'):
            self.row, self.column = self.saved
        elif self.escape == '\x1bD':
            self.row = min(bottom, self.row + 1)
        elif self.escape.startswith('\x1b[') and self.escape[-1] in 'ABCDGHfd':
            values = self.escape[2:-1].split(';')
            if not all(value.isdecimal() or not value for value in values):
                return
            first = int(values[0] or '1') or 1
            final = self.escape[-1]
            if final in 'AB':
                self.row += first if final == 'B' else -first
            elif final in 'CD':
                self.column += first if final == 'C' else -first
            elif final == 'G':
                self.column = first
            else:
                self.row = first
                if final in 'Hf':
                    self.column = int(values[1] or '1') if len(values) > 1 else 1
            self.row, self.column = max(1, min(bottom, self.row)), max(1, min(width, self.column))
            self.pending_wrap = False
