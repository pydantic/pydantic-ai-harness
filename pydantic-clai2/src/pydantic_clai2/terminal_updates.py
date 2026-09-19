"""Keep nested editor and output paints inside one synchronized terminal update."""

from collections.abc import Generator
from contextlib import contextmanager

from prompt_toolkit.application.current import get_app
from prompt_toolkit.output.vt100 import Vt100_Output


class TerminalUpdates:
    """Share update ownership between output batches and normal editor refreshes."""

    def __init__(self) -> None:
        """Start outside any editor or output paint."""
        self._depth = 0

    def begin(self) -> None:
        """Defer terminal presentation until the outermost paint finishes."""
        terminal = get_app().output
        if self._depth == 0 and isinstance(terminal, Vt100_Output):
            terminal.write_raw('\x1b[?2026h')
            terminal.flush()
        self._depth += 1

    def end(self) -> None:
        """Present the completed paint, not a nested editor redraw."""
        self._depth -= 1
        terminal = get_app().output
        if self._depth == 0 and isinstance(terminal, Vt100_Output):
            terminal.write_raw('\x1b[?2026l')
            terminal.flush()

    @contextmanager
    def batch(self) -> Generator[None]:
        """Release the terminal even if output fails or the worker is cancelled."""
        self.begin()
        try:
            yield
        finally:
            self.end()
