"""Write complete output lines above a live prompt without splitting its editor."""

import asyncio
import io
import threading
from contextvars import copy_context

from prompt_toolkit.application import get_app, in_terminal
from prompt_toolkit.output import Output


class PromptOutput(io.StringIO):
    """Buffer Rich and Termflow chunks until a complete line can share one redraw.

    Adapted from the persistent-prompt prototype. Worker-thread writes are handed
    back to the editor's event loop; terminal writes stay ordered with its exit.
    """

    def __init__(self, output: Output) -> None:
        """Share the editor's output device and event loop."""
        super().__init__()
        self._output = output
        self._loop = asyncio.get_running_loop()
        self._thread = threading.current_thread()
        self._context = copy_context()
        self._lock = threading.Lock()
        self._partial = ''
        self._ready = ''
        self._writer: asyncio.Task[None] | None = None

    def write(self, text: str) -> int:
        """Schedule complete lines without interrupting partial Markdown chunks."""
        with self._lock:
            self._partial += text
            if '\n' not in self._partial:
                return len(text)
            lines, self._partial = self._partial.rsplit('\n', 1)
            self._ready += lines + '\n'
        if threading.current_thread() is self._thread:
            self._start_writer()
        else:
            self._loop.call_soon_threadsafe(self._start_writer, context=self._context)
        return len(text)

    def flush(self) -> None:
        """Partial lines wait for a newline or an explicit async drain."""

    def isatty(self) -> bool:
        """Preserve the underlying terminal's rendering capabilities."""
        stdout = self._output.stdout
        return stdout is not None and stdout.isatty()

    async def drain(self) -> None:
        """Finish pending output before transferring terminal ownership."""
        with self._lock:
            self._ready += self._partial
            self._partial = ''
        self._start_writer()
        if self._writer is not None:
            await self._writer

    def _start_writer(self) -> None:
        if self._writer is None and self._ready:
            self._writer = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            while self._ready:
                async with in_terminal():
                    with self._lock:
                        text, self._ready = self._ready, ''
                    # in_terminal normally enables echo for subprocesses. Our writes
                    # must not echo keys that arrive while the editor is redrawn.
                    with get_app().input.raw_mode():
                        self._output.write_raw(text)
                        self._output.flush()
        finally:
            self._writer = None
