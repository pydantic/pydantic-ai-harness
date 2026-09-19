"""Who has to step aside when a plugin takes the whole terminal mid-run."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from rich.console import Console

from . import theme
from .plugins import FullScreen, bare_screen


class Screen:
    """The shell's `FullScreen`: bound to the live renderer and status row for the length of one prompt.

    Plugin hosts are created once at load time, but what has to stop before a widget can draw
    changes every prompt. Hosts hold `screen.full`; the prompt loop binds what it means.

    One widget owns the screen at a time: a second `full()` (a parallel tool call, say) waits for
    the first to exit. It is not re-entrant; a widget that opens another widget does so inside
    its own block, not through a nested `full()`.
    """

    def __init__(self) -> None:
        """Start without a stream or editor to suspend."""
        self._take: FullScreen = bare_screen
        self._owner = asyncio.Lock()
        self.editor: FullScreen | None = None
        self._busy = 0
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()

    @contextmanager
    def session(self) -> Generator[None]:
        """Keep notices suspended after input ends until plugin workers are unloaded."""
        self._closed = False
        if not self._busy:
            self._idle.set()
        try:
            yield
        finally:
            self._closed = True
            self._idle.clear()

    @contextmanager
    def busy(self) -> Generator[None]:
        """Defer background notices until turns and menus have released output."""
        self._busy += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._busy -= 1
            if not self._busy and not self._closed:
                self._idle.set()

    async def notify(self, message: str, *, console: Console) -> None:
        """Print only at an idle boundary, without taking input away from the editor."""
        while self._busy or self._closed:
            await self._idle.wait()
        console.print(message, style=theme.current().info, markup=False, highlight=False)

    @contextmanager
    def bound(self, take: FullScreen) -> Generator[None]:
        """While active, `full()` defers to `take`; afterwards it is a no-op again."""
        self._take = take
        try:
            yield
        finally:
            self._take = bare_screen

    @asynccontextmanager
    async def full(self) -> AsyncGenerator[None]:
        """Own the terminal until the block exits. Give this to `PluginHost` as its `full_screen`."""
        with self.busy():
            async with self._owner, self._take(), (self.editor or bare_screen)():
                yield
