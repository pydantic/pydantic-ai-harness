"""Pinned editor and scrollback ownership, without a PromptSession renderer."""

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

import anyio
from anyio.to_thread import run_sync
from PIL import Image
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.history import History
from rich.console import Console
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.completion import CompleteEvent, Completion, Document  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.layout import truncate  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from .commands import Commands, is_command_input
from .image_input import ImageInput, clipboard_images, pasted_paths, read_images
from .interrupts import Interrupts
from .prompt_buffer import PromptBuffer
from .prompt_keys import PromptKeys
from .prompt_surface import PromptSurface
from .tool_output import terminal_text


class LivePrompt:
    """One terminal surface, one keyboard reader, sequential queued submissions."""

    def __init__(
        self,
        *,
        console: Console,
        commands: Commands,
        history: History,
        images: ImageInput,
        interrupts: Interrupts,
        toolbar: Callable[[], list[tuple[str, str]]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind editing state, terminal ownership and per-session services."""
        self.console = console
        self.commands = commands
        self.history = history
        self.images = images
        self.interrupts = interrupts
        self.toolbar = toolbar
        self.clock = clock
        self.buffer = PromptBuffer(history=list(reversed(list(history.load_history_strings()))))
        self.output = PromptSurface(output=console.file, size=lambda: (console.width, console.height))
        self.keys = PromptKeys(source=get_app_session().input, feed=self.feed, eof=lambda: self.submit(EOFError()))
        self._submissions: deque[str | KeyboardInterrupt | EOFError] = deque()
        self._submitted = asyncio.Event()
        self._suspended = False
        self._completions: list[Completion] = []
        self._selection = -1
        self._complete = anyio.Event()
        self._completion_owner = anyio.Lock()
        self._opened = False

    @property
    def queued_messages(self) -> tuple[str, ...]:
        """Pending text, excluding control signals."""
        return tuple(item for item in self._submissions if isinstance(item, str))

    def submit(self, value: str | KeyboardInterrupt | EOFError) -> None:
        """Publish a submission without ending or replacing the editor."""
        self._submissions.append(value)
        self._submitted.set()
        self.paint()

    async def read(self) -> str:
        """Consume queued submissions in order."""
        await self._submitted.wait()
        value = self._submissions.popleft()
        if not self._submissions:
            self._submitted.clear()
        self.paint()
        if isinstance(value, BaseException):
            raise value
        return value

    def paste(self, text: str | None) -> None:
        """Attach clipboard/path images, or insert a literal bracketed paste."""
        self.images.retain([self.buffer.text, *self.queued_messages])
        self.images.notice = ''
        try:
            paths = pasted_paths(text) if text is not None else []
            if text is not None and not paths:
                self.buffer.insert(text)
            else:
                self.buffer.insert(self.images.attach(read_images(paths) if text is not None else clipboard_images()))
        except (OSError, ValueError, NotImplementedError, Image.DecompressionBombError) as exc:
            self.images.notice = f'Image paste failed: {exc}. Linux requires wl-paste (Wayland) or xclip (X11).'

    def feed(self, key: str, data: str = '') -> None:
        """Route editing, completion and interrupts without rendering a widget tree."""
        if key == 'ctrl-c':
            if not self.interrupts.cancel():
                self.buffer.replace('')
                self.buffer.search = None
                self.submit(KeyboardInterrupt())
        elif key == 'escape' and self.interrupts.active:
            self.interrupts.cancel(exit_on_repeat=False)
        elif key == 'ctrl-d':
            if self.buffer.text:
                self.buffer.edit('delete')
            else:
                self.submit(EOFError())
        elif key in ('paste', 'ctrl-v', 'alt-v'):
            self.paste(data if key == 'paste' else None)
        elif self.buffer.search is not None:
            self.buffer.search_key(key)
        elif key in ('tab', 'backtab'):
            self.complete(backwards=key == 'backtab')
        elif key == 'enter':
            self.accept()
        elif key in ('alt-enter', 'ctrl-j'):
            self.buffer.insert('\n')
        elif key in ('up', 'down') and self._completions:
            self._selection = (self._selection + (-1 if key == 'up' else 1)) % len(self._completions)
        elif key == 'escape':
            self._completions = []
            self._selection = -1
        else:
            self.buffer.edit(key)
        if key not in ('tab', 'backtab', 'up', 'down', 'escape'):
            self.refresh_completions()
        self.paint()

    def accept(self) -> None:
        """Accept a selected completion, or queue the current nonempty draft."""
        if self._selection >= 0:
            self.accept_completion()
            return
        text = self.buffer.text.strip()
        if text:
            self.history.append_string(text)
            self.buffer.history.append(text)
            self.buffer.history_index = None
            self.buffer.replace('')
            self.submit(text)

    def complete(self, *, backwards: bool) -> None:
        """Cycle suggestions, accepting a sole candidate immediately."""
        if len(self._completions) == 1:
            self._selection = 0
            self.accept_completion()
        elif self._completions:
            self._selection = (
                len(self._completions) - 1
                if backwards and self._selection < 0
                else (self._selection + (-1 if backwards else 1)) % len(self._completions)
            )

    def accept_completion(self) -> None:
        """Apply the selected Termflow completion to its original prefix."""
        item = self._completions[self._selection]
        start = max(0, self.buffer.cursor + item.start_position)
        self.buffer.text = self.buffer.text[:start] + item.text + self.buffer.text[self.buffer.cursor :]
        self.buffer.cursor = start + len(item.text)
        self._completions = []
        self._selection = -1

    def refresh_completions(self) -> None:
        """Compute file/command suggestions off the input loop, ignoring stale results."""
        self._completions = []
        self._selection = -1
        self._complete.set()

    async def completion_loop(self) -> None:
        """Serialize completion lookups and join their worker on cancellation."""
        while True:
            await self._complete.wait()
            self._complete = anyio.Event()
            text, cursor = self.buffer.text, self.buffer.cursor
            if not text or self.buffer.search is not None:
                continue
            async with self._completion_owner:
                items = await run_sync(
                    lambda: list(
                        self.commands.get_completions(Document(text, cursor), CompleteEvent(text_inserted=True))
                    )[:100]
                )
            if (text, cursor) == (self.buffer.text, self.buffer.cursor):
                self._completions = items
                self.paint()

    def frame(self) -> tuple[str, ...]:
        """Build the reserved rows; transcript contents are deliberately absent."""
        width, height = max(1, self.console.width), max(2, self.console.height)
        muted, reset = theme.sgr(theme.MUTED), '\x1b[0m'
        if width < 6 or height < 6:
            return tuple(self.buffer.rows(width=width, limit=1))
        rows: list[str] = []
        queue_limit = max(1, height // 6)
        for text in self.queued_messages[:queue_limit]:
            label = 'Command' if is_command_input(text) else 'Follow-up'
            rows.append(muted + truncate(f'{label}: {" ".join(text.split())}', width) + reset)
        if len(self.queued_messages) > queue_limit:
            rows.append(muted + f'+{len(self.queued_messages) - queue_limit} more queued' + reset)
        title = ''
        if self.interrupts.active:
            spinner = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'[int(self.clock() * 10) % 10]
            title = f' Working {spinner} '
        title = truncate(title, width - 2)
        rows.append(muted + '┌' + title + '─' * max(0, width - 2 - visible_length(title)) + '┐' + reset)
        available = max(1, min(height // 3, height - len(rows) - 4))
        draft = self.buffer.rows(width=width - 4, limit=available)
        for index, row in enumerate(draft):
            prefix = '> ' if index == 0 else '  '
            rows.append(
                muted + '│' + reset + prefix + row + ' ' * max(0, width - 4 - visible_length(row)) + muted + '│' + reset
            )
        rows.append(muted + '└' + '─' * (width - 2) + '┘' + reset)
        popup_limit = max(0, min(6, height - len(rows) - 2))
        start = max(0, self._selection - popup_limit + 1)
        for index, item in enumerate(self._completions[start : start + popup_limit], start=start):
            line = truncate(
                ' '.join(terminal_text(f'{item.display or item.text}  {item.display_meta or ""}').split()), width
            )
            rows.append(('\x1b[7m' if index == self._selection else muted) + line + reset)
        if self.buffer.search is not None:
            footer = f'reverse-i-search: {self.buffer.search}'
        else:
            footer = (
                ' '.join(terminal_text(self.images.notice).split())
                if self.images.notice
                else ''.join(
                    (theme.sgr(style) if style else muted) + ' '.join(terminal_text(text).splitlines())
                    for style, text in self.toolbar()
                )
            )
            if self.queued_messages:
                footer += f' | queued: {len(self.queued_messages)}'
        rows.append(muted + truncate(footer, width) + reset)
        return tuple(rows)

    def paint(self) -> None:
        """Draw only when the editor owns the terminal."""
        if self._opened and not self._suspended:
            self.output.paint(self.frame())

    @asynccontextmanager
    async def suspended(self) -> AsyncGenerator[None]:
        """Hand input and terminal margins to a menu without losing the draft."""
        if self._suspended:
            yield
            return
        self._suspended = True
        self.keys.stop()
        try:
            async with self._completion_owner:
                await self.output.drain()
                self.output.release()
                yield
        finally:
            self._suspended = False
            self.paint()
            self.keys.start()

    @asynccontextmanager
    async def opened(self) -> AsyncGenerator[None]:
        """Scope keyboard, resize/status polling and output ownership to the shell."""

        async def refresh() -> None:
            while True:
                self.paint()
                await anyio.sleep(0.1)

        original = self.console.file
        self._opened = True
        self.console.file = self.output
        try:
            self.paint()
            self.keys.start()
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(refresh)
                tasks.start_soon(self.completion_loop)
                try:
                    yield
                    await self.output.drain()
                finally:
                    tasks.cancel_scope.cancel()
        finally:
            self._opened = False
            self.keys.stop()
            self.console.file = original
            self.output.release()
