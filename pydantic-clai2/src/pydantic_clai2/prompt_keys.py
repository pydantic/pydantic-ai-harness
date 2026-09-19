"""Keyboard decoding only, with no prompt-toolkit application or renderer."""

import asyncio
from collections.abc import Callable
from contextlib import ExitStack

from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys


class PromptKeys:
    """Reuse the portable escape/paste decoder while owning input attachment.

    Termflow's read_key currently drops bracketed paste and modified-key data.
    Keep the existing decoder until those protocols are supported there too.
    No prompt-toolkit layout, cursor writer or event loop is started.
    """

    def __init__(self, *, source: Input, feed: Callable[[str, str], None], eof: Callable[[], None]) -> None:
        """Keep decoder and callbacks local to this editor."""
        self.source = source
        self.feed = feed
        self.eof = eof
        self._stack: ExitStack | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._escape = False

    def start(self) -> None:
        """Attach one input reader, with raw mode owned by its lifetime."""
        stack = ExitStack()
        try:
            stack.enter_context(self.source.raw_mode())
            stack.enter_context(self.source.attach(self.read))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        self.read()

    def stop(self) -> None:
        """Detach before a menu reads input, and cancel pending escape decoding."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._escape = False
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def read(self) -> None:
        """Consume available decoded keys without awaiting or starting a thread."""
        if self._timer is not None:
            self._timer.cancel()
        for key in self.source.read_keys():
            self.dispatch(key)
        if self.source.closed:
            self.eof()
        else:
            self._timer = asyncio.get_running_loop().call_later(0.05, self.flush)

    def flush(self) -> None:
        """Resolve a lone Escape without treating an Alt chord as cancellation."""
        self._timer = None
        for key in self.source.flush_keys():
            self.dispatch(key)
        if self._escape:
            self._escape = False
            self.feed('escape', '')

    def dispatch(self, key: KeyPress) -> None:
        """Translate decoder tokens into editor actions and literal paste payloads."""
        if key.key == Keys.Escape:
            self._escape = True
            return
        name = key.key.value if isinstance(key.key, Keys) else key.key
        name = {
            'c-m': 'enter',
            'c-j': 'enter',
            'c-i': 'tab',
            'c-h': 'backspace',
            's-tab': 'backtab',
            '<bracketed-paste>': 'paste',
        }.get(name, name)
        if name.startswith('c-'):
            name = 'ctrl-' + name[2:]
        if self._escape:
            name = 'alt-' + name
            self._escape = False
        self.feed(name, key.data)
