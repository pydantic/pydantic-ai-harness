"""Keyboard decoding only, with no prompt-toolkit application or renderer."""

import asyncio
from collections.abc import Callable
from contextlib import ExitStack

from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

_KEY_NAMES = {8: 'backspace', 9: 'tab', 13: 'enter', 27: 'escape', 32: 'space', 127: 'backspace'}
_MODIFIER_NAMES = ((4, 'ctrl'), (2, 'alt'), (1, 'shift'))


def modified_key(sequence: str) -> str | None:
    """Name an xterm `CSI 27;mods;code~` or kitty `CSI code;mods u` key report, or None.

    Only the shift, alt, and control bits are named. Terminals disagree about the
    higher bits (xterm's meta is kitty's super) and the lock bits carry no meaning
    here. Printable keys with no modifiers are reported only when a terminal has
    been asked to report every key, which the editor never does, so they are not
    turned into draft text. Shift alone on a printable key (or space) returns the
    character it types: modifyOtherKeys level 2 reports `Shift+Q` as
    `CSI 27;2;81~` and `Shift+1` as `CSI 27;2;33~`.
    """
    if not sequence.startswith('\x1b[') or sequence[-1] not in '~u':
        return None
    fields = [field.split(':')[0] for field in sequence[2:-1].split(';')]
    if not all(field.isdecimal() for field in fields):
        return None
    numbers = [int(field) for field in fields]
    if sequence.endswith('~'):
        if len(numbers) != 3 or numbers[0] != 27:
            return None
        _, modifiers, code = numbers
    elif len(numbers) == 1:
        code, modifiers = numbers[0], 1
    elif len(numbers) == 2:
        code, modifiers = numbers
    else:
        return None
    bits = (modifiers - 1) & 7
    char = chr(code) if 32 <= code <= 0x10FFFF else '\x00'
    if bits == 1 and char.isprintable():
        # iTerm2 and xterm report the shifted character; kitty reports the unshifted letter.
        return char.upper() if len(char.upper()) == 1 else char
    base = _KEY_NAMES.get(code)
    if base is None:
        if not bits or not char.isprintable():
            return None
        # iTerm2 reports the shifted letter where xterm and kitty report the unshifted one.
        base = char.lower() if bits & 1 else char
    if bits == 1 and base == 'tab':
        return 'backtab'
    return '-'.join([name for bit, name in _MODIFIER_NAMES if bits & bit] + [base])


class PromptKeys:
    """Reuse the portable escape/paste decoder while owning input attachment.

    Termflow's read_key currently drops bracketed paste and modified-key data.
    Keep the existing decoder until those protocols are supported there too.
    No prompt-toolkit layout, cursor writer or event loop is started.
    """

    def __init__(
        self,
        *,
        source: Input,
        feed: Callable[[str, str], None],
        eof: Callable[[], None],
    ) -> None:
        """Keep decoder and callbacks local to this editor."""
        self.source = source
        self.feed = feed
        self.eof = eof
        self._stack: ExitStack | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._escape = False
        self._csi = ''

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
        self._csi = ''
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
        self._csi = ''

    def _feed_modified(self, name: str, sequence: str) -> None:
        """Feed a shifted printable report as the text it types, like an unmodified key."""
        self.feed(name, name if len(name) == 1 else sequence)

    def dispatch(self, key: KeyPress) -> None:
        """Translate decoder tokens into editor actions and literal paste payloads."""
        if key.key == Keys.CPRResponse:
            # Ignore a late response left over from a previous renderer/menu.
            return
        name = modified_key(key.data) if key.key != Keys.BracketedPaste else None
        if name is not None:
            self._escape = False
            self._feed_modified(name, key.data)
            return
        if self._csi:
            # The installed decoder splits unrecognized modified keys into individual
            # keys. Reassemble it here, without modifying its global key table.
            self._csi += key.data
            if len(self._csi) > 32 or len(key.data) != 1 or '@' <= key.data <= '~':
                sequence, self._csi = self._csi, ''
                name = modified_key(sequence)
                if name is not None:
                    self._feed_modified(name, sequence)
            return
        if key.key == Keys.Escape:
            self._escape = True
            return
        if self._escape and key.data == '[':
            self._escape = False
            self._csi = '\x1b['
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
