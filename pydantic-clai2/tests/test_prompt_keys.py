"""Decoder attachment and terminal-input handoff, without a renderer."""

from collections.abc import Callable, Generator
from contextlib import contextmanager

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

from pydantic_clai2.prompt_keys import PromptKeys


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_decoding_meta_paste_arrows_and_lone_escape() -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        keys.dispatch(KeyPress(Keys.ControlLeft, ''))
        keys.dispatch(KeyPress(Keys.BackTab, ''))
        keys.dispatch(KeyPress(Keys.Escape, '\x1b'))
        keys.dispatch(KeyPress('v', 'v'))
        keys.dispatch(KeyPress(Keys.BracketedPaste, 'one\ntwo'))
        keys.dispatch(KeyPress(Keys.Escape, '\x1b'))
        keys.flush()
        keys.flush()
        assert events == [('ctrl-left', ''), ('backtab', ''), ('alt-v', 'v'), ('paste', 'one\ntwo'), ('escape', '')]
        keys.start()
        keys.stop()
        keys.stop()


async def test_attach_failure_unwinds_raw_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    restored: list[bool] = []

    @contextmanager
    def raw_mode() -> Generator[None]:
        try:
            yield
        finally:
            restored.append(True)

    @contextmanager
    def fail(callback: Callable[[], None]) -> Generator[None]:
        raise OSError('attach failed')
        yield  # pragma: no cover -- makes this a context manager that fails on entry.

    with create_pipe_input() as pipe:
        monkeypatch.setattr(pipe, 'raw_mode', raw_mode)
        monkeypatch.setattr(pipe, 'attach', fail)
        keys = PromptKeys(source=pipe, feed=lambda key, data: None, eof=lambda: None)
        with pytest.raises(OSError, match='attach failed'):
            keys.start()
        keys.stop()
    assert restored == [True]
