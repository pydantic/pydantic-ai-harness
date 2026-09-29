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


@pytest.mark.parametrize('sequence', ['\x1b[13;2u', '\x1b[27;2;13~'])
async def test_shift_enter_through_actual_decoder_with_split_input(sequence: str) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            for char in sequence:
                pipe.send_text(char)
                keys.read()
            assert events == [('shift-enter', sequence)]
        finally:
            keys.stop()


def test_csi_partial_unknown_and_oversized_sequences_do_not_become_draft_text() -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        for sequence in ('\x1b[13;', '\x1b[999u', '\x1b[' + '9' * 31):
            for char in sequence:
                keys.dispatch(KeyPress(Keys.Escape if char == '\x1b' else char, char))
            keys.flush()
        assert events == []
        keys.dispatch(KeyPress(Keys.ControlM, '\x1b[13;2u'))
        assert events == [('shift-enter', '\x1b[13;2u')]
        keys.dispatch(KeyPress(Keys.BracketedPaste, '\x1b[13;2u'))
        assert events[-1] == ('paste', '\x1b[13;2u')
        keys.dispatch(KeyPress(Keys.Escape, '\x1b'))
        keys.dispatch(KeyPress('[', '['))
        keys.dispatch(KeyPress(Keys.Left, '\x1b[D'))
        assert len(events) == 2


@pytest.mark.parametrize(
    'sequence',
    ['\x1b\x7f', '\x1b\x08', '\x1b[27;3;127~', '\x1b[27;3;8~', '\x1b[127;3u', '\x1b[8;3u'],
)
@pytest.mark.parametrize('split', [False, True])
async def test_alt_backspace_through_actual_decoder(sequence: str, split: bool) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            for chunk in sequence if split else [sequence]:
                pipe.send_text(chunk)
                keys.read()
            assert [key for key, _ in events] == ['alt-backspace']
        finally:
            keys.stop()


@pytest.mark.parametrize('sequence', ['\x1b[27;3;127~', '\x1b[27;3;8~', '\x1b[127;3u', '\x1b[8;3u'])
def test_modified_alt_backspace_tokens_and_literal_paste(sequence: str) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        keys.dispatch(KeyPress(Keys.ControlH, sequence))
        keys.dispatch(KeyPress(Keys.BracketedPaste, sequence))
        assert events == [('alt-backspace', sequence), ('paste', sequence)]


@pytest.mark.parametrize(
    ('sequence', 'expected'),
    [
        ('\x1b[27;3;13~', 'alt-enter'),
        ('\x1b[13;3u', 'alt-enter'),
        ('\x1b[27;5;99~', 'ctrl-c'),
        ('\x1b[99;5u', 'ctrl-c'),
        ('\x1b[27;5;100~', 'ctrl-d'),
        ('\x1b[27;5;106~', 'ctrl-j'),
        ('\x1b[27;5;13~', 'ctrl-enter'),
        ('\x1b[27;5;127~', 'ctrl-backspace'),
        ('\x1b[27;6;86~', 'ctrl-shift-v'),
        ('\x1b[118;6u', 'ctrl-shift-v'),
        ('\x1b[27;2;9~', 'backtab'),
        ('\x1b[27;7;13~', 'ctrl-alt-enter'),
        ('\x1b[27;4;13~', 'alt-shift-enter'),
        ('\x1b[27;3;32~', 'alt-space'),
        ('\x1b[27u', 'escape'),
        ('\x1b[13;3:1u', 'alt-enter'),
        ('\x1b[97:65;3u', 'alt-a'),
        ('\x1b[27;65;13~', 'enter'),
        ('\x1b[27;9;13~', 'enter'),
    ],
)
async def test_modified_key_reports_name_shift_alt_and_control(sequence: str, expected: str) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            pipe.send_text(sequence)
            keys.read()
            assert events == [(expected, sequence)]
        finally:
            keys.stop()


@pytest.mark.parametrize(
    ('sequence', 'text'),
    [
        ('\x1b[27;2;81~', 'Q'),
        ('\x1b[27;2;113~', 'Q'),
        ('\x1b[113;2u', 'Q'),
        ('\x1b[27;2;33~', '!'),
        ('\x1b[27;2;32~', ' '),
        ('\x1b[27;2;223~', '\u00df'),
    ],
)
@pytest.mark.parametrize('split', [False, True])
async def test_shifted_printable_reports_are_typed_text(sequence: str, text: str, split: bool) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            for chunk in sequence if split else [sequence]:
                pipe.send_text(chunk)
                keys.read()
            assert events == [(text, text)]
        finally:
            keys.stop()


@pytest.mark.parametrize(
    ('sequence', 'expected'),
    [('\x1b[27;2;13~', 'shift-enter'), ('\x1b[27;2;127~', 'shift-backspace'), ('\x1b[27;2;27~', 'shift-escape')],
)
async def test_shifted_named_keys_keep_their_names(sequence: str, expected: str) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        try:
            pipe.send_text(sequence)
            keys.read()
            assert events == [(expected, sequence)]
        finally:
            keys.stop()


@pytest.mark.parametrize(
    'sequence',
    [
        '\x1b[97u',
        '\x1b[999u',
        '\x1b[97;65u',
        '\x1b[27;3;0~',
        '\x1b[27;3;1114112~',
        '\x1b[57358;3u',
        '\x1b[27;3~',
        '\x1b[3;5;13~',
        '\x1b[?13;3u',
        '\x1b[1;2;3u',
        '\x1b[\u00b2u',
        '\x1b[27;2;\u00b3~',
    ],
)
def test_unmodified_and_functional_reports_are_not_draft_keys(sequence: str) -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(source=pipe, feed=lambda key, data: events.append((key, data)), eof=lambda: None)
        for char in sequence:
            keys.dispatch(KeyPress(Keys.Escape if char == '\x1b' else char, char))
        keys.flush()
        assert events == []


async def test_cursor_reports_are_not_draft_keys() -> None:
    events: list[tuple[str, str]] = []
    with create_pipe_input() as pipe:
        keys = PromptKeys(
            source=pipe,
            feed=lambda key, data: events.append((key, data)),
            eof=lambda: None,
        )
        try:
            pipe.send_text('\x1b[12;34R')
            keys.read()
            assert events == []
        finally:
            keys.stop()
