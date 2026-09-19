"""Theme selection through commands, menus, settings, and terminal rendering."""

import io
from pathlib import Path

import anyio
import pytest
from anyio.to_thread import run_sync
from menu_script import Script, make_context, pick
from prompt_toolkit.application import create_app_session, get_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from pydantic import ValidationError
from pydantic_ai import Agent, PartStartEvent, TextPart, ThinkingPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.filesystem import FileEditedEvent
from rich.console import Console
from rich.text import Text
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import StreamRenderer, chat, theme
from pydantic_clai2.commands import config_command, config_completions, set_completions
from pydantic_clai2.config import Settings
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.set_menu import open_settings_menu
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.status import Status, StatusLine
from pydantic_clai2.theme_picker import build_theme_picker, theme_command
from pydantic_clai2.tool_output import print_tool_header


async def test_picker_and_settings_share_validation_and_persistence(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    assert build_theme_picker(context).highlighted == MenuItem('pydantic (current)', value='pydantic')
    script = Script(lists=[pick('light')], choices=[], texts=[])
    with theme.use(lambda: context.settings.theme):
        assert await theme_command(context, [], runners=script.runners) == 'Saved display.theme. Applied.'
        assert theme.current() == theme.THEMES['light']
        assert SettingsStore(context.store.path).load().theme == 'light'
        assert build_theme_picker(context).highlighted == MenuItem('light (current)', value='light')
        assert await theme_command(context, ['system']) == 'Saved display.theme. Applied.'
        assert theme.current().syntax == 'ansi_dark'
        for args in (['unknown'], ['light', 'extra']):
            with pytest.raises(ValueError, match='Choose pydantic, light, or system'):
                await theme_command(context, args)
        with pytest.raises(ValidationError):
            context.set_setting(['display.theme', 'unknown'])
        assert context.settings.theme == context.store.load().theme == 'system'

        def edit(menu: FieldMenu) -> list[str]:
            row = menu.row_for('display.theme')
            assert row is not None and row.choices == ('pydantic', 'light', 'system')
            assert theme.current() == theme.THEMES['system']
            result = menu.apply(row, 'light')
            assert theme.current() == theme.THEMES['light']
            return [result]

        assert await open_settings_menu(context, run=edit) == 'Saved display.theme. Applied.'
        assert theme.current().syntax == 'friendly'
        assert context.reset_setting('display.theme').startswith('Reset')
        assert theme.current() == theme.THEMES['pydantic']
    assert applied == ['display.theme'] * 4
    assert context.store.overrides() == {}
    config_command(context.store, ['set', 'display.theme', 'light'])
    assert config_command(context.store, ['get', 'display.theme']) == '"light"'
    config_command(context.store, ['reset', 'display.theme'])
    assert context.store.load().theme == 'pydantic'
    assert tuple(set_completions(['display.theme', ''])) == ('pydantic', 'light', 'system')
    assert tuple(config_completions(['set', 'display.theme', ''])) == ('pydantic', 'light', 'system')


@pytest.mark.parametrize('result', [MenuResult(cancelled=True), MenuResult(), pick(0)])
async def test_cancel_keeps_preference(tmp_path: Path, result: MenuResult) -> None:
    context, applied = make_context(tmp_path)
    script = Script(lists=[result], choices=[], texts=[])
    assert await theme_command(context, [], runners=script.runners) == 'No changes.'
    assert applied == [] and context.store.overrides() == {}


@pytest.mark.parametrize('key', ['escape', 'ctrl-c', 'enter'])
def test_picker_keyboard_and_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    context, _ = make_context(tmp_path)
    output = io.StringIO()
    monkeypatch.setattr('sys.stdout', output)
    monkeypatch.setenv('COLUMNS', '120')
    monkeypatch.setenv('LINES', '30')
    keys = iter(['l', key])
    monkeypatch.setattr('pydantic_clai2.theme_picker.menu_key', lambda: next(keys))
    result = build_theme_picker(context).run()
    if key == 'enter':
        assert result.item == MenuItem('light', value='light')
    else:
        assert result.cancelled
    assert 'Select theme' in output.getvalue()
    assert 'Darker accents' in output.getvalue()
    assert 'Heading / tool name' in output.getvalue()
    assert context.store.overrides() == {}


@pytest.mark.parametrize('name', ['pydantic', 'light', 'system'])
@pytest.mark.parametrize('truecolor', [False, True])
async def test_roles_reach_markdown_status_and_tool_output(
    name: theme.ThemeName, truecolor: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor' if truecolor else '')
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor', width=80, height=24)
    with theme.use(lambda: name):
        colors = theme.current()
        print_tool_header(console, name='read_file', argument='example.py')
        renderer = StreamRenderer(console, stop_loading=lambda: None, show_tool_output=True, smooth_seconds=0.1)
        await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='Considering options')))
        await renderer.on_stream_event(PartStartEvent(index=1, part=TextPart(content='# Answer\n')))
        await renderer.finish()
        await renderer.on_stream_event(
            FileEditedEvent(
                root_dir='.',
                path='example.py',
                content_hash='hash',
                truncated=False,
                diff='--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n',
            )
        )
        status = Status(context_alert=True, context_tokens=90)
        assert status.toolbar()[1] == (colors.warning, '90')
        async with StatusLine(console, status):
            pass
        painted = output.getvalue()
        rendered = Text.from_ansi(painted)
        assert f'{theme.sgr(colors.warning)}9{theme.sgr(colors.warning)}0' in painted
        assert 'read_file' in rendered.plain and 'Considering options' in rendered.plain and 'Answer' in rendered.plain
        assert 'old' in rendered.plain and 'new' in rendered.plain
        if name == 'system':
            assert '38;2;' not in painted and '48;2;' not in painted
        else:
            assert rendered.get_style_at_offset(console, rendered.plain.index('read_file')) == console.get_style(
                colors.accent
            )
    assert theme.current() == theme.THEMES['pydantic']


def test_all_palette_colours_have_basic_ansi_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('COLORTERM', raising=False)
    for colors in theme.THEMES.values():
        for color in (
            colors.primary,
            colors.info,
            colors.warning,
            colors.error,
            colors.muted,
            colors.thinking,
            colors.surface,
            colors.panel,
            colors.link,
            colors.text,
            colors.highlight,
            colors.diff_addition,
            colors.diff_deletion,
        ):
            assert int(theme.sgr(color)[2:-1]) in (*range(30, 40), *range(90, 98))


async def test_fenced_code_uses_the_selected_syntax_theme() -> None:
    styles: list[str] = []
    name: theme.ThemeName
    for name in theme.THEMES:
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, color_system='truecolor')

        def selected_theme() -> theme.ThemeName:
            return name

        with theme.use(selected_theme):
            renderer = StreamRenderer(console, stop_loading=lambda: None)
            await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart('```python\nreturn True\n```\n')))
            await renderer.finish()
        rendered = Text.from_ansi(output.getvalue())
        styles.append(str(rendered.get_style_at_offset(console, rendered.plain.index('return'))))
    assert len(set(styles)) == 3


async def test_theme_scopes_isolate_concurrent_sessions(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    selected = anyio.Event()
    checked = anyio.Event()

    async def select() -> None:
        with theme.use(lambda: context.settings.theme):
            await run_sync(lambda: context.set_setting(['display.theme', 'light']))
            selected.set()
            await checked.wait()
            assert theme.current() is theme.THEMES['light']
        assert theme.current() is theme.THEMES['pydantic']

    async def check() -> None:
        with theme.use(lambda: 'system'):
            await selected.wait()
            assert theme.current() is theme.THEMES['system']
            checked.set()
        assert theme.current() is theme.THEMES['pydantic']

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(select)
            tasks.start_soon(check)
    assert theme.current() is theme.THEMES['pydantic']


async def test_typed_command_in_shell_and_restart(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/help\n/theme light\n/theme invalid\n/theme light extra\n/exit\n')
        await chat(Agent(TestModel()), deps=None, store=store, console=Console(file=output))
    assert '/theme: Select terminal colours' in output.getvalue()
    assert 'Unknown theme: invalid' in output.getvalue() and 'Usage: /theme' in output.getvalue()
    assert store.load().theme == 'light'
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('hi\n/exit\n')
        await chat(
            Agent(TestModel(custom_output_text='# Restored\n')),
            deps=None,
            settings=Settings(model=None, theme=SettingsStore(store.path).load().theme),
            store=store,
            console=Console(file=output),
        )
    assert '\x1b[38;2;152;15;156m' in output.getvalue()
    assert theme.current() == theme.THEMES['pydantic']


async def test_live_prompt_repaints_after_selection(tmp_path: Path) -> None:
    painted = {name: anyio.Event() for name in theme.THEMES}
    done = anyio.Event()

    class Output(io.StringIO):
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            if '\x1b[6n' in text:
                pipe.send_text('\x1b[10;1R')
            return super().write(text)

        def flush(self) -> None:
            app = get_app()
            if app.renderer.last_rendered_screen is not None:
                attrs = app.renderer.style.get_attrs_for_style_str('class:bottom-toolbar')
                for name, colors in theme.THEMES.items():
                    if attrs.color == colors.thinking.lstrip('#'):
                        painted[name].set()

    output = Output()
    store = SettingsStore(tmp_path / 'config.db')
    terminal = Vt100_Output(output, lambda: Size(rows=24, columns=80), term='xterm-256color')

    async def run() -> None:
        await chat(
            Agent(TestModel()),
            deps=None,
            store=store,
            console=Console(file=output, force_terminal=True, width=80, height=24),
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=terminal), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            await painted['pydantic'].wait()
            pipe.send_text('/theme light\n')
            await painted['light'].wait()
            pipe.send_text('/theme system\n')
            await painted['system'].wait()
            pipe.send_text('/exit\n')
            await done.wait()
    assert store.load().theme == 'system'
