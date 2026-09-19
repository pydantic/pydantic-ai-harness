"""Background notices wait for output ownership without taking over the editor."""

import asyncio
import io
from pathlib import Path

import anyio
import pytest
from packaging.version import Version
from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import chat, updates
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.screen import Screen
from pydantic_clai2.sessions import Sessions
from pydantic_clai2.settings_store import SettingsStore

READINESS_TIMEOUT = 10


async def test_notices_wait_until_all_screen_owners_leave() -> None:
    output = io.StringIO()
    console = Console(file=output)
    screen = Screen()
    started = anyio.Event()
    finished = anyio.Event()

    async def notify() -> None:
        started.set()
        await screen.notify('[literal] notice', console=console)
        finished.set()

    async with anyio.create_task_group() as workers:
        with screen.busy():
            async with screen.full():
                workers.start_soon(notify)
                with anyio.fail_after(READINESS_TIMEOUT):
                    await started.wait()
                assert output.getvalue() == ''
            assert not finished.is_set()
        with anyio.fail_after(READINESS_TIMEOUT):
            await finished.wait()
    assert output.getvalue() == '[literal] notice\n'


async def test_closed_session_keeps_notices_blocked_until_unload() -> None:
    output = io.StringIO()
    console = Console(file=output)
    screen = Screen()
    started = anyio.Event()

    async def notify() -> None:
        started.set()
        await screen.notify('stale notice', console=console)

    with anyio.fail_after(READINESS_TIMEOUT):
        async with anyio.create_task_group() as tasks:
            with screen.busy():
                with screen.session():
                    tasks.start_soon(notify)
                    await started.wait()
                    await anyio.wait_all_tasks_blocked()
            await anyio.wait_all_tasks_blocked()
            assert output.getvalue() == ''
            tasks.cancel_scope.cancel()
    with screen.session():
        await screen.notify('fresh notice', console=console)
    assert output.getvalue() == 'fresh notice\n'


async def test_exit_discards_a_notice_that_becomes_ready_during_goodbye(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = anyio.Event()
    before = asyncio.all_tasks()

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if 'Goodbye.' in text:
                ready.set()
            return super().write(text)

    async def check() -> Version | None:
        await ready.wait()
        return Version('999999')

    monkeypatch.setattr(updates, 'latest_version', check)
    output = Output()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\n')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'config.db'),
            builtin_plugins=[PluginSettings(id='updates', factory='pydantic_clai2.updates')],
        )
    assert ready.is_set()
    assert 'Goodbye.' in output.getvalue()
    assert 'available (installed' not in output.getvalue()
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('terminal', [False, True])
async def test_shell_defers_plugin_notice_until_command_returns(tmp_path: Path, terminal: bool) -> None:
    before = asyncio.all_tasks()
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'notifier.py').write_text(
        'import asyncio\n'
        'from pydantic_clai2.commands import Command\n'
        'def activate(host):\n'
        '    async def command(args):\n'
        '        started = asyncio.Event()\n'
        '        async def notice():\n'
        '            started.set()\n'
        "            await host.notify('background notice')\n"
        '        task = asyncio.create_task(notice())\n'
        '        tasks.append(task)\n'
        '        await started.wait()\n'
        "        host.console.print('menu closed')\n"
        "        return 'command complete'\n"
        '    tasks = []\n'
        "    @host.on('session_end')\n"
        '    async def end(event):\n'
        '        for task in tasks:\n'
        '            task.cancel()\n'
        '        await asyncio.gather(*tasks, return_exceptions=True)\n'
        "    host.commands.register(Command(name='notice', description='test', handler=command))\n"
    )
    text = io.StringIO()
    console = Console(file=text, force_terminal=terminal, color_system='truecolor')
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        anyio.fail_after(READINESS_TIMEOUT),
    ):
        async with anyio.create_task_group() as workers:

            async def run() -> None:
                await chat(Agent(TestModel()), deps=None, store=store, console=console)

            workers.start_soon(run)
            pipe.send_text('/notice\n')
            await anyio.wait_all_tasks_blocked()
            assert text.getvalue().index('command complete') < text.getvalue().index('background notice')
            pipe.send_text('/exit\n')
    assert asyncio.all_tasks() == before


async def test_startup_resume_defers_update_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started = anyio.Event()
    checked = anyio.Event()
    output = io.StringIO()

    async def check() -> Version:
        await started.wait()
        checked.set()
        return Version('99999')

    async def resume(self: Sessions[None, str], args: list[str]) -> str:
        assert args == []
        started.set()
        await checked.wait()
        assert 'available (installed' not in output.getvalue()
        return 'browser closed'

    monkeypatch.setattr(updates, 'latest_version', check)
    monkeypatch.setattr(Sessions, 'command', resume)
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        anyio.fail_after(READINESS_TIMEOUT),
    ):
        async with anyio.create_task_group() as workers:

            async def run() -> None:
                await chat(
                    Agent(TestModel()),
                    deps=None,
                    store=SettingsStore(tmp_path / 'config.db'),
                    console=Console(file=output, width=300),
                    builtin_plugins=[PluginSettings(id='updates', factory='pydantic_clai2.updates')],
                    resume='',
                )

            workers.start_soon(run)
            await anyio.wait_all_tasks_blocked()
            text = output.getvalue()
            assert text.index('browser closed') < text.index('CLAI2 99999 available')
            pipe.send_text('/exit\n')


async def test_notice_preserves_live_editor_draft() -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system='truecolor')
    screen = Screen()
    before = asyncio.all_tasks()
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        anyio.fail_after(READINESS_TIMEOUT),
    ):
        prompt = PromptSession[str]()
        live = LivePrompt(prompt, console, prepare=lambda: None, interrupts=Interrupts())
        screen.editor = live.suspended
        drafted = anyio.Event()

        def changed(buffer: Buffer) -> None:
            if buffer.text == 'unfinished draft':
                drafted.set()

        prompt.default_buffer.on_text_changed += changed
        async with live.opened():
            pipe.send_text('unfinished draft')
            await drafted.wait()
            await screen.notify('background notice', console=console)
            await live.output.drain()
            assert 'background notice' in output.getvalue()
            assert prompt.default_buffer.text == 'unfinished draft'
            assert prompt.app.is_running
            pipe.send_text('\n')
            assert await live.read() == 'unfinished draft'
    assert asyncio.all_tasks() == before
