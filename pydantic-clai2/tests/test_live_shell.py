"""Exercise the interactive shell with a real editor and a deterministic model."""

import asyncio
import io
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from rich.console import Console

from pydantic_clai2 import chat
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_steering_and_queued_turn_use_distinct_requests(tmp_path: Path) -> None:
    requests: list[list[str]] = []
    edited = asyncio.Event()
    streaming = asyncio.Event()
    release = asyncio.Event()

    def text_changed(buffer: Buffer) -> None:
        if buffer.text == 'draft':
            edited.set()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        texts = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        ]
        requests.append(texts)
        if len(requests) == 1:
            get_app().current_buffer.on_text_changed += text_changed
            yield 'first '
            streaming.set()
            await release.wait()
        yield 'answer\n'

    output = io.StringIO()
    with create_pipe_input() as input, create_app_session(input=input, output=DummyOutput()):
        task = asyncio.create_task(
            chat(
                Agent(FunctionModel(stream_function=stream)),
                deps=None,
                console=Console(file=output, force_terminal=True, width=80, height=24),
                store=SettingsStore(tmp_path / 'settings.db'),
            )
        )
        input.send_text('original\r')
        with anyio.fail_after(10):
            await streaming.wait()
            input.send_text('correction\rnext turn\x1b\r/exit\rdraft')
            await edited.wait()
            release.set()
            await task
    assert requests == [
        ['original'],
        ['original', 'correction'],
        ['original', 'correction', 'next turn'],
    ]
    assert '> next turn' in output.getvalue()
    assert 'Goodbye.' in output.getvalue()


async def test_cancellation_clears_queue_before_returning_to_idle(tmp_path: Path) -> None:
    streaming = asyncio.Event()
    edited = asyncio.Event()
    cancelled = asyncio.Event()
    requests = 0

    class Output(io.StringIO):
        def write(self, text: str) -> int:
            if 'Turn cancelled.' in text:
                cancelled.set()
            return super().write(text)

    def text_changed(buffer: Buffer) -> None:
        if buffer.text == 'draft':
            edited.set()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal requests
        requests += 1
        get_app().current_buffer.on_text_changed += text_changed
        yield 'partial '
        streaming.set()
        await asyncio.Event().wait()

    output = Output()
    with create_pipe_input() as input, create_app_session(input=input, output=DummyOutput()):
        task = asyncio.create_task(
            chat(
                Agent(FunctionModel(stream_function=stream)),
                deps=None,
                console=Console(file=output, force_terminal=True, width=80, height=24),
                store=SettingsStore(tmp_path / 'settings.db'),
            )
        )
        input.send_text('original\r')
        with anyio.fail_after(10):
            await streaming.wait()
            input.send_text('must not run\x1b\rdraft')
            await edited.wait()
            input.send_text('\x03')
            await cancelled.wait()
            input.send_text('\x15/exit\r')
            await task
    assert requests == 1
    assert 'Queued submissions cleared after cancellation.' in output.getvalue()
    assert 'Goodbye.' in output.getvalue()
