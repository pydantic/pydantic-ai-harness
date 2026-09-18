"""Real editor keys, steering delivery, and terminal handoffs without a provider."""

import asyncio
import io
from collections.abc import AsyncIterator

import anyio
import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Never
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import Session
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.plugins import PluginHost
from pydantic_clai2.prompt_output import PromptOutput
from pydantic_clai2.status import Status


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_editor_steers_queues_and_preserves_edited_draft() -> None:
    steered: list[str] = []
    started = asyncio.Event()
    edited = asyncio.Event()
    console = Console(file=io.StringIO())
    with create_pipe_input() as input:
        prompt = PromptSession[str](input=input, output=DummyOutput())
        live = LivePrompt(
            prompt=prompt,
            console=console,
            status=Status(),
            steer=lambda text: steered.append(text) is None,
            prepare=lambda: None,
            show_frame=Never(),
        )

        def text_changed(buffer: Buffer) -> None:
            if buffer.text == 'draft!':
                edited.set()

        prompt.default_buffer.on_text_changed += text_changed

        async def operation() -> bool:
            started.set()
            await edited.wait()
            return True

        task = asyncio.create_task(live.run(operation))
        await started.wait()
        input.send_text('\x04\rsteer this\rqueue this\x1b\r/plugins list\rdraft?\x7f!')
        with anyio.fail_after(5):
            assert await task
        assert steered == ['steer this']
        assert list(live.queue) == ['queue this', '/plugins list']
        assert live.draft.text == 'draft!'
        assert live.draft.cursor_position == 6
        input.send_text('\r')
        with anyio.fail_after(5):
            assert await prompt.prompt_async(default=live.draft) == 'draft!'
        assert steered == ['steer this']


async def test_late_steering_is_queued_instead_of_lost() -> None:
    started = asyncio.Event()
    submitted = asyncio.Event()
    with create_pipe_input() as input:
        prompt = PromptSession[str](input=input, output=DummyOutput())

        def steer(text: str) -> bool:
            submitted.set()
            return False

        live = LivePrompt(
            prompt=prompt,
            console=Console(file=io.StringIO()),
            status=Status(),
            steer=steer,
            prepare=lambda: None,
            show_frame=Never(),
        )

        async def operation() -> bool:
            started.set()
            await submitted.wait()
            return True

        task = asyncio.create_task(live.run(operation))
        await started.wait()
        input.send_text('late message\r')
        with anyio.fail_after(5):
            await task
        assert list(live.queue) == ['late message']


@pytest.mark.parametrize('external', [False, True])
async def test_cancel_restores_output_and_keeps_draft(external: bool) -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    edited = asyncio.Event()
    output = io.StringIO()
    console = Console(file=output)
    with create_pipe_input() as input:
        prompt = PromptSession[str](input=input, output=DummyOutput())
        live = LivePrompt(
            prompt=prompt,
            console=console,
            status=Status(),
            steer=lambda _: True,
            prepare=lambda: None,
            show_frame=Never(),
        )

        def text_changed(buffer: Buffer) -> None:
            edited.set()

        prompt.default_buffer.on_text_changed += text_changed

        async def operation() -> bool:
            try:
                started.set()
                await asyncio.Event().wait()
                return True
            finally:
                cleaned.set()

        async def run_editor() -> None:
            await live.run(operation)

        task = asyncio.create_task(Interrupts().run(run_editor()))
        await started.wait()
        input.send_text('draft')
        await edited.wait()
        if external:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            input.send_text('\x03')
            with anyio.fail_after(5):
                assert not await task
        assert cleaned.is_set()
        assert live.draft.text == 'draft'
        assert console.file is output


async def test_full_screen_releases_input_and_restores_draft() -> None:
    edited = asyncio.Event()
    menu_ready = asyncio.Event()
    answered = asyncio.Event()
    output = io.StringIO()
    console = Console(file=output)
    with create_pipe_input() as input:
        prompt = PromptSession[str](input=input, output=DummyOutput())
        live = LivePrompt(
            prompt=prompt,
            console=console,
            status=Status(),
            steer=lambda _: True,
            prepare=lambda: None,
            show_frame=Never(),
        )

        def text_changed(buffer: Buffer) -> None:
            edited.set()

        prompt.default_buffer.on_text_changed += text_changed
        answer: list[str] = []

        def menu_keys() -> None:
            answer.extend(key.data for key in input.read_keys())
            answered.set()

        async def operation() -> bool:
            input.send_text('draft')
            await edited.wait()
            async with live.paused():
                assert console.file is output
                with input.raw_mode(), input.attach(menu_keys):
                    menu_ready.set()
                    await answered.wait()
            assert console.file is not output
            return True

        task = asyncio.create_task(live.run(operation))
        await menu_ready.wait()
        input.send_text('menu answer')
        with anyio.fail_after(5):
            assert await task
        assert ''.join(answer) == 'menu answer'
        assert live.draft.text == 'draft'


@pytest.mark.parametrize('before_stream', [False, True])
async def test_session_steering_reaches_core_and_history(before_stream: bool) -> None:
    streaming = asyncio.Event()
    release = asyncio.Event()
    requests: list[list[str]] = []

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
            yield 'initial '
            streaming.set()
            await release.wait()
        yield 'answer'

    model = FunctionModel(stream_function=stream)
    session = Session(Agent(model), deps=None)
    resolving = asyncio.Event()
    resolved = asyncio.Event()

    async def resolve(name: str) -> FunctionModel:
        resolving.set()
        await resolved.wait()
        return model

    if before_stream:
        session.model = 'test'
        session.resolve_model = resolve
    assert not session.steer('no run')
    task = asyncio.create_task(session.prompt('original'))
    await (resolving if before_stream else streaming).wait()
    assert session.steer('correction')
    resolved.set()
    await streaming.wait()
    release.set()
    with anyio.fail_after(5):
        await task
    assert requests == [['original'], ['original', 'correction']]
    assert not session.steer('already finished')
    assert any(
        isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) and p.content == 'correction' for p in m.parts)
        for m in session.messages
    )


async def test_model_request_guard_blocks_steering_before_provider_call() -> None:
    streaming = asyncio.Event()
    release = asyncio.Event()
    requests: list[list[ModelMessage]] = []
    plugin = PluginHost[None](name='input-guard', console=Console(file=io.StringIO()), settings={})

    @plugin.on('before_model_request')
    async def guard(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        if any(
            isinstance(part, UserPromptPart) and part.content == 'blocked steering'
            for message in request_context.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            raise ValueError('Input rejected by guard')
        return request_context

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        requests.append(list(messages))
        yield 'initial '
        streaming.set()
        await release.wait()
        yield 'answer'

    session = Session(Agent(FunctionModel(stream_function=stream)), deps=None, plugins=plugin.capabilities)
    task = asyncio.create_task(session.prompt('allowed input'))
    with anyio.fail_after(5):
        await streaming.wait()
        assert session.steer('blocked steering')
        release.set()
        with pytest.raises(ValueError, match='Input rejected by guard'):
            await task
    assert len(requests) == 1
    assert not session.steer('after failed run')


async def test_rejected_steering_is_returned_to_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    agent = Agent(TestModel(call_tools=['wait'], custom_output_text='done'))

    @agent.tool_plain
    async def wait() -> str:
        started.set()
        await finish.wait()
        return 'done'

    session = Session(agent, deps=None)
    task = asyncio.create_task(session.prompt('original'))
    await started.wait()

    def closed(ctx: RunContext[None], text: str, *, priority: str) -> None:
        raise UserError('Run queue closed')

    monkeypatch.setattr(RunContext, 'enqueue', closed)
    assert not session.steer('late correction')
    assert not session.steer('another correction')
    finish.set()
    with anyio.fail_after(5):
        await task


async def test_output_buffers_lines_and_drains_worker_writes() -> None:
    class Output(DummyOutput):
        def write_raw(self, data: str) -> None:
            written.append(data)

    written: list[str] = []
    output = PromptOutput(Output())
    assert output.write('part') == 4
    output.flush()
    assert written == []
    await asyncio.to_thread(output.write, 'ial\nlast')
    await output.drain()
    assert ''.join(written) == 'partial\nlast'
    assert not output.isatty()
