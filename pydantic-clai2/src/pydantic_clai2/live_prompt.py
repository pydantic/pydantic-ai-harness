"""An editable prompt during a turn, with steering and between-turn submissions."""

import signal
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import anyio
from prompt_toolkit import PromptSession
from prompt_toolkit.application import in_terminal
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Filter
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from rich.console import Console

from .prompt_output import PromptOutput
from .status import Status


class LivePrompt:
    """Keep draft editing independent of the agent task and its streamed output."""

    def __init__(
        self,
        *,
        prompt: PromptSession[str],
        console: Console,
        status: Status,
        steer: Callable[[str], bool],
        prepare: Callable[[], None],
        show_frame: Filter,
    ) -> None:
        """Bind editing to a session's steering callback, not to its agent loop."""
        self.prompt = prompt
        self.console = console
        self.status = status
        self.steer = steer
        self.prepare = prepare
        self.show_frame = show_frame
        self.draft = Document()
        self.queue: deque[str] = deque()
        self._output: PromptOutput | None = None
        self._original = console.file
        self.keys = KeyBindings()

        @self.keys.add('enter')
        def steer_input(event: KeyPressEvent) -> None:
            self._submit(queue=False)

        @self.keys.add('escape', 'enter')
        def queue_input(event: KeyPressEvent) -> None:
            self._submit(queue=True)

        @self.keys.add('c-c')
        def interrupt(event: KeyPressEvent) -> None:
            signal.raise_signal(signal.SIGINT)

        @self.keys.add('c-d')
        def delete(event: KeyPressEvent) -> None:
            # EOF must not abandon a still-running agent.
            event.current_buffer.delete()

    def _submit(self, *, queue: bool) -> None:
        buffer = self.prompt.default_buffer
        text = buffer.text.strip()
        if not text:
            return
        buffer.append_to_history()
        buffer.reset()
        queued = queue or text.startswith('/') or not self.steer(text)
        if queued:
            self.queue.append(text)
        self.status.input_hint = f'{len(self.queue)} queued' if queued else 'Steering sent'
        self.prompt.app.invalidate()

    async def run(self, operation: Callable[[], Awaitable[bool]]) -> bool:
        """Run the agent inside the prompt's application context, then retain its draft."""
        completed = False
        self.status.input_hint = 'Enter: steer | Alt+Enter: queue'
        output = PromptOutput(self.prompt.output)
        self._output = output
        self._original = self.console.file
        self.console.file = output
        original_keys = self.prompt.key_bindings
        try:
            async with anyio.create_task_group() as tasks:

                async def work() -> None:
                    nonlocal completed
                    try:
                        completed = await operation()
                    finally:
                        with anyio.CancelScope(shield=True):
                            await output.drain()
                        app = self.prompt.app
                        if app.is_running and not app.is_done:
                            app.exit(result='')

                def start() -> None:
                    self.prepare()
                    self.prompt.app.erase_when_done = True
                    tasks.start_soon(work)

                try:
                    await self.prompt.prompt_async(
                        '> ',
                        default=self.draft,
                        key_bindings=self.keys,
                        show_frame=self.show_frame,
                        pre_run=start,
                        refresh_interval=0.1,
                        handle_sigint=False,
                    )
                finally:
                    self.draft = self.prompt.default_buffer.document
        finally:
            self.console.file = self._original
            self.prompt.key_bindings = original_keys
            self._output = None
            self.status.input_hint = ''
        return completed

    @asynccontextmanager
    async def paused(self) -> AsyncGenerator[None]:
        """Release both editor and output for a full-screen widget, preserving the draft."""
        assert self._output is not None
        await self._output.drain()
        async with in_terminal():
            self.console.file = self._original
            try:
                yield
            finally:
                self.console.file = self._output
