"""The interactive session: reads prompts, drives runs, and forwards Ctrl+C and steer messages."""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TextIO

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.run import AgentRun
from termflow.ansi import DIM_OFF, DIM_ON  # pyright: ignore[reportMissingTypeStubs]


class Lines:
    """Lines the user submits, in order. `read` returns `None` once input has ended.

    One reader serves the whole session: a line read while no run is in flight becomes the next
    prompt, and a line read during a run is enqueued into it as a steer message.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    def push(self, line: str | None) -> None:
        """Append a line, or `None` to mark the end of input. Call from the event loop's thread."""
        self._queue.put_nowait(line)

    async def read(self) -> str | None:
        return await self._queue.get()

    @classmethod
    def from_stdin(cls) -> Lines:
        """Pump `sys.stdin` from a daemon thread so lines arrive while a run is in flight."""
        lines = cls()
        loop = asyncio.get_running_loop()

        def pump() -> None:
            for raw in sys.stdin:
                loop.call_soon_threadsafe(lines.push, raw.rstrip('\n'))
            loop.call_soon_threadsafe(lines.push, None)

        threading.Thread(target=pump, name='harness-stdin', daemon=True).start()
        return lines


@contextmanager
def _sigint_calls(handler: Callable[[], None]) -> Generator[None]:
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, handler)
    try:
        yield
    finally:
        loop.remove_signal_handler(signal.SIGINT)


@dataclass(kw_only=True)
class Repl:
    """Drive `agent` from a terminal: prompts in, rendered runs out, Ctrl+C cancels, typing steers.

    Rendering is `CliBridge`'s job; the REPL only writes the prompt and its own status lines to
    `output`. Message history carries across prompts, including the partial history of a run
    that was cancelled.
    """

    agent: AbstractAgent[None, str]
    model: Model | KnownModelName | str
    output: TextIO
    lines: Lines = field(default_factory=Lines)
    """Where prompts and steer messages come from. `Lines.from_stdin()` for a terminal session."""
    prompt: str = 'harness> '
    """Written before each prompt is read."""
    history: list[ModelMessage] = field(default_factory=list[ModelMessage])
    """The conversation so far; each run starts from it and replaces it, a cancelled run included."""
    _run: AgentRun[None, object] | None = field(init=False, default=None, repr=False)

    async def run(self) -> None:
        """Read prompts until the end of input, running each one. Ctrl+C while idle re-shows the prompt."""
        with _sigint_calls(self.interrupt):
            while True:
                self._show_prompt()
                line = await self.lines.read()
                if line is None:
                    self.output.write('\n')
                    return
                if line.strip():
                    await self.submit(line)

    async def run_once(self, prompt: str) -> None:
        """Run one prompt and return. Ctrl+C cancels it; lines typed meanwhile steer it."""
        with _sigint_calls(self.interrupt):
            await self.submit(prompt)

    async def submit(self, prompt: str) -> None:
        """Run `prompt` against the current history, forwarding lines read meanwhile as steer messages."""
        try:
            async with self.agent.iter(prompt, model=self.model, message_history=self.history) as run:
                self._run = run
                steer = asyncio.create_task(self._steer(run))
                try:
                    async for _ in run:
                        pass
                finally:
                    self._run = None
                    steer.cancel()
                    await asyncio.gather(steer, return_exceptions=True)
                self.history = run.all_messages()
        except RunCancelled as exc:
            self.history = exc.all_messages()
            self._status('cancelled')

    def interrupt(self) -> None:
        """Ctrl+C: cancel the run in flight, or start a fresh prompt line when idle."""
        if self._run is not None:
            self._run.cancel()
        else:
            self.output.write('\n')
            self._show_prompt()

    async def _steer(self, run: AgentRun[None, object]) -> None:
        while True:
            line = await self.lines.read()
            if line is None:
                self.lines.push(None)
                return
            if line.strip():
                run.enqueue(line)
                self._status(f'steer queued: {line}')

    def _show_prompt(self) -> None:
        self.output.write(self.prompt)
        self.output.flush()

    def _status(self, text: str) -> None:
        self.output.write(f'{DIM_ON}({text}){DIM_OFF}\n')
        self.output.flush()
