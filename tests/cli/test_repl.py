from __future__ import annotations

import asyncio
import io
import os
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from termflow.ansi import visible  # pyright: ignore[reportMissingTypeStubs]

from pydantic_ai_harness.cli import CliBridge, Lines, Repl

pytestmark = pytest.mark.anyio


def _user_prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


@dataclass
class _Harness:
    """A REPL over an agent whose first model step blocks in a tool until `release` is set."""

    repl: Repl
    buffer: io.StringIO
    entered_tool: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    requests: list[list[str]] = field(default_factory=list[list[str]])

    def transcript(self) -> str:
        return visible(self.buffer.getvalue())

    async def type(self, line: str | None, *, at_prompt: int) -> None:
        """Push `line` once the REPL shows its `at_prompt`-th prompt, i.e. once it is idle again."""
        while self.transcript().count(self.repl.prompt) < at_prompt:
            await asyncio.sleep(0)
        self.repl.lines.push(line)


def _harness() -> _Harness:
    buffer = io.StringIO()
    harness: _Harness | None = None

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert harness is not None
        harness.requests.append(_user_prompts(messages))
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='wait', json_args='{}')}
        else:
            yield f'saw {_user_prompts(messages)}'

    agent: Agent[None, str] = Agent(capabilities=[CliBridge(output=buffer, width=80)])

    @agent.tool_plain
    async def wait() -> str:
        """Block until the test releases it."""
        assert harness is not None
        harness.entered_tool.set()
        await harness.release.wait()
        return 'released'

    repl = Repl(agent=agent, model=FunctionModel(stream_function=model), output=buffer)
    harness = _Harness(repl=repl, buffer=buffer)
    return harness


class TestRepl:
    async def test_prompts_run_in_order_and_share_history(self) -> None:
        harness = _harness()
        harness.release.set()

        async def type_prompts() -> None:
            await harness.type('one', at_prompt=1)
            await harness.type('   ', at_prompt=2)
            await harness.type('two', at_prompt=3)
            await harness.type(None, at_prompt=4)

        await asyncio.gather(harness.repl.run(), type_prompts())

        assert harness.requests == [['one'], ['one'], ['one', 'two']]
        assert harness.transcript() == (
            "harness> > wait {}\n< wait released\nsaw ['one']\nharness> harness> saw ['one', 'two']\nharness> \n"
        )

    async def test_ctrl_c_cancels_the_run_and_keeps_its_history(self) -> None:
        harness = _harness()

        async def press_ctrl_c_then_continue() -> None:
            await harness.type('start', at_prompt=1)
            await harness.entered_tool.wait()
            os.kill(os.getpid(), signal.SIGINT)
            await harness.type('after', at_prompt=2)
            await harness.type(None, at_prompt=3)

        await asyncio.gather(harness.repl.run(), press_ctrl_c_then_continue())

        assert harness.transcript() == "harness> > wait {}\n(cancelled)\nharness> saw ['start', 'after']\nharness> \n"
        assert harness.requests == [['start'], ['start', 'after']]
        assert isinstance(harness.repl.history[-1], ModelResponse)

    async def test_line_typed_during_a_run_steers_it(self) -> None:
        harness = _harness()

        async def type_meanwhile() -> None:
            await harness.entered_tool.wait()
            harness.repl.lines.push('')
            harness.repl.lines.push('also do X')
            await asyncio.sleep(0)
            harness.release.set()

        await asyncio.gather(harness.repl.run_once('start'), type_meanwhile())

        assert harness.requests == [['start'], ['start', 'also do X']]
        assert (
            harness.transcript()
            == "> wait {}\n(steer queued: also do X)\n< wait released\nsaw ['start', 'also do X']\n"
        )

    async def test_end_of_input_during_a_run_ends_the_session_after_it(self) -> None:
        harness = _harness()

        async def close_input() -> None:
            await harness.type('start', at_prompt=1)
            await harness.entered_tool.wait()
            harness.repl.lines.push(None)
            await asyncio.sleep(0)
            harness.release.set()

        await asyncio.gather(harness.repl.run(), close_input())

        assert harness.requests == [['start'], ['start']]
        assert harness.transcript() == "harness> > wait {}\n< wait released\nsaw ['start']\nharness> \n"

    async def test_ctrl_c_while_idle_starts_a_fresh_prompt_line(self) -> None:
        buffer = io.StringIO()
        agent: Agent[None, str] = Agent(TestModel(), capabilities=[CliBridge(output=buffer, width=80)])
        lines = Lines()
        repl = Repl(agent=agent, model='test', lines=lines, output=buffer)
        session = asyncio.create_task(repl.run())
        await asyncio.sleep(0)

        repl.interrupt()
        lines.push(None)
        await session

        assert buffer.getvalue() == 'harness> \nharness> \n'


class TestLines:
    async def test_from_stdin_pumps_lines_then_end_of_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('sys.stdin', io.StringIO('first\nsecond\n'))

        lines = Lines.from_stdin()

        assert [await lines.read() for _ in range(3)] == ['first', 'second', None]
