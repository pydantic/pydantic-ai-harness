from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaThinkingCalls, DeltaThinkingPart, FunctionModel
from pydantic_ai.models.test import TestModel
from termflow.ansi import visible  # pyright: ignore[reportMissingTypeStubs]

from pydantic_ai_harness.cli import CliBridge
from pydantic_ai_harness.coder import Coder

pytestmark = pytest.mark.anyio


def _transcript(buffer: io.StringIO) -> str:
    return visible(buffer.getvalue())


class TestCliBridge:
    async def test_renders_markdown_and_tool_calls(self, tmp_path: Path) -> None:
        (tmp_path / 'notes.txt').write_text('hello\n')
        buffer = io.StringIO()
        model = TestModel(call_tools=['list_directory'], custom_output_text='# Done\n\nListed **one** file.')
        agent = Agent(model, capabilities=[Coder(tmp_path), CliBridge(output=buffer, width=80)])

        result = await agent.run('what is here?')

        assert result.output == '# Done\n\nListed **one** file.'
        assert _transcript(buffer) == (
            '> list_directory {}\n< list_directory notes.txt  (6 bytes)\nDone\n════════\n\nListed one file.\n'
        )

    async def test_tool_result_is_one_bounded_line(self) -> None:
        buffer = io.StringIO()
        agent = Agent(TestModel(custom_output_text='ok\n'), capabilities=[CliBridge(output=buffer, width=40)])

        @agent.tool_plain
        def big() -> str:
            """A tool whose result would not fit on one line."""
            return 'x' * 60 + '\nsecond\nthird'

        await agent.run('go')

        lines = _transcript(buffer).splitlines()
        assert lines[0] == '> big {}'
        assert lines[1].startswith('< big xxxx')
        assert len(lines[1]) <= 40
        assert lines[2:] == ['ok']

    async def test_retry_prompt_is_rendered(self) -> None:
        buffer = io.StringIO()
        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[CliBridge(output=buffer, width=80)])
        calls = 0

        @agent.tool_plain
        def flaky() -> str:
            """Fails once."""
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ModelRetry('try again')
            return 'fine'

        await agent.run('go')

        assert _transcript(buffer) == '> flaky {}\n! flaky try again (+2 lines)\n> flaky {}\n< flaky fine\nok\n'

    async def test_thinking_is_not_rendered(self) -> None:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaThinkingCalls]:
            yield {0: DeltaThinkingPart(content='pondering')}
            yield {0: DeltaThinkingPart(content=' more')}
            yield 'visible '
            yield 'answer'

        buffer = io.StringIO()
        agent = Agent(FunctionModel(stream_function=stream), capabilities=[CliBridge(output=buffer, width=80)])

        result = await agent.run('go')

        assert result.output == 'visible answer'
        assert _transcript(buffer) == 'visible answer\n'

    async def test_output_defaults_to_stdout_at_run_start(self, capsys: pytest.CaptureFixture[str]) -> None:
        agent = Agent(TestModel(custom_output_text='to stdout'), capabilities=[CliBridge(width=80)])

        await agent.run('go')

        assert capsys.readouterr().out == 'to stdout\n'

    async def test_each_run_starts_clean(self) -> None:
        buffer = io.StringIO()
        agent = Agent(TestModel(custom_output_text='again'), capabilities=[CliBridge(output=buffer, width=80)])

        await agent.run('one')
        await agent.run('two')

        assert _transcript(buffer) == 'again\nagain\n'
