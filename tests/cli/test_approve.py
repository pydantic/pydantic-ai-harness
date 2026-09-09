from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from termflow.ansi import DIM_ON, visible  # pyright: ignore[reportMissingTypeStubs]

from pydantic_ai_harness.cli import (
    Approver,
    CliBridge,
    CliDeps,
    DeclineAll,
    Lines,
    Repl,
    TerminalApprover,
    Verdict,
    allow_all,
)
from pydantic_ai_harness.shell import Shell, ShellCommandRequestEvent

pytestmark = pytest.mark.anyio


def _tool_results(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, (RetryPromptPart, ToolReturnPart))
    ]


def _shell_model(*commands: str) -> FunctionModel:
    """Run each command on its own step, then answer `done`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        step = len(_tool_results(messages))
        if step < len(commands):
            args = json.dumps({'command': commands[step]})
            yield {0: DeltaToolCall(name='run_command', json_args=args, tool_call_id=f'call_{step}')}
        else:
            yield 'done'

    return FunctionModel(stream_function=stream)


def _agent(tmp_path: Path, bridge: CliBridge[CliDeps], *commands: str) -> Agent[CliDeps, str]:
    shell = Shell[CliDeps](cwd=tmp_path, denied_commands=[])
    return Agent(_shell_model(*commands), deps_type=CliDeps, capabilities=[shell, bridge])


class TestShellRendering:
    async def test_allowed_command_renders_start_lines_and_exit(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=allow_all)
        agent = _agent(tmp_path, bridge, 'echo out; echo err >&2')

        result = await agent.run('go', deps=CliDeps(approver=allow_all))

        assert result.output == 'done'
        lines = visible(buffer.getvalue()).splitlines()
        assert lines[0] == '> run_command {"command": "echo out; echo err >&2"}'
        assert lines[1] == '$ echo out; echo err >&2'
        assert sorted(lines[2:4]) == ['err', 'out']
        assert lines[4].startswith('exit 0 (')
        assert lines[5] == 'done'
        assert not any(line.startswith('< run_command') for line in lines)
        assert f'{DIM_ON}err' in buffer.getvalue()

    async def test_declined_command_is_cancelled_with_the_reason(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=DeclineAll(reason='not today'))
        agent = _agent(tmp_path, bridge, 'echo never')

        result = await agent.run('go', deps=CliDeps(approver=allow_all))

        assert _tool_results(result.all_messages()) == ['[Command was not run: not today]']
        assert visible(buffer.getvalue()) == (
            '> run_command {"command": "echo never"}\n< run_command [Command was not run: not today]\ndone\n'
        )

    async def test_approver_comes_from_deps_when_the_bridge_has_none(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        agent = _agent(tmp_path, CliBridge[CliDeps](output=buffer, width=80), 'echo hi')

        result = await agent.run('go', deps=CliDeps(approver=allow_all))

        assert _tool_results(result.all_messages()) == ['[stdout]\nhi\n']

    async def test_without_any_approver_every_request_is_declined(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        shell = Shell[None](cwd=tmp_path, denied_commands=[])
        agent = Agent(_shell_model('echo hi'), deps_type=type(None), capabilities=[shell, CliBridge(output=buffer)])

        result = await agent.run('go')

        (outcome,) = _tool_results(result.all_messages())
        assert outcome.startswith('[Command was not run: nobody can approve this')

    async def test_timed_out_and_truncated_commands_say_so(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        bridge = CliBridge[CliDeps](output=buffer, width=80, approver=allow_all)
        shell = Shell[CliDeps](cwd=tmp_path, denied_commands=[], max_output_chars=20)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            step = len(_tool_results(messages))
            if step == 0:
                yield {0: DeltaToolCall(name='run_command', json_args=json.dumps({'command': 'seq 1 100'}))}
            elif step == 1:
                args = json.dumps({'command': 'sleep 5', 'timeout_seconds': 0.2})
                yield {0: DeltaToolCall(name='run_command', json_args=args)}
            else:
                yield 'done'

        agent = Agent(FunctionModel(stream_function=stream), deps_type=CliDeps, capabilities=[shell, bridge])

        await agent.run('go', deps=CliDeps(approver=allow_all))

        lines = visible(buffer.getvalue()).splitlines()
        assert any(line.startswith('exit 0 (') and line.endswith('s, output truncated)') for line in lines)
        assert any(line.startswith('timed out (') for line in lines)


class TestLinesAsk:
    async def test_a_pushed_line_answers_a_pending_question_before_readers(self) -> None:
        lines = Lines()
        reader = asyncio.create_task(lines.read())
        asker = asyncio.create_task(lines.ask())
        await asyncio.sleep(0)

        lines.push('y')
        lines.push('next prompt')

        assert await asker == 'y'
        assert await reader == 'next prompt'

    async def test_end_of_input_answers_and_still_ends_the_session(self) -> None:
        lines = Lines()
        asker = asyncio.create_task(lines.ask())
        await asyncio.sleep(0)

        lines.push(None)

        assert await asker is None
        assert await lines.read() is None

    async def test_one_question_at_a_time(self) -> None:
        lines = Lines()
        first = asyncio.create_task(lines.ask())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match='already waiting'):
            await lines.ask()

        lines.push('y')
        assert await first == 'y'


class TestTerminalApprover:
    @pytest.mark.parametrize(
        ('answer', 'allowed'), [('y', True), ('YES', True), ('n', False), ('', False), (None, False)]
    )
    async def test_only_yes_allows(self, answer: str | None, allowed: bool) -> None:
        lines = Lines()
        buffer = io.StringIO()
        approver: Approver = TerminalApprover(answers=lines, output=buffer)
        pending = asyncio.create_task(approver(_event(), description='run rm -rf build'))
        await asyncio.sleep(0)

        lines.push(answer)
        verdict = await pending

        assert verdict == (Verdict(allowed=True) if allowed else Verdict(allowed=False, reason='declined by the user'))
        assert visible(buffer.getvalue()) == '? run rm -rf build [y/N] '

    async def test_repl_asks_on_the_terminal_and_the_answer_is_not_a_steer(self, tmp_path: Path) -> None:
        buffer = io.StringIO()
        lines = Lines()
        shell = Shell[CliDeps](cwd=tmp_path, denied_commands=[])
        agent = Agent(deps_type=CliDeps, capabilities=[shell, CliBridge[CliDeps](output=buffer, width=80)])
        repl = Repl(agent=agent, model=_shell_model('echo approved'), lines=lines, output=buffer)

        async def answer_then_quit() -> None:
            lines.push('go')
            while '[y/N]' not in visible(buffer.getvalue()):
                await asyncio.sleep(0)
            lines.push('y')
            while 'done' not in visible(buffer.getvalue()):
                await asyncio.sleep(0)
            lines.push(None)

        await asyncio.gather(repl.run(), answer_then_quit())

        transcript = visible(buffer.getvalue())
        assert '? run echo approved [y/N] ' in transcript
        assert '$ echo approved\napproved\nexit 0 (' in transcript
        assert 'steer queued' not in transcript


def _event() -> ShellCommandRequestEvent:
    return ShellCommandRequestEvent(command='rm -rf build', cwd='/tmp', timeout=30.0, background=False)
