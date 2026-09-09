"""Events emitted by the Shell capability."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest
from pydantic_ai import Agent, RunCancelled, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import CapabilityEvent, ModelMessage, RetryPromptPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRun

from pydantic_ai_harness.shell import (
    MAX_EVENT_LINE_CHARS,
    Shell,
    ShellCommandEndEvent,
    ShellCommandRequestEvent,
    ShellCommandStartEvent,
    ShellOutputLineEvent,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _tool_results(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, (RetryPromptPart, ToolReturnPart))
    ]


def _calls_model(calls: Sequence[tuple[str, str]]) -> FunctionModel:
    """Issue each `(tool_name, json_args)` on its own step, then finish."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        results = _tool_results(messages)
        step = len(results)
        if step < len(calls):
            name, json_args = calls[step]
            if '$ID' in json_args:
                command_id = next(result.split('ID: ')[1].strip() for result in results if 'ID: ' in result)
                json_args = json_args.replace('$ID', command_id)
            yield {0: DeltaToolCall(name=name, json_args=json_args, tool_call_id=f'call_{step}')}
        else:
            yield 'done'

    return FunctionModel(stream_function=stream)


@dataclass
class Listener(AbstractCapability[None]):
    """Records every shell event and applies a scripted decision to requests."""

    decision: str | None = None
    rewrite_to: str | None = None
    events: list[CapabilityEvent] = field(default_factory=list[CapabilityEvent])

    @on_event(ShellCommandRequestEvent)
    async def _on_request(self, ctx: RunContext[None], event: ShellCommandRequestEvent) -> None:
        self.events.append(event)
        if self.decision == 'cancel':
            event.cancel('the user said no')
        elif self.decision == 'cancel_silently':
            event.cancel()
        elif self.rewrite_to is not None:
            event.rewrite(self.rewrite_to, reason='proxy')

    @on_event(ShellCommandStartEvent, ShellOutputLineEvent, ShellCommandEndEvent)
    async def _on_lifecycle(
        self, ctx: RunContext[None], event: ShellCommandStartEvent | ShellOutputLineEvent | ShellCommandEndEvent
    ) -> None:
        self.events.append(event)


async def _run(
    tmp_path: Path,
    calls: Sequence[tuple[str, str]],
    *,
    listener: Listener | None = None,
    max_output_chars: int = 50_000,
) -> tuple[Listener, list[str]]:
    listener = listener or Listener()
    shell = Shell[None](cwd=tmp_path, denied_commands=[], id='shell', max_output_chars=max_output_chars)
    result = await Agent(_calls_model(calls), deps_type=type(None), capabilities=[shell, listener]).run('go')
    return listener, _tool_results(result.all_messages())


def _run_command(command: str, **extra: float) -> tuple[str, str]:
    return 'run_command', json.dumps({'command': command, **extra})


class TestForegroundEvents:
    async def test_request_start_lines_end_in_order(self, tmp_path: Path) -> None:
        listener, results = await _run(tmp_path, [_run_command('echo one; echo two >&2; echo three')])

        request, start, *lines, end = listener.events
        assert isinstance(request, ShellCommandRequestEvent)
        assert request == ShellCommandRequestEvent(
            command='echo one; echo two >&2; echo three',
            cwd=str(tmp_path.resolve()),
            timeout=30.0,
            background=False,
            capability_id='shell',
            tool_call_id='call_0',
            tool_name='run_command',
        )
        assert isinstance(start, ShellCommandStartEvent)
        assert start.command == request.command
        assert start.cwd == request.cwd
        assert start.timeout == 30.0
        assert start.background is False
        assert start.pid > 0

        assert sorted((line.stream, line.line) for line in lines if isinstance(line, ShellOutputLineEvent)) == [
            ('stderr', 'two'),
            ('stdout', 'one'),
            ('stdout', 'three'),
        ]
        assert all(isinstance(line, ShellOutputLineEvent) and line.command_id == start.command_id for line in lines)

        assert isinstance(end, ShellCommandEndEvent)
        assert end.command_id == start.command_id
        assert end.command == request.command
        assert end.background is False
        assert end.exit_code == 0
        assert end.timed_out is False
        assert end.duration_seconds > 0
        assert (end.stdout, end.stderr, end.truncated) == ('one\nthree\n', 'two\n', False)
        assert results == ['[stdout]\none\nthree\n\n[stderr]\ntwo\n']

    async def test_stdin_is_closed_so_a_reader_exits_at_once(self, tmp_path: Path) -> None:
        listener, results = await _run(tmp_path, [_run_command('cat', timeout_seconds=5)])

        end = listener.events[-1]
        assert isinstance(end, ShellCommandEndEvent)
        assert end.exit_code == 0
        assert end.duration_seconds < 5
        assert results == ['(no output)']

    async def test_timeout_ends_with_timed_out(self, tmp_path: Path) -> None:
        listener, results = await _run(tmp_path, [_run_command('echo partial; sleep 5', timeout_seconds=0.3)])

        end = listener.events[-1]
        assert isinstance(end, ShellCommandEndEvent)
        assert end.timed_out is True
        assert end.exit_code != 0
        assert end.stdout == 'partial\n'
        assert results == ['[Command timed out after 0.3s]']

    async def test_long_line_is_cut_for_the_event_but_not_the_model(self, tmp_path: Path) -> None:
        width = MAX_EVENT_LINE_CHARS * 20
        command = f'{sys.executable} -c "print(\'x\' * {width})"'
        listener, results = await _run(tmp_path, [_run_command(command)], max_output_chars=width * 2)

        [line] = [event for event in listener.events if isinstance(event, ShellOutputLineEvent)]
        assert line == ShellOutputLineEvent(
            command_id=line.command_id,
            stream='stdout',
            line='x' * MAX_EVENT_LINE_CHARS,
            truncated=True,
            capability_id='shell',
            tool_call_id='call_0',
            tool_name='run_command',
        )
        assert results == [f'[stdout]\n{"x" * width}\n']

    async def test_end_tails_follow_the_output_cap(self, tmp_path: Path) -> None:
        listener, _ = await _run(tmp_path, [_run_command('seq 1 200')], max_output_chars=40)

        end = listener.events[-1]
        assert isinstance(end, ShellCommandEndEvent)
        assert end.truncated is True
        assert len(end.stdout) <= 40
        assert end.stdout.endswith('199\n200\n')

    async def test_carriage_returns_are_stripped_from_lines(self, tmp_path: Path) -> None:
        listener, _ = await _run(tmp_path, [_run_command("printf 'a\\r\\nb'")])

        assert [event.line for event in listener.events if isinstance(event, ShellOutputLineEvent)] == ['a', 'b']

    async def test_denied_command_emits_no_request(self, tmp_path: Path) -> None:
        listener, results = await _run(tmp_path, [_run_command('vim file')])

        assert listener.events == []
        assert results == ["Interactive commands are not allowed. Command: 'vim file'"]


class TestRequestDecisions:
    async def test_cancel_returns_the_reason_to_the_model(self, tmp_path: Path) -> None:
        marker = tmp_path / 'ran'
        listener, results = await _run(
            tmp_path, [_run_command(f'touch {marker}')], listener=Listener(decision='cancel')
        )

        assert not marker.exists()
        assert [type(event) for event in listener.events] == [ShellCommandRequestEvent]
        assert results == ['[Command was not run: the user said no]']

    async def test_cancel_without_a_reason(self, tmp_path: Path) -> None:
        _, results = await _run(tmp_path, [_run_command('echo hi')], listener=Listener(decision='cancel_silently'))

        assert results == ['[Command was not run: cancelled by a listener]']

    async def test_rewrite_runs_the_new_command_and_tells_the_model(self, tmp_path: Path) -> None:
        listener, results = await _run(
            tmp_path, [_run_command('echo original')], listener=Listener(rewrite_to='echo rewritten')
        )

        start = listener.events[1]
        assert isinstance(start, ShellCommandStartEvent)
        assert start.command == 'echo rewritten'
        assert results == ['[Command rewritten (proxy): echo rewritten]\n[stdout]\nrewritten\n']

    async def test_rewrite_is_checked_against_the_policy(self, tmp_path: Path) -> None:
        listener, results = await _run(tmp_path, [_run_command('echo hi')], listener=Listener(rewrite_to='vim x'))

        assert [type(event) for event in listener.events] == [ShellCommandRequestEvent]
        assert results == ["Interactive commands are not allowed. Command: 'vim x'"]


@dataclass
class CancelOnStart(AbstractCapability[None]):
    """Cancels the run as soon as the command is running, the way a host's Ctrl+C does."""

    run: AgentRun[None, str] | None = None
    pid: int | None = None

    @on_event(ShellCommandStartEvent)
    async def _on_start(self, ctx: RunContext[None], event: ShellCommandStartEvent) -> None:
        assert self.run is not None
        self.pid = event.pid
        self.run.cancel()


async def _process_group_is_gone(pgid: int) -> bool:
    with anyio.move_on_after(3):
        while True:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return True
            await anyio.sleep(0.05)
    return False


class TestCancellation:
    async def test_cancelling_the_run_kills_the_whole_process_group(self, tmp_path: Path) -> None:
        shell = Shell[None](cwd=tmp_path, denied_commands=[], id='shell')
        canceller = CancelOnStart()
        agent = Agent(
            _calls_model([_run_command('sleep 30; echo never')]), deps_type=type(None), capabilities=[shell, canceller]
        )
        started = time.monotonic()
        with pytest.raises(RunCancelled):
            async with agent.iter('go') as run:
                canceller.run = run
                async for _ in run:
                    pass

        assert canceller.pid is not None
        assert time.monotonic() - started < 10
        # The shell is the group leader; `sleep` is its child and would
        # outlive a kill aimed at the shell alone.
        assert await _process_group_is_gone(canceller.pid)


class TestBackgroundEvents:
    async def test_start_check_stop(self, tmp_path: Path) -> None:
        listener, results = await _run(
            tmp_path,
            [
                ('start_command', '{"command": "echo bg; sleep 30"}'),
                ('check_command', '{"command_id": "$ID"}'),
                ('stop_command', '{"command_id": "$ID"}'),
            ],
        )

        request, start, end = listener.events
        assert isinstance(request, ShellCommandRequestEvent)
        assert (request.command, request.timeout, request.background) == ('echo bg; sleep 30', None, True)
        assert isinstance(start, ShellCommandStartEvent)
        assert (start.timeout, start.background) == (None, True)
        assert isinstance(end, ShellCommandEndEvent)
        assert end.command_id == start.command_id
        assert end.background is True
        assert end.timed_out is False
        assert end.exit_code not in (None, 0)
        assert end.stdout == 'bg\n'
        assert results[2].endswith('[stopped]\n[exit code: -15]')

    async def test_check_emits_end_once_when_it_sees_the_exit(self, tmp_path: Path) -> None:
        listener, _ = await _run(
            tmp_path,
            [
                ('start_command', '{"command": "true"}'),
                ('run_command', '{"command": "sleep 0.3"}'),
                ('check_command', '{"command_id": "$ID"}'),
                ('check_command', '{"command_id": "$ID"}'),
                ('stop_command', '{"command_id": "$ID"}'),
            ],
        )

        ends = [event for event in listener.events if isinstance(event, ShellCommandEndEvent) and event.background]
        assert len(ends) == 1
        assert ends[0].exit_code == 0

    async def test_cancelled_background_request(self, tmp_path: Path) -> None:
        listener, results = await _run(
            tmp_path, [('start_command', '{"command": "sleep 30"}')], listener=Listener(decision='cancel')
        )

        assert [type(event) for event in listener.events] == [ShellCommandRequestEvent]
        assert results == ['[Command was not run: the user said no]']
