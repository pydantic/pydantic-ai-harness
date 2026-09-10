"""Events emitted by the Shell capability."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest
import sniffio
from pydantic_ai import Agent, RunCancelled, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import CapabilityEvent, ModelMessage, ModelResponse, RetryPromptPart, ToolReturnPart
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
from pydantic_ai_harness.shell._process import kill_process_group

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


Call = tuple[str, str]


def _calls_model(steps: Sequence[Call | list[Call]]) -> FunctionModel:
    """Issue each `(tool_name, json_args)` on its own step, a list of them in parallel, then finish."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        results = _tool_results(messages)
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step >= len(steps):
            yield 'done'
            return
        calls = steps[step]
        calls = calls if isinstance(calls, list) else [calls]
        deltas: DeltaToolCalls = {}
        for index, (name, json_args) in enumerate(calls):
            if '$ID' in json_args:
                command_id = next(result.split('ID: ')[1].strip() for result in results if 'ID: ' in result)
                json_args = json_args.replace('$ID', command_id)
            tool_call_id = f'call_{step}' if len(calls) == 1 else f'call_{step}_{index}'
            deltas[index] = DeltaToolCall(name=name, json_args=json_args, tool_call_id=tool_call_id)
        yield deltas

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
    calls: Sequence[Call | list[Call]],
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

    async def test_timeout_still_delivers_the_unterminated_line(self, tmp_path: Path) -> None:
        # The deadline cancels the readers before end of file, so the tail is
        # flushed by the drain that follows the kill instead.
        listener, _ = await _run(tmp_path, [_run_command('printf hello; sleep 5', timeout_seconds=0.3)])

        lines = [(event.line, event.truncated) for event in listener.events if isinstance(event, ShellOutputLineEvent)]
        end = listener.events[-1]
        assert isinstance(end, ShellCommandEndEvent)
        assert lines == [('hello', False)]
        assert end.stdout == 'hello'

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

    async def test_line_split_across_reads_is_reassembled(self, tmp_path: Path) -> None:
        # Two writes with a pause between them arrive as two pipe reads, and
        # the boundary falls inside a three-byte character.
        listener, results = await _run(tmp_path, [_run_command("printf '\\344\\270'; sleep 0.2; printf '\\255!\\n'")])

        lines = [(event.line, event.truncated) for event in listener.events if isinstance(event, ShellOutputLineEvent)]
        assert lines == [('中!', False)]
        assert results == ['[stdout]\n中!\n']

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
        assert results == ['[Command rewritten: proxy]\n[stdout]\nrewritten\n']

    async def test_rewrite_is_checked_against_the_policy(self, tmp_path: Path) -> None:
        listener, results = await _run(tmp_path, [_run_command('echo hi')], listener=Listener(rewrite_to='vim x'))

        assert [type(event) for event in listener.events] == [ShellCommandRequestEvent]
        assert results == ["[Command rewritten: proxy]\nInteractive commands are not allowed. Command: 'vim x'"]

    async def test_cancel_beats_rewrite_whatever_the_listener_order(self, tmp_path: Path) -> None:
        canceller = Listener(decision='cancel')
        rewriter = Listener(rewrite_to='echo rewritten')
        shell = Shell[None](cwd=tmp_path, denied_commands=[], id='shell')
        for order in ([canceller, rewriter], [rewriter, canceller]):
            agent = Agent(_calls_model([_run_command('echo hi')]), deps_type=type(None), capabilities=[shell, *order])

            result = await agent.run('go')

            assert _tool_results(result.all_messages()) == ['[Command was not run: the user said no]']

    async def test_a_later_listener_cannot_lift_an_earlier_cancel(self, tmp_path: Path) -> None:
        marker = tmp_path / 'ran'
        lifter = LiftCancel()
        shell = Shell[None](cwd=tmp_path, denied_commands=[], id='shell')
        agent = Agent(
            _calls_model([_run_command(f'touch {marker}')]),
            deps_type=type(None),
            capabilities=[shell, Listener(decision='cancel'), lifter],
        )

        result = await agent.run('go')

        assert not lifter.lifted
        assert not marker.exists()
        assert _tool_results(result.all_messages()) == ['[Command was not run: the user said no]']

    async def test_last_rewrite_wins(self, tmp_path: Path) -> None:
        first = Listener(rewrite_to='echo first')
        second = Listener(rewrite_to='echo second')
        shell = Shell[None](cwd=tmp_path, denied_commands=[], id='shell')
        agent = Agent(
            _calls_model([_run_command('echo hi')]), deps_type=type(None), capabilities=[shell, first, second]
        )

        result = await agent.run('go')

        assert _tool_results(result.all_messages()) == ['[Command rewritten: proxy]\n[stdout]\nsecond\n']


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


@dataclass
class RaiseOnStart(AbstractCapability[None]):
    """Raises in the start listener, the way a faulty host handler could."""

    pid: int | None = None
    command_id: str | None = None

    @on_event(ShellCommandStartEvent)
    async def _on_start(self, ctx: RunContext[None], event: ShellCommandStartEvent) -> None:
        self.pid = event.pid
        self.command_id = event.command_id
        raise RuntimeError('listener blew up')


@dataclass
class LiftCancel(AbstractCapability[None]):
    """Tries to lift an earlier listener's veto by assigning the field directly."""

    lifted: bool = False

    @on_event(ShellCommandRequestEvent)
    async def _on_request(self, ctx: RunContext[None], event: ShellCommandRequestEvent) -> None:
        try:
            event.cancelled = False  # type: ignore[prop-value]
            self.lifted = True
        except AttributeError:
            pass


def _live_group_members(pgid: int) -> list[str]:
    """`ps` rows for group members that are not zombies.

    `os.killpg(pgid, 0)` still succeeds while an orphaned member sits unreaped
    under a PID 1 that never waits (some container sandboxes), so the check
    reads process state instead. `ps -eo pgid=,stat=` is the same on Linux
    and macOS; a zombie's state starts with `Z`.
    """
    table = subprocess.run(['ps', '-eo', 'pgid=,stat='], capture_output=True, text=True, check=True).stdout
    rows = [row.split() for row in table.splitlines()]
    return [' '.join(row) for row in rows if len(row) == 2 and row[0] == str(pgid) and not row[1].startswith('Z')]


async def _process_group_is_gone(pgid: int) -> bool:
    """Usually true on the first check; the polling below only runs when the kill lags."""
    with anyio.move_on_after(3):
        while True:
            if not _live_group_members(pgid):
                return True
            await anyio.sleep(0.05)  # pragma: no cover
    return False  # pragma: no cover


class TestRunFailure:
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

    async def test_a_raising_start_listener_kills_the_process_group(self, tmp_path: Path) -> None:
        shell = Shell[None](cwd=tmp_path, denied_commands=[], id='shell')
        raiser = RaiseOnStart()
        agent = Agent(
            _calls_model([_run_command('sleep 30; echo never')]), deps_type=type(None), capabilities=[shell, raiser]
        )
        with pytest.raises(RuntimeError, match='listener blew up'):
            await agent.run('go')

        assert raiser.pid is not None
        # The listener's exception skips the foreground wait, the only other
        # cleanup path, so the kill must come from the toolset itself.
        assert await _process_group_is_gone(raiser.pid)

    async def test_a_raising_start_listener_stops_the_background_command(self, tmp_path: Path) -> None:
        shell = Shell[None](cwd=tmp_path, denied_commands=[], id='shell')
        raiser = RaiseOnStart()
        agent = Agent(
            _calls_model([('start_command', json.dumps({'command': 'sleep 30'}))]),
            deps_type=type(None),
            capabilities=[shell, raiser],
        )
        with pytest.raises(RuntimeError, match='listener blew up'):
            await agent.run('go')

        assert raiser.pid is not None
        assert raiser.command_id is not None
        # The ID never reached the model, so no one is left to call
        # `stop_command`; the kill must come from the toolset itself.
        assert await _process_group_is_gone(raiser.pid)


class TestKillAfterLeaderExit:
    @pytest.mark.anyio(backends=['asyncio'])
    async def test_the_sweep_reaches_group_members_after_the_leader_was_reaped(self, tmp_path: Path) -> None:
        """A leader that exits leaving a child in the group is reaped, but the sweep still reaches the child."""
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('start_new_session is an asyncio spawn option')
        pidfile = tmp_path / 'child.pid'
        # The shell exits right after forking; the backgrounded sleep keeps the group alive.
        proc = await anyio.open_process(['sh', '-c', f'sleep 60 & echo $! > {pidfile}'], start_new_session=True)
        # The sweep runs in `finally` so a failing check leaves no member behind.
        try:
            await proc.wait()
            child_pid = int(pidfile.read_text())
            # The leader is reaped, but the group must still show the surviving member.
            os.kill(child_pid, 0)
            assert _live_group_members(proc.pid)
        finally:
            await kill_process_group(proc)
        assert await _process_group_is_gone(proc.pid)


class TestBackgroundEvents:
    async def test_start_check_stop(self, tmp_path: Path) -> None:
        # The foreground wait orders the steps: on a loaded machine `stop` can
        # otherwise land before the background shell has even run `echo`.
        listener, results = await _run(
            tmp_path,
            [
                ('start_command', '{"command": "echo bg; touch ready; sleep 30"}'),
                _run_command('while [ ! -e ready ]; do sleep 0.05; done'),
                ('check_command', '{"command_id": "$ID"}'),
                ('stop_command', '{"command_id": "$ID"}'),
            ],
        )

        request, start, end = [
            event
            for event in listener.events
            if isinstance(event, (ShellCommandRequestEvent, ShellCommandStartEvent, ShellCommandEndEvent))
            and event.background
        ]
        assert isinstance(request, ShellCommandRequestEvent)
        assert (request.command, request.timeout, request.background) == ('echo bg; touch ready; sleep 30', None, True)
        assert isinstance(start, ShellCommandStartEvent)
        assert (start.timeout, start.background) == (None, True)
        assert isinstance(end, ShellCommandEndEvent)
        assert end.command_id == start.command_id
        assert end.background is True
        assert end.timed_out is False
        assert end.exit_code not in (None, 0)
        assert end.stdout == 'bg\n'
        assert results[3].endswith('[stopped]\n[exit code: -15]')

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

    async def test_parallel_check_and_stop_during_a_stop_emit_one_end(self, tmp_path: Path) -> None:
        # Ignoring SIGTERM makes the stop last the whole grace period, so the
        # parallel calls are certain to run while it is in progress. Whether
        # the check lands before or after the stop claims the process, it must
        # leave the end event to the stop; the duplicate stop waits for the
        # first and reports the same exit.
        listener, results = await _run(
            tmp_path,
            [
                ('start_command', '{"command": "trap \'\' TERM; sleep 30"}'),
                [
                    ('stop_command', '{"command_id": "$ID"}'),
                    ('check_command', '{"command_id": "$ID"}'),
                    ('stop_command', '{"command_id": "$ID"}'),
                ],
            ],
        )

        ends = [event for event in listener.events if isinstance(event, ShellCommandEndEvent)]
        assert len(ends) == 1
        assert ends[0].exit_code == -9
        first_stop, check, second_stop = results[1:]
        assert first_stop == second_stop == '(no output)\n[stopped]\n[exit code: -9]'
        assert check in ('(no output yet)\n[status: running]', '(no output yet)\n[status: finished]')

    async def test_rewritten_background_command_is_not_echoed(self, tmp_path: Path) -> None:
        listener, results = await _run(
            tmp_path,
            [('start_command', '{"command": "echo original"}'), ('stop_command', '{"command_id": "$ID"}')],
            listener=Listener(rewrite_to='echo rewritten'),
        )

        start = listener.events[1]
        assert isinstance(start, ShellCommandStartEvent)
        assert start.command == 'echo rewritten'
        assert results[0].startswith('[Command rewritten: proxy]\nStarted background command\nID: ')
        assert 'echo rewritten' not in results[0]

    async def test_cancelled_background_request(self, tmp_path: Path) -> None:
        listener, results = await _run(
            tmp_path, [('start_command', '{"command": "sleep 30"}')], listener=Listener(decision='cancel')
        )

        assert [type(event) for event in listener.events] == [ShellCommandRequestEvent]
        assert results == ['[Command was not run: the user said no]']
