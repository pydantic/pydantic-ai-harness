"""Shell toolset -- gives agents the ability to run commands."""

from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool

from pydantic_ai_harness._output import truncate_tail
from pydantic_ai_harness.shell._events import (
    OutputStream,
    ShellCommandEndEvent,
    ShellCommandRequestEvent,
    ShellCommandStartEvent,
    ShellOutputLineEvent,
)
from pydantic_ai_harness.shell._process import (
    BackgroundProcess,
    LineSink,
    OutputReader,
    cleanup_bg_files,
    is_interactive_command,
    kill_process_group,
    read_bg_output,
    recoverable,
    run_to_exit,
)


def _format_output(stdout: str, stderr: str, *, empty: str) -> str:
    sections: list[str] = []
    if stdout:
        sections.append(f'[stdout]\n{stdout}')
    if stderr:
        sections.append(f'[stderr]\n{stderr}')
    return '\n'.join(sections) if sections else empty


class ShellToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to execute shell commands.

    Supports synchronous execution (run_command) and background processes
    (start_command / check_command / stop_command). Output is streamed,
    truncated to fit model context, and labelled with stdout/stderr/exit code.

    Optionally tracks the working directory across calls so ``cd`` persists.

    Inside an agent run every command announces itself with a
    `ShellCommandRequestEvent` a listener can cancel or rewrite, then emits
    start, per-line output, and end events. The public methods run the same
    code outside a run and emit nothing.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        allowed_commands: Sequence[str],
        denied_commands: Sequence[str],
        denied_operators: Sequence[str],
        default_timeout: float,
        max_output_chars: int,
        persist_cwd: bool,
        allow_interactive: bool,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self._cwd = cwd.resolve()
        # The configured starting directory, never mutated by persist_cwd, so
        # `for_run` can hand each run a fresh instance rooted back here.
        self._initial_cwd = self._cwd
        self._allowed_commands = list(allowed_commands)
        self._denied_commands = list(denied_commands)
        self._denied_operators = list(denied_operators)
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._persist_cwd = persist_cwd
        self._allow_interactive = allow_interactive
        self._env = dict(env) if env is not None else None
        self._denied_env_patterns = list(denied_env_patterns)
        self._background: dict[str, BackgroundProcess] = {}

        if self._allowed_commands and self._denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')
        if max_output_chars <= 0:
            raise ValueError('max_output_chars must be a positive integer.')

        self.add_function(
            self._run_command_tool,
            name='run_command',
            metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'},
        )
        self.add_function(
            self._start_command_tool,
            name='start_command',
            metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'},
        )
        self.add_function(self._check_command_tool, name='check_command')
        self.add_function(self._stop_command_tool, name='stop_command')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so cwd and background processes are isolated.

        `get_toolset` builds one shared instance at agent construction (see
        `AbstractToolset.for_run`, which defaults to returning `self`). This
        toolset holds mutable per-run state (`_cwd`, `_background`), so without
        an override two concurrent runs would corrupt each other's cwd and kill
        each other's background processes.
        """
        return ShellToolset(
            cwd=self._initial_cwd,
            allowed_commands=self._allowed_commands,
            denied_commands=self._denied_commands,
            denied_operators=self._denied_operators,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            persist_cwd=self._persist_cwd,
            allow_interactive=self._allow_interactive,
            env=self._env,
            denied_env_patterns=self._denied_env_patterns,
        )

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Enforce the model-visible output cap at the tool dispatch seam.

        Tools place control metadata (status, exit code, `start_command`'s ID
        line) at the end of their responses, so keeping the tail preserves it
        without any per-tool cases here. Only `str` results are capped; a
        future tool returning rich content (e.g. `ToolReturn`) needs this seam
        extended.
        """
        result = await super().call_tool(name, tool_args, ctx, tool)
        if not isinstance(result, str):
            return result
        return truncate_tail(result, self._max_output_chars)

    def _resolve_env(self) -> dict[str, str] | None:
        """Compute the environment passed to spawned subprocesses.

        Returns `None` -- meaning the subprocess inherits the parent env -- only
        when neither `env` nor `denied_env_patterns` is configured, so the
        default behavior is unchanged. An explicit `env` replaces inheritance
        entirely; `denied_env_patterns` then strips matching names (glob, via
        `fnmatch`) from whichever base applies, so the two compose: patterns
        filter an explicit `env` just as they filter the inherited environment.
        """
        if self._env is None and not self._denied_env_patterns:
            return None
        base = dict(self._env) if self._env is not None else dict(os.environ)
        if not self._denied_env_patterns:
            return base
        return {
            name: value
            for name, value in base.items()
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in self._denied_env_patterns)
        }

    async def __aexit__(self, *args: Any) -> None:
        """Terminate all remaining background processes and clean up temp files."""
        for bg in self._background.values():
            if not bg.finished:
                await kill_process_group(bg.proc)
                with anyio.CancelScope(shield=True):
                    await bg.proc.wait()
                await bg.proc.aclose()
            cleanup_bg_files(bg)
        self._background.clear()

    def _first_denied_operator(self, command: str) -> str | None:
        """Return the first denied operator found in command, or None."""
        return next((op for op in self._denied_operators if op in command), None)

    def _check_command(self, command: str) -> None:
        """Validate command against allow/deny lists.

        These checks are best-effort and are not a security boundary -- a
        sufficiently motivated agent can bypass them. Use OS-level isolation
        (containers, sandboxes) for hard enforcement.

        Rejecting a command the OS could not accept belongs here rather than in
        `recoverable`: `anyio.open_process` reports a NUL byte or an
        unencodable character as the same `ValueError` whether it came from
        `command`, the working directory, or a configured `env`, and only the
        first of those is the model's to fix.
        """
        if not command.strip():
            raise ModelRetry('The command is empty.')
        if '\x00' in command:
            raise ModelRetry('The command contains a NUL byte, which cannot be passed to a process.')
        try:
            # `os.fsencode`, not `str.encode`: the spawn encodes with the
            # filesystem encoding and `surrogateescape`, which accepts the
            # \udc80-\udcff range as the raw bytes it round-trips from. Encoding
            # as plain UTF-8 here would reject commands the OS runs happily.
            os.fsencode(command)
        except UnicodeEncodeError as e:
            raise ModelRetry('The command contains characters that cannot be encoded for the operating system.') from e

        if not self._allow_interactive and is_interactive_command(command):
            raise PermissionError(f'Interactive commands are not allowed. Command: {command!r}')

        matched_op = self._first_denied_operator(command)
        if matched_op:
            raise PermissionError(f'Shell operator {matched_op!r} is not allowed.')

        try:
            tokens = shlex.split(command)
        except ValueError:
            return
        # A blank command was rejected above, so a parse that succeeds has a token.
        executable = tokens[0]

        if self._denied_commands and executable in self._denied_commands:
            raise PermissionError(f'Command {executable!r} is denied.')
        if self._allowed_commands and executable not in self._allowed_commands:
            raise PermissionError(f'Command {executable!r} is not in the allowed list.')

    async def _request(
        self, ctx: RunContext[AgentDepsT], command: str, *, timeout: float | None, background: bool
    ) -> tuple[str, str | None]:
        """Announce a command and apply the listeners' decision.

        Returns the command to run (possibly rewritten) and a note for the
        model, or `None` as the command when a listener cancelled it. A
        rewrite goes through `_check_command` again so a listener cannot hand
        the model a command the policy would have refused; the retry names the
        rewrite so the model is not blamed for a command it never proposed.
        The rewritten command itself stays out of the note: a host may have
        put a credential in it, and the reason is the host's to word.
        """
        request = ShellCommandRequestEvent(command=command, cwd=str(self._cwd), timeout=timeout, background=background)
        await ctx.emit(request)
        if request.cancelled:
            return '', f'[Command was not run: {request.cancel_reason or "cancelled by a listener"}]'
        if request.rewrite_reason is None:
            return command, None
        note = f'[Command rewritten: {request.rewrite_reason}]'
        try:
            self._check_command(request.command)
        except (PermissionError, ModelRetry) as e:
            raise ModelRetry(f'{note}\n{e}') from e
        return request.command, note

    def _line_sink(self, ctx: RunContext[AgentDepsT] | None, command_id: str, stream: OutputStream) -> LineSink | None:
        if ctx is None:
            return None

        async def sink(line: str, truncated: bool) -> None:
            await ctx.emit(ShellOutputLineEvent(command_id=command_id, stream=stream, line=line, truncated=truncated))

        return sink

    def _end_event(
        self,
        *,
        command_id: str,
        command: str,
        background: bool,
        exit_code: int,
        timed_out: bool,
        started_at: float,
        stdout: str,
        stderr: str,
    ) -> ShellCommandEndEvent:
        stdout_tail = truncate_tail(stdout, self._max_output_chars)
        stderr_tail = truncate_tail(stderr, self._max_output_chars)
        return ShellCommandEndEvent(
            command_id=command_id,
            command=command,
            background=background,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started_at,
            stdout=stdout_tail,
            stderr=stderr_tail,
            truncated=stdout_tail != stdout or stderr_tail != stderr,
        )

    def _build_cwd_capture(self, command: str) -> tuple[str, Path | None]:
        """Wrap a command to record its final working directory out-of-band.

        `pwd` is written to a private temp file whose random path the agent's
        command can't address, so command output can never spoof the tracked
        cwd -- unlike parsing a sentinel out of stdout, where any command that
        prints the sentinel string (or one using `;` to skip success-gating)
        could redirect the cwd. Returns the wrapped command plus the temp-file
        path, or the command unchanged and `None` when cwd tracking is off.
        """
        if not self._persist_cwd:
            return command, None
        fd, name = tempfile.mkstemp(prefix='harness_cwd_')
        os.close(fd)
        wrapped = f'{command}\n__harness_ec=$?\npwd > {shlex.quote(name)}\nexit $__harness_ec'
        return wrapped, Path(name)

    def _apply_captured_cwd(self, cwd_file: Path) -> None:
        """Update the persistent cwd from the capture file, ignoring junk.

        The whole read-and-check is guarded, not just the read: the command it
        belongs to already succeeded, so a capture that isn't UTF-8 (a
        `UnicodeDecodeError`, which is a `ValueError` rather than an `OSError`)
        or a recorded path the OS refuses to stat (`ENAMETOOLONG`, which
        `Path.is_dir` propagates before 3.14 and swallows from 3.14 on) is
        bookkeeping the toolset can drop, not a tool failure to report.
        """
        try:
            recorded = cwd_file.read_text(encoding='utf-8').strip()
            if not recorded:
                return
            candidate = Path(recorded)
            if candidate.is_dir():
                self._cwd = candidate
        except (OSError, ValueError):
            return

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Execute a shell command directly, outside an agent run."""
        return await self._run(None, command, timeout_seconds=timeout_seconds)

    async def _run_command_tool(
        self, ctx: RunContext[AgentDepsT], command: str, *, timeout_seconds: float | None = None
    ) -> str:
        """Execute a shell command and return its output.

        Args:
            ctx: The current agent run context.
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        return await self._run(ctx, command, timeout_seconds=timeout_seconds)

    @recoverable
    async def _run(
        self, ctx: RunContext[AgentDepsT] | None, command: str, *, timeout_seconds: float | None = None
    ) -> str:
        self._check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        note: str | None = None
        if ctx is not None:
            command, note = await self._request(ctx, command, timeout=timeout, background=False)
            if not command:
                return note or ''

        command_id = uuid.uuid4().hex[:12]
        started_at = time.monotonic()
        actual_command, cwd_file = self._build_cwd_capture(command)
        try:
            proc = await anyio.open_process(
                actual_command,
                cwd=self._cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=self._resolve_env(),
            )
            if ctx is not None:
                try:
                    await ctx.emit(
                        ShellCommandStartEvent(
                            command_id=command_id,
                            command=command,
                            cwd=str(self._cwd),
                            timeout=timeout,
                            background=False,
                            pid=proc.pid,
                        )
                    )
                except BaseException:
                    # A raising or cancelled listener ends the run; killing
                    # the group first is what keeps the process from
                    # outliving it.
                    await kill_process_group(proc)
                    raise
            assert proc.stdout is not None
            assert proc.stderr is not None
            stdout = OutputReader(proc.stdout, on_line=self._line_sink(ctx, command_id, 'stdout'))
            stderr = OutputReader(proc.stderr, on_line=self._line_sink(ctx, command_id, 'stderr'))
            exit_code, timed_out = await run_to_exit(proc, stdout, stderr, timeout=timeout)

            if ctx is not None:
                await ctx.emit(
                    self._end_event(
                        command_id=command_id,
                        command=command,
                        background=False,
                        exit_code=exit_code,
                        timed_out=timed_out,
                        started_at=started_at,
                        stdout=stdout.text,
                        stderr=stderr.text,
                    )
                )

            if timed_out:
                output = f'[Command timed out after {timeout}s]'
            else:
                output = _format_output(stdout.text, stderr.text, empty='(no output)')
                if cwd_file is not None and exit_code == 0:
                    self._apply_captured_cwd(cwd_file)
                if exit_code != 0:
                    output = f'{output}\n[exit code: {exit_code}]'
            return output if note is None else f'{note}\n{output}'
        finally:
            if cwd_file is not None:
                cwd_file.unlink(missing_ok=True)

    async def start_command(self, command: str) -> str:
        """Start a background command directly, outside an agent run."""
        return await self._start(None, command)

    async def _start_command_tool(self, ctx: RunContext[AgentDepsT], command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Callers MUST call `stop_command(command_id)` when done to terminate the
        process and clean up temporary output files.

        Args:
            ctx: The current agent run context.
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        return await self._start(ctx, command)

    @recoverable
    async def _start(self, ctx: RunContext[AgentDepsT] | None, command: str) -> str:
        self._check_command(command)
        note: str | None = None
        if ctx is not None:
            command, note = await self._request(ctx, command, timeout=None, background=True)
            if not command:
                return note or ''

        command_id = uuid.uuid4().hex[:12]

        stdout_file = tempfile.NamedTemporaryFile(mode='w+b', prefix=f'harness_{command_id}_out_', delete=False)
        stderr_file = tempfile.NamedTemporaryFile(mode='w+b', prefix=f'harness_{command_id}_err_', delete=False)

        try:
            proc = await anyio.open_process(
                command,
                cwd=self._cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                env=self._resolve_env(),
            )
        except BaseException:
            stdout_file.close()
            stderr_file.close()
            os.unlink(stdout_file.name)
            os.unlink(stderr_file.name)
            raise

        stdout_file.close()
        stderr_file.close()

        bg = BackgroundProcess(
            command=command,
            command_id=command_id,
            proc=proc,
            stdout_path=stdout_file.name,
            stderr_path=stderr_file.name,
        )
        self._background[command_id] = bg
        if ctx is not None:
            try:
                await ctx.emit(
                    ShellCommandStartEvent(
                        command_id=command_id,
                        command=command,
                        cwd=str(self._cwd),
                        timeout=None,
                        background=True,
                        pid=proc.pid,
                    )
                )
            except BaseException:
                # Symmetric with `_run`: the run ends and the ID never
                # reaches the model, so nothing is left to stop the
                # process. Kill it and drop the record.
                await kill_process_group(bg.proc)
                with anyio.CancelScope(shield=True):
                    await bg.proc.wait()
                cleanup_bg_files(bg)
                self._background.pop(command_id)
                await bg.proc.aclose()
                raise

        if note is None:
            return f'Started background command: {command!r}\nID: {command_id}'
        # The rewritten command stays out of the result, as in `_request`.
        return f'{note}\nStarted background command\nID: {command_id}'

    async def check_command(self, command_id: str) -> str:
        """Check a background command directly, outside an agent run."""
        return await self._check_status(None, command_id)

    async def _check_command_tool(self, ctx: RunContext[AgentDepsT], command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            ctx: The current agent run context.
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        return await self._check_status(ctx, command_id)

    async def _check_status(self, ctx: RunContext[AgentDepsT] | None, command_id: str) -> str:
        bg = self._background.get(command_id)
        if bg is None:
            return f'[Error: unknown command ID {command_id!r}]'

        just_exited = None if bg.finished else bg.proc.returncode
        if just_exited is not None:
            bg.exit_code = just_exited
            bg.finished = True

        stdout, stderr = read_bg_output(bg)
        if just_exited is not None and ctx is not None:
            await ctx.emit(self._bg_end_event(bg, exit_code=just_exited, stdout=stdout, stderr=stderr))

        status = 'finished' if bg.finished else 'running'
        parts = [_format_output(stdout, stderr, empty='(no output yet)'), f'[status: {status}]']
        if bg.finished and bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]')
        return '\n'.join(parts)

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command directly, outside an agent run."""
        return await self._stop(None, command_id)

    async def _stop_command_tool(self, ctx: RunContext[AgentDepsT], command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            ctx: The current agent run context.
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        return await self._stop(ctx, command_id)

    async def _stop(self, ctx: RunContext[AgentDepsT] | None, command_id: str) -> str:
        bg = self._background.get(command_id)
        if bg is None:
            return f'[Error: unknown command ID {command_id!r}]'

        async with bg.stop_lock:
            stopped: int | None = None
            if not bg.finished:
                # Claimed before the first await: a check running while the
                # kill is in progress must not report the exit and emit a
                # second end.
                bg.finished = True
                await kill_process_group(bg.proc)
                with anyio.CancelScope(shield=True):
                    stopped = await bg.proc.wait()
                bg.exit_code = stopped

            stdout, stderr = read_bg_output(bg)

            cleanup_bg_files(bg)
            self._background.pop(command_id, None)
            await bg.proc.aclose()

        if stopped is not None and ctx is not None:
            await ctx.emit(self._bg_end_event(bg, exit_code=stopped, stdout=stdout, stderr=stderr))

        parts = [_format_output(stdout, stderr, empty='(no output)'), '[stopped]']
        if bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]')
        return '\n'.join(parts)

    def _bg_end_event(self, bg: BackgroundProcess, *, exit_code: int, stdout: str, stderr: str) -> ShellCommandEndEvent:
        return self._end_event(
            command_id=bg.command_id,
            command=bg.command,
            background=True,
            exit_code=exit_code,
            timed_out=False,
            started_at=bg.started_at,
            stdout=stdout,
            stderr=stderr,
        )
