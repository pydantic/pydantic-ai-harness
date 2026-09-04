"""Shell toolset -- gives agents the ability to run commands."""

from __future__ import annotations

import errno
import fnmatch
import functools
import math
import os
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Concatenate, ParamSpec

import anyio
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.sandboxes import Sandbox, SandboxError, SandboxTimeoutError, SandboxUnavailableError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool

from pydantic_ai_harness._output import truncate_tail
from pydantic_ai_harness._sandbox import sandbox_path

_KILL_GRACE_PERIOD: float = 2.0

_P = ParamSpec('_P')

# Spawning a command fails with a bare `OSError` for causes that have no
# dedicated subclass, and with `FileNotFoundError`/`NotADirectoryError` for
# causes that do. The errno says whose fault it is: these are the model's, and
# it can act on them. Every other errno (EMFILE, ENOMEM) is the sandbox's, and
# must keep aborting the run rather than sending the model into a retry loop it
# can't win.
#
# ENOENT and ENOTDIR reach here only from the working directory, since the
# command string is handed to a shell that always exists -- a command whose own
# executable is missing is reported by that shell on stderr, not by the spawn.
#
# Keyed by `OSError.errno`, which the stdlib types as `int | None`.
_RECOVERABLE_ERRNOS: dict[int | None, str] = {
    errno.ENOENT: 'The working directory no longer exists.',
    errno.ENOTDIR: 'The working directory is no longer a directory.',
}


def _recoverable(
    fn: Callable[Concatenate[ShellToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[ShellToolset, _P], Awaitable[str]]:
    """Convert model-correctable errors into `ModelRetry`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run. A denied command, a command
    the sandbox refuses to spawn, and a working directory the model's own earlier
    command destroyed are all things the model can recover from, so surface them
    as a retry instead of crashing the agent.
    """

    @functools.wraps(fn)
    async def wrapper(self: ShellToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except PermissionError as e:
            raise ModelRetry(str(e)) from e
        # A dead sandbox and a misconfigured one (`UserError`, e.g. no sandbox attached) are the
        # application's to fix; deliberate backend failures are recoverable, but programming
        # errors still propagate.
        except (SandboxUnavailableError, UserError):
            raise
        except SandboxError as e:
            raise ModelRetry(str(e)) from e
        except OSError as e:
            reason = _RECOVERABLE_ERRNOS.get(e.errno)
            if reason is None:
                raise
            # `str(e)` embeds the absolute path; the reason alone doesn't.
            raise ModelRetry(reason) from e

    return wrapper


def _is_interactive_command(command: str) -> bool:
    """Detect commands that typically require interactive input."""
    interactive_patterns = [
        r'^(vi|vim|nano|emacs|less|more|top|htop|man)\b',
        r'^sudo\s',
        r'^passwd\b',
        r'^ssh\b',
        r'^telnet\b',
        r'^ftp\b',
    ]
    return any(re.match(p, command.strip()) for p in interactive_patterns)


class _BackgroundProcess:
    """State for a background command running inside the sandbox."""

    __slots__ = ('sandbox', 'pid', 'stdout_path', 'stderr_path', 'exit_code_path', 'finished', 'exit_code')

    def __init__(
        self,
        sandbox: Sandbox,
        pid: int,
        stdout_path: str,
        stderr_path: str,
        exit_code_path: str,
    ) -> None:
        self.sandbox = sandbox
        self.pid = pid
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.exit_code_path = exit_code_path
        self.finished = False
        self.exit_code: int | None = None


class ShellToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to execute shell commands inside the run's sandbox.

    Supports synchronous execution (run_command) and background processes
    (start_command / check_command / stop_command). Output is truncated to fit
    model context and labelled with stdout/stderr/exit code.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        allowed_commands: Sequence[str],
        denied_commands: Sequence[str],
        denied_operators: Sequence[str],
        default_timeout: float,
        max_timeout: float,
        max_output_chars: int,
        allow_interactive: bool,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
    ) -> None:
        super().__init__()
        # The configured starting directory: a sandbox path, absolute or relative to the
        # sandbox working directory.
        self._initial_cwd = cwd
        self._cwd = sandbox_path(cwd)
        self._allowed_commands = list(allowed_commands)
        self._denied_commands = list(denied_commands)
        self._denied_operators = list(denied_operators)
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._max_output_chars = max_output_chars
        self._allow_interactive = allow_interactive
        self._env = dict(env) if env is not None else None
        self._denied_env_patterns = list(denied_env_patterns)
        self._background: dict[str, _BackgroundProcess] = {}

        if self._allowed_commands and self._denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')
        if max_output_chars <= 0:
            raise ValueError('max_output_chars must be a positive integer.')
        for name, value in (('default_timeout', default_timeout), ('max_timeout', max_timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number.')
        if default_timeout > max_timeout:
            raise ValueError('default_timeout must not exceed max_timeout.')

        self.add_function(
            self.run_command,
            name='run_command',
            metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'},
        )
        self.add_function(
            self.start_command,
            name='start_command',
            metadata={'code_arg_name': 'command', 'code_arg_language': 'shell'},
        )
        self.add_function(self.check_command, name='check_command')
        self.add_function(self.stop_command, name='stop_command')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so cwd and background processes are isolated.

        `get_toolset` builds one shared instance at agent construction (see
        `AbstractToolset.for_run`, which defaults to returning `self`). This
        toolset holds mutable per-run background state, so without an override
        two concurrent runs could kill each other's background processes.
        """
        return ShellToolset(
            cwd=self._initial_cwd,
            allowed_commands=self._allowed_commands,
            denied_commands=self._denied_commands,
            denied_operators=self._denied_operators,
            default_timeout=self._default_timeout,
            max_timeout=self._max_timeout,
            max_output_chars=self._max_output_chars,
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

    def _filter_env(self, env: Mapping[str, str]) -> dict[str, str]:
        """Remove environment names denied by the configured glob patterns."""
        if not self._denied_env_patterns:
            return dict(env)
        return {
            name: value
            for name, value in env.items()
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in self._denied_env_patterns)
        }

    def _run_env(self) -> dict[str, str] | None:
        """The environment handed to the sandbox: the explicit `env`, deny-filtered.

        `None` leaves environment selection to the sandbox backend; denied patterns
        only filter an explicit `env` mapping.
        """
        return None if self._env is None else self._filter_env(self._env)

    async def _cwd_for(self, ctx: RunContext[AgentDepsT]) -> str:
        """Resolve the configured working directory for this command."""
        return await ctx.sandbox.resolve(self._cwd)

    async def __aexit__(self, *args: Any) -> None:
        """Terminate all remaining background processes and clean up their output files."""
        errors: list[Exception] = []
        for command_id, bg in list(self._background.items()):
            process_errors: list[Exception] = []
            try:
                if not bg.finished:
                    await self._terminate(bg)
            except Exception as e:
                process_errors.append(e)
            try:
                await self._cleanup_bg_files(bg)
            except Exception as e:
                process_errors.append(e)
            if process_errors:
                errors.extend(process_errors)
            else:
                del self._background[command_id]
        if errors:
            raise errors[0]

    def _first_denied_operator(self, command: str) -> str | None:
        """Return the first denied operator found in command, or None."""
        return next((op for op in self._denied_operators if op in command), None)

    def _check_command(self, command: str) -> None:
        """Validate command against allow/deny lists.

        These checks are best-effort and are not a security boundary -- a
        sufficiently motivated agent can bypass them. The sandbox is the
        isolation boundary.

        Rejecting a command the OS could not accept belongs here rather than in
        `_recoverable`: a spawn reports a NUL byte or an unencodable character
        as the same `ValueError` whether it came from `command`, the working
        directory, or a configured `env`, and only the first of those is the
        model's to fix.
        """
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

        if not self._allow_interactive and _is_interactive_command(command):
            raise PermissionError(f'Interactive commands are not allowed. Command: {command!r}')

        matched_op = self._first_denied_operator(command)
        if matched_op:
            raise PermissionError(f'Shell operator {matched_op!r} is not allowed.')

        try:
            tokens = shlex.split(command)
        except ValueError:
            return
        if not tokens:
            return
        executable = tokens[0]

        if self._denied_commands and executable in self._denied_commands:
            raise PermissionError(f'Command {executable!r} is denied.')
        if self._allowed_commands and executable not in self._allowed_commands:
            raise PermissionError(f'Command {executable!r} is not in the allowed list.')

    @_recoverable
    async def run_command(
        self,
        ctx: RunContext[AgentDepsT],
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        """Execute a shell command and return its output.

        Each command starts in the configured working directory; `cd` affects only that command.

        Args:
            ctx: The current agent run context.
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: configured default).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        self._check_command(command)
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > self._max_timeout
        ):
            raise ModelRetry(
                f'timeout_seconds must be greater than 0 and at most {self._max_timeout}; '
                'use start_command for longer work.'
            )
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        try:
            result = await ctx.sandbox.run(
                command,
                shell=True,
                timeout=timeout,
                cwd=await self._cwd_for(ctx),
                env=self._run_env(),
            )
        except SandboxTimeoutError as e:
            parts: list[str] = []
            if e.stdout:
                parts.append(f'[stdout]\n{e.stdout}')
            if e.stderr:
                parts.append(f'[stderr]\n{e.stderr}')
            parts.append(f'[Command timed out after {timeout}s]')
            return '\n'.join(parts)

        parts = []
        if result.stdout:
            parts.append(f'[stdout]\n{result.stdout}')
        if result.stderr:
            parts.append(f'[stderr]\n{result.stderr}')
        output = '\n'.join(parts) if parts else '(no output)'

        if result.exit_code != 0:
            output = f'{output}\n[exit code: {result.exit_code}]'
        return output

    @_recoverable
    async def start_command(self, ctx: RunContext[AgentDepsT], command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Callers MUST call `stop_command(command_id)` when done to terminate the
        process and clean up temporary output files.

        Args:
            ctx: The current agent run context.
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        self._check_command(command)
        command_id = uuid.uuid4().hex[:12]
        stdout_path = f'/tmp/harness_{command_id}_out'
        stderr_path = f'/tmp/harness_{command_id}_err'
        exit_code_path = f'/tmp/harness_{command_id}_ec'
        inner = f'sh -c {shlex.quote(command)}; __harness_ec=$?; echo $__harness_ec > {exit_code_path}'
        # `setsid` + output files + `kill` through `run()` is the one background path that
        # works on every provider: the protocol has no background-process API, and Modal has
        # no per-process kill operation or output-so-far handle.
        wrapped = f'setsid sh -c {shlex.quote(inner)} < /dev/null > {stdout_path} 2> {stderr_path} & echo $!'
        result = await ctx.sandbox.run(
            wrapped,
            shell=True,
            cwd=await self._cwd_for(ctx),
            env=self._run_env(),
        )
        try:
            pid = int(result.stdout.strip())
        except ValueError as e:
            message = result.stderr.strip() or 'Sandbox did not return a background process ID.'
            raise ModelRetry(message) from e

        self._background[command_id] = _BackgroundProcess(
            sandbox=ctx.sandbox,
            pid=pid,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            exit_code_path=exit_code_path,
        )
        return f'Started background command: {command!r}\nID: {command_id}'

    @_recoverable
    async def check_command(self, ctx: RunContext[AgentDepsT], command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            ctx: The current agent run context.
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        bg = self._background.get(command_id)
        if bg is None:
            return f'[Error: unknown command ID {command_id!r}]'
        await self._refresh(bg)
        stdout = await self._read_bg_file(bg, bg.stdout_path)
        stderr = await self._read_bg_file(bg, bg.stderr_path)

        status = 'finished' if bg.finished else 'running'
        output_sections: list[str] = []
        if stdout:
            output_sections.append(f'[stdout]\n{stdout}')
        if stderr:
            output_sections.append(f'[stderr]\n{stderr}')
        parts = ['\n'.join(output_sections) if output_sections else '(no output yet)', f'[status: {status}]']
        if bg.finished and bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]')
        return '\n'.join(parts)

    @_recoverable
    async def stop_command(self, ctx: RunContext[AgentDepsT], command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            ctx: The current agent run context.
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        bg = self._background.get(command_id)
        if bg is None:
            return f'[Error: unknown command ID {command_id!r}]'

        await self._refresh(bg)
        if not bg.finished:
            await self._terminate(bg)
        await self._refresh(bg)
        stdout = await self._read_bg_file(bg, bg.stdout_path)
        stderr = await self._read_bg_file(bg, bg.stderr_path)

        await self._cleanup_bg_files(bg)
        del self._background[command_id]

        output_sections: list[str] = []
        if stdout:
            output_sections.append(f'[stdout]\n{stdout}')
        if stderr:
            output_sections.append(f'[stderr]\n{stderr}')
        parts = ['\n'.join(output_sections) if output_sections else '(no output)', '[stopped]']
        if bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]')
        return '\n'.join(parts)

    async def _refresh(self, bg: _BackgroundProcess) -> None:
        if bg.finished:
            return
        try:
            value = (await bg.sandbox.read_bytes(bg.exit_code_path)).decode('utf-8', errors='replace').strip()
        except FileNotFoundError:
            return
        try:
            bg.exit_code = int(value)
        except ValueError:
            return
        bg.finished = True

    async def _read_bg_file(self, bg: _BackgroundProcess, path: str) -> str:
        try:
            result = await bg.sandbox.run(
                ['tail', '-c', str(self._max_output_chars * 4), path], timeout=self._default_timeout
            )
            if result.exit_code != 0:
                if 'No such file' in result.stderr:
                    return ''
                raise RuntimeError(result.stderr.strip() or f'Unable to read background output file {path!r}.')
            return result.stdout
        except FileNotFoundError:
            return ''

    async def _terminate(self, bg: _BackgroundProcess) -> None:
        """SIGTERM the process group, escalating to SIGKILL after the grace period."""
        result = await bg.sandbox.run(['kill', '-TERM', f'-{bg.pid}'])
        if result.exit_code != 0:
            await self._refresh(bg)
            if bg.finished or 'No such process' in result.stderr:
                return
            raise RuntimeError(result.stderr.strip() or f'Failed to terminate background process {bg.pid}.')
        await anyio.sleep(_KILL_GRACE_PERIOD)
        result = await bg.sandbox.run(['kill', '-KILL', f'-{bg.pid}'])
        if result.exit_code != 0:  # pragma: no branch - macOS may reap the group after SIGTERM
            await self._refresh(bg)
            if bg.finished or 'No such process' in result.stderr:
                return
            raise RuntimeError(result.stderr.strip() or f'Failed to kill background process {bg.pid}.')

    async def _cleanup_bg_files(self, bg: _BackgroundProcess) -> None:
        errors: list[Exception] = []
        for path in (bg.stdout_path, bg.stderr_path, bg.exit_code_path):
            try:
                await bg.sandbox.remove(path)
            except FileNotFoundError:
                pass
            except Exception as e:
                errors.append(e)
        if errors:
            raise errors[0]
