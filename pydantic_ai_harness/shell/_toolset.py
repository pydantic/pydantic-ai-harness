"""Shell toolset -- gives agents the ability to run commands inside the run's workspace."""

from __future__ import annotations

import fnmatch
import logging
import os
import posixpath
import shlex
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import anyio
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import WorkspaceTimeoutError

from pydantic_ai_harness._output import truncate_tail
from pydantic_ai_harness._workspace import metadata_dir
from pydantic_ai_harness.shell._jobs import CONTROL_TIMEOUT, Job
from pydantic_ai_harness.shell._limits import file_limit_status, limited_script, validate_file_limit
from pydantic_ai_harness.shell._persistent import MAX_FOREGROUND_WAIT, CommandMode, run_persistent_command
from pydantic_ai_harness.shell._policy import is_interactive_command, recoverable

_logger = logging.getLogger(__name__)

RUN_SCOPED_TOOL_NAMES: tuple[str, ...] = ('run_command', 'start_command', 'check_command', 'stop_command')
"""The default tools. Their commands are killed when the agent run ends."""

PERSISTENT_TOOL_NAME = 'shell'
"""The opt-in tool whose commands outlive the agent run."""

SHELL_TOOL_NAMES: tuple[str, ...] = (*RUN_SCOPED_TOOL_NAMES, PERSISTENT_TOOL_NAME)
"""Every tool `Shell` can register, in registration order."""

_OUTPUT_BYTES_PER_CHAR = 4
"""UTF-8 bytes per character at most: reading `4 * max_output_chars` bytes of a log keeps every character the cap keeps."""


class _BackgroundProcess:
    """State for a run-scoped background command: its job and, once seen, how it ended."""

    __slots__ = ('job', 'finished', 'exit_code')

    def __init__(self, job: Job) -> None:
        self.job = job
        self.finished = False
        self.exit_code: int | None = None

    async def refresh(self) -> None:
        if not self.finished:
            running, exit_code = await self.job.status()
            self.finished = not running
            self.exit_code = exit_code


class ShellToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to execute shell commands in the run's workspace.

    Supports synchronous execution (run_command) and background processes
    (start_command / check_command / stop_command). Output is truncated to fit
    model context and labelled with stdout/stderr/exit code. The opt-in `shell`
    tool instead starts commands that outlive the run and returns handles to
    their output and exit status. Every command, file, and signal goes through
    `ctx.workspace`.

    Optionally tracks the working directory across calls so `cd` persists.
    """

    def __init__(
        self,
        *,
        allowed_commands: Sequence[str],
        denied_commands: Sequence[str],
        denied_operators: Sequence[str],
        default_timeout: float,
        max_output_chars: int,
        persist_cwd: bool,
        allow_interactive: bool,
        max_file_bytes: int | None = None,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
        tools: Sequence[str] = RUN_SCOPED_TOOL_NAMES,
    ) -> None:
        super().__init__()
        # The absolute workspace path `persist_cwd` last recorded; `None` means the working directory.
        self._cwd: str | None = None
        self._allowed_commands = list(allowed_commands)
        self._denied_commands = list(denied_commands)
        self._denied_operators = list(denied_operators)
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        validate_file_limit(max_file_bytes, persistent=PERSISTENT_TOOL_NAME in tools)
        if max_file_bytes is not None and persist_cwd:
            raise ValueError(
                'max_file_bytes is not supported with persist_cwd; cwd capture writes a file in the child.'
            )
        self._max_file_bytes = max_file_bytes
        self._persist_cwd = persist_cwd
        self._allow_interactive = allow_interactive
        self._env = dict(env) if env is not None else None
        self._denied_env_patterns = list(denied_env_patterns)
        self._tools = tuple(tools)
        self._background: dict[str, _BackgroundProcess] = {}
        self._jobs_dir: str | None = None

        if self._allowed_commands and self._denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')
        if max_output_chars <= 0:
            raise ValueError('max_output_chars must be a positive integer.')
        if unknown := sorted(set(self._tools) - set(SHELL_TOOL_NAMES)):
            raise ValueError(f'Unknown shell tools: {", ".join(unknown)}. Available: {", ".join(SHELL_TOOL_NAMES)}.')
        if PERSISTENT_TOOL_NAME in self._tools and not 0 < default_timeout <= MAX_FOREGROUND_WAIT:
            raise ValueError(
                f'default_timeout must be greater than zero and at most {MAX_FOREGROUND_WAIT:g} seconds '
                'for the shell tool.'
            )

        command_metadata = {'code_arg_name': 'command', 'code_arg_language': 'shell'}
        registrations: dict[str, Callable[..., Awaitable[str]]] = {
            'run_command': self.run_command,
            'start_command': self.start_command,
            'check_command': self.check_command,
            'stop_command': self.stop_command,
            PERSISTENT_TOOL_NAME: self.shell,
        }
        for name in SHELL_TOOL_NAMES:
            if name in self._tools:
                metadata = command_metadata if name in ('run_command', 'start_command', PERSISTENT_TOOL_NAME) else None
                self.add_function(registrations[name], name=name, metadata=metadata)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so cwd and background processes are isolated.

        `get_toolset` builds one shared instance at agent construction (see
        `AbstractToolset.for_run`, which defaults to returning `self`). This
        toolset holds mutable per-run state (`_cwd`, `_background`), so without
        an override two concurrent runs would corrupt each other's cwd and kill
        each other's background processes.
        """
        return ShellToolset(
            allowed_commands=self._allowed_commands,
            denied_commands=self._denied_commands,
            denied_operators=self._denied_operators,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            max_file_bytes=self._max_file_bytes,
            persist_cwd=self._persist_cwd,
            allow_interactive=self._allow_interactive,
            env=self._env,
            denied_env_patterns=self._denied_env_patterns,
            tools=self._tools,
        )

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Offer no tools on a read-only workspace: it refuses `run`, so no shell tool could succeed."""
        if ctx.workspace.read_only:
            return {}
        return await super().get_tools(ctx)

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
        """The variables handed to the workspace for each command, on top of its own environment.

        The workspace decides the base environment; nothing comes from the agent process.
        `None` adds nothing. An explicit `env` is added, minus names that match
        `denied_env_patterns` (glob, via `fnmatch`).
        """
        if self._env is None:
            return None
        if not self._denied_env_patterns:
            return dict(self._env)
        return {
            name: value
            for name, value in self._env.items()
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in self._denied_env_patterns)
        }

    async def _cwd_for(self, ctx: RunContext[AgentDepsT]) -> str:
        """The absolute workspace directory the next run-scoped command starts in."""
        return self._cwd if self._cwd is not None else await ctx.workspace.working_dir()

    async def _jobs_base(self, ctx: RunContext[AgentDepsT]) -> str:
        """The workspace directory holding this run's job and capture files, looked up once per run."""
        if self._jobs_dir is None:
            self._jobs_dir = await metadata_dir(ctx.workspace, 'shell')
        return self._jobs_dir

    async def __aexit__(self, *args: Any) -> None:
        """Terminate all remaining background processes and remove their files from the workspace.

        Cleanup is best-effort and shielded: the run may be ending because it was cancelled or
        because the workspace became unusable, and neither should leave the exit stack half-run.
        Any error cleaning up one job is logged at debug level and the next job is tried; a durable
        workspace may refuse calls outside an activity with errors that are not `WorkspaceError`.
        """
        with anyio.move_on_after(CONTROL_TIMEOUT * 2, shield=True):
            for bg in self._background.values():
                try:
                    await bg.refresh()
                    if not bg.finished:
                        await bg.job.kill()
                    await bg.job.cleanup()
                except Exception:
                    _logger.debug('Could not clean up background job %s', bg.job.directory, exc_info=True)
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
        `recoverable`: a spawn reports a NUL byte or an unencodable character as
        the same `ValueError` whether it came from `command`, the working
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

        if not self._allow_interactive and is_interactive_command(command):
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

    async def _build_cwd_capture(self, ctx: RunContext[AgentDepsT], command: str) -> tuple[str, str | None]:
        """Wrap a command to record its final working directory out-of-band.

        `pwd` is written to a file inside the workspace whose random path the
        agent's command can't address, so command output can never spoof the
        tracked cwd -- unlike parsing a sentinel out of stdout, where any command
        that prints the sentinel string (or one using `;` to skip success-gating)
        could redirect the cwd. Returns the wrapped command plus the capture
        path, or the command unchanged and `None` when cwd tracking is off.
        """
        if not self._persist_cwd:
            return command, None
        name = posixpath.join(await self._jobs_base(ctx), f'cwd-{uuid.uuid4().hex}')
        wrapped = f'{command}\n__harness_ec=$?\npwd > {shlex.quote(name)}\nexit $__harness_ec'
        return wrapped, name

    async def _apply_captured_cwd(self, ctx: RunContext[AgentDepsT], cwd_file: str) -> None:
        """Update the persistent cwd from the capture file, ignoring junk.

        The whole read-and-check is guarded, not just the read: the command it
        belongs to already succeeded, so a capture that isn't UTF-8 (a
        `UnicodeDecodeError`, which is a `ValueError` rather than an `OSError`)
        or a recorded path the workspace refuses to stat (`ENAMETOOLONG`) is
        bookkeeping the toolset can drop, not a tool failure to report.
        """
        try:
            recorded = (await ctx.workspace.read_bytes(cwd_file)).decode('utf-8').strip()
            if not posixpath.isabs(recorded):
                return
            if (await ctx.workspace.stat(recorded)).is_dir:
                self._cwd = posixpath.normpath(recorded)
        except (OSError, ValueError):
            return

    async def _remove_capture(self, ctx: RunContext[AgentDepsT], cwd_file: str | None) -> None:
        if cwd_file is None:
            return
        with anyio.move_on_after(CONTROL_TIMEOUT, shield=True):
            try:
                await ctx.workspace.remove(cwd_file)
            except FileNotFoundError:
                pass

    @recoverable
    async def run_command(
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
        self._check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        actual_command, cwd_file = await self._build_cwd_capture(ctx, command)
        try:
            try:
                result = await ctx.workspace.run(
                    limited_script(actual_command, self._max_file_bytes),
                    shell=True,
                    cwd=await self._cwd_for(ctx),
                    env=self._resolve_env(),
                    timeout=timeout,
                )
            except WorkspaceTimeoutError as e:
                return _labelled(e.stdout, e.stderr, empty=None, trailer=f'[Command timed out after {timeout}s]')

            output = _labelled(result.stdout, result.stderr, empty='(no output)')
            exit_code = result.exit_code

            if cwd_file is not None and exit_code == 0:
                await self._apply_captured_cwd(ctx, cwd_file)

            if exit_code != 0:
                output = f'{output}\n[exit code: {exit_code}]'
                output += file_limit_status(exit_code, self._max_file_bytes)
            return output
        finally:
            await self._remove_capture(ctx, cwd_file)

    @recoverable
    async def shell(
        self,
        ctx: RunContext[AgentDepsT],
        command: str,
        *,
        mode: CommandMode = 'foreground',
        timeout: float | None = None,
    ) -> str:
        """Run a command that keeps running after this call and after the agent run.

        Foreground waits up to `timeout` seconds (at most 270) for the command to
        exit, then returns handles to the same still-running process. Background
        returns the handles immediately. Both return the PID, the path of the
        combined stdout/stderr log, and the path of a JSON status file whose
        `exit_code` is null until the command exits. Read those files with your
        other tools and stop the process with `kill` and the returned PID; no
        notification arrives when it finishes.

        Args:
            ctx: The current agent run context.
            command: The shell command to run.
            mode: `foreground` to wait, `background` to return at once.
            timeout: Seconds to wait in foreground mode (default: the configured timeout).
        """
        self._check_command(command)
        return await run_persistent_command(
            ctx,
            command,
            base=await self._jobs_base(ctx),
            cwd=await ctx.workspace.working_dir(),
            env=self._resolve_env(),
            mode=mode,
            timeout=self._default_timeout if timeout is None else timeout,
        )

    @recoverable
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
        job = await Job.launch(
            ctx.workspace,
            command,
            base=await self._jobs_base(ctx),
            cwd=await self._cwd_for(ctx),
            env=self._resolve_env(),
            combined=False,
            file_limit=self._max_file_bytes,
        )
        self._background[command_id] = _BackgroundProcess(job)
        return f'Started background command: {command!r}\nID: {command_id}'

    async def _read_bg_output(self, bg: _BackgroundProcess) -> tuple[str, str]:
        """The retained tail of a background command's stdout and stderr logs."""
        limit = self._max_output_chars * _OUTPUT_BYTES_PER_CHAR
        stdout = await bg.job.tail(bg.job.output_path, limit)
        stderr = await bg.job.tail(bg.job.stderr_path, limit)
        return stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')

    @recoverable
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

        await bg.refresh()
        stdout, stderr = await self._read_bg_output(bg)

        status = 'finished' if bg.finished else 'running'
        parts = [_labelled(stdout, stderr, empty='(no output yet)'), f'[status: {status}]']
        if bg.finished and bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]' + file_limit_status(bg.exit_code, self._max_file_bytes))
        return '\n'.join(parts)

    @recoverable
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

        await bg.refresh()
        if not bg.finished:
            with anyio.CancelScope(shield=True):
                await bg.job.kill()
                # A group that ignored SIGTERM is killed with SIGKILL, which its wrapper cannot
                # outlive to publish a status: the command is stopped, with no exit code to report.
                await bg.refresh()
            bg.finished = True

        stdout, stderr = await self._read_bg_output(bg)

        await bg.job.cleanup()
        del self._background[command_id]

        parts = [_labelled(stdout, stderr, empty='(no output)'), '[stopped]']
        if bg.exit_code is not None:
            parts.append(f'[exit code: {bg.exit_code}]' + file_limit_status(bg.exit_code, self._max_file_bytes))
        return '\n'.join(parts)


def _labelled(stdout: str, stderr: str, *, empty: str | None, trailer: str | None = None) -> str:
    """`[stdout]`/`[stderr]` sections, `empty` when both are blank, and an optional last line."""
    sections: list[str] = []
    if stdout:
        sections.append(f'[stdout]\n{stdout}')
    if stderr:
        sections.append(f'[stderr]\n{stderr}')
    if not sections and empty is not None:
        sections.append(empty)
    if trailer is not None:
        sections.append(trailer)
    return '\n'.join(sections)
