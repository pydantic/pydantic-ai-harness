"""Tests for the Shell capability and the model-facing surface of `ShellToolset`."""

from __future__ import annotations

import errno
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import anyio
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.sandboxes import (
    CommandResult,
    LocalSandbox,
    Sandbox,
    SandboxCommand,
    SandboxError,
    SandboxFileEntry,
    SandboxRef,
    SandboxResult,
    SandboxTimeoutError,
    SandboxUnavailableError,
)

from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.filesystem import FileSystemToolset
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.shell._toolset import ShellToolset
from tests.shell.conftest import (  # pyright: ignore[reportMissingTypeStubs]
    background_toolset,
    call_tool,
    command_id,
    run_context,
    shell_toolset,
)


class TestInteractiveCommands:
    @pytest.mark.parametrize(
        'command',
        [
            'vi file.txt',
            'vim file.txt',
            'nano file.txt',
            'emacs file.txt',
            'less file.txt',
            'more file.txt',
            'top',
            'htop',
            'man ls',
            'sudo rm -rf /',
            'passwd',
            'ssh host',
            'telnet localhost 80',
            'ftp host',
            '  vi file.txt',  # leading whitespace must not hide the command name
        ],
    )
    async def test_interactive_commands_are_blocked_by_default(self, command: str, sandbox: Sandbox) -> None:
        with pytest.raises(ModelRetry, match='Interactive commands are not allowed'):
            await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command=command)

    @pytest.mark.parametrize(
        ('command', 'output'),
        [
            ('viewer=x; printf $viewer', 'x'),  # an interactive name must match a whole word
            ('printf sudo', 'sudo'),  # ...and only at the start of the command
        ],
    )
    async def test_commands_that_merely_resemble_interactive_ones_run(
        self, command: str, output: str, sandbox: Sandbox
    ) -> None:
        result = await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command=command)
        assert result == f'[stdout]\n{output}'

    async def test_allow_interactive_permits_them(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(allow_interactive=True)
        result = await call_tool(toolset, run_context(sandbox), 'run_command', command='man() { printf ran; }; man ls')
        assert result == '[stdout]\nran'


class TestCommandPolicy:
    async def test_denied_command_is_reported_as_a_retry(self, sandbox: Sandbox) -> None:
        # A denied command is model-correctable, so it surfaces as ModelRetry (which pyai feeds
        # back to the model) rather than aborting the run.
        toolset = shell_toolset(denied_commands=('rm',))
        with pytest.raises(ModelRetry, match="Command 'rm' is denied."):
            await call_tool(toolset, run_context(sandbox), 'run_command', command='rm -rf /')

    async def test_allowlist_blocks_unlisted_commands(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(allowed_commands=('echo',))
        with pytest.raises(ModelRetry, match="Command 'cat' is not in the allowed list."):
            await call_tool(toolset, run_context(sandbox), 'run_command', command='cat file.txt')

    async def test_allowlist_permits_listed_commands(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(allowed_commands=('echo',))
        assert await call_tool(toolset, run_context(sandbox), 'run_command', command='echo hi') == '[stdout]\nhi\n'

    def test_allowlist_and_denylist_together_are_rejected(self) -> None:
        with pytest.raises(ValueError, match=r'^Specify allowed_commands or denied_commands, not both\.$'):
            shell_toolset(allowed_commands=('echo',), denied_commands=('rm',))

    def test_max_output_chars_must_be_positive(self) -> None:
        # A cap of 0 would blank every response, including start_command's ID line,
        # leaving its process unstoppable.
        with pytest.raises(ValueError, match=r'^max_output_chars must be a positive integer\.$'):
            shell_toolset(max_output_chars=0)

    @pytest.mark.parametrize('operator', ['>', '>>'])
    async def test_denied_operator_blocks_the_command(self, operator: str, sandbox: Sandbox) -> None:
        toolset = shell_toolset(denied_operators=(operator,))
        with pytest.raises(ModelRetry, match=f'Shell operator {operator!r} is not allowed.'):
            await call_tool(toolset, run_context(sandbox), 'run_command', command=f'echo hi {operator} f')

    async def test_command_free_of_denied_operators_runs(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(denied_operators=('>', '>>'))
        assert await call_tool(toolset, run_context(sandbox), 'run_command', command='echo hi') == '[stdout]\nhi\n'

    async def test_unparseable_command_skips_the_name_check(self, sandbox: Sandbox) -> None:
        # `shlex` cannot find the command name in an unterminated quote, so the shell rejects
        # the command instead of the denylist.
        toolset = shell_toolset(denied_commands=('echo',))
        result = await call_tool(toolset, run_context(sandbox), 'run_command', command="echo 'unterminated")
        assert '[exit code:' in result

    async def test_empty_command_has_no_name_to_check(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(allowed_commands=('echo',))
        assert await call_tool(toolset, run_context(sandbox), 'run_command', command='') == '(no output)'


class TestOutputRendering:
    async def test_stdout_and_stderr_are_labelled_separately(self, sandbox: Sandbox) -> None:
        result = await call_tool(
            shell_toolset(), run_context(sandbox), 'run_command', command='printf out; printf err >&2'
        )
        assert result == '[stdout]\nout\n[stderr]\nerr'

    async def test_command_without_output_says_so(self, sandbox: Sandbox) -> None:
        assert await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command='true') == '(no output)'


class TestOutputCap:
    """The model-visible cap is enforced once, at the tool dispatch seam."""

    async def test_string_tool_results_are_capped(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(max_output_chars=1)

        def text() -> str:
            return 'xx'

        toolset.add_function(text)
        assert await call_tool(toolset, run_context(sandbox), 'text') == 'x'

    async def test_non_string_tool_results_are_left_alone(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(max_output_chars=1)

        def number() -> int:
            return 42

        toolset.add_function(number)
        ctx = run_context(sandbox)
        tools = await toolset.get_tools(ctx)
        assert await toolset.call_tool('number', {}, ctx, tools['number']) == 42

    async def test_cap_keeps_the_exit_code_tail(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(max_output_chars=200)
        result = await call_tool(toolset, run_context(sandbox), 'run_command', command="printf '%0400d' 0; exit 7")
        assert len(result) == 200
        assert result.endswith('[exit code: 7]')

    async def test_cap_keeps_the_start_command_id_tail(self, tmp_path: Path, sandbox: Sandbox) -> None:
        # The ID line is the tail, so a truncated echo still leaves the process stoppable.
        toolset = background_toolset(tmp_path, max_output_chars=80)
        ctx = run_context(sandbox)
        # `sleep` keeps it running, so stopping it is what is being tested, not a race with its exit.
        result = await call_tool(toolset, ctx, 'start_command', command='sleep 30 #' + 'x' * 200)
        assert len(result) == 80
        assert await call_tool(toolset, ctx, 'stop_command', command_id=command_id(result)) == '(no output)\n[stopped]'


class TestTimeouts:
    async def test_default_timeout_applies_when_none_is_given(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(default_timeout=0.05)
        result = await call_tool(toolset, run_context(sandbox), 'run_command', command='sleep 10')
        assert result == '[Command timed out after 0.05s]'

    @pytest.mark.parametrize('field', ['default_timeout', 'max_timeout'])
    @pytest.mark.parametrize('value', [0.0, -1.0, math.inf, math.nan], ids=['zero', 'negative', 'infinity', 'nan'])
    def test_timeout_configuration_must_be_positive_and_finite(
        self, field: Literal['default_timeout', 'max_timeout'], value: float
    ) -> None:
        with pytest.raises(ValueError, match=f'{field} must be a positive finite number'):
            if field == 'default_timeout':
                shell_toolset(default_timeout=value)
            else:
                shell_toolset(max_timeout=value)

    def test_default_timeout_cannot_exceed_max_timeout(self) -> None:
        with pytest.raises(ValueError, match='default_timeout must not exceed max_timeout'):
            shell_toolset(default_timeout=2.0, max_timeout=1.0)

    async def test_model_timeout_above_maximum_recommends_start_command(self, tmp_path: Path) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = shell_toolset(default_timeout=1.0, max_timeout=1.0)
            with pytest.raises(ModelRetry, match='at most 1.0.*start_command'):
                await call_tool(
                    toolset, run_context(Sandbox.wrap(backend)), 'run_command', command='true', timeout_seconds=2
                )
            assert backend.timeouts == []

    async def test_model_timeout_at_maximum_is_forwarded(self, tmp_path: Path) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = shell_toolset(default_timeout=1.0, max_timeout=1.0)
            await call_tool(
                toolset, run_context(Sandbox.wrap(backend)), 'run_command', command='true', timeout_seconds=1.0
            )
            assert backend.timeouts == [1.0]

    async def test_tool_description_uses_the_configured_default(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(default_timeout=12.5)
        tools = await toolset.get_tools(run_context(sandbox))
        description = str(tools['run_command'].tool_def.parameters_json_schema)
        assert 'configured default' in description
        assert 'default: 30' not in description
        assert 'Each command starts in the configured working directory' in str(
            tools['run_command'].tool_def.description
        )


class TestWorkingDirectory:
    async def test_each_command_starts_in_the_configured_directory(self, tmp_path: Path, sandbox: Sandbox) -> None:
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'root.txt').write_text('root\n')
        toolset = shell_toolset()
        ctx = run_context(sandbox)
        await call_tool(toolset, ctx, 'run_command', command='cd sub')
        assert await call_tool(toolset, ctx, 'run_command', command='pwd') == f'[stdout]\n{tmp_path}\n'

        filesystem = FileSystemToolset(
            root_dir=tmp_path,
            allowed_patterns=(),
            denied_patterns=(),
            protected_patterns=(),
            max_read_lines=100,
            max_list_results=100,
            max_search_results=100,
            max_find_results=100,
        )
        tools = await filesystem.get_tools(ctx)
        result: object = await filesystem.call_tool('read_file', {'path': 'root.txt'}, ctx, tools['read_file'])
        assert isinstance(result, str)
        assert result.startswith('[root.txt | 1 lines | hash:')
        assert result.endswith('\n     1\troot\n')


class TestForRunIsolation:
    """`for_run` gives each run independent background-process state."""

    async def test_each_run_keeps_the_configured_environment(self, sandbox: Sandbox) -> None:
        shared = shell_toolset(
            env={'HARNESS_VISIBLE': 'yes', 'HARNESS_DENIED': 'secret'},
            denied_env_patterns=('HARNESS_DENIED',),
        )
        ctx = run_context(sandbox)
        per_run = await shared.for_run(ctx)
        assert isinstance(per_run, ShellToolset)
        result = await call_tool(
            per_run,
            ctx,
            'run_command',
            command='printf \'%s:%s\' "$HARNESS_VISIBLE" "${HARNESS_DENIED-absent}"',
        )
        assert result == '[stdout]\nyes:absent'


class TestSpawnFailures:
    """Failures raised by the spawn itself, split by whose mistake they are."""

    async def test_missing_working_directory_is_a_path_free_retry(self, tmp_path: Path, sandbox: Sandbox) -> None:
        # The model's own earlier command can do this: `mv "$PWD" "$PWD-old"` passes the
        # denylist, which only inspects the first token.
        (tmp_path / 'sub').mkdir()
        toolset = shell_toolset(Path('sub'))
        shutil.rmtree(tmp_path / 'sub')
        with pytest.raises(ModelRetry, match='The working directory no longer exists.') as exc_info:
            await call_tool(toolset, run_context(sandbox), 'run_command', command='echo hello')
        assert str(tmp_path) not in str(exc_info.value)  # the retry prompt names no sandbox path

    async def test_working_directory_replaced_by_a_file_is_a_retry(self, tmp_path: Path, sandbox: Sandbox) -> None:
        (tmp_path / 'sub').write_text('not a directory\n')
        toolset = shell_toolset(Path('sub'))
        with pytest.raises(ModelRetry, match='The working directory is no longer a directory.'):
            await call_tool(toolset, run_context(sandbox), 'run_command', command='echo hello')

    @pytest.mark.parametrize(
        ('command', 'expected'),
        [('echo hi\x00there', 'NUL byte'), ('echo \ud800', 'cannot be encoded for the operating system')],
    )
    async def test_unspawnable_command_string_is_a_retry(self, command: str, expected: str, sandbox: Sandbox) -> None:
        with pytest.raises(ModelRetry, match=expected):
            await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command=command)

    @pytest.mark.parametrize('escaped', ['\udc80', '\udcff'])
    async def test_surrogateescape_command_still_runs(self, escaped: str, sandbox: Sandbox) -> None:
        # The spawn encodes with `surrogateescape`, which round-trips this range back to the raw
        # byte it came from. Screening the command as plain UTF-8 would reject a command that runs.
        result = await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command=f'echo {escaped}')
        assert '[exit code' not in result

    @pytest.mark.parametrize('env', [{'FOO': 'bar\x00baz'}, {'FO\x00O': 'bar'}, {'FOO': 'bar\ud800'}])
    async def test_unspawnable_environment_aborts_the_run(self, env: dict[str, str], sandbox: Sandbox) -> None:
        # The spawn reports a NUL or an unencodable character the same way wherever it came from.
        # This one came from the application's `env`, so the model cannot fix it and must not retry.
        toolset = shell_toolset(env=env)
        with pytest.raises(ValueError):
            await call_tool(toolset, run_context(sandbox), 'run_command', command='echo hello')


class TestShellCapability:
    async def test_registers_the_four_command_tools(self, sandbox: Sandbox) -> None:
        toolset = Shell[None]().get_toolset()
        assert set(await toolset.get_tools(run_context(sandbox))) == {
            'run_command',
            'start_command',
            'check_command',
            'stop_command',
        }

    def test_defaults(self) -> None:
        shell = Shell[None]()
        assert (shell.cwd, shell.default_timeout, shell.max_timeout, shell.max_output_chars) == (
            '.',
            30.0,
            600.0,
            50_000,
        )
        assert shell.allow_interactive is False
        assert (shell.env, list(shell.denied_env_patterns)) == (None, [])

    @pytest.mark.parametrize(
        'destructive', ['rm', 'rmdir', 'mkfs', 'dd', 'format', 'shutdown', 'reboot', 'halt', 'poweroff', 'init']
    )
    async def test_default_denylist_blocks_destructive_commands(self, destructive: str, sandbox: Sandbox) -> None:
        toolset = Shell[None]().get_toolset()
        with pytest.raises(ModelRetry, match=f"Command '{destructive}' is denied."):
            await call_tool(toolset, run_context(sandbox), 'run_command', command=f'{destructive} --version')

    async def test_empty_allowlist_keeps_the_default_denylist(self, sandbox: Sandbox) -> None:
        toolset = Shell[None](allowed_commands=[]).get_toolset()
        with pytest.raises(ModelRetry, match="Command 'rm' is denied."):
            await call_tool(toolset, run_context(sandbox), 'run_command', command='rm --version')

    async def test_allowlist_replaces_the_default_denylist(self, sandbox: Sandbox) -> None:
        # Without the swap in `__post_init__` the two lists would collide and `get_toolset` raise.
        toolset = Shell[None](allowed_commands=['echo']).get_toolset()
        assert await call_tool(toolset, run_context(sandbox), 'run_command', command='echo hi') == '[stdout]\nhi\n'

    def test_allowlist_with_an_explicitly_passed_default_denylist_is_rejected(self) -> None:
        shell = Shell[None](allowed_commands=['rm'], denied_commands=Shell[None]().denied_commands)
        with pytest.raises(ValueError, match='Specify allowed_commands or denied_commands'):
            shell.get_toolset()

    def test_environment_denylist_requires_an_explicit_environment(self) -> None:
        with pytest.raises(ValueError, match='denied_env_patterns requires an explicit env mapping'):
            Shell(denied_env_patterns=['SECRET_*'])

    async def test_capability_passes_the_environment_to_the_toolset(self, sandbox: Sandbox) -> None:
        toolset = Shell[None](
            env={'HARNESS_VISIBLE': 'yes', 'HARNESS_DENIED': 'secret'},
            denied_env_patterns=['HARNESS_DENIED'],
        ).get_toolset()
        result = await call_tool(
            toolset,
            run_context(sandbox),
            'run_command',
            command='printf \'%s:%s\' "$HARNESS_VISIBLE" "${HARNESS_DENIED-absent}"',
        )
        assert result == '[stdout]\nyes:absent'

    async def test_llm_api_key_patterns_strip_provider_credentials(self, sandbox: Sandbox) -> None:
        secrets = {pattern.replace('*', 'KEY'): 'leak-me' for pattern in LLM_API_KEY_ENV_PATTERNS}
        toolset = Shell[None](
            env={**secrets, 'HARNESS_KEEP': 'kept', 'PATH': os.environ['PATH']},
            denied_env_patterns=list(LLM_API_KEY_ENV_PATTERNS),
        ).get_toolset()
        result = await call_tool(toolset, run_context(sandbox), 'run_command', command='env')
        assert 'leak-me' not in result
        assert 'HARNESS_KEEP=kept' in result  # a name no pattern matches is still passed through


async def _tools_offered_to_model(*, shell_first: bool) -> dict[str, str | None]:
    """Run an agent with Shell and CodeMode and return the tools the model was offered."""
    offered: dict[str, str | None] = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.update({tool.name: tool.description for tool in info.function_tools})
        return ModelResponse(parts=[TextPart('done')])

    shell = Shell[object]()
    code_mode = CodeMode[object]()
    capabilities: list[AbstractCapability[object]] = [shell, code_mode] if shell_first else [code_mode, shell]
    agent: Agent[None, str] = Agent(FunctionModel(capture), capabilities=capabilities)
    await agent.run('go')
    return offered


class TestCodeModeInterop:
    """`run_command` and `start_command` take a command line, so CodeMode leaves them native.

    Folding them into `run_code` would make the model write a Monty script whose argument is a
    shell script quoted as a Python string. The command-id tools carry no command line, so they
    stay sandboxed like any other tool.
    """

    @pytest.mark.parametrize('shell_first', [True, False], ids=['shell-first', 'code-mode-first'])
    async def test_command_tools_stay_native(self, shell_first: bool) -> None:
        tools = await _tools_offered_to_model(shell_first=shell_first)

        assert 'run_command' in tools
        assert 'start_command' in tools
        run_code_description = tools['run_code']
        assert run_code_description is not None
        assert 'async def run_command' not in run_code_description
        assert 'async def start_command' not in run_code_description

    @pytest.mark.parametrize('shell_first', [True, False], ids=['shell-first', 'code-mode-first'])
    async def test_command_id_tools_are_still_sandboxed(self, shell_first: bool) -> None:
        tools = await _tools_offered_to_model(shell_first=shell_first)

        assert 'check_command' not in tools
        assert 'stop_command' not in tools
        run_code_description = tools['run_code']
        assert run_code_description is not None
        assert 'async def check_command' in run_code_description
        assert 'async def stop_command' in run_code_description


class _RecordingLocalBackend:
    def __init__(self, backend: LocalSandbox) -> None:
        self.backend = backend
        self.remove_error: RuntimeError | None = None
        self.environments: list[Mapping[str, str] | None] = []
        self.timeouts: list[float | None] = []
        self.run_error: Exception | None = None
        self.raise_after_kill = False
        self.kill_failure: str | None = None
        self.kill_failure_stderr = 'kill failed'
        self.hold_on_term = False
        self.tail_failure = False

    @property
    def ref(self) -> SandboxRef | None:
        return self.backend.ref

    async def working_dir(self) -> str:
        return await self.backend.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self.backend.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.backend.write_bytes(path, data)

    async def stat(self, path: str) -> SandboxFileEntry:
        return await self.backend.stat(path)

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
        return await self.backend.list_dir(path)

    async def make_dir(self, path: str) -> None:
        await self.backend.make_dir(path)

    async def remove(self, path: str) -> None:
        if self.remove_error is not None:
            raise self.remove_error
        await self.backend.remove(path)

    async def exists(self, path: str) -> bool:
        return await self.backend.exists(path)

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        self.environments.append(env)
        self.timeouts.append(timeout)
        if self.run_error is not None:
            raise self.run_error
        if self.kill_failure is not None and not isinstance(command, str) and command[0] == 'kill':
            # A held TERM reports success without signalling, so the escalation to KILL is reached.
            if self.hold_on_term and command[1] == '-TERM':
                return CommandResult(exit_code=0, stdout='', stderr='')
            return CommandResult(exit_code=1, stdout='', stderr=self.kill_failure_stderr)
        if self.tail_failure and not isinstance(command, str) and command[0] == 'tail':
            return CommandResult(exit_code=1, stdout='', stderr='tail failed')
        result = await self.backend.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
        if self.raise_after_kill and not isinstance(command, str) and command[:2] == ['kill', '-TERM']:
            raise RuntimeError('cleanup failed')
        return result


async def test_recording_backend_delegates_the_complete_flat_filesystem(tmp_path: Path) -> None:
    async with LocalSandbox(root=tmp_path) as local:
        backend = _RecordingLocalBackend(local)
        directory = str(tmp_path / 'nested')
        path = f'{directory}/file.txt'

        await backend.make_dir(directory)
        await backend.write_bytes(path, b'data')

        assert await backend.read_bytes(path) == b'data'
        assert (await backend.stat(path)).size == 4
        assert [entry.name for entry in await backend.list_dir(directory)] == ['file.txt']
        assert await backend.exists(path) is True

        await backend.remove(path)
        assert await backend.exists(path) is False


class _FailingBackend:
    ref = SandboxRef(sandbox_id='failing-1')

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def working_dir(self) -> str:
        return '/work'

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        raise self.error


class _TimeoutBackend(_FailingBackend):
    def __init__(self) -> None:
        super().__init__(SandboxTimeoutError('timed out', stdout='before\n', stderr='problem\n'))


class _ResultBackend(_FailingBackend):
    def __init__(self, stdout: str, stderr: str = '') -> None:
        super().__init__(RuntimeError('unused'))
        self.stdout = stdout
        self.stderr = stderr

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        return CommandResult(exit_code=0, stdout=self.stdout, stderr=self.stderr)


async def _await_finished(toolset: ShellToolset[None], ctx: RunContext[None], started_id: str) -> str:
    """Poll `check_command` until the background command reports it has finished."""
    with anyio.fail_after(5):
        while True:
            result = await call_tool(toolset, ctx, 'check_command', command_id=started_id)
            if '[status: finished]' in result:
                return result
            await anyio.sleep(0.02)


class TestRunCommand:
    async def test_runs_in_sandbox_root_and_labels_output(self, tmp_path: Path, sandbox: Sandbox) -> None:
        result = await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command='pwd')
        assert result == f'[stdout]\n{tmp_path}\n'

    async def test_relative_cwd_resolves_against_the_sandbox_working_directory(
        self, tmp_path: Path, sandbox: Sandbox
    ) -> None:
        (tmp_path / 'sub').mkdir()
        result = await call_tool(shell_toolset(Path('sub')), run_context(sandbox), 'run_command', command='pwd')
        assert result == f'[stdout]\n{tmp_path / "sub"}\n'

    async def test_nonzero_exit_code_is_rendered(self, sandbox: Sandbox) -> None:
        result = await call_tool(
            shell_toolset(), run_context(sandbox), 'run_command', command='printf error >&2; exit 7'
        )
        assert result == '[stderr]\nerror\n[exit code: 7]'

    async def test_default_environment_selection_is_delegated_to_the_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('HARNESS_HOST_ONLY', 'secret')
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            await call_tool(shell_toolset(), run_context(Sandbox.wrap(backend)), 'run_command', command='true')
        assert backend.environments == [None]

    async def test_explicit_environment_is_filtered(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset(
            env={'HARNESS_VISIBLE': 'yes', 'HARNESS_DENIED': 'secret'},
            denied_env_patterns=('HARNESS_DENIED',),
        )
        result = await call_tool(
            toolset,
            run_context(sandbox),
            'run_command',
            command='printf \'%s:%s\' "$HARNESS_VISIBLE" "${HARNESS_DENIED-absent}"',
        )
        assert result == '[stdout]\nyes:absent'

    async def test_timeout_is_returned(self, sandbox: Sandbox) -> None:
        result = await call_tool(
            shell_toolset(), run_context(sandbox), 'run_command', command='sleep 10', timeout_seconds=0.05
        )
        assert result == '[Command timed out after 0.05s]'

    async def test_timeout_includes_partial_output(self) -> None:
        result = await call_tool(
            shell_toolset(), run_context(Sandbox.wrap(_TimeoutBackend())), 'run_command', command='slow'
        )
        assert result == '[stdout]\nbefore\n\n[stderr]\nproblem\n\n[Command timed out after 10.0s]'

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (SandboxError('temporary failure'), ModelRetry),
            (SandboxUnavailableError('gone'), SandboxUnavailableError),
            (RuntimeError('backend bug'), RuntimeError),
        ],
    )
    async def test_error_mapping(self, error: RuntimeError, expected: type[RuntimeError]) -> None:
        sandbox = Sandbox.wrap(_FailingBackend(error))
        assert await sandbox.working_dir() == '/work'
        with pytest.raises(expected, match=str(error)):
            await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command='echo hello')

    async def test_non_recoverable_errno_propagates(self) -> None:
        # A sandbox that cannot fork is not something the model can retry its way out of, so it
        # must keep aborting the run rather than starting an unwinnable retry loop.
        sandbox = Sandbox.wrap(_FailingBackend(OSError(errno.ENOMEM, 'Cannot allocate memory')))
        with pytest.raises(OSError, match='Cannot allocate memory'):
            await call_tool(shell_toolset(), run_context(sandbox), 'run_command', command='echo hello')

    async def test_missing_sandbox_asks_the_application_to_attach_one(self) -> None:
        # The unavailable default raises `UserError`, itself a `RuntimeError`: without an explicit
        # re-raise the attachment instructions would reach the model as a retry prompt instead.
        with pytest.raises(UserError, match='No sandbox is attached'):
            await call_tool(shell_toolset(), run_context(), 'run_command', command='echo hello')


class TestBackgroundCommands:
    async def test_short_command_finishes_with_output(self, tmp_path: Path, sandbox: Sandbox) -> None:
        toolset = background_toolset(tmp_path)
        ctx = run_context(sandbox)
        started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='printf background; false'))

        assert await _await_finished(toolset, ctx, started_id) == (
            '[stdout]\nbackground\n[status: finished]\n[exit code: 1]'
        )
        await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_exit_command_records_its_exit_code(self, tmp_path: Path, sandbox: Sandbox) -> None:
        toolset = background_toolset(tmp_path)
        ctx = run_context(sandbox)
        started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='exit 3'))

        assert await _await_finished(toolset, ctx, started_id) == '(no output yet)\n[status: finished]\n[exit code: 3]'
        await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_stderr_and_junk_exit_capture_are_rendered_as_running(self, tmp_path: Path, sandbox: Sandbox) -> None:
        toolset = background_toolset(tmp_path)
        ctx = run_context(sandbox)
        started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='printf problem >&2; sleep 30'))
        await sandbox.write_bytes(f'/tmp/harness_{started_id}_ec', b'junk')

        with anyio.fail_after(5):
            while True:
                result = await call_tool(toolset, ctx, 'check_command', command_id=started_id)
                if '[stderr]\nproblem' in result:
                    break
                await anyio.sleep(0.02)  # pragma: no cover - retry timing is race-dependent
        assert result == '[stderr]\nproblem\n[status: running]'
        await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_missing_output_files_are_empty(self, tmp_path: Path, sandbox: Sandbox) -> None:
        toolset = background_toolset(tmp_path)
        ctx = run_context(sandbox)
        started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
        await sandbox.remove(f'/tmp/harness_{started_id}_out')
        await sandbox.remove(f'/tmp/harness_{started_id}_err')

        result = await call_tool(toolset, ctx, 'check_command', command_id=started_id)
        assert result == '(no output yet)\n[status: running]'
        await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_starts_in_the_configured_directory_with_the_configured_environment(
        self, tmp_path: Path, sandbox: Sandbox
    ) -> None:
        (tmp_path / 'sub').mkdir()
        toolset = background_toolset(tmp_path / 'sub', env={'BG_TOKEN': 'bg-present'})
        ctx = run_context(sandbox)
        start = await call_tool(toolset, ctx, 'start_command', command='printf \'%s %s\' "$BG_TOKEN" "$(pwd)"')
        started_id = command_id(start)

        assert await _await_finished(toolset, ctx, started_id) == (
            f'[stdout]\nbg-present {tmp_path / "sub"}\n[status: finished]\n[exit code: 0]'
        )
        await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_stop_kills_command_and_removes_record(self, tmp_path: Path, sandbox: Sandbox) -> None:
        toolset = background_toolset(tmp_path)
        ctx = run_context(sandbox)
        started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))

        result = await call_tool(toolset, ctx, 'stop_command', command_id=started_id)
        assert result == '(no output)\n[stopped]'
        assert await call_tool(toolset, ctx, 'check_command', command_id=started_id) == (
            f'[Error: unknown command ID {started_id!r}]'
        )

    async def test_exit_cleans_up_unfinished_command(self, tmp_path: Path, sandbox: Sandbox) -> None:
        toolset = background_toolset(tmp_path)
        ctx = run_context(sandbox)
        started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
        paths = [Path(f'/tmp/harness_{started_id}_{suffix}') for suffix in ('out', 'err', 'ec')]

        await toolset.__aexit__(None, None, None)

        assert not any(path.exists() for path in paths)
        assert await call_tool(toolset, ctx, 'check_command', command_id=started_id) == (
            f'[Error: unknown command ID {started_id!r}]'
        )

    async def test_exit_cleans_up_finished_command(self, tmp_path: Path, sandbox: Sandbox) -> None:
        toolset = background_toolset(tmp_path)
        ctx = run_context(sandbox)
        started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='true'))
        await _await_finished(toolset, ctx, started_id)

        await toolset.__aexit__(None, None, None)
        assert await call_tool(toolset, ctx, 'check_command', command_id=started_id) == (
            f'[Error: unknown command ID {started_id!r}]'
        )

    async def test_exit_surfaces_sandbox_cleanup_failure_and_keeps_record(self, tmp_path: Path) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            sandbox = Sandbox.wrap(backend)
            toolset = background_toolset(tmp_path)
            ctx = run_context(sandbox)
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.remove_error = RuntimeError('cleanup failed')
            # The protocol members the shell tools never consult still work through the facade.
            assert sandbox.ref == local.ref
            assert await sandbox.working_dir() == str(tmp_path)

            with pytest.raises(RuntimeError, match='cleanup failed'):
                await toolset.__aexit__(None, None, None)

            assert await call_tool(toolset, ctx, 'check_command', command_id=started_id) == (
                '(no output yet)\n[status: running]'
            )
            backend.remove_error = None
            await toolset.__aexit__(None, None, None)

            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.run_error = RuntimeError('kill cleanup failed')
            with pytest.raises(RuntimeError, match='kill cleanup failed'):
                await toolset.__aexit__(None, None, None)
            backend.run_error = None
            await toolset.__aexit__(None, None, None)

    async def test_background_output_read_failure_is_reported(self, tmp_path: Path) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.tail_failure = True

            with pytest.raises(RuntimeError, match='tail failed'):
                await call_tool(toolset, ctx, 'check_command', command_id=started_id)
            backend.tail_failure = False
            await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_exit_surfaces_kill_failure_and_keeps_record(self, tmp_path: Path) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.raise_after_kill = True

            with pytest.raises(RuntimeError, match='cleanup failed'):
                await call_tool(toolset, ctx, 'stop_command', command_id=started_id)
            assert '[status: running]' in await call_tool(toolset, ctx, 'check_command', command_id=started_id)
            backend.raise_after_kill = False
            await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_missing_background_output_file_is_empty(self, tmp_path: Path) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.run_error = FileNotFoundError('missing')

            assert await call_tool(toolset, ctx, 'check_command', command_id=started_id) == (
                '(no output yet)\n[status: running]'
            )
            backend.run_error = None
            await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    @pytest.mark.parametrize('signal', ['-TERM', '-KILL'])
    async def test_stop_surfaces_kill_failure_and_keeps_record(self, tmp_path: Path, signal: str) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.kill_failure = signal
            backend.hold_on_term = signal == '-KILL'

            with pytest.raises(RuntimeError, match='kill failed'):
                await call_tool(toolset, ctx, 'stop_command', command_id=started_id)
            assert '[status: running]' in await call_tool(toolset, ctx, 'check_command', command_id=started_id)
            backend.kill_failure = None
            await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    @pytest.mark.parametrize('signal', ['-TERM', '-KILL'])
    async def test_stop_accepts_a_kill_that_failed_because_the_group_is_gone(self, tmp_path: Path, signal: str) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.kill_failure = signal
            backend.kill_failure_stderr = 'kill: No such process'
            backend.hold_on_term = signal == '-KILL'

            assert await call_tool(toolset, ctx, 'stop_command', command_id=started_id) == '(no output)\n[stopped]'

    async def test_stop_accepts_successful_kill(self, tmp_path: Path) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.hold_on_term = True

            await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    async def test_unknown_id_messages_are_unchanged(self, sandbox: Sandbox) -> None:
        toolset = shell_toolset()
        ctx = run_context(sandbox)
        assert await call_tool(toolset, ctx, 'check_command', command_id='missing') == (
            "[Error: unknown command ID 'missing']"
        )
        assert await call_tool(toolset, ctx, 'stop_command', command_id='missing') == (
            "[Error: unknown command ID 'missing']"
        )

    @pytest.mark.parametrize(
        ('backend', 'message', 'expected'),
        [
            (_ResultBackend('not-a-pid', 'setsid failed'), 'setsid failed', ModelRetry),
            (_ResultBackend('', ''), 'Sandbox did not return a background process ID.', ModelRetry),
            (_FailingBackend(SandboxError('temporary failure')), 'temporary failure', ModelRetry),
            (_FailingBackend(SandboxUnavailableError('gone')), 'gone', SandboxUnavailableError),
            (_FailingBackend(RuntimeError('backend bug')), 'backend bug', RuntimeError),
        ],
    )
    async def test_start_error_mapping(
        self, backend: _FailingBackend, message: str, expected: type[BaseException]
    ) -> None:
        with pytest.raises(expected, match=message):
            await call_tool(shell_toolset(), run_context(Sandbox.wrap(backend)), 'start_command', command='echo hello')

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (SandboxError('temporary failure'), ModelRetry),
            (SandboxUnavailableError('gone'), SandboxUnavailableError),
            (RuntimeError('backend bug'), RuntimeError),
        ],
    )
    async def test_check_error_mapping(
        self,
        tmp_path: Path,
        error: RuntimeError,
        expected: type[RuntimeError],
    ) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.run_error = error
            with pytest.raises(expected, match=str(error)):
                await call_tool(toolset, ctx, 'check_command', command_id=started_id)
            backend.run_error = None
            await call_tool(toolset, ctx, 'stop_command', command_id=started_id)

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (SandboxError('temporary failure'), ModelRetry),
            (SandboxUnavailableError('gone'), SandboxUnavailableError),
            (RuntimeError('backend bug'), RuntimeError),
        ],
    )
    async def test_stop_error_mapping(
        self,
        tmp_path: Path,
        error: RuntimeError,
        expected: type[RuntimeError],
    ) -> None:
        async with LocalSandbox(root=tmp_path) as local:
            backend = _RecordingLocalBackend(local)
            toolset = background_toolset(tmp_path)
            ctx = run_context(Sandbox.wrap(backend))
            started_id = command_id(await call_tool(toolset, ctx, 'start_command', command='sleep 30'))
            backend.run_error = error
            with pytest.raises(expected, match=str(error)):
                await call_tool(toolset, ctx, 'stop_command', command_id=started_id)
            backend.run_error = None
            await call_tool(toolset, ctx, 'stop_command', command_id=started_id)


async def test_shell_capability_runs_through_an_agent_with_a_sandbox(tmp_path: Path) -> None:
    responses = [
        ModelResponse(parts=[ToolCallPart('run_command', {'command': 'printf hello'})]),
        ModelResponse(parts=[TextPart('done')]),
    ]

    def model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return responses.pop(0)

    async with LocalSandbox(root=tmp_path) as sandbox:
        result = await Agent(FunctionModel(model), capabilities=[Shell()]).run('run', sandbox=sandbox)

    assert result.output == 'done'


async def test_shell_capability_requires_a_sandbox_on_the_public_agent_path() -> None:
    with pytest.raises(UserError, match='No sandbox is attached'):
        await Agent(TestModel(call_tools=['run_command']), capabilities=[Shell()]).run('run')
