"""Tests for the Shell capability and ShellToolset."""

from __future__ import annotations

import errno
import json
import logging
import os
import shlex
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import patch

import anyio
import anyio.to_thread
import pytest
import sniffio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness._workspace import READ_ONLY_FAILURE
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.shell._policy import is_interactive_command
from pydantic_ai_harness.shell._toolset import ShellToolset

from .._tool_calls import call_tool
from .._workspace import local_workspace


def _env_toolset(
    shell_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
    denied_env_patterns: Sequence[str] = (),
) -> ShellToolset[None]:
    """Build a ShellToolset wired for env-control tests, with safe defaults."""
    return ShellToolset(
        cwd=shell_dir,
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=10.0,
        max_output_chars=50_000,
        persist_cwd=False,
        allow_interactive=False,
        env=env,
        denied_env_patterns=denied_env_patterns,
    )


def _shell_toolset(
    shell_dir: Path,
    *,
    max_output_chars: int = 50_000,
    default_timeout: float = 10.0,
) -> ShellToolset[None]:
    return ShellToolset(
        cwd=shell_dir,
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=default_timeout,
        max_output_chars=max_output_chars,
        persist_cwd=False,
        allow_interactive=False,
    )


def _raise_oserror(code: int, message: str) -> Callable[..., Awaitable[NoReturn]]:
    """Build a stand-in for `anyio.open_process` that fails with a given errno."""

    async def fail(*args: object, **kwargs: object) -> NoReturn:
        raise OSError(code, message)

    return fail


def _read_env_var(name: str) -> str:
    """Shell command that prints an env var's value, or ABSENT if unset."""
    return f'{sys.executable} -c "import os; print(os.environ.get({name!r}, \'ABSENT\'))"'


def _run_context(workspace: Workspace | None = None) -> RunContext[None]:
    """Minimal `RunContext` for invoking toolset methods directly in tests, with a local workspace."""
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        workspace=workspace if workspace is not None else Workspace(local_workspace('/')),
    )


def _ctx() -> RunContext[None]:
    """A run context whose workspace is this machine; toolsets under test are configured with absolute cwds."""
    return _run_context()


async def _call_shell_tool(toolset: ShellToolset[None], name: str, **tool_args: Any) -> str:
    ctx = _run_context()
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool(name, tool_args, ctx, tools[name])
    assert isinstance(result, str)
    return result


def _parse_command_id(result: str) -> str:
    assert 'ID: ' in result, f'Expected "ID: " in result: {result!r}'
    return result.split('ID: ')[1].strip()


class TestIsInteractiveCommand:
    def test_vi(self) -> None:
        assert is_interactive_command('vi file.txt') is True

    def test_vim(self) -> None:
        assert is_interactive_command('vim file.txt') is True

    def test_nano(self) -> None:
        assert is_interactive_command('nano file.txt') is True

    def test_less(self) -> None:
        assert is_interactive_command('less file.txt') is True

    def test_top(self) -> None:
        assert is_interactive_command('top') is True

    def test_sudo(self) -> None:
        assert is_interactive_command('sudo rm -rf /') is True

    def test_ssh(self) -> None:
        assert is_interactive_command('ssh host') is True

    def test_regular_command(self) -> None:
        assert is_interactive_command('ls -la') is False

    def test_echo(self) -> None:
        assert is_interactive_command('echo hello') is False

    def test_grep(self) -> None:
        assert is_interactive_command('grep pattern file') is False

    def test_emacs(self) -> None:
        assert is_interactive_command('emacs file.txt') is True

    def test_man(self) -> None:
        assert is_interactive_command('man ls') is True

    def test_htop(self) -> None:
        assert is_interactive_command('htop') is True

    def test_telnet(self) -> None:
        assert is_interactive_command('telnet localhost 80') is True

    def test_ftp(self) -> None:
        assert is_interactive_command('ftp host') is True

    def test_passwd(self) -> None:
        assert is_interactive_command('passwd') is True

    def test_more(self) -> None:
        assert is_interactive_command('more file.txt') is True

    def test_not_prefix_match(self) -> None:
        assert is_interactive_command('view file.txt') is False
        assert is_interactive_command('vishnu') is False

    def test_leading_spaces(self) -> None:
        assert is_interactive_command('  vi file.txt') is True
        assert is_interactive_command('  sudo rm') is True


@pytest.fixture
def shell_dir(tmp_path: Path) -> Path:
    (tmp_path / 'test.txt').write_text('hello\n')
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'nested.txt').write_text('nested\n')
    return tmp_path


@pytest.fixture
def toolset(shell_dir: Path) -> ShellToolset[None]:
    return ShellToolset(
        cwd=shell_dir,
        allowed_commands=[],
        denied_commands=['rm', 'rmdir'],
        denied_operators=[],
        default_timeout=10.0,
        max_output_chars=50_000,
        persist_cwd=False,
        allow_interactive=False,
    )


@pytest.fixture
def persist_toolset(shell_dir: Path) -> ShellToolset[None]:
    return ShellToolset(
        cwd=shell_dir,
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=10.0,
        max_output_chars=50_000,
        persist_cwd=True,
        allow_interactive=False,
    )


class TestCommandValidation:
    async def test_denied_command_blocked(self, toolset: ShellToolset[None]) -> None:
        with pytest.raises(PermissionError, match="'rm' is denied"):
            toolset._check_command('rm -rf /')

    async def test_allowed_command_permitted(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=['echo', 'cat'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('echo hello')
        ts._check_command('cat file.txt')

    async def test_allowed_blocks_non_matching(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=['echo'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match='not in the allowed list'):
            ts._check_command('cat file.txt')

    async def test_both_allow_and_deny_raises(self, shell_dir: Path) -> None:
        with pytest.raises(ValueError, match='Specify allowed_commands or denied_commands'):
            ShellToolset(
                cwd=shell_dir,
                allowed_commands=['echo'],
                denied_commands=['rm'],
                denied_operators=[],
                default_timeout=10.0,
                max_output_chars=50_000,
                persist_cwd=False,
                allow_interactive=False,
            )

    async def test_interactive_blocked_by_default(self, toolset: ShellToolset[None]) -> None:
        with pytest.raises(PermissionError, match='Interactive commands'):
            toolset._check_command('vim file.txt')

    async def test_interactive_allowed_when_enabled(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=True,
        )
        ts._check_command('vim file.txt')

    async def test_denied_operator_blocked(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>', '>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match="'>' is not allowed"):
            ts._check_command('echo hello > file.txt')

    async def test_denied_operator_passes_when_not_present(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>', '>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('echo hello')

    async def test_unparseable_command_allowed(self, toolset: ShellToolset[None]) -> None:
        toolset._check_command("echo 'unterminated")

    async def test_empty_command_allowed(self, toolset: ShellToolset[None]) -> None:
        toolset._check_command('')

    async def test_denied_operator_substring_match(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['>>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(PermissionError, match="'>>' is not allowed"):
            ts._check_command('echo hello >> file.txt')

    async def test_shlex_error_returns_early(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=['rm'],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command("echo 'unterminated")

    async def test_empty_tokens(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=['echo'],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ts._check_command('')

    def test_first_denied_operator_match(self, toolset: ShellToolset[None]) -> None:
        ts = ShellToolset(
            cwd=Path('/tmp'),
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['|', '>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        assert ts._first_denied_operator('echo hi | cat') == '|'

    def test_first_denied_operator_no_match(self, toolset: ShellToolset[None]) -> None:
        ts = ShellToolset(
            cwd=Path('/tmp'),
            allowed_commands=[],
            denied_commands=[],
            denied_operators=['|', '>'],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        assert ts._first_denied_operator('echo hello') is None

    def test_first_denied_operator_empty_list(self, toolset: ShellToolset[None]) -> None:
        assert toolset._first_denied_operator('echo hi | cat') is None


class TestCwdCapture:
    """The persistent-cwd mechanism records `pwd` out-of-band to a private file inside the
    workspace, so command output can never spoof the tracked directory."""

    async def test_capture_disabled_returns_command_unchanged(self, toolset: ShellToolset[None]) -> None:
        wrapped, cwd_file = await toolset._build_cwd_capture(_ctx(), 'echo hi')
        assert wrapped == 'echo hi'
        assert cwd_file is None

    async def test_capture_records_pwd_out_of_band(self, persist_toolset: ShellToolset[None]) -> None:
        wrapped, cwd_file = await persist_toolset._build_cwd_capture(_ctx(), 'echo hi')
        assert cwd_file is not None
        # pwd is redirected to the private capture file, never echoed to stdout
        assert f'pwd > {shlex.quote(cwd_file)}' in wrapped
        assert wrapped.startswith('echo hi')

    async def test_apply_valid_dir_updates_cwd(
        self, persist_toolset: ShellToolset[None], shell_dir: Path, tmp_path: Path
    ) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text(f'{shell_dir / "subdir"}\n')
        await persist_toolset._apply_captured_cwd(_ctx(), str(capture))
        assert persist_toolset._cwd == str(shell_dir / 'subdir')

    async def test_apply_empty_file_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text('')
        await persist_toolset._apply_captured_cwd(_ctx(), str(capture))
        assert persist_toolset._cwd is None

    async def test_apply_non_dir_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text(str(tmp_path / 'does_not_exist'))
        await persist_toolset._apply_captured_cwd(_ctx(), str(capture))
        assert persist_toolset._cwd is None

    async def test_apply_file_keeps_cwd(
        self, persist_toolset: ShellToolset[None], shell_dir: Path, tmp_path: Path
    ) -> None:
        capture = tmp_path / 'cwd'
        capture.write_text(str(shell_dir / 'test.txt'))
        await persist_toolset._apply_captured_cwd(_ctx(), str(capture))
        assert persist_toolset._cwd is None

    async def test_capture_not_utf8_keeps_cwd(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        # The wrapper runs `pwd` in the same shell as the model's command, so a
        # shell function named `pwd` decides the bytes written to the capture
        # file. Decoding them raises `UnicodeDecodeError`, a `ValueError` and
        # not an `OSError`, so the guard has to cover both.
        result = await persist_toolset.run_command(_ctx(), r"""pwd() { printf '\377\376'; }""")
        assert '[exit code' not in result
        assert persist_toolset._cwd is None

    async def test_capture_path_too_long_keeps_cwd(
        self, persist_toolset: ShellToolset[None], shell_dir: Path, tmp_path: Path
    ) -> None:
        # The recorded path is junk the workspace refuses to stat (ENAMETOOLONG); the
        # tracked cwd must survive.
        capture = tmp_path / 'cwd'
        capture.write_text(f'/{"x" * 300}')
        await persist_toolset._apply_captured_cwd(_ctx(), str(capture))
        assert persist_toolset._cwd is None

    async def test_relative_capture_keeps_cwd(self, persist_toolset: ShellToolset[None], tmp_path: Path) -> None:
        # A `pwd` function can print anything; a relative path names no workspace directory.
        result = await persist_toolset.run_command(_ctx(), 'pwd() { echo subdir; }')
        assert '[exit code' not in result
        assert persist_toolset._cwd is None

    async def test_command_that_exits_early_writes_no_capture(self, persist_toolset: ShellToolset[None]) -> None:
        # `exit` skips the capture line; the tracked cwd and the cleanup both tolerate its absence.
        result = await persist_toolset.run_command(_ctx(), 'cd subdir && exit 0')
        assert result == '(no output)'
        assert persist_toolset._cwd is None

    async def test_capture_file_is_removed(self, persist_toolset: ShellToolset[None]) -> None:
        # The capture lives in the workspace's job directory and is removed after each command,
        # including one that deletes it first.
        await persist_toolset.run_command(_ctx(), 'true')
        base = Path(await persist_toolset._jobs_base(_ctx()))
        assert not list(base.glob('cwd-*'))
        await persist_toolset.run_command(_ctx(), f'rm -f {shlex.quote(str(base))}/cwd-*')
        assert not list(base.glob('cwd-*'))


class TestForRunIsolation:
    """B3: `get_toolset` builds one shared instance at agent construction, so
    `for_run` must hand each run a fresh copy -- otherwise concurrent runs share
    `_cwd`/`_background` and corrupt each other."""

    async def test_for_run_returns_fresh_instance(self, persist_toolset: ShellToolset[None]) -> None:
        run1 = await persist_toolset.for_run(_run_context())
        run2 = await persist_toolset.for_run(_run_context())
        assert run1 is not persist_toolset
        assert run2 is not run1

    async def test_persist_cwd_isolated_across_runs(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        run1 = await persist_toolset.for_run(_run_context())
        assert isinstance(run1, ShellToolset)
        await run1.run_command(_ctx(), 'cd subdir')
        assert run1._cwd == str(shell_dir / 'subdir')
        # A second run must start back at the configured root, not inherit run1's cd.
        run2 = await persist_toolset.for_run(_run_context())
        assert isinstance(run2, ShellToolset)
        assert run2._cwd is None


class TestPersistCwdHardening:
    """B4: regression tests for the old stdout-sentinel footguns -- a command's
    output spoofing the cwd, and `;` silently disabling tracking."""

    async def test_cd_persists_even_with_semicolon(self, persist_toolset: ShellToolset[None]) -> None:
        # The old mechanism skipped tracking whenever ';' appeared, silently
        # dropping a real `cd`. The out-of-band capture records it regardless.
        await persist_toolset.run_command(_ctx(), 'cd subdir ; true')
        result = await persist_toolset.run_command(_ctx(), 'pwd')
        assert 'subdir' in result

    async def test_output_cannot_spoof_cwd(self, persist_toolset: ShellToolset[None], shell_dir: Path) -> None:
        # The old mechanism parsed cwd from stdout, so a command printing the
        # sentinel string could redirect the tracked cwd with no real cd.
        spoof = f'true ; echo __HARNESS_PWD__{shell_dir / "subdir"}'
        await persist_toolset.run_command(_ctx(), spoof)
        assert persist_toolset._cwd == str(shell_dir)


class TestSpawnFailures:
    """Failures raised by the spawn itself, which reached past `_recoverable`
    when it only caught `PermissionError` and aborted the whole run."""

    def _toolset_in(self, cwd: Path) -> ShellToolset[None]:
        return ShellToolset(
            cwd=cwd,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )

    async def test_cwd_deleted(self, shell_dir: Path) -> None:
        # The model's own earlier command can do this: `mv "$PWD" "$PWD-old"`
        # passes the denylist, which only inspects the first token.
        target = shell_dir / 'subdir'
        ts = self._toolset_in(target)
        shutil.rmtree(target)
        with pytest.raises(ModelRetry, match='working directory no longer exists'):
            await ts.run_command(_ctx(), 'echo hello')

    async def test_cwd_replaced_by_file(self, shell_dir: Path) -> None:
        target = shell_dir / 'subdir'
        ts = self._toolset_in(target)
        shutil.rmtree(target)
        target.write_text('not a directory\n')
        with pytest.raises(ModelRetry, match='no longer a directory'):
            await ts.run_command(_ctx(), 'echo hello')

    async def test_cwd_deleted_start_command(self, shell_dir: Path) -> None:
        target = shell_dir / 'subdir'
        ts = self._toolset_in(target)
        shutil.rmtree(target)
        with pytest.raises(ModelRetry, match='working directory no longer exists'):
            await ts.start_command(_ctx(), 'sleep 30')

    async def test_message_omits_host_path(self, shell_dir: Path) -> None:
        target = shell_dir / 'subdir'
        ts = self._toolset_in(target)
        shutil.rmtree(target)
        with pytest.raises(ModelRetry) as exc_info:
            await ts.run_command(_ctx(), 'echo hello')
        assert str(target) not in str(exc_info.value)

    @pytest.mark.parametrize(
        ('command', 'expected'),
        [('echo hi\x00there', 'NUL byte'), ('echo \ud800', 'cannot be encoded for the operating system')],
    )
    async def test_unspawnable_command_string(self, toolset: ShellToolset[None], command: str, expected: str) -> None:
        with pytest.raises(ModelRetry, match=expected):
            await toolset.run_command(_ctx(), command)

    async def test_unspawnable_command_string_start_command(self, toolset: ShellToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='NUL byte'):
            await toolset.start_command(_ctx(), 'echo \x00')

    @pytest.mark.parametrize('escaped', ['\udc80', '\udcff'])
    async def test_surrogateescape_command_still_runs(self, toolset: ShellToolset[None], escaped: str) -> None:
        # The spawn encodes with `surrogateescape`, which round-trips this range
        # back to the raw byte it came from. Screening the command as plain
        # UTF-8 would reject a command the OS runs.
        result = await toolset.run_command(_ctx(), f'echo {escaped}')
        assert '[exit code' not in result

    @pytest.mark.parametrize('env', [{'FOO': 'bar\x00baz'}, {'FO\x00O': 'bar'}, {'FOO': 'bar\ud800'}])
    async def test_unspawnable_env_aborts(self, shell_dir: Path, env: dict[str, str]) -> None:
        # The spawn reports a NUL or an unencodable character as the same
        # `ValueError` wherever it came from. This one came from the
        # application's `env`, so the model cannot fix it and must not be asked
        # to retry.
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
            env=env,
        )
        with pytest.raises(ValueError) as exc_info:
            await ts.run_command(_ctx(), 'echo hello')
        assert not isinstance(exc_info.value, ModelRetry)

    async def test_argument_or_environment_too_long_propagates(
        self, toolset: ShellToolset[None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # E2BIG does not identify whether the model's command or the
        # application's environment crossed the combined platform limit. An
        # application configuration error must not become an unwinnable retry.
        monkeypatch.setattr(anyio, 'open_process', _raise_oserror(errno.E2BIG, 'Argument list too long'))
        with pytest.raises(OSError, match='Argument list too long'):
            await toolset.run_command(_ctx(), 'echo hello')

    async def test_non_recoverable_errno_propagates(
        self, toolset: ShellToolset[None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A host that can't fork is not something the model can retry its way
        # out of, so it must keep aborting the run.
        monkeypatch.setattr(anyio, 'open_process', _raise_oserror(errno.ENOMEM, 'Cannot allocate memory'))
        with pytest.raises(OSError, match='Cannot allocate memory'):
            await toolset.run_command(_ctx(), 'echo hello')


class TestRunCommand:
    async def test_basic_echo(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'echo hello')
        assert '[stdout]' in result
        assert 'hello' in result

    async def test_stderr_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'echo error >&2')
        assert '[stderr]' in result
        assert 'error' in result

    async def test_mixed_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'echo out && echo err >&2')
        assert '[stdout]' in result
        assert '[stderr]' in result

    async def test_exit_code_reported(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'exit 42')
        assert '[exit code: 42]' in result

    async def test_exit_code_zero_not_shown(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'echo ok')
        assert 'exit code' not in result

    async def test_no_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'true')
        assert result == '(no output)'

    async def test_output_truncation(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=50)
        result = await _call_shell_tool(ts, 'run_command', command=f'{sys.executable} -c "print(\'x\' * 200)"')
        assert len(result) == 50
        assert 'truncated, showing last 5 chars' in result

    async def test_output_truncation_caps_complete_failure_response(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=200)
        command = f'{sys.executable} -c "import sys; sys.stdout.write(\'x\' * 400); sys.exit(7)"'
        result = await _call_shell_tool(ts, 'run_command', command=command)
        assert len(result) == 200
        assert result.startswith('[... output truncated, showing last 153 chars]\n')
        assert result.endswith('[exit code: 7]')

    async def test_persist_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        await ts.run_command(_ctx(), 'cd subdir')
        result = await ts.run_command(_ctx(), 'pwd')
        assert 'subdir' in result

    async def test_persist_cwd_only_on_success(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        original = ts._cwd
        await ts.run_command(_ctx(), 'cd nonexistent_dir_xyz && false')
        assert ts._cwd == original

    async def test_denied_command_in_run(self, toolset: ShellToolset[None]) -> None:
        # B2: a denied command is model-correctable, so it surfaces as ModelRetry
        # (which pyai feeds back to the model) rather than aborting the run.
        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await toolset.run_command(_ctx(), 'rm -rf /')

    async def test_cwd_used(self, toolset: ShellToolset[None], shell_dir: Path) -> None:
        result = await toolset.run_command(_ctx(), 'cat test.txt')
        assert 'hello' in result

    async def test_multiline_output(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), f'{sys.executable} -c "print(\'a\\nb\\nc\\n\')"')
        assert '[stdout]' in result

    async def test_timeout_reports_value(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), 'sleep 10')
        assert 'timed out after 0.5s' in result

    async def test_custom_timeout_overrides_default(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), 'sleep 10', timeout_seconds=0.5)
        assert 'timed out after 0.5s' in result

    async def test_persist_cwd_disabled_no_update(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        original = ts._cwd
        await ts.run_command(_ctx(), 'cd subdir')
        assert ts._cwd == original

    async def test_nonzero_exit_shows_code(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'exit 1')
        assert '[exit code: 1]' in result

    async def test_stdout_stderr_separated_by_newline(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), 'echo out && echo err >&2')
        assert '[stdout]\nout\n\n[stderr]\nerr' in result

    async def test_non_ascii_stdout(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(
            _ctx(), f'{sys.executable} -c "import sys; sys.stdout.buffer.write(b\'hello \\xff\\xfe world\\n\')"'
        )
        assert 'hello' in result

    async def test_non_ascii_stderr(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(
            _ctx(), f'{sys.executable} -c "import sys; sys.stderr.buffer.write(b\'err \\xff\\xfe msg\\n\')"'
        )
        assert 'err' in result

    async def test_stdout_chunk_join(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.run_command(_ctx(), f"{sys.executable} -c \"print('A' * 100 + 'B' * 100)\"")
        assert 'A' * 100 + 'B' * 100 in result

    async def test_exit_code_fallback_to_zero(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), 'echo ok')
        assert 'exit code' not in result

    async def test_error_message_content(self, shell_dir: Path) -> None:
        with pytest.raises(ValueError, match='^Specify allowed_commands or denied_commands, not both\\.$'):
            ShellToolset(
                cwd=shell_dir,
                allowed_commands=['echo'],
                denied_commands=['rm'],
                denied_operators=[],
                default_timeout=10.0,
                max_output_chars=50_000,
                persist_cwd=False,
                allow_interactive=False,
            )

    def test_non_positive_max_output_chars_rejected(self, shell_dir: Path) -> None:
        # Matches LocalStackToolset: a cap of 0 would blank every response,
        # including start_command's ID line, leaving its process unstoppable.
        with pytest.raises(ValueError, match='max_output_chars must be a positive integer.'):
            _shell_toolset(shell_dir, max_output_chars=0)

    async def test_stdout_chunks_joined_cleanly(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=500_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), "printf '%05000d\\n' $(seq 1 100)")
        assert 'XXXX' not in result

    async def test_stderr_chunks_joined_cleanly(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=500_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), "printf '%0500d\\n' $(seq 1 100) >&2")
        assert 'XXXX' not in result

    async def test_persist_cwd_updates_after_cd(self, shell_dir: Path) -> None:
        """CWD should update to the actual directory after a successful cd."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        await ts.run_command(_ctx(), 'cd subdir')
        assert ts._cwd == str(shell_dir / 'subdir')

    async def test_persist_cwd_not_updated_on_failure(self, shell_dir: Path) -> None:
        """CWD should not update if command fails (exit code non-zero)."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        original = ts._cwd
        await ts.run_command(_ctx(), 'false')
        assert ts._cwd == original


class TestProcessGroupKill:
    async def test_timeout_kills_subprocess_tree(self, shell_dir: Path) -> None:
        """On timeout, the entire process group should be killed."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), 'bash -c "sleep 100 & sleep 100"')
        assert 'timed out' in result

    async def test_timeout_with_output_before_timeout(self, shell_dir: Path) -> None:
        """Output produced before timeout should still result in timeout message."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=0.5,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), 'echo before_timeout && sleep 100')
        assert 'timed out' in result

    async def test_start_new_session_used(self, shell_dir: Path) -> None:
        """Verify the child is in a different process group from the parent."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        parent_pgrp = os.getpgrp()
        result = await ts.run_command(_ctx(), f'{sys.executable} -c "import os; print(os.getpgrp() != {parent_pgrp})"')
        assert 'True' in result


class TestBackgroundCommands:
    async def test_start_command_returns_id(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        result = await _call_shell_tool(ts, 'start_command', command='sleep 100')
        assert 'ID:' in result
        assert 'Started background command' in result
        command_id = _parse_command_id(result)
        await ts.stop_command(_ctx(), command_id)

    async def test_start_command_long_echo_is_capped_keeping_id(self, shell_dir: Path) -> None:
        # The command echo is subject to the cap like any other output; the ID
        # line is the tail, so truncation keeps it usable for check/stop calls.
        ts = _shell_toolset(shell_dir, max_output_chars=80)
        result = await _call_shell_tool(ts, 'start_command', command='true ' + 'x' * 200)
        assert len(result) == 80
        assert 'output truncated' in result
        command_id = _parse_command_id(result)
        assert len(command_id) == 12
        await ts.stop_command(_ctx(), command_id)

    async def test_check_unknown_id(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.check_command(_ctx(), 'nonexistent_id')
        assert 'unknown command ID' in result

    async def test_stop_unknown_id(self, toolset: ShellToolset[None]) -> None:
        result = await toolset.stop_command(_ctx(), 'nonexistent_id')
        assert 'unknown command ID' in result

    async def test_start_and_stop(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'echo hello_bg')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(), command_id)
        assert 'stopped' in stop_result
        assert 'hello_bg' in stop_result
        assert stop_result.splitlines()[-2:] == ['[stopped]', '[exit code: 0]']

    async def test_start_and_check_running(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'sleep 100')
        command_id = _parse_command_id(start_result)

        check_result = await ts.check_command(_ctx(), command_id)
        assert 'running' in check_result
        assert check_result.endswith('[status: running]')

        await ts.stop_command(_ctx(), command_id)

    async def test_check_and_stop_respect_output_cap(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=200)
        start_result = await ts.start_command(_ctx(), "printf '%0400d' 0; sleep 30")
        command_id = _parse_command_id(start_result)
        await anyio.sleep(0.5)

        try:
            check_result = await _call_shell_tool(ts, 'check_command', command_id=command_id)
            assert len(check_result) == 200
            assert 'output truncated' in check_result
            assert check_result.endswith('[status: running]')
        finally:
            stop_result = await _call_shell_tool(ts, 'stop_command', command_id=command_id)
        assert len(stop_result) == 200
        assert 'output truncated' in stop_result
        stop_lines = stop_result.splitlines()
        assert stop_lines[-2] == '[stopped]'
        assert stop_lines[-1].startswith('[exit code:')

    async def test_new_string_tool_is_capped_at_dispatch(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=1)

        def text() -> str:
            return 'xx'

        ts.add_function(text)
        assert await _call_shell_tool(ts, 'text') == 'x'

    async def test_non_string_tool_result_is_unchanged(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir, max_output_chars=1)

        def number() -> int:
            return 42

        ts.add_function(number)
        ctx = _run_context()
        tools = await ts.get_tools(ctx)
        assert await ts.call_tool('number', {}, ctx, tools['number']) == 42

    async def test_start_and_check_finished(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'echo done_quick')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        check_result = await ts.check_command(_ctx(), command_id)
        assert 'finished' in check_result
        assert 'done_quick' in check_result
        assert check_result.splitlines()[-2:] == ['[status: finished]', '[exit code: 0]']

        await ts.stop_command(_ctx(), command_id)

    async def test_start_denied_command_raises(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=['rm'],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await ts.start_command(_ctx(), 'rm -rf /')

    async def test_stop_captures_stderr(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'echo err_bg >&2')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(), command_id)
        assert 'err_bg' in stop_result

    async def test_stop_no_output(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'true')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(), command_id)
        assert '(no output)' in stop_result

    async def test_check_no_output_yet(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'sleep 100')
        command_id = _parse_command_id(start_result)

        check_result = await ts.check_command(_ctx(), command_id)
        assert 'no output yet' in check_result

        await ts.stop_command(_ctx(), command_id)

    async def test_check_command_captures_stderr(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'echo err_check >&2')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        check_result = await ts.check_command(_ctx(), command_id)
        assert '[stderr]' in check_result
        assert 'err_check' in check_result

        await ts.stop_command(_ctx(), command_id)

    async def test_start_command_uses_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'pwd')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        stop_result = await ts.stop_command(_ctx(), command_id)
        assert str(shell_dir) in stop_result

    async def test_stop_removes_from_registry(self, shell_dir: Path) -> None:
        """After stop, the command_id should no longer be known."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        start_result = await ts.start_command(_ctx(), 'true')
        command_id = _parse_command_id(start_result)

        await anyio.sleep(0.5)

        await ts.stop_command(_ctx(), command_id)

        # Should now be unknown
        check_result = await ts.check_command(_ctx(), command_id)
        assert 'unknown command ID' in check_result

    async def test_start_command_cleans_temp_files_on_failure(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        with patch('anyio.open_process', side_effect=OSError('spawn failed')):
            with pytest.raises(OSError, match='spawn failed'):
                await ts.start_command(_ctx(), 'echo hi')
        assert not ts._background

    async def test_aexit_terminates_background_processes(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        pid_pipe = shell_dir / 'pid'
        os.mkfifo(pid_pipe)
        result = await ts.start_command(_ctx(), f'echo $$ > {pid_pipe}; exec sleep 300')
        command_id = _parse_command_id(result)
        job_dir = Path(ts._background[command_id].job.directory)
        assert (job_dir / 'stdout.log').exists()
        assert (job_dir / 'stderr.log').exists()
        # Reading the FIFO blocks until the command has written its PID: no polling, so no timing.
        with anyio.fail_after(10):
            pid = int(await anyio.to_thread.run_sync(pid_pipe.read_text))

        await ts.__aexit__(None, None, None)

        assert not ts._background
        assert not job_dir.exists()
        await _wait_for_exit(pid)

    # Cleanup reads the status (`read_bytes`), signals the job (`run`), then removes its files
    # (`remove`); `None` is the control, where the fake refuses nothing.
    @pytest.mark.parametrize('refused', [None, 'read_bytes', 'run', 'remove'])
    async def test_aexit_survives_any_workspace_exception(
        self, shell_dir: Path, caplog: pytest.LogCaptureFixture, refused: str | None
    ) -> None:
        # A durable workspace may refuse calls outside an activity with an arbitrary error.
        ts = _shell_toolset(shell_dir)
        ids = [_parse_command_id(await ts.start_command(_ctx(), 'exec sleep 300')) for _ in range(2)]
        jobs = [ts._background[command_id].job for command_id in ids]
        for job in jobs:
            job.workspace = Workspace(_Refusing('/', refused))
        try:
            with caplog.at_level(logging.DEBUG, logger='pydantic_ai_harness.shell._toolset'):
                async with AsyncExitStack() as stack:
                    await stack.enter_async_context(ts)
            assert not ts._background
            logged = [record.message for record in caplog.records]
            expected = [] if refused is None else [f'Could not clean up background job {job.directory}' for job in jobs]
            assert logged == expected
            assert all(Path(job.directory).exists() == (refused is not None) for job in jobs)
        finally:
            for job in jobs:
                job.workspace = _ctx().workspace
                await job.kill()
                await job.cleanup()

    async def test_aexit_tolerates_a_workspace_error(self, shell_dir: Path) -> None:
        # Cleanup is best-effort: a job whose directory is already gone does not stop the others.
        ts = _shell_toolset(shell_dir)
        first = _parse_command_id(await ts.start_command(_ctx(), 'exec sleep 300'))
        second = _parse_command_id(await ts.start_command(_ctx(), 'exec sleep 300'))
        first_job = ts._background[first].job
        second_dir = Path(ts._background[second].job.directory)
        first_job.workspace = Workspace(_FailingKill('/'))
        try:
            await ts.__aexit__(None, None, None)
            assert not ts._background
            assert not second_dir.exists()
        finally:
            first_job.workspace = _ctx().workspace
            await first_job.kill()
            await first_job.cleanup()

    async def test_aexit_noop_when_no_background(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        await ts.__aexit__(None, None, None)
        assert not ts._background

    async def test_aexit_cleans_already_finished_process(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.start_command(_ctx(), 'echo done')
        command_id = _parse_command_id(result)
        await anyio.sleep(0.5)
        # Mark as finished via check_command
        await ts.check_command(_ctx(), command_id)
        bg = ts._background[command_id]
        assert bg.finished

        await ts.__aexit__(None, None, None)
        assert not ts._background


class TestEdgeCases:
    async def test_toolset_tool_names(self, toolset: ShellToolset[None]) -> None:
        tool_names = list(toolset.tools.keys())
        assert 'run_command' in tool_names
        assert 'start_command' in tool_names
        assert 'check_command' in tool_names
        assert 'stop_command' in tool_names

    async def test_run_command_uses_actual_cwd(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        result = await ts.run_command(_ctx(), 'pwd')
        assert str(shell_dir) in result

    async def test_persist_cwd_requires_all_three_conditions(self, shell_dir: Path) -> None:
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=True,
            allow_interactive=False,
        )
        # Successful echo -- sentinel shows same dir, cwd should remain valid
        await ts.run_command(_ctx(), 'echo hi')
        assert ts._cwd == str(shell_dir)


class TestShellCapability:
    def test_default_construction(self) -> None:
        shell = Shell()
        assert shell.cwd == '.'
        assert shell.default_timeout == 30.0
        assert 'rm' in shell.denied_commands

    def test_custom_construction(self) -> None:
        shell = Shell(
            cwd='/tmp',
            allowed_commands=['echo', 'cat'],
            denied_commands=[],
            default_timeout=60.0,
        )
        assert shell.default_timeout == 60.0
        shell.get_toolset()

    async def test_empty_allowlist_keeps_default_denylist(self) -> None:
        toolset = Shell(allowed_commands=[]).get_toolset()

        with pytest.raises(ModelRetry, match="'rm' is denied"):
            await toolset.run_command(_ctx(), 'rm --version')
        assert 'hello' in await toolset.run_command(_ctx(), 'echo hello')

    def test_explicit_default_denylist_conflicts_with_allowlist(self) -> None:
        denied_commands = Shell().denied_commands
        shell = Shell(allowed_commands=['rm'], denied_commands=denied_commands)

        with pytest.raises(ValueError, match='Specify allowed_commands or denied_commands'):
            shell.get_toolset()

    def test_agent_accepts_allowlist_without_explicit_denylist(self, tmp_path: Path) -> None:
        Agent(TestModel(), capabilities=[Shell(cwd=tmp_path, allowed_commands=['ls', 'cat', 'rg'])])

    def test_get_toolset_returns_toolset(self, tmp_path: Path) -> None:
        shell = Shell(cwd=tmp_path)
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)

    def test_default_denied_commands(self) -> None:
        shell = Shell()
        assert 'rm' in shell.denied_commands
        assert 'dd' in shell.denied_commands
        assert 'shutdown' in shell.denied_commands

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_integration(self, tmp_path: Path) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[Shell(cwd=tmp_path)])
        result = await agent.run('run echo hello')
        assert result.output == 'done'


async def _tools_offered_to_model(cwd: Path, *, shell_first: bool) -> dict[str, str | None]:
    """Run an agent with Shell and CodeMode and return the tools the model was offered."""
    offered: dict[str, str | None] = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.update({tool.name: tool.description for tool in info.function_tools})
        return ModelResponse(parts=[TextPart('done')])

    shell = Shell[object](cwd=cwd)
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

    @pytest.mark.anyio(backends=['asyncio'])
    @pytest.mark.parametrize('shell_first', [True, False], ids=['shell-first', 'code-mode-first'])
    async def test_command_tools_stay_native(self, tmp_path: Path, shell_first: bool) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        tools = await _tools_offered_to_model(tmp_path, shell_first=shell_first)

        assert 'run_command' in tools
        assert 'start_command' in tools
        run_code_description = tools['run_code']
        assert run_code_description is not None
        assert 'async def run_command' not in run_code_description
        assert 'async def start_command' not in run_code_description

    @pytest.mark.anyio(backends=['asyncio'])
    @pytest.mark.parametrize('shell_first', [True, False], ids=['shell-first', 'code-mode-first'])
    async def test_command_id_tools_are_still_sandboxed(self, tmp_path: Path, shell_first: bool) -> None:

        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        tools = await _tools_offered_to_model(tmp_path, shell_first=shell_first)

        assert 'check_command' not in tools
        assert 'stop_command' not in tools
        run_code_description = tools['run_code']
        assert run_code_description is not None
        assert 'async def check_command' in run_code_description
        assert 'async def stop_command' in run_code_description


class TestStopEscalation:
    async def test_stop_escalates_to_sigkill(self, shell_dir: Path) -> None:
        """A group that ignores SIGTERM is killed after the grace period, with no exit code to report."""
        ts = _shell_toolset(shell_dir)
        ready = shell_dir / 'ready'
        start = await ts.start_command(_ctx(), f"trap '' TERM; echo $$ > {ready}; while :; do sleep 1; done")
        command_id = _parse_command_id(start)
        with anyio.fail_after(10):
            while not ready.exists() or not ready.read_text().strip():
                await anyio.sleep(0.01)
        pid = int(ready.read_text())
        with patch('pydantic_ai_harness.shell._jobs._KILL_GRACE_PERIOD', 0.2):
            result = await ts.stop_command(_ctx(), command_id)
        assert result.endswith('[stopped]')
        await _wait_for_exit(pid)

    async def test_stop_after_process_already_exited(self, shell_dir: Path) -> None:
        """A job that exited between the status read and the signal is not an error."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(), 'true'))
        bg = ts._background[command_id]
        with anyio.fail_after(10):
            while (await bg.job.status())[0]:
                await anyio.sleep(0.01)
        bg.job.pgid = None
        bg.job.pid = 2**22 + 12345  # beyond any live PID, so `kill` finds no process
        await bg.job.kill()


class _Refusing(LocalWorkspaceBackend):
    """A local backend that raises an error that is not a `WorkspaceError` from one chosen operation."""

    def __init__(self, working_dir: str, refused: str | None) -> None:
        super().__init__(working_dir)
        self.refused = refused

    def _check(self, operation: str) -> None:
        if operation == self.refused:
            raise RuntimeError('workspace call outside an activity')

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self._check('run')
        return await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def read_bytes(self, path: str) -> bytes:
        self._check('read_bytes')
        return await super().read_bytes(path)

    async def remove(self, path: str) -> None:
        self._check('remove')
        await super().remove(path)


_KILL_SCRIPT = 'kill -s "$1" -- "$2"'


class _RecordingKill(LocalWorkspaceBackend):
    """A local backend that records every argv command and can answer the signal command itself."""

    def __init__(self, working_dir: str, *, kill_result: CommandResult | None = None) -> None:
        super().__init__(working_dir)
        self.argv: list[list[str]] = []
        self.kill_result = kill_result

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if not isinstance(command, str):
            self.argv.append(list(command))
            if self.kill_result is not None and command[:3] == ['sh', '-c', _KILL_SCRIPT]:
                return self.kill_result
        return await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)


class TestSignalling:
    async def test_signals_go_through_the_shell_builtin(self, shell_dir: Path) -> None:
        # Slim images ship no `kill` executable, so no argv may start with a bare `kill`.
        backend = _RecordingKill('/')
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(backend))
        command_id = _parse_command_id(await ts.start_command(ctx, 'exec sleep 300'))
        job = ts._background[command_id].job
        stopped = await ts.stop_command(ctx, command_id)
        assert stopped.splitlines()[-2:] == ['[stopped]', '[exit code: 143]']
        signals = [argv for argv in backend.argv if argv[:3] == ['sh', '-c', _KILL_SCRIPT]]
        target = f'-{job.pgid}' if job.pgid is not None else str(job.pid)
        assert signals == [['sh', '-c', _KILL_SCRIPT, 'kill', 'TERM', target]]
        assert all(argv[0] != 'kill' for argv in backend.argv)
        await _wait_for_exit(job.pid)

    async def test_failed_signal_is_not_reported_as_stopped(self, shell_dir: Path) -> None:
        backend = _RecordingKill(
            '/', kill_result=CommandResult(exit_code=1, stdout='', stderr='kill: Operation not permitted')
        )
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(backend))
        command_id = _parse_command_id(await ts.start_command(ctx, 'exec sleep 300'))
        job = ts._background[command_id].job
        try:
            tools = await ts.get_tools(ctx)
            with pytest.raises(ToolFailed, match='Operation not permitted'):
                await ts.call_tool('stop_command', {'command_id': command_id}, ctx, tools['stop_command'])
            assert (await job.status())[0]
        finally:
            backend.kill_result = None
            await job.kill()
            await job.cleanup()

    async def test_failed_signal_without_stderr_names_the_signal(self, shell_dir: Path) -> None:
        backend = _RecordingKill('/', kill_result=CommandResult(exit_code=2, stdout='', stderr=''))
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_run_context(Workspace(backend)), 'exec sleep 300'))
        job = ts._background[command_id].job
        try:
            with pytest.raises(WorkspaceError, match=f'Unable to send SIGTERM to job {job.pid}'):
                await job.kill()
        finally:
            backend.kill_result = None
            await job.kill()
            await job.cleanup()


class _FailingKill(LocalWorkspaceBackend):
    """A local backend whose commands fail as a broken workspace would.

    `__aexit__` reads status and removes files through the filesystem methods, so the only
    command it runs is the signal to a still-running job.
    """

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        raise WorkspaceError('kill failed')


async def _wait_for_exit(pid: int) -> None:
    """Wait for `pid` to be reaped; a process init reaps may linger as a zombie for a moment."""
    with anyio.fail_after(10):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await anyio.sleep(0.01)  # pragma: lax no cover


class TestReadBgOutputEdgeCases:
    async def test_missing_logs_read_as_empty(self, shell_dir: Path) -> None:
        """A log removed from the workspace reads as empty rather than failing the check."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(), 'exec sleep 300'))
        job = ts._background[command_id].job
        (Path(job.directory) / 'stdout.log').unlink()
        (Path(job.directory) / 'stderr.log').unlink()
        try:
            result = await ts.check_command(_ctx(), command_id)
            assert result == '(no output yet)\n[status: running]'
        finally:
            await ts.stop_command(_ctx(), command_id)

    async def test_unreadable_log_is_a_failed_call(self, shell_dir: Path) -> None:
        """A log the workspace cannot read is reported to the model as a failed tool call."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(), 'exec sleep 300'))
        job = ts._background[command_id].job
        stdout_log = Path(job.directory) / 'stdout.log'
        stdout_log.write_text('secret')
        stdout_log.chmod(0)
        try:
            if os.access(stdout_log, os.R_OK):  # pragma: no cover - root reads regardless of mode bits
                pytest.skip('mode bits do not bind this user')
            with pytest.raises(ToolFailed) as exc_info:
                await ts.check_command(_ctx(), command_id)
            assert exc_info.value.message == 'Unable to read job log ' + repr(str(stdout_log)) + '.'
        finally:
            stdout_log.chmod(0o600)
            await ts.stop_command(_ctx(), command_id)


class TestJobStatusEdgeCases:
    async def test_unparseable_status_counts_as_running(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(), 'exec sleep 300'))
        job = ts._background[command_id].job
        try:
            (Path(job.directory) / 'status.json').write_text('not json')
            assert (await ts.check_command(_ctx(), command_id)).endswith('[status: running]')
            (Path(job.directory) / 'status.json').unlink()
            assert (await ts.check_command(_ctx(), command_id)).endswith('[status: running]')
        finally:
            await ts.stop_command(_ctx(), command_id)


class TestCleanupBgFilesEdgeCases:
    async def test_cleanup_of_removed_directory(self, shell_dir: Path) -> None:
        """A job directory already removed from the workspace is not an error."""
        ts = _shell_toolset(shell_dir)
        command_id = _parse_command_id(await ts.start_command(_ctx(), 'true'))
        job = ts._background[command_id].job
        with anyio.fail_after(10):
            # Removing the directory while the wrapper still publishes its status races the wrapper.
            while (await job.status())[0]:
                await anyio.sleep(0.01)  # pragma: lax no cover
        await job.cleanup()
        await job.cleanup()
        assert not Path(job.directory).exists()
        del ts._background[command_id]


class TestStopCommandAlreadyFinished:
    async def test_stop_already_finished_process(self, shell_dir: Path) -> None:
        """stop_command on an already-finished process skips kill."""
        ts = ShellToolset(
            cwd=shell_dir,
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        )
        # Start a command that finishes immediately
        start_result = await ts.start_command(_ctx(), 'echo done')
        command_id = _parse_command_id(start_result)

        # Wait for the process to finish
        await anyio.sleep(0.5)

        # Manually mark as finished with exit_code = None (simulates edge case
        # where finished is True but exit_code was never captured)
        bg = ts._background[command_id]
        bg.finished = True
        bg.exit_code = None

        # stop_command should skip the kill branch and handle None exit_code
        result = await ts.stop_command(_ctx(), command_id)
        assert result.endswith('[stopped]')
        assert '[exit code:' not in result


class TestResolveEnv:
    """Unit coverage for the env-resolution branches."""

    def test_adds_nothing_when_unconfigured(self, shell_dir: Path) -> None:
        # No env -> None: commands get the workspace's own environment unchanged.
        assert _env_toolset(shell_dir)._resolve_env() is None

    def test_patterns_alone_add_nothing(self, shell_dir: Path) -> None:
        # Patterns filter only an explicit `env`; the workspace environment is its provider's.
        assert _env_toolset(shell_dir, denied_env_patterns=['OPENAI_*'])._resolve_env() is None

    def test_explicit_env_is_added(self, shell_dir: Path) -> None:
        resolved = _env_toolset(shell_dir, env={'FOO': 'bar'})._resolve_env()
        assert resolved == {'FOO': 'bar'}

    def test_explicit_empty_env(self, shell_dir: Path) -> None:
        assert _env_toolset(shell_dir, env={})._resolve_env() == {}

    def test_patterns_strip_from_explicit_env(self, shell_dir: Path) -> None:
        resolved = _env_toolset(
            shell_dir,
            env={'OPENAI_API_KEY': 'secret', 'PATH': '/usr/bin'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()
        assert resolved == {'PATH': '/usr/bin'}

    def test_patterns_no_match_keeps_base(self, shell_dir: Path) -> None:
        resolved = _env_toolset(
            shell_dir,
            env={'FOO': 'bar'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()
        assert resolved == {'FOO': 'bar'}

    def test_pattern_match_is_case_sensitive(self, shell_dir: Path) -> None:
        # Env var names are case-sensitive on POSIX; lowercase must not match.
        resolved = _env_toolset(
            shell_dir,
            env={'openai_api_key': 'secret'},
            denied_env_patterns=['OPENAI_*'],
        )._resolve_env()
        assert resolved == {'openai_api_key': 'secret'}


class TestEnvControlExecution:
    """End-to-end: the resolved env actually reaches spawned subprocesses."""

    async def test_explicit_env_seen_by_command(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'MY_TOKEN': 'present', 'PATH': os.environ['PATH']})
        result = await ts.run_command(_ctx(), _read_env_var('MY_TOKEN'))
        assert 'present' in result

    async def test_explicit_env_hides_inherited_secret(self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('OPENROUTER_API_KEY', 'leak-me')
        ts = _env_toolset(shell_dir, env={'PATH': os.environ['PATH']})
        result = await ts.run_command(_ctx(), _read_env_var('OPENROUTER_API_KEY'))
        assert 'ABSENT' in result
        assert 'leak-me' not in result

    async def test_host_environment_does_not_reach_commands(
        self, shell_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The local workspace passes only PATH, HOME, LANG, and TMPDIR from the host.
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'leak-me')
        monkeypatch.setenv('HARNESS_INHERITED', 'yes')
        ts = _env_toolset(shell_dir)
        result = await ts.run_command(
            _ctx(), _read_env_var('ANTHROPIC_API_KEY') + '; ' + _read_env_var('HARNESS_INHERITED')
        )
        assert 'leak-me' not in result and 'yes' not in result

    async def test_workspace_environment_passes_through(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, denied_env_patterns=['ANTHROPIC_*'])
        result = await ts.run_command(_ctx(), 'printf "$PATH"')
        assert os.environ['PATH'] in result

    async def test_env_and_patterns_compose_at_spawn(self, shell_dir: Path) -> None:
        # Both set: a pattern strips a key from the explicit env, the rest survives.
        ts = _env_toolset(
            shell_dir,
            env={'SECRET_KEY': 'leak-me', 'KEEP_VAR': 'kept', 'PATH': os.environ['PATH']},
            denied_env_patterns=['SECRET_*'],
        )
        stripped = await ts.run_command(_ctx(), _read_env_var('SECRET_KEY'))
        assert 'ABSENT' in stripped
        assert 'leak-me' not in stripped
        survived = await ts.run_command(_ctx(), _read_env_var('KEEP_VAR'))
        assert 'kept' in survived

    async def test_background_command_honors_env(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'BG_TOKEN': 'bg-present', 'PATH': os.environ['PATH']})
        start_result = await ts.start_command(_ctx(), _read_env_var('BG_TOKEN'))
        command_id = _parse_command_id(start_result)
        await anyio.sleep(0.5)
        stop_result = await ts.stop_command(_ctx(), command_id)
        assert 'bg-present' in stop_result


class TestEnvControlPropagation:
    """The capability and `for_run` carry env control through unchanged."""

    async def test_for_run_propagates_env(self, shell_dir: Path) -> None:
        ts = _env_toolset(shell_dir, env={'FOO': 'bar'}, denied_env_patterns=['OPENAI_*'])
        run_ts = await ts.for_run(_run_context())
        assert isinstance(run_ts, ShellToolset)
        assert run_ts._resolve_env() == {'FOO': 'bar'}

    def test_capability_defaults_add_nothing(self) -> None:
        shell = Shell()
        assert shell.env is None
        assert list(shell.denied_env_patterns) == []

    def test_capability_passes_env_to_toolset(self, tmp_path: Path) -> None:
        shell = Shell(
            cwd=tmp_path,
            env={'FOO': 'bar'},
            denied_env_patterns=['OPENAI_*'],
        )
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)
        assert toolset._resolve_env() == {'FOO': 'bar'}

    def test_llm_pattern_constant_strips_provider_keys(self, tmp_path: Path) -> None:
        env = {name: 'secret' for name in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY')}
        env |= {'GEMINI_API_KEY': 'secret', 'GOOGLE_API_KEY': 'secret', 'GATEWAY_KEY': 'secret'}
        env |= {'PYDANTIC_AI_GATEWAY_API_KEY': 'secret', 'PATH': '/usr/bin'}
        shell = Shell(cwd=tmp_path, env=env, denied_env_patterns=list(LLM_API_KEY_ENV_PATTERNS))
        toolset = shell.get_toolset()
        assert isinstance(toolset, ShellToolset)
        # None of the provider-credential prefixes survive.
        assert toolset._resolve_env() == {'PATH': '/usr/bin'}


class TestReadOnlyWorkspace:
    """A read-only workspace refuses `run`, so the shell offers no tools and reports a refusal as a failure."""

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_is_offered_no_shell_tools(self, tmp_path: Path) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        model = TestModel(call_tools=[])
        capabilities: list[AbstractCapability[None]] = [
            Shell[None](cwd=tmp_path, tools=['run_command', 'shell']),
            LocalWorkspace[None](tmp_path, read_only=True),
        ]
        await Agent(model, deps_type=type(None), capabilities=capabilities).run('Inspect tools')
        assert model.last_model_request_parameters is not None
        assert model.last_model_request_parameters.function_tools == []

    async def test_get_tools_is_empty(self, tmp_path: Path) -> None:
        ts = _shell_toolset(tmp_path)
        read_only = _run_context(ReadOnlyWorkspace(Workspace(local_workspace(tmp_path))))
        assert await ts.get_tools(read_only) == {}
        assert set(await ts.get_tools(_ctx())) == {'run_command', 'start_command', 'check_command', 'stop_command'}

    async def test_refusal_is_a_failed_tool_call(self, tmp_path: Path) -> None:
        ts = _shell_toolset(tmp_path)
        writable = _ctx()
        tools = await ts.get_tools(writable)
        read_only = _run_context(ReadOnlyWorkspace(Workspace(local_workspace(tmp_path))))
        with pytest.raises(ToolFailed) as exc_info:
            await ts.call_tool('run_command', {'command': 'echo hi'}, read_only, tools['run_command'])
        assert exc_info.value.message == READ_ONLY_FAILURE


class TestDetachedJobRoundTrip:
    async def test_start_check_stop(self, tmp_path: Path) -> None:
        ts = _shell_toolset(tmp_path)
        command_id = _parse_command_id(
            await ts.start_command(_ctx(), 'echo started; echo warn >&2; echo made > made.txt; exec sleep 300')
        )
        job_dir = Path(ts._background[command_id].job.directory)
        with anyio.fail_after(10):
            while 'started' not in (checked := await ts.check_command(_ctx(), command_id)):
                await anyio.sleep(0.05)
        assert checked.endswith('[status: running]')
        assert '[stderr]\nwarn' in checked
        assert (tmp_path / 'made.txt').read_text() == 'made\n'
        assert json.loads((job_dir / 'status.json').read_text())['exit_code'] is None

        stopped = await ts.stop_command(_ctx(), command_id)
        assert stopped.splitlines()[-2:] == ['[stopped]', '[exit code: 143]']
        assert 'started' in stopped
        assert not job_dir.exists()
        assert 'unknown command ID' in await ts.check_command(_ctx(), command_id)

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_persistent_background_job(self, tmp_path: Path) -> None:
        if sniffio.current_async_library() != 'asyncio':  # pragma: no cover
            pytest.skip('Agent.run() requires asyncio')
        shell = Shell[None](cwd=tmp_path, tools=['shell'])
        output = await call_tool(
            [shell],
            'shell',
            {'command': 'exec sleep 300', 'mode': 'background'},
            workspace=local_workspace(tmp_path),
        )
        pid = int(output.split('PID: ')[1].split()[0])
        stop = output.split('use `')[1].split('`')[0]
        status = Path(output.split('Status: ')[1].splitlines()[0])
        try:
            assert json.loads(status.read_text()) == {'pid': pid, 'exit_code': None}
            assert '"exit_code": null' in output
            # The wait for the published exit code runs in the workspace shell, with the model's own tool.
            wait = f'{stop} && while grep -q \'"exit_code": null\' {shlex.quote(str(status))}; do sleep 0.05; done'
            result = await call_tool(
                [Shell[None](cwd=tmp_path, tools=['run_command'], denied_commands=[])],
                'run_command',
                {'command': wait, 'timeout_seconds': 10},
                workspace=local_workspace(tmp_path),
            )
            assert 'exit code' not in result and 'timed out' not in result
            assert json.loads(status.read_text()) == {'pid': pid, 'exit_code': 143}
        finally:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            shutil.rmtree(status.parent, ignore_errors=True)


class _RaisingWorkspace(LocalWorkspaceBackend):
    """A local backend whose every command raises the configured workspace error."""

    def __init__(self, working_dir: str, error: WorkspaceError) -> None:
        super().__init__(working_dir)
        self.error = error

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        raise self.error


class TestWorkspaceFailures:
    """Deliberate workspace failures reach the model as failed calls; a vanished workspace ends the run."""

    @pytest.mark.parametrize(
        ('error', 'message'),
        [
            (WorkspaceTimeoutError('slow', timeout=30), 'The workspace operation timed out after 30s.'),
            (WorkspaceTimeoutError('slow'), 'The workspace operation timed out.'),
            (WorkspaceError('backend refused'), 'backend refused'),
            (WorkspaceError(), 'The workspace operation failed (WorkspaceError).'),
        ],
    )
    async def test_failed_call(self, shell_dir: Path, error: WorkspaceError, message: str) -> None:
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(_RaisingWorkspace('/', error)))
        tools = await ts.get_tools(ctx)
        with pytest.raises(ToolFailed) as failed:
            await ts.call_tool('start_command', {'command': 'true'}, ctx, tools['start_command'])
        assert failed.value.message == message
        assert not ts._background

    async def test_unavailable_workspace_ends_the_run(self, shell_dir: Path) -> None:
        ts = _shell_toolset(shell_dir)
        ctx = _run_context(Workspace(_RaisingWorkspace('/', WorkspaceUnavailableError('sandbox expired'))))
        tools = await ts.get_tools(ctx)
        with pytest.raises(WorkspaceUnavailableError, match='sandbox expired'):
            await ts.call_tool('run_command', {'command': 'true'}, ctx, tools['run_command'])
