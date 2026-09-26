"""Tests for `E2BSandboxBackend`, the E2B implementation of the sandbox protocol."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from e2b.exceptions import (
    AuthenticationException,
    InvalidArgumentException,
    NotEnoughSpaceException,
    RateLimitException,
    SandboxException,
    SandboxNotFoundException,
    ServiceBusyException,
    TimeoutException,
)
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

from .fake_e2b import FakeE2B


async def started(**settings: Any) -> E2BSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting `get_sandbox()` is that touch.
    """
    backend = E2BSandboxBackend(**settings)
    await backend.get_sandbox()
    return backend


class TestConformance:
    async def test_get_sandbox_is_lazy_and_reuses_the_sandbox(self, fake_e2b: FakeE2B) -> None:
        backend = E2BSandboxBackend()
        assert not fake_e2b.sandboxes
        sandbox = await backend.get_sandbox()
        assert await backend.get_sandbox() is sandbox
        assert fake_e2b.sandboxes == [sandbox]

    @pytest.mark.parametrize('operation', ['run', 'write_bytes'])
    async def test_ref_is_recorded_by_the_first_operation(self, fake_e2b: FakeE2B, operation: str) -> None:
        backend = E2BSandboxBackend()
        assert backend.ref is None
        if operation == 'run':
            await backend.run(['true'])
        else:
            await backend.write_bytes('/tmp/file', b'data')
        assert backend.ref == WorkspaceRef(provider='e2b', id=fake_e2b.sandboxes[0].sandbox_id)

    async def test_identity_is_e2b_sandbox_id(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
        assert await backend.get_sandbox() is fake_e2b.sandboxes[0]


class TestCreate:
    async def test_creates_from_config(self, fake_e2b: FakeE2B) -> None:
        backend = await started(
            template='base',
            sandbox_timeout=120,
            env={'FOO': 'bar'},
            allow_internet_access=False,
        )
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
        call = fake_e2b.create_calls[-1]
        assert (call.template, call.timeout, call.envs, call.allow_internet_access) == (
            'base',
            120,
            {'FOO': 'bar'},
            False,
        )

    async def test_defaults(self, fake_e2b: FakeE2B) -> None:
        await started()
        call = fake_e2b.create_calls[-1]
        # The most E2B's Hobby plan allows, pausing rather than killing at the end of it.
        assert (call.template, call.timeout, call.envs, call.lifecycle) == (None, 3_600, None, {'on_timeout': 'pause'})
        # Passed explicitly: it decides whether the sandbox is reachable without its token.
        assert (call.secure, call.allow_internet_access) == (True, True)

    @pytest.mark.parametrize(
        ('error', 'message'),
        [
            (
                SandboxException('404: template xyz not found', status_code=404),
                'Could not start E2B sandbox: 404: template xyz not found',
            ),
            (
                SandboxException('400: Timeout cannot be greater than 1 hours', status_code=400),
                'Hobby plans allow at most 3600 seconds; pass `E2BSandbox(sandbox_timeout=3600)`.',
            ),
        ],
    )
    async def test_a_refused_create_is_unavailable(self, fake_e2b: FakeE2B, error: Exception, message: str) -> None:
        # A refused request fails the same way on every retry, so it ends the run.
        fake_e2b.create_error = error
        with pytest.raises(WorkspaceUnavailableError, match=re.escape(message)) as exc:
            await started()
        assert exc.value.__cause__ is error

    @pytest.mark.parametrize(
        'error',
        [SandboxException('500: internal', status_code=500), SandboxException('no status')],
        ids=['server-error', 'no-status'],
    )
    async def test_an_unrefused_create_failure_is_an_operation_error(self, fake_e2b: FakeE2B, error: Exception) -> None:
        fake_e2b.create_error = error
        with pytest.raises(WorkspaceError, match='Could not start E2B sandbox') as exc:
            await started()
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    async def test_hanging_create_does_not_hang_the_caller(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The client-side bound prevents a wedged control plane from hanging acquisition; like
        # any unreachable service, it propagates as a transient failure.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._CREATE_TIMEOUT', 0.05)
        fake_e2b.create_hangs = True
        with anyio.fail_after(5):
            with pytest.raises(TimeoutError, match='did not complete within') as exc:
                await started()
        assert type(exc.value) is TimeoutError

    async def test_rejects_relative_working_dir(self, fake_e2b: FakeE2B) -> None:
        with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
            await started(working_dir='repo')


class TestConnect:
    async def test_connects_to_an_existing_sandbox(self, fake_e2b: FakeE2B) -> None:
        # E2B resumes a paused sandbox on connect, so no separate liveness probe is needed:
        # a sandbox that is really gone raises instead of handing back a dead handle.
        backend = await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep'))
        assert fake_e2b.connect_calls == [('sbx-keep', 3_600)]
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-keep')

    async def test_a_refused_lifetime_on_connect_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        # Attaching with a lifetime over the plan's limit fails the same way on every retry.
        error = SandboxException('400: Timeout cannot be greater than 1 hours', status_code=400)
        fake_e2b.connect_error = error
        with pytest.raises(
            WorkspaceUnavailableError, match=r"'sbx-keep'.*pass `E2BSandbox\(sandbox_timeout=3600\)`"
        ) as exc:
            await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep'), sandbox_timeout=86_400)
        assert exc.value.__cause__ is error

    async def test_connect_to_a_missing_sandbox_fails(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = SandboxNotFoundException('not found')
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-gone'"):
            await started(ref=WorkspaceRef(provider='e2b', id='sbx-gone'))
        assert not fake_e2b.create_calls

    async def test_an_operation_on_a_gone_sandbox_does_not_create_a_replacement(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = SandboxNotFoundException('not found')
        backend = E2BSandboxBackend(ref=WorkspaceRef(provider='e2b', id='sbx-gone'))
        for _ in range(2):
            with pytest.raises(WorkspaceUnavailableError, match="'sbx-gone'"):
                await backend.run(['true'])
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-gone')
        assert not fake_e2b.create_calls
        assert not fake_e2b.sandboxes


class TestRun:
    async def test_argv_is_quoted_into_one_shell_word_string(self, fake_e2b: FakeE2B) -> None:
        # E2B has no argv form: every command goes through `/bin/bash -l -c`, so the quoting
        # is what keeps an argument with a space or a `$` one literal word.
        backend = await started()
        await backend.run(['echo', 'a b', '$HOME'])
        assert fake_e2b.sandboxes[0].commands.calls[-1].command == "echo 'a b' '$HOME'"

    async def test_shell_string_runs_under_sh(self, fake_e2b: FakeE2B) -> None:
        # E2B's login bash would otherwise interpret it; `sh -c` matches every other workspace.
        backend = await started()
        await backend.run('echo hi | wc -c', shell=True)
        assert fake_e2b.sandboxes[0].commands.calls[-1].command == "/bin/sh -c 'echo hi | wc -c'"

    async def test_reports_streams_and_exit_code(self, fake_e2b: FakeE2B) -> None:
        # E2B raises `CommandExitException` on a non-zero exit; the protocol calls that a
        # normal result, so the backend unwraps it instead of propagating.
        fake_e2b.responder = lambda command, timeout: ('out', 'err', 2)
        backend = await started()
        result = await backend.run(['false'])
        assert (result.stdout, result.stderr, result.exit_code) == ('out', 'err', 2)

    async def test_cwd_and_env_reach_the_command(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.run(['env'], cwd='/srv', env={'FOO': 'bar'})
        call = fake_e2b.sandboxes[0].commands.calls[-1]
        assert (call.cwd, call.envs) == ('/srv', {'FOO': 'bar'})

    async def test_configured_env_reaches_an_attached_sandbox_under_the_commands_own(self, fake_e2b: FakeE2B) -> None:
        # Creation `envs` never reach a sandbox made elsewhere, so the configured env rides on
        # every command, with the command's own values winning.
        backend = await started(
            ref=WorkspaceRef(provider='e2b', id='sbx-keep'), env={'BASE': 'configured', 'SHARED': 'configured'}
        )
        await backend.run(['env'], env={'SHARED': 'command'})
        assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'BASE': 'configured', 'SHARED': 'command'}

    async def test_rejects_relative_cwd(self, fake_e2b: FakeE2B) -> None:
        backend = await started()

        with pytest.raises(ValueError, match='cwd must be an absolute workspace path'):
            await backend.run(['pwd'], cwd='repo')

    async def test_configured_working_dir_is_the_default_cwd(self, fake_e2b: FakeE2B) -> None:
        # E2B has no create-time working directory, so the backend applies it per command.
        backend = await started(working_dir='/work')
        await backend.run(['pwd'])
        assert fake_e2b.sandboxes[0].commands.calls[-1].cwd == '/work'

    async def test_started_in_background_with_the_sdk_deadline_off(self, fake_e2b: FakeE2B) -> None:
        # E2B's own `timeout` abandons the stream and leaves the command running, so it is
        # switched off and the deadline is enforced (and killed) client-side instead.
        backend = await started()
        await backend.run(['x'], timeout=30)
        call = fake_e2b.sandboxes[0].commands.calls[-1]
        assert (call.background, call.timeout) == (True, 0)

    @pytest.mark.parametrize('timeout', [0, -1.0, float('inf'), float('nan')])
    async def test_invalid_timeout_rejected(self, fake_e2b: FakeE2B, timeout: float) -> None:
        backend = await started()
        with pytest.raises(ValueError, match='timeout must be a positive finite number'):
            await backend.run(['x'], timeout=timeout)

    async def test_deadline_kills_and_reports_the_output_so_far(self, fake_e2b: FakeE2B) -> None:
        # The protocol says an expired deadline raises a `TimeoutError`; the output the
        # command produced before the kill rides on the exception, which is the only place
        # the result-or-raise shape leaves for it.
        fake_e2b.responder = lambda command, timeout: ('partial', 'oops', 0)
        fake_e2b.command_hangs = True
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError) as exc:
            await backend.run(['sleep', '99'], timeout=0.05)
        assert (exc.value.stdout, exc.value.stderr) == ('partial', 'oops')
        assert fake_e2b.sandboxes[0].commands.killed_pids == [4242]

    async def test_a_cancelled_run_kills_the_command(self, fake_e2b: FakeE2B) -> None:
        # The protocol's cancellation contract: a cancelled `run()` must not knowingly leave
        # the command running. E2B has a per-command kill, so the backend uses it.
        fake_e2b.command_hangs = True
        backend = await started()
        async with anyio.create_task_group() as tg:
            tg.start_soon(backend.run, ['sleep', '99'])
            await anyio.wait_all_tasks_blocked()
            tg.cancel_scope.cancel()
        assert fake_e2b.sandboxes[0].commands.killed_pids == [4242]

    async def test_a_run_cancelled_while_the_command_starts_kills_it(self, fake_e2b: FakeE2B) -> None:
        # E2B starts the process before it returns the pid, so the start is waited out and the
        # command killed rather than left running with nothing to kill it by.
        fake_e2b.start_response_held = held = anyio.Event()
        backend = await started()
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tg:
                tg.start_soon(backend.run, ['sleep', '99'])
                await anyio.wait_all_tasks_blocked()
                scope.cancel()
                held.set()
        assert fake_e2b.sandboxes[0].commands.killed_pids == [4242]

    async def test_a_failed_kill_does_not_replace_the_timeout(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.command_hangs = True
        fake_e2b.kill_command_error = SandboxException('kill refused')
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '99'], timeout=0.05)

    async def test_a_failed_operation_names_what_failed(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = SandboxException('no such user')
        backend = await started()
        with pytest.raises(WorkspaceError, match='Command could not run in the E2B sandbox: no such user'):
            await backend.run(['x'])

    async def test_a_failing_health_probe_leaves_the_original_error(self, fake_e2b: FakeE2B) -> None:
        # The classifying probe can itself fail; the error being classified propagates as the
        # transient failure it most likely is, rather than a guess replacing it.
        fake_e2b.run_error = TimeoutException('slow')
        fake_e2b.sandbox_is_running = False
        fake_e2b.is_running_error = ConnectionResetError('transport gone')
        backend = await started()
        with pytest.raises(TimeoutException, match='slow'):
            await backend.run(['x'])

    async def test_a_gone_sandbox_names_itself_and_its_lifetime(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = TimeoutException('unavailable')
        fake_e2b.sandbox_is_running = False
        backend = await started()
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-1' is no longer running: .*`sandbox_timeout`"):
            await backend.run(['x'])

    async def test_an_attached_sandbox_names_itself_when_gone(self, fake_e2b: FakeE2B) -> None:
        backend = await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep'))
        fake_e2b.run_error = SandboxNotFoundException('gone')
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-keep' is no longer running"):
            await backend.run(['x'])

    async def test_run_wait_failure_is_a_sandbox_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.wait_error = SandboxException('stream broke')
        backend = await started()
        with pytest.raises(WorkspaceError, match='stream broke') as exc:
            await backend.run(['x'])
        assert 'the command may still be running' in str(exc.value)
        # The command may still be running, so it is killed on the way out.
        assert fake_e2b.sandboxes[0].commands.killed_pids == [4242]


class TestKilledSandbox:
    """A sandbox killed through its native handle is gone for every later operation.

    The fake follows the SDK after `kill()`: connecting 404s into `SandboxNotFoundException`,
    and envd calls on a handle that is still held fail with the 502 `TimeoutException` the
    SDK blames on the sandbox timeout, which only the health probe tells from a slow request.
    """

    async def test_attaching_after_kill_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        owner = await started()
        assert owner.ref is not None
        assert await (await owner.get_sandbox()).kill() is True
        attached = E2BSandboxBackend(ref=owner.ref)
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-1' is no longer running"):
            await attached.working_dir()
        assert await fake_e2b.sandboxes[0].kill() is False

    async def test_a_command_on_a_killed_sandbox_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await (await backend.get_sandbox()).kill()
        with pytest.raises(WorkspaceUnavailableError, match='it was killed'):
            await backend.run(['true'])

    async def test_a_filesystem_call_on_a_killed_sandbox_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await (await backend.get_sandbox()).kill()
        with pytest.raises(WorkspaceUnavailableError, match='it was killed'):
            await backend.read_bytes('/tmp/a.txt')

    async def test_a_backend_attached_before_the_kill_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        owner = await started()
        assert owner.ref is not None
        attached = await started(ref=owner.ref)
        await (await owner.get_sandbox()).kill()
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-1' is no longer running"):
            await attached.write_bytes('/tmp/a.txt', b'x')


class TestWorkingDir:
    async def test_a_configured_working_dir_is_resolved_and_initializes_ref(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/real/work\n', '', 0)
        backend = E2BSandboxBackend(working_dir='/work')
        assert await backend.working_dir() == '/real/work'
        assert backend.ref is not None
        assert fake_e2b.sandboxes[0].commands.calls[0].cwd == '/work'

    async def test_probed_once_and_cached(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/home/user\n', '', 0)
        backend = await started()
        assert await backend.working_dir() == '/home/user'
        assert await backend.working_dir() == '/home/user'
        assert [call.command for call in fake_e2b.sandboxes[0].commands.calls] == ['pwd -P']

    async def test_the_probe_carries_a_deadline(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        # The probe is a command like any other, so it is bounded and killed rather than left
        # to hang a run that only wanted to resolve a path.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._INTERNAL_EXEC_TIMEOUT', 0.05)
        fake_e2b.command_hangs = True
        backend = await started()
        with anyio.fail_after(5):
            with pytest.raises(WorkspaceTimeoutError):
                await backend.working_dir()

    @pytest.mark.parametrize(
        ('stdout', 'exit_code'),
        [('', 0), ('relative/dir\n', 0), ('/home/user\n', 1)],
    )
    async def test_an_unusable_answer_is_refused(self, fake_e2b: FakeE2B, stdout: str, exit_code: int) -> None:
        # Caching anything but an absolute path would hand every later `resolve()` a working
        # directory that is not one, mis-resolving relative paths with no error.
        fake_e2b.responder = lambda command, timeout: (stdout, '', exit_code)
        backend = await started()
        with pytest.raises(WorkspaceError, match='Could not determine the working directory'):
            await backend.working_dir()

    async def test_the_facade_resolves_relative_paths_against_it(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/home/user\n', '', 0)
        workspace = Workspace(await started())
        assert await workspace.resolve('src/main.py') == '/home/user/src/main.py'

    async def test_the_facade_preserves_trailing_space_in_probed_working_dir(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/srv/project \n' if command == 'pwd -P' else '', '', 0)
        backend = await started()
        fake_e2b.sandboxes[0].files.files['/srv/project /marker.txt'] = b'found'
        workspace = Workspace(backend)
        assert await workspace.read_text('marker.txt') == 'found'


class TestFilesystem:
    async def test_stat_reports_size_for_files(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        entry = await backend.stat('/tmp/a.txt')
        assert (entry.name, entry.path, entry.is_dir, entry.size) == ('a.txt', '/tmp/a.txt', False, 4)

    async def test_stat_reports_no_size_for_directories(self, fake_e2b: FakeE2B) -> None:
        # A directory's reported size is a filesystem implementation detail, not a content
        # length, so the protocol carrier reports none.
        backend = await started()
        await backend.make_dir('/tmp/pkg')
        entry = await backend.stat('/tmp/pkg')
        assert (entry.is_dir, entry.size) == (True, None)

    async def test_list_dir_returns_absolute_paths(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/srv/a.py', b'print(1)')
        await backend.make_dir('/srv/pkg')
        entries = await backend.list_dir('/srv')
        assert [(entry.name, entry.path, entry.is_dir, entry.size) for entry in entries] == [
            ('a.py', '/srv/a.py', False, 8),
            ('pkg', '/srv/pkg', True, None),
        ]

    async def test_symlinks_report_their_target(self, fake_e2b: FakeE2B, tmp_path: Path) -> None:
        fake_e2b.host_root = tmp_path
        (tmp_path / 'data.txt').write_bytes(b'12345')
        (tmp_path / 'pkg').mkdir()
        (tmp_path / 'to-file').symlink_to('data.txt')
        (tmp_path / 'to-dir').symlink_to('pkg')
        (tmp_path / 'dangling').symlink_to('missing')
        entries = await E2BSandboxBackend().list_dir(str(tmp_path))
        assert {entry.name: (entry.is_dir, entry.size) for entry in entries} == {
            'dangling': (False, None),
            'data.txt': (False, 5),
            'pkg': (True, None),
            'to-dir': (True, None),
            'to-file': (False, 5),
        }

    async def test_symlink_whose_target_is_gone_reads_as_dangling(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.sandboxes[0].files.symlinks['/srv/link'] = '/srv/removed'
        entry = await backend.stat('/srv/link')
        assert (entry.is_dir, entry.size) == (False, None)

    async def test_remove_deletes_a_directory_tree(self, fake_e2b: FakeE2B) -> None:
        # One call covers both halves of the protocol's `remove`: E2B deletes a file or a
        # directory with everything under it.
        backend = await started()
        await backend.write_bytes('/tmp/pkg/nested/a.txt', b'body')
        await backend.remove('/tmp/pkg')
        assert await backend.exists('/tmp/pkg/nested/a.txt') is False

    @pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove'])
    async def test_a_missing_path_raises_the_builtin_error(self, fake_e2b: FakeE2B, operation: str) -> None:
        # The protocol's contract: backends translate their SDK's own missing-file exception
        # into the builtin `FileNotFoundError` every consumer already handles.
        backend = await started()
        with pytest.raises(FileNotFoundError, match="'/tmp/missing.txt'"):
            await getattr(backend, operation)('/tmp/missing.txt')

    @pytest.mark.parametrize(
        ('operation', 'path', 'expected'),
        [
            ('read_bytes', '/tmp/pkg', IsADirectoryError),
            ('write_bytes', '/tmp/pkg', IsADirectoryError),
            ('list_dir', '/tmp/a.txt', NotADirectoryError),
            ('stat', '/tmp/a.txt/inner', NotADirectoryError),
            ('make_dir', '/tmp/a.txt', FileExistsError),
            ('write_bytes', '/etc/hosts', PermissionError),
        ],
    )
    async def test_a_path_failure_raises_the_builtin_error(
        self, fake_e2b: FakeE2B, operation: str, path: str, expected: type[OSError]
    ) -> None:
        # envd types only a missing path; the rest arrive as a 400 or 500 whose message names
        # the failure, which is what makes them the protocol's builtin file errors.
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        await backend.make_dir('/tmp/pkg')
        fake_e2b.sandboxes[0].files.denied.add('/etc')
        args = (path, b'x') if operation == 'write_bytes' else (path,)
        with pytest.raises(expected, match=re.escape(repr(path))) as exc:
            await getattr(backend, operation)(*args)
        assert type(exc.value) is expected

    async def test_another_invalid_read_stays_a_workspace_error(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        fake_e2b.read_error = InvalidArgumentException('bad request')
        with pytest.raises(WorkspaceError, match='bad request'):
            await backend.read_bytes('/tmp/a.txt')

    async def test_removing_a_missing_path_does_not_reach_e2b(self, fake_e2b: FakeE2B) -> None:
        # envd's remove succeeds on a missing path, so the backend checks first to report it.
        backend = await started()
        with pytest.raises(FileNotFoundError):
            await backend.remove('/tmp/missing')
        assert fake_e2b.sandboxes[0].files.removed == []

    async def test_exists_still_reports_other_failures(self, fake_e2b: FakeE2B) -> None:
        # Only "there is nothing at that path" is an answer; anything else is a failure.
        backend = await started()
        fake_e2b.fs_error = SandboxException('input/output error')
        with pytest.raises(WorkspaceError, match='input/output error'):
            await backend.exists('/root/x')


# What E2B raises, whether the sandbox still runs when asked, and what the protocol caller sees.
_OPERATION_ERRORS = [
    pytest.param(AuthenticationException('bad key'), True, WorkspaceUnavailableError, id='rejected-credentials'),
    pytest.param(SandboxNotFoundException('gone'), True, WorkspaceUnavailableError, id='sandbox-not-found'),
    # E2B raises `TimeoutException` for a request the sandbox never answered, slow or dead alike;
    # the health probe tells them apart.
    pytest.param(TimeoutException('unanswered'), False, WorkspaceUnavailableError, id='timeout-sandbox-gone'),
    pytest.param(TimeoutException('unanswered'), True, TimeoutException, id='timeout-sandbox-running'),
    pytest.param(NotEnoughSpaceException('disk full'), True, WorkspaceError, id='operation-failed'),
    pytest.param(RateLimitException('slow down'), True, RateLimitException, id='rate-limited'),
    pytest.param(ServiceBusyException('busy'), True, ServiceBusyException, id='service-busy'),
    pytest.param(ConnectionResetError('reset'), True, ConnectionResetError, id='transport'),
]


@pytest.mark.parametrize('operation', ['run', 'read_bytes'])
@pytest.mark.parametrize(('error', 'running', 'expected'), _OPERATION_ERRORS)
async def test_operation_errors_map_to_protocol_failures(
    fake_e2b: FakeE2B, operation: str, error: Exception, running: bool, expected: type[Exception]
) -> None:
    backend = await started()
    fake_e2b.sandbox_is_running = running
    if operation == 'run':
        fake_e2b.run_error = error
        call = backend.run(['x'])
    else:
        fake_e2b.fs_error = error
        call = backend.read_bytes('/x')
    with pytest.raises(expected) as exc:
        await call
    assert type(exc.value) is expected
    if expected is type(error):
        assert exc.value is error


@pytest.mark.parametrize('attach', [False, True], ids=['create', 'connect'])
@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        pytest.param(AuthenticationException('bad key'), WorkspaceUnavailableError, id='rejected-credentials'),
        pytest.param(SandboxException('no capacity'), WorkspaceError, id='operation-failed'),
        # Nothing is acquired yet to probe, so an unanswered request is transient.
        pytest.param(TimeoutException('unanswered'), TimeoutException, id='timeout'),
        pytest.param(RateLimitException('slow down'), RateLimitException, id='rate-limited'),
        # The SDK's own timeout, not the backend's creation deadline, so it is not rewritten.
        pytest.param(TimeoutError('socket connect timed out'), TimeoutError, id='transport-timeout'),
    ],
)
async def test_acquisition_errors_map_to_protocol_failures(
    fake_e2b: FakeE2B, attach: bool, error: Exception, expected: type[Exception]
) -> None:
    if attach:
        fake_e2b.connect_error = error
    else:
        fake_e2b.create_error = error
    with pytest.raises(expected) as exc:
        await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep') if attach else None)
    assert type(exc.value) is expected
    if expected is type(error):
        assert exc.value is error


def test_missing_e2b_extra_has_an_install_hint() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            "import sys; sys.modules['e2b'] = None; import pydantic_ai_harness.e2b_sandbox",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert 'Install `pydantic-ai-harness[e2b]`' in result.stderr


async def test_command_timeout_starts_once_the_sandbox_is_acquired(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_response_held = held = anyio.Event()
    backend = E2BSandboxBackend()

    async def release() -> None:
        # Creating the sandbox outlasts the timeout; the command itself fits in it comfortably.
        await anyio.sleep(1.1)
        held.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(release)
        result = await backend.run(['echo', 'ready'], timeout=1)
        assert (result.exit_code, result.stdout) == (0, 'echo ready\n')


async def test_filesystem_first_use_preserves_auth_error(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_error = AuthenticationException('denied')
    with pytest.raises(WorkspaceUnavailableError):
        await E2BSandboxBackend().read_bytes('/file')
