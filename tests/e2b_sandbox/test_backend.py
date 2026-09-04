"""Tests for `E2BSandboxBackend`, the E2B implementation of the sandbox protocol."""

from __future__ import annotations

import builtins
import functools
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import pytest
from pydantic_ai.sandboxes import (
    Sandbox,
    SandboxBackend,
    SandboxError,
    SandboxRef,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SupportsFilesystem,
)

from pydantic_ai_harness.e2b_sandbox import (
    E2BSandboxAuthError,
    E2BSandboxBackend,
    E2BSandboxError,
    E2BSandboxUnavailableError,
)

from ..sandbox_conformance import (
    check_command_validation,
    check_missing_file,
    check_timeout,
)
from .fake_e2b import FakeE2B

_EntryPoint = Callable[[], Awaitable[object]]


async def started(**settings: Any) -> E2BSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting the property is that touch.
    """
    backend = E2BSandboxBackend(**settings)
    await backend.sandbox
    return backend


_CREATE: _EntryPoint = functools.partial(started)
_CONNECT: _EntryPoint = functools.partial(started, ref=SandboxRef(sandbox_id='sbx-keep'))
_CREATE_OR_CONNECT: _EntryPoint = functools.partial(started, identity={'k': 'v'})
_KILL_BY_ID: _EntryPoint = functools.partial(E2BSandboxBackend.kill_by_id, 'missing')


def _hide_e2b(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import e2b` fail, as it does without the extra installed."""
    real_import = builtins.__import__

    def no_e2b(name: str, *args: object, **kwargs: object) -> object:
        if name == 'e2b':
            raise ImportError('No module named e2b')
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, 'e2b', raising=False)
    monkeypatch.setattr(builtins, '__import__', no_e2b)


class TestConformance:
    async def test_backend_implements_run_and_filesystem_protocols(self, fake_e2b: FakeE2B) -> None:
        # `isinstance` is shallow (member presence only); the signature half is pinned
        # statically by the `if TYPE_CHECKING` block in `_backend.py`.
        backend = await started()
        assert isinstance(backend, SandboxBackend)
        assert isinstance(backend, SupportsFilesystem)

    async def test_identity_is_e2b_sandbox_id(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        assert backend.ref == SandboxRef(sandbox_id='sbx-1')
        assert await backend.sandbox is fake_e2b.sandboxes[0]

    async def test_shared_command_validation(self, fake_e2b: FakeE2B) -> None:
        await check_command_validation(started)

    async def test_shared_missing_file(self, fake_e2b: FakeE2B) -> None:
        await check_missing_file(started)

    async def test_shared_timeout(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.command_hangs = True
        await check_timeout(started)

    async def test_shared_run_and_nonzero_result(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('', '', 2)
        backend = await started()
        result = await backend.run(['false'])
        assert result.exit_code != 0


class TestCreate:
    async def test_creates_from_config(self, fake_e2b: FakeE2B) -> None:
        backend = await started(
            template='base',
            sandbox_timeout=120,
            env={'FOO': 'bar'},
            metadata={'owner': 'harness'},
            allow_internet_access=False,
        )
        assert backend.ref == SandboxRef(sandbox_id='sbx-1')
        call = fake_e2b.create_calls[-1]
        assert (call.template, call.timeout, call.envs) == ('base', 120, {'FOO': 'bar'})
        assert (call.metadata, call.allow_internet_access) == ({'owner': 'harness'}, False)

    async def test_defaults(self, fake_e2b: FakeE2B) -> None:
        await started()
        call = fake_e2b.create_calls[-1]
        assert (call.template, call.timeout, call.envs, call.metadata) == (None, 300, None, None)
        # Passed explicitly: it decides whether the sandbox is reachable without its token.
        assert (call.secure, call.allow_internet_access) == (True, True)

    async def test_e2b_error_becomes_a_start_failure(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.create_error = fake_e2b.error_type('no capacity')
        with pytest.raises(E2BSandboxError, match='Could not start E2B sandbox: SandboxException: no capacity'):
            await started()

    async def test_auth_error_is_terminal(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.create_error = fake_e2b.auth_type('bad key')
        with pytest.raises(E2BSandboxAuthError, match='E2B rejected the credentials'):
            await started()

    async def test_hanging_create_does_not_hang_the_caller(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The client-side bound prevents a wedged control plane from hanging acquisition.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._CREATE_TIMEOUT', 0.05)
        fake_e2b.create_hangs = True
        with anyio.fail_after(5):
            with pytest.raises(E2BSandboxError, match='did not complete within'):
                await started()

    async def test_rejects_relative_working_dir(self, fake_e2b: FakeE2B) -> None:
        with pytest.raises(ValueError, match='working_dir must be an absolute sandbox path'):
            await started(working_dir='repo')


class TestConnect:
    async def test_connects_to_an_existing_sandbox(self, fake_e2b: FakeE2B) -> None:
        # E2B resumes a paused sandbox on connect, so no separate liveness probe is needed:
        # a sandbox that is really gone raises instead of handing back a dead handle.
        backend = await started(ref=SandboxRef(sandbox_id='sbx-keep'))
        assert fake_e2b.connect_calls == [('sbx-keep', None)]
        assert backend.ref == SandboxRef(sandbox_id='sbx-keep')

    async def test_attaching_leaves_the_lifetime_alone(self, fake_e2b: FakeE2B) -> None:
        # E2B substitutes its own 300-second default when `timeout` is `None` at connect time,
        # which would silently extend a shorter remaining lifetime just by looking at the
        # sandbox. Attaching asks for no timeout at all, so nothing about the lifetime moves.
        await started(ref=SandboxRef(sandbox_id='sbx-keep'))

        assert fake_e2b.connect_calls == [('sbx-keep', None)]

    async def test_connect_to_a_missing_sandbox_fails(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = fake_e2b.sandbox_gone_type('not found')
        with pytest.raises(E2BSandboxUnavailableError, match="'sbx-gone'"):
            await started(ref=SandboxRef(sandbox_id='sbx-gone'))

    async def test_connect_auth_error_is_terminal(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = fake_e2b.auth_type('bad key')
        with pytest.raises(E2BSandboxAuthError, match='E2B rejected the credentials'):
            await started(ref=SandboxRef(sandbox_id='sbx-keep'))

    async def test_other_connect_failures_are_recoverable(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = fake_e2b.error_type('service unavailable')
        with pytest.raises(E2BSandboxError, match='Could not connect to E2B sandbox') as exc:
            await started(ref=SandboxRef(sandbox_id='sbx-keep'))
        assert not isinstance(exc.value, SandboxUnavailableError)


class TestClose:
    async def test_an_unused_backend_has_nothing_to_close(self, fake_e2b: FakeE2B) -> None:
        # Building one does no I/O, so closing it must not reach E2B either -- resolving here
        # would create the very sandbox being released.
        await E2BSandboxBackend().close(terminate=True)

        assert fake_e2b.sandboxes == []
        assert fake_e2b.kill_ids == []

    async def test_kills_when_owned(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.close(terminate=True)
        assert fake_e2b.sandboxes[0].killed is True

    async def test_leaves_an_attached_sandbox_running(self, fake_e2b: FakeE2B) -> None:
        backend = await started(ref=SandboxRef(sandbox_id='sbx-keep'))
        await backend.close(terminate=False)
        assert fake_e2b.sandboxes[0].killed is False

    async def test_kill_failure_is_visible(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.kill_error = RuntimeError('kill boom')
        with pytest.raises(E2BSandboxError, match='kill boom'):
            await backend.close(terminate=True)

    async def test_already_gone_sandbox_is_not_an_error(self, fake_e2b: FakeE2B) -> None:
        # An owned run that outlived its `sandbox_timeout` self-terminates; the teardown kill
        # then hits "already gone", which is success, not a failure to raise.
        backend = await started()
        fake_e2b.kill_error = fake_e2b.sandbox_gone_type('already gone')
        await backend.close(terminate=True)

    async def test_auth_failure_during_kill_is_typed(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.kill_error = fake_e2b.auth_type('bad key')

        with pytest.raises(E2BSandboxAuthError, match='E2B rejected the credentials'):
            await backend.close(terminate=True)

    async def test_hanging_kill_is_bounded(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        # Teardown runs shielded, so a hanging kill would be uncancellable; its own deadline
        # is the only bound between a wedged control plane and a hung process.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        backend = await started()
        fake_e2b.kill_hangs = True
        with anyio.fail_after(5):
            with pytest.raises(E2BSandboxError, match='Timed out'):
                await backend.close(terminate=True)


@pytest.mark.parametrize(
    'entry_point',
    [_CREATE, _CONNECT, _CREATE_OR_CONNECT, _KILL_BY_ID],
    ids=['create', 'connect', 'create_or_connect', 'kill_by_id'],
)
async def test_missing_e2b_package_is_named(monkeypatch: pytest.MonkeyPatch, entry_point: _EntryPoint) -> None:
    _hide_e2b(monkeypatch)
    with pytest.raises(E2BSandboxError, match="The 'e2b' package is required"):
        await entry_point()


class TestRun:
    async def test_argv_is_quoted_into_one_shell_word_string(self, fake_e2b: FakeE2B) -> None:
        # E2B has no argv form: every command goes through `/bin/bash -l -c`, so the quoting
        # is what keeps an argument with a space or a `$` one literal word.
        backend = await started()
        await backend.run(['echo', 'a b', '$HOME'])
        assert fake_e2b.sandboxes[0].commands.calls[-1].command == "echo 'a b' '$HOME'"

    async def test_shell_string_is_passed_through(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.run('echo hi | wc -c', shell=True)
        assert fake_e2b.sandboxes[0].commands.calls[-1].command == 'echo hi | wc -c'

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

    async def test_rejects_relative_cwd(self, fake_e2b: FakeE2B) -> None:
        backend = await started()

        with pytest.raises(ValueError, match='cwd must be an absolute sandbox path'):
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

    @pytest.mark.parametrize(
        ('command', 'shell', 'message'),
        [
            (['ls'], True, 'an argv sequence cannot be combined with shell=True'),
            ('ls -la', False, 'a string command requires shell=True'),
            ([], False, 'the argv sequence is empty'),
        ],
    )
    async def test_command_shape_mismatches_are_rejected(
        self, fake_e2b: FakeE2B, command: str | list[str], shell: bool, message: str
    ) -> None:
        backend = await started()
        with pytest.raises(TypeError, match=message):
            await backend.run(command, shell=shell)

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
        with pytest.raises(SandboxTimeoutError) as exc:
            await backend.run(['sleep', '99'], timeout=0.05)
        assert isinstance(exc.value, TimeoutError)
        assert (exc.value.stdout, exc.value.stderr, exc.value.timeout) == ('partial', 'oops', 0.05)
        assert fake_e2b.sandboxes[0].commands.killed_pids == [4242]

    async def test_a_cancelled_run_kills_the_command(self, fake_e2b: FakeE2B) -> None:
        # The protocol's cancellation contract: a cancelled `run()` must not knowingly leave
        # the command running. E2B has a per-command kill, so the backend uses it.
        fake_e2b.command_hangs = True
        backend = await started()
        with anyio.move_on_after(0.05):
            await backend.run(['sleep', '99'])
        assert fake_e2b.sandboxes[0].commands.killed_pids == [4242]

    async def test_a_failed_kill_does_not_replace_the_timeout(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.command_hangs = True
        fake_e2b.kill_command_error = fake_e2b.error_type('kill refused')
        backend = await started()
        with pytest.raises(SandboxTimeoutError):
            await backend.run(['sleep', '99'], timeout=0.05)

    async def test_run_failure_is_a_recoverable_sandbox_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = fake_e2b.error_type('transient blip')
        backend = await started()
        with pytest.raises(E2BSandboxError, match='Command could not run in the sandbox: transient blip') as exc:
            await backend.run(['x'])
        assert isinstance(exc.value, SandboxError)
        assert not isinstance(exc.value, SandboxUnavailableError)

    async def test_a_non_e2b_failure_names_its_type(self, fake_e2b: FakeE2B) -> None:
        # Transport failures are not `SandboxException`; they must still surface as a typed,
        # recoverable sandbox error rather than abort the run.
        fake_e2b.run_error = ValueError('connection reset')
        backend = await started()
        with pytest.raises(E2BSandboxError, match='ValueError: connection reset'):
            await backend.run(['x'])

    @pytest.mark.parametrize(
        ('exc_property', 'match'),
        [
            ('sandbox_gone_type', 'no longer running'),
            ('auth_type', 'E2B rejected the credentials'),
        ],
    )
    async def test_terminal_run_failures(self, fake_e2b: FakeE2B, exc_property: str, match: str) -> None:
        exc_type: type[Exception] = getattr(fake_e2b, exc_property)
        fake_e2b.run_error = exc_type('terminal failure')
        backend = await started()
        with pytest.raises(SandboxUnavailableError, match=match):
            await backend.run(['x'])

    async def test_a_dead_sandbox_behind_a_timeout_is_terminal(self, fake_e2b: FakeE2B) -> None:
        # E2B reports an unanswered request as `TimeoutException` whether the sandbox is slow
        # or gone; the health probe is what keeps the model out of a retry loop against a
        # sandbox that expired.
        fake_e2b.run_error = fake_e2b.ambiguous_type('unavailable')
        fake_e2b.sandbox_is_running = False
        backend = await started()
        with pytest.raises(E2BSandboxUnavailableError, match='sandbox_timeout of 300s'):
            await backend.run(['x'])

    async def test_a_timeout_on_a_live_sandbox_stays_recoverable(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = fake_e2b.ambiguous_type('slow')
        backend = await started()
        with pytest.raises(E2BSandboxError, match='slow') as exc:
            await backend.run(['x'])
        assert isinstance(exc.value, SandboxError)
        assert not isinstance(exc.value, SandboxUnavailableError)

    async def test_a_failing_health_probe_preserves_the_original_error(self, fake_e2b: FakeE2B) -> None:
        # The classifying probe can itself fail with a raw transport error; that must not
        # abort the run in place of the error we were classifying.
        fake_e2b.run_error = fake_e2b.ambiguous_type('slow')
        fake_e2b.is_running_error = ValueError('transport gone')
        backend = await started()
        with pytest.raises(E2BSandboxError, match='slow'):
            await backend.run(['x'])

    async def test_an_attached_sandbox_names_itself_when_gone(self, fake_e2b: FakeE2B) -> None:
        # A connected backend does not know the lifetime it was created with, so it points at
        # the sandbox instead of quoting a `sandbox_timeout` it never set.
        backend = await started(ref=SandboxRef(sandbox_id='sbx-keep'))
        fake_e2b.run_error = fake_e2b.sandbox_gone_type('gone')
        with pytest.raises(E2BSandboxUnavailableError, match="'sbx-keep' is no longer running"):
            await backend.run(['x'])

    async def test_run_wait_failure_is_a_sandbox_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.wait_error = fake_e2b.error_type('stream broke')
        backend = await started()
        with pytest.raises(E2BSandboxError, match='stream broke') as exc:
            await backend.run(['x'])
        assert 'the command may still be running' in str(exc.value)
        # The command may still be running, so it is killed on the way out.
        assert fake_e2b.sandboxes[0].commands.killed_pids == [4242]


class TestWorkingDir:
    async def test_a_configured_working_dir_needs_no_probe(self, fake_e2b: FakeE2B) -> None:
        backend = await started(working_dir='/work')
        assert await backend.working_dir() == '/work'
        assert fake_e2b.sandboxes[0].commands.calls == []

    async def test_probed_once_and_cached(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/home/user\n', '', 0)
        backend = await started()
        assert await backend.working_dir() == '/home/user'
        assert await backend.working_dir() == '/home/user'
        assert [call.command for call in fake_e2b.sandboxes[0].commands.calls] == ['pwd']

    async def test_the_probe_carries_a_deadline(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        # The probe is a command like any other, so it is bounded and killed rather than left
        # to hang a run that only wanted to resolve a path.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._INTERNAL_EXEC_TIMEOUT', 0.05)
        fake_e2b.command_hangs = True
        backend = await started()
        with anyio.fail_after(5):
            with pytest.raises(SandboxTimeoutError):
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
        with pytest.raises(E2BSandboxError, match='Could not determine the working directory'):
            await backend.working_dir()

    async def test_the_facade_resolves_relative_paths_against_it(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/home/user\n', '', 0)
        sandbox = Sandbox(await started())
        assert await sandbox.resolve('src/main.py') == '/home/user/src/main.py'


class TestFilesystem:
    async def test_write_then_read_round_trips(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        assert await backend.read_bytes('/tmp/a.txt') == b'body'

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

    async def test_remove_deletes_a_directory_tree(self, fake_e2b: FakeE2B) -> None:
        # One call covers both halves of the protocol's `remove`: E2B deletes a file or a
        # directory with everything under it.
        backend = await started()
        await backend.write_bytes('/tmp/pkg/nested/a.txt', b'body')
        await backend.remove('/tmp/pkg')
        assert await backend.exists('/tmp/pkg/nested/a.txt') is False

    async def test_exists(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        assert await backend.exists('/tmp/a.txt') is True
        assert await backend.exists('/tmp/missing.txt') is False

    @pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove'])
    async def test_a_missing_path_raises_the_builtin_error(self, fake_e2b: FakeE2B, operation: str) -> None:
        # The protocol's contract: backends translate their SDK's own missing-file exception
        # into the builtin `FileNotFoundError` every consumer already handles.
        backend = await started()
        with pytest.raises(FileNotFoundError, match="'/tmp/missing.txt'"):
            await getattr(backend, operation)('/tmp/missing.txt')

    async def test_a_filesystem_error_is_recoverable_while_the_sandbox_runs(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.fs_error = fake_e2b.error_type('Permission denied')
        with pytest.raises(E2BSandboxError, match='Permission denied') as exc:
            await backend.write_bytes('/root/x', b'data')
        assert isinstance(exc.value, SandboxError)
        assert not isinstance(exc.value, SandboxUnavailableError)

    async def test_a_filesystem_error_on_a_dead_sandbox_is_terminal(self, fake_e2b: FakeE2B) -> None:
        # E2B reports an envd request the sandbox never answered as a timeout, whether it is
        # slow or gone; the health probe is what tells the model to stop retrying.
        backend = await started()
        fake_e2b.fs_error = fake_e2b.ambiguous_type('request failed')
        fake_e2b.sandbox_is_running = False
        with pytest.raises(E2BSandboxUnavailableError):
            await backend.read_bytes('/x')

    async def test_a_missing_sandbox_is_terminal(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.fs_error = fake_e2b.sandbox_gone_type('sandbox gone')
        with pytest.raises(E2BSandboxUnavailableError):
            await backend.list_dir('/x')

    async def test_an_auth_failure_is_terminal(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.fs_error = fake_e2b.auth_type('bad key')
        with pytest.raises(E2BSandboxAuthError, match='E2B rejected the credentials'):
            await backend.make_dir('/x')

    async def test_exists_still_reports_other_failures(self, fake_e2b: FakeE2B) -> None:
        # Only "there is nothing at that path" is an answer; anything else is a failure.
        backend = await started()
        fake_e2b.fs_error = fake_e2b.error_type('Permission denied')
        with pytest.raises(E2BSandboxError, match='Permission denied'):
            await backend.exists('/root/x')
