"""Substantive tests for the Modal workspace backend."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

from .fake_modal import FakeImage, FakeModal, FileInfo

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def started(**settings: Any) -> ModalSandboxBackend:
    backend = ModalSandboxBackend(**settings)
    await backend.get_client()
    return backend


class TestRun:
    async def test_argv_runs_without_a_shell(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: (' '.join(argv), '', 0)
        backend = await started()
        result = await backend.run(['echo', 'hi'])
        assert result.stdout == 'echo hi'
        assert fake_modal.sandboxes[0].exec_calls[-1].argv == ['echo', 'hi']

    async def test_shell_wraps_in_sh(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.run('echo hi | wc -c', shell=True)
        assert fake_modal.sandboxes[0].exec_calls[-1].argv == ['/bin/sh', '-c', 'echo hi | wc -c']

    async def test_reports_streams_and_exit_code(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('out', 'err', 2)
        backend = await started()
        result = await backend.run(['false'])
        assert (result.stdout, result.stderr, result.exit_code) == ('out', 'err', 2)

    async def test_cwd_and_env_reach_the_command(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.run(['env'], cwd='/srv', env={'FOO': 'bar'})
        call = fake_modal.sandboxes[0].exec_calls[-1]
        assert call.workdir == '/srv'
        assert call.env == {'FOO': 'bar'}

    async def test_working_dir_and_env_apply_to_every_command_of_an_attached_sandbox(
        self, fake_modal: FakeModal
    ) -> None:
        # Given per command rather than only at creation, so a sandbox attached by reference
        # honors them too; a command's own `cwd` and `env` take precedence.
        backend = await started(
            ref=WorkspaceRef(provider='modal', id='sb-keep'), working_dir='/work', env={'A': '1', 'B': '1'}
        )
        await backend.run(['env'])
        await backend.run(['env'], cwd='/srv', env={'B': '2'})
        calls = fake_modal.sandboxes[0].exec_calls
        assert [(call.workdir, call.env) for call in calls] == [
            ('/work', {'A': '1', 'B': '1'}),
            ('/srv', {'A': '1', 'B': '2'}),
        ]

    async def test_rejects_relative_cwd(self, fake_modal: FakeModal) -> None:
        backend = await started()

        with pytest.raises(ValueError, match='cwd must be an absolute workspace path'):
            await backend.run(['pwd'], cwd='repo')

    async def test_preserves_parent_segments_in_cwd(self, fake_modal: FakeModal) -> None:
        backend = await started()

        await backend.run(['pwd'], cwd='/linked/../target')

        assert fake_modal.sandboxes[0].exec_calls[-1].workdir == '/linked/../target'

    async def test_fractional_timeout_rounds_up_to_a_modal_deadline(self, fake_modal: FakeModal) -> None:
        # Modal takes whole seconds and reads 0 as "no timeout", so a sub-second deadline
        # must not floor to unbounded.
        backend = await started()
        await backend.run(['x'], timeout=0.5)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 1

    async def test_timeout_none_stays_unbounded(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.run(['x'])
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout is None

    @pytest.mark.parametrize('timeout', [0, -1.0, float('inf'), float('nan')])
    async def test_invalid_timeout_rejected(self, fake_modal: FakeModal, timeout: float) -> None:
        backend = await started()
        with pytest.raises(ValueError, match='timeout must be a positive finite number'):
            await backend.run(['x'], timeout=timeout)

    async def test_client_deadline_sentinel_raises_a_timeout(self, fake_modal: FakeModal) -> None:
        # Modal's -1 is its client-side deadline sentinel; the protocol says a deadline
        # raises, and the output produced before the kill rides on the exception.
        fake_modal.responder = lambda argv, timeout: ('partial', 'oops', -1)
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError) as exc:
            await backend.run(['sleep', '99'], timeout=5)
        assert isinstance(exc.value, TimeoutError)
        assert (exc.value.stdout, exc.value.stderr) == ('partial', 'oops')

    async def test_a_timeout_message_quotes_the_callers_timeout(self, fake_modal: FakeModal) -> None:
        # Modal enforces whole seconds, so 0.5 runs as a 1-second deadline; the message still
        # names the timeout the caller asked for.
        fake_modal.responder = lambda argv, timeout: ('', '', -1)
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError, match=r'^Command timed out after 0\.5 seconds\.$'):
            await backend.run(['sleep', '99'], timeout=0.5)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 1

    async def test_sentinel_without_a_deadline_is_a_real_exit(self, fake_modal: FakeModal) -> None:
        # -1 is only the timeout sentinel when we set a deadline; from another cause it is
        # the honest exit code.
        fake_modal.responder = lambda argv, timeout: ('', '', -1)
        backend = await started()
        assert (await backend.run(['x'])).exit_code == -1

    async def test_server_side_deadline_kill_is_a_timeout(self, fake_modal: FakeModal) -> None:
        # The server enforces the deadline before the client's own clock fires, so its
        # SIGKILL (exit 137) can beat Modal's -1 sentinel; a 137 that consumed the whole
        # deadline window is a timeout, not a mysterious ordinary exit.
        def deadline_kill(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            time.sleep(1.05)  # the deadline is consumed inside the exec RPC, before `wait()`
            return '', '', 137

        fake_modal.responder = deadline_kill
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '99'], timeout=1)

    async def test_early_sigkill_is_a_real_exit(self, fake_modal: FakeModal) -> None:
        # A command that dies by SIGKILL well before the deadline (an OOM kill, a `kill -9`
        # it asked for) reports the exit code it really had.
        fake_modal.responder = lambda argv, timeout: ('', '', 137)
        backend = await started()
        assert (await backend.run(['kill-self'], timeout=15)).exit_code == 137

    async def test_early_sigkill_with_slow_output_is_a_real_exit(self, fake_modal: FakeModal) -> None:
        # The exit is timed when the process ends, so output that drains past the deadline
        # does not turn an early SIGKILL into a timeout.
        fake_modal.responder = lambda argv, timeout: ('', '', 137)
        fake_modal.stdout_delay = 1.1
        backend = await started()
        assert (await backend.run(['kill-self'], timeout=1)).exit_code == 137

    @pytest.mark.parametrize('stage', ['exec_error', 'wait_error'])
    async def test_an_sdk_timeout_error_is_not_a_command_timeout(self, fake_modal: FakeModal, stage: str) -> None:
        # A `TimeoutError` from Modal's transport is transient; only the command's own deadline
        # is a `WorkspaceTimeoutError`.
        setattr(fake_modal, stage, TimeoutError('transport'))
        backend = await started()
        with pytest.raises(TimeoutError, match='transport') as exc:
            await backend.run(['x'])
        assert not isinstance(exc.value, WorkspaceTimeoutError)

    async def test_invalid_utf8_output_uses_replacement_characters(self, fake_modal: FakeModal) -> None:
        # Modal's text mode decodes strictly; reading bytes and decoding with replacement
        # keeps a command printing binary from aborting the run.
        fake_modal.responder = lambda argv, timeout: (b'\xff\xfe', b'', 0)
        backend = await started()
        assert (await backend.run(['cat', 'binary'])).stdout == '��'

    @pytest.mark.parametrize(
        ('name', 'expected', 'match'),
        [
            ('ExecutionError', WorkspaceError, 'Command could not run in the workspace: failed'),
            ('SandboxTimeoutError', WorkspaceUnavailableError, "'sb-owned' is no longer running"),
            ('AuthError', WorkspaceUnavailableError, 'Modal rejected the credentials'),
        ],
    )
    async def test_exec_failures_are_mapped(
        self, fake_modal: FakeModal, name: str, expected: type[Exception], match: str
    ) -> None:
        fake_modal.exec_error = fake_modal.exception(name)('failed')
        backend = await started()
        with pytest.raises(expected, match=match) as exc:
            await backend.run(['x'])
        assert type(exc.value) is expected

    async def test_exec_transport_failure_propagates(self, fake_modal: FakeModal) -> None:
        fake_modal.exec_error = fake_modal.exception('ConnectionError')('connection reset')
        backend = await started()
        with pytest.raises(fake_modal.exception('ConnectionError')):
            await backend.run(['x'])

    async def test_dead_workspace_conflict_is_terminal(self, fake_modal: FakeModal) -> None:
        # A first exec on a dead sandbox surfaces as Modal's ambiguous ConflictError; the
        # poll disambiguates it from a transient abort.
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('Sandbox already finished')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['x'])

    async def test_shutting_down_conflict_is_terminal(self, fake_modal: FakeModal) -> None:
        # Right after `terminate()`, Modal still polls the sandbox as running but refuses exec
        # with this ConflictError; the sandbox will not come back, so it is not retryable.
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('Modal Sandbox is shutting down.')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['x'])

    async def test_transient_conflict_stays_recoverable(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        with pytest.raises(WorkspaceError, match='aborted') as exc:
            await backend.run(['x'])
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    async def test_failing_poll_preserves_the_original_error(self, fake_modal: FakeModal) -> None:
        # The classifying poll can itself fail with a raw transport error; that must not
        # abort the run in place of the error we were classifying.
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        fake_modal.sandboxes[0].poll_error = ValueError('transport gone')
        with pytest.raises(WorkspaceError, match='aborted'):
            await backend.run(['x'])

    async def test_poll_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.exception('AuthError')('unauthenticated')
        with pytest.raises(WorkspaceUnavailableError, match='Modal rejected the credentials'):
            await backend.run(['x'])

    async def test_poll_reporting_a_missing_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.exception('NotFoundError')('gone')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.run(['x'])

    async def test_attached_sandbox_names_itself_when_gone(self, fake_modal: FakeModal) -> None:
        backend = await started(ref=WorkspaceRef(provider='modal', id='sb-keep'))
        fake_modal.exec_error = fake_modal.exception('SandboxTerminatedError')('gone')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-keep' is no longer running"):
            await backend.run(['x'])

    async def test_run_wait_failure_is_a_workspace_error(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = fake_modal.exception('ExecutionError')('wait failed')
        backend = await started()
        with pytest.raises(WorkspaceError, match='Could not read the command result'):
            await backend.run(['x'])

    async def test_raw_run_wait_failure_propagates(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = RuntimeError('raw wait failed')
        backend = await started()
        with pytest.raises(RuntimeError, match='raw wait failed'):
            await backend.run(['x'])

    async def test_cancelling_run_propagates_the_cancellation(self, fake_modal: FakeModal) -> None:
        # A cancelled run abandons the result collection (reaping its readers) and re-raises
        # the cancellation untranslated; the command itself runs on until its Modal deadline.
        fake_modal.wait_hangs = True
        backend = await started()
        waiter = asyncio.create_task(backend.run(['x'], timeout=5))
        await anyio.wait_all_tasks_blocked()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter


class TestWorkingDir:
    async def test_configured_working_dir_is_resolved_before_first_operation(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/canonical/work\n', '', 0)
        backend = ModalSandboxBackend(working_dir='/alias')
        assert await backend.working_dir() == '/canonical/work'
        assert backend.ref is not None

    async def test_probed_once_and_cached(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        backend = await started()
        assert await backend.working_dir() == '/srv'
        assert await backend.working_dir() == '/srv'
        assert [call.argv for call in fake_modal.sandboxes[0].exec_calls] == [['pwd', '-P']]

    async def test_the_probe_carries_a_deadline(self, fake_modal: FakeModal) -> None:
        # Modal has no per-command kill, so even the internal probe is bounded.
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        backend = await started()
        await backend.working_dir()
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 10

    @pytest.mark.parametrize(
        ('stdout', 'exit_code'),
        [('', 0), ('relative/dir\n', 0), ('/srv\n', 1)],
    )
    async def test_an_unusable_answer_is_refused(self, fake_modal: FakeModal, stdout: str, exit_code: int) -> None:
        # Caching anything but an absolute path would hand every later `resolve()` a working
        # directory that is not one, mis-resolving relative paths with no error.
        fake_modal.responder = lambda argv, timeout: (stdout, '', exit_code)
        backend = await started()
        with pytest.raises(WorkspaceError, match='Could not determine the working directory'):
            await backend.working_dir()

    async def test_the_facade_resolves_relative_paths_against_it(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        workspace = Workspace(await started())
        assert await workspace.resolve('src/main.py') == '/srv/src/main.py'

    async def test_trailing_space_in_working_dir_is_preserved(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv/project \n', '', 0)
        backend = await started()
        fake_modal.sandboxes[0].files['/srv/project /marker.txt'] = b'marker'
        workspace = Workspace(backend)
        assert await workspace.read_text('marker.txt') == 'marker'


class TestCreate:
    @pytest.mark.parametrize('operation', ['run', 'write_bytes'])
    async def test_ref_is_recorded_by_the_first_operation(self, fake_modal: FakeModal, operation: str) -> None:
        backend = ModalSandboxBackend()
        assert backend.ref is None
        if operation == 'run':
            await backend.run(['true'])
        else:
            await backend.write_bytes('/tmp/file', b'data')
        assert backend.ref == WorkspaceRef(provider='modal', id=fake_modal.sandboxes[0].object_id)

    async def test_creates_from_config(self, fake_modal: FakeModal) -> None:
        backend = await started(
            image='ubuntu:22.04',
            app_name='my-app',
            create_app_if_missing=False,
            sandbox_timeout=120,
            idle_timeout=60,
            working_dir='/work',
            env={'FOO': 'bar'},
        )
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-owned')
        assert fake_modal.app_lookups[-1] == {'name': 'my-app', 'create_if_missing': False}
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.create_kwargs[-1]['timeout'] == 120
        assert fake_modal.create_kwargs[-1]['idle_timeout'] == 60
        assert fake_modal.create_kwargs[-1]['workdir'] == '/work'
        assert fake_modal.create_kwargs[-1]['env'] == {'FOO': 'bar'}

    async def test_an_image_object_is_used_as_given(self, fake_modal: FakeModal) -> None:
        image: Any = object()
        await started(image=image)
        assert fake_modal.create_kwargs[-1]['image'] is image
        assert fake_modal.image_tags == []

    async def test_default_image_has_git_and_ripgrep(self, fake_modal: FakeModal) -> None:
        await started()
        image = fake_modal.create_kwargs[-1]['image']
        assert isinstance(image, FakeImage)
        assert {'git', 'ripgrep'} <= set(image.apt_packages)

    async def test_default_app(self, fake_modal: FakeModal) -> None:
        await started()
        assert fake_modal.app_lookups[-1] == {'name': 'pydantic-ai-harness', 'create_if_missing': True}
        assert fake_modal.create_kwargs[-1]['env'] is None
        # Modal's maximum lifetime, and no idle termination: Modal's idle termination is permanent,
        # so it would end a conversation that pauses for a while.
        assert fake_modal.create_kwargs[-1]['timeout'] == 86_400
        assert fake_modal.create_kwargs[-1]['idle_timeout'] is None

    @pytest.mark.parametrize(
        ('name', 'match'),
        [
            ('InvalidError', 'Could not start Modal sandbox: failed'),
            ('NotFoundError', 'Could not start Modal sandbox: failed'),
            ('AlreadyExistsError', 'Could not start Modal sandbox: failed'),
            ('ExecutionError', 'Could not start Modal sandbox: failed'),
            ('AuthError', 'Modal rejected the credentials'),
        ],
    )
    async def test_a_refused_create_is_unavailable(self, fake_modal: FakeModal, name: str, match: str) -> None:
        # Modal refusing to create the sandbox (an unknown app or image, an invalid argument such
        # as a `sandbox_timeout` above its limit) cannot be fixed by the model or a retry, so it
        # ends the run instead of going back to the model as a `WorkspaceError`.
        fake_modal.create_error = fake_modal.exception(name)('failed')
        with pytest.raises(WorkspaceUnavailableError, match=match) as exc:
            await started()
        assert type(exc.value) is WorkspaceUnavailableError

    async def test_create_transport_failure_propagates(self, fake_modal: FakeModal) -> None:
        fake_modal.create_error = fake_modal.exception('ResourceExhaustedError')('rate limited')
        with pytest.raises(fake_modal.exception('ResourceExhaustedError')):
            await started()

    async def test_rejects_relative_working_dir(self, fake_modal: FakeModal) -> None:
        with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
            await started(working_dir='repo')

    async def test_preserves_parent_segments_in_working_dir(self, fake_modal: FakeModal) -> None:
        await started(working_dir='/linked/../target')

        assert fake_modal.create_kwargs[-1]['workdir'] == '/linked/../target'


class TestConnect:
    async def test_connects_to_a_running_sandbox(self, fake_modal: FakeModal) -> None:
        backend = await started(ref=WorkspaceRef(provider='modal', id='sb-keep'))
        assert fake_modal.attach_ids == ['sb-keep']
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-keep')

    async def test_connect_to_a_finished_sandbox_fails(self, fake_modal: FakeModal) -> None:
        # Modal hands back a handle for a sandbox it still knows about even after it has
        # terminated, so a ref must not resolve to a dead environment.
        fake_modal.attach_poll_result = 0
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await started(ref=WorkspaceRef(provider='modal', id='sb-gone'))
        assert not fake_modal.create_kwargs

    async def test_connect_to_an_unknown_id_fails(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_error = fake_modal.exception('NotFoundError')('not found')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-nope'"):
            await started(ref=WorkspaceRef(provider='modal', id='sb-nope'))
        assert not fake_modal.create_kwargs

    async def test_an_operation_on_a_terminated_sandbox_still_shutting_down_fails(self, fake_modal: FakeModal) -> None:
        owner = await started()
        assert owner.ref is not None
        await fake_modal.sandboxes[0].terminate.aio()
        backend = ModalSandboxBackend(ref=owner.ref)
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['true'])

    async def test_an_operation_on_a_gone_sandbox_does_not_create_a_replacement(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_error = fake_modal.exception('NotFoundError')('not found')
        backend = ModalSandboxBackend(ref=WorkspaceRef(provider='modal', id='sb-nope'))
        for _ in range(2):
            with pytest.raises(WorkspaceUnavailableError, match="'sb-nope'"):
                await backend.run(['true'])
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-nope')
        assert not fake_modal.create_kwargs
        assert not fake_modal.sandboxes


class TestFilesystem:
    async def test_write_then_read_round_trips(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        assert await backend.read_bytes('/tmp/a.txt') == b'body'

    async def test_stat_reports_size_for_files(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        entry = await backend.stat('/tmp/a.txt')
        assert (entry.name, entry.path, entry.is_dir, entry.size) == ('a.txt', '/tmp/a.txt', False, 4)

    async def test_stat_reports_no_size_for_directories(self, fake_modal: FakeModal) -> None:
        # A directory's reported size is a filesystem implementation detail, not a content
        # length, so the protocol carrier reports none.
        backend = await started()
        await backend.make_dir('/tmp/pkg')
        entry = await backend.stat('/tmp/pkg')
        assert (entry.is_dir, entry.size) == (True, None)

    async def test_list_dir_returns_absolute_paths(self, fake_modal: FakeModal) -> None:
        fake_modal.sandboxes.clear()
        backend = await started()
        fake_modal.sandboxes[0].listing = [FileInfo('a.py', False, size=7), FileInfo('pkg', True)]
        entries = await backend.list_dir('/srv')
        assert [(entry.name, entry.path, entry.is_dir, entry.size) for entry in entries] == [
            ('a.py', '/srv/a.py', False, 7),
            ('pkg', '/srv/pkg', True, None),
        ]

    async def test_symlinks_follow_relative_chained_dangling_and_looping_targets(
        self, fake_modal: FakeModal, tmp_path: Path
    ) -> None:
        fake_modal.host_root = tmp_path
        (tmp_path / 'data.txt').write_bytes(b'12345')
        (tmp_path / 'relative').symlink_to('data.txt')
        (tmp_path / 'chained').symlink_to('relative')
        (tmp_path / 'dangling').symlink_to('missing')
        (tmp_path / 'looping').symlink_to('looping')
        entries = await ModalSandboxBackend().list_dir(str(tmp_path))
        assert {entry.name: (entry.is_dir, entry.size) for entry in entries} == {
            'chained': (False, 5),
            'dangling': (False, None),
            'data.txt': (False, 5),
            'looping': (False, None),
            'relative': (False, 5),
        }

    async def test_remove_is_recursive(self, fake_modal: FakeModal) -> None:
        # One call covers both halves of the protocol's `remove`: on a file `recursive`
        # changes nothing, and on a directory it is what removes a non-empty one.
        backend = await started()
        await backend.make_dir('/tmp/pkg')
        await backend.remove('/tmp/pkg')
        assert fake_modal.sandboxes[0].removals == [('/tmp/pkg', True)]

    async def test_exists(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        assert await backend.exists('/tmp/a.txt') is True
        assert await backend.exists('/tmp/missing.txt') is False

    async def test_exists_is_false_through_a_non_directory(self, fake_modal: FakeModal) -> None:
        # Modal splits "there is nothing at that path" in two, and a non-leaf path component
        # that is a file is the other half.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemNotADirectoryError')('nope')
        assert await backend.exists('/tmp/a.txt/deeper') is False

    @pytest.mark.parametrize('operation', ['read_bytes', 'stat'])
    async def test_a_missing_path_raises_the_builtin_error(self, fake_modal: FakeModal, operation: str) -> None:
        # The protocol's contract: backends translate their SDK's own missing-file exception
        # into the builtin `FileNotFoundError` every consumer already handles.
        backend = await started()
        with pytest.raises(FileNotFoundError, match="'/tmp/missing.txt'"):
            await getattr(backend, operation)('/tmp/missing.txt')

    async def test_exists_still_reports_other_failures(self, fake_modal: FakeModal) -> None:
        # Only "there is nothing at that path" is an answer; anything else is a failure.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('Permission denied')
        with pytest.raises(WorkspaceError, match='Permission denied'):
            await backend.exists('/root/x')

    async def test_a_filesystem_error_is_recoverable_while_the_sandbox_runs(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('Permission denied')
        with pytest.raises(WorkspaceError, match='Permission denied') as exc:
            await backend.write_bytes('/root/x', b'data')
        assert isinstance(exc.value, WorkspaceError)
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    async def test_a_filesystem_error_on_a_dead_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        # Modal's filesystem wraps a dead sandbox as an ordinary-looking error, so the poll
        # is what keeps the model out of a retry loop against a corpse.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('request failed')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(WorkspaceUnavailableError):
            await backend.read_bytes('/x')

    async def test_a_filesystem_error_on_a_sandbox_still_shutting_down_is_terminal(self, fake_modal: FakeModal) -> None:
        # While a terminated sandbox shuts down it polls as running and its filesystem fails
        # generically; the exec probe is what names the state.
        backend = await started()
        await fake_modal.sandboxes[0].terminate.aio()
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await backend.read_bytes('/x')
        assert fake_modal.sandboxes[0].exec_calls[-1].argv == ['true']

    async def test_a_wrapped_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('request failed')
        fake_modal.sandboxes[0].poll_error = fake_modal.exception('AuthError')('unauthenticated')
        with pytest.raises(WorkspaceUnavailableError, match='Modal rejected the credentials'):
            await backend.list_dir('/x')


class TestErrorMapping:
    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('AuthError', WorkspaceUnavailableError),
            ('PermissionDeniedError', WorkspaceUnavailableError),
            ('NotFoundError', WorkspaceUnavailableError),
            ('SandboxTerminatedError', WorkspaceUnavailableError),
            ('SandboxTimeoutError', WorkspaceUnavailableError),
            ('SandboxFilesystemNotFoundError', FileNotFoundError),
            ('SandboxFilesystemIsADirectoryError', IsADirectoryError),
            ('SandboxFilesystemNotADirectoryError', NotADirectoryError),
            ('SandboxFilesystemPermissionError', PermissionError),
            ('SandboxFilesystemPathAlreadyExistsError', FileExistsError),
            ('InvalidError', WorkspaceError),
            ('ConflictError', WorkspaceError),
            ('AlreadyExistsError', WorkspaceError),
            ('ExecutionError', WorkspaceError),
            ('RequestSizeError', WorkspaceError),
            ('SandboxFilesystemError', WorkspaceError),
            ('FilesystemExecutionError', WorkspaceError),
            ('ConnectionError', None),
            ('ServiceError', None),
            ('ResourceExhaustedError', None),
            ('InternalError', None),
            ('Error', None),
        ],
    )
    async def test_modal_exception_maps_to_the_protocol_failure(
        self, fake_modal: FakeModal, name: str, expected: type[Exception] | None
    ) -> None:
        # `None` propagates the SDK exception unchanged: transport, rate-limit, and unknown
        # failures are what a durable engine retries. The fake replaces an exec-level failure as
        # Modal's filesystem layer does, so this also checks the original is classified, not
        # its stand-in. The sandbox still polls as running, so the ambiguous kinds stay
        # operation failures.
        backend = await started()
        error = fake_modal.exception(name)('boom')
        fake_modal.sandboxes[0].fs_error = error
        with pytest.raises(Exception) as exc_info:
            await backend.read_bytes('/p')
        if expected is None:
            assert exc_info.value is error
        else:
            assert type(exc_info.value) is expected
            assert exc_info.value.__cause__ is error
