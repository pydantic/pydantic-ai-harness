"""Substantive tests for the Modal workspace backend."""

from __future__ import annotations

import asyncio
import time
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

from pydantic_ai_harness.modal_workspace import ModalWorkspaceBackend

from .fake_modal import FakeModal, FileInfo

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def started(**settings: Any) -> ModalWorkspaceBackend:
    backend = ModalWorkspaceBackend(**settings)
    await backend.workspace
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

    async def test_rejects_relative_cwd(self, fake_modal: FakeModal) -> None:
        backend = await started()

        with pytest.raises(ValueError, match='cwd must be an absolute workspace path'):
            await backend.run(['pwd'], cwd='repo')

    async def test_preserves_parent_segments_in_cwd(self, fake_modal: FakeModal) -> None:
        backend = await started()

        await backend.run(['pwd'], cwd='/linked/../target')

        assert fake_modal.sandboxes[0].exec_calls[-1].workdir == '/linked/../target'

    @pytest.mark.parametrize(
        ('command', 'shell', 'message'),
        [
            (['ls'], True, 'an argv sequence cannot be combined with shell=True'),
            ('ls -la', False, 'a string command requires shell=True'),
            ([], False, 'the argv sequence is empty'),
        ],
    )
    async def test_command_shape_mismatches_are_rejected(
        self, fake_modal: FakeModal, command: str | list[str], shell: bool, message: str
    ) -> None:
        backend = await started()
        with pytest.raises(TypeError, match=message):
            await backend.run(command, shell=shell)

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
        assert (exc.value.stdout, exc.value.stderr, exc.value.timeout) == ('partial', 'oops', 5)

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

    async def test_invalid_utf8_output_uses_replacement_characters(self, fake_modal: FakeModal) -> None:
        # Modal's text mode decodes strictly; reading bytes and decoding with replacement
        # keeps a command printing binary from aborting the run.
        fake_modal.responder = lambda argv, timeout: (b'\xff\xfe', b'', 0)
        backend = await started()
        assert (await backend.run(['cat', 'binary'])).stdout == '��'

    async def test_exec_failure_is_a_recoverable_sandbox_error(self, fake_modal: FakeModal) -> None:
        fake_modal.exec_error = fake_modal.error_type('transient blip')
        backend = await started()
        with pytest.raises(WorkspaceError, match='Command could not run in the workspace: transient blip') as exc:
            await backend.run(['x'])
        assert isinstance(exc.value, WorkspaceError)
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    @pytest.mark.parametrize(
        ('exc_property', 'match'),
        [
            ('unavailable_type', 'no longer running'),
            ('workspace_terminated_type', 'no longer running'),
            ('workspace_timeout_type', 'no longer running'),
            ('auth_type', 'Modal rejected the credentials'),
        ],
    )
    async def test_terminal_exec_failures(self, fake_modal: FakeModal, exc_property: str, match: str) -> None:
        # All three Modal spellings of "the sandbox is gone" classify the same way; sandbox
        # expiry is the one an owned run outliving its lifetime actually produces, and
        # rejected credentials are the non-sandbox case.
        exc_type: type[Exception] = getattr(fake_modal, exc_property)
        fake_modal.exec_error = exc_type('terminal failure')
        backend = await started()
        with pytest.raises(WorkspaceUnavailableError, match=match):
            await backend.run(['x'])

    async def test_dead_workspace_conflict_is_terminal(self, fake_modal: FakeModal) -> None:
        # A first exec on a dead sandbox surfaces as Modal's ambiguous ConflictError; the
        # poll disambiguates it from a transient abort.
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('Sandbox already finished')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(WorkspaceUnavailableError, match='sandbox_timeout of 300s'):
            await backend.run(['x'])

    async def test_transient_conflict_stays_recoverable(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        with pytest.raises(WorkspaceError, match='aborted') as exc:
            await backend.run(['x'])
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    async def test_failing_poll_preserves_the_original_error(self, fake_modal: FakeModal) -> None:
        # The classifying poll can itself fail with a raw transport error; that must not
        # abort the run in place of the error we were classifying.
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        fake_modal.sandboxes[0].poll_error = ValueError('transport gone')
        with pytest.raises(WorkspaceError, match='aborted'):
            await backend.run(['x'])

    async def test_poll_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(WorkspaceUnavailableError, match='Modal rejected the credentials'):
            await backend.run(['x'])

    async def test_poll_reporting_a_missing_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.unavailable_type('gone')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.run(['x'])

    async def test_attached_sandbox_names_itself_when_gone(self, fake_modal: FakeModal) -> None:
        # A connected backend does not know the lifetime it was created with, so it points
        # at the sandbox instead of quoting a `sandbox_timeout` it never set.
        backend = await started(ref=WorkspaceRef(provider='modal', id='sb-keep'))
        fake_modal.exec_error = fake_modal.workspace_terminated_type('gone')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-keep' is no longer running"):
            await backend.run(['x'])

    async def test_run_wait_failure_is_a_workspace_error(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = fake_modal.error_type('wait failed')
        backend = await started()
        with pytest.raises(WorkspaceError, match='wait failed'):
            await backend.run(['x'])

    async def test_raw_run_wait_failure_is_a_workspace_error(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = RuntimeError('raw wait failed')
        backend = await started()
        with pytest.raises(WorkspaceError, match='raw wait failed'):
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
    async def test_configured_workdir_is_resolved_before_first_operation(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/canonical/work\n', '', 0)
        backend = ModalWorkspaceBackend(workdir='/alias')
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
    async def test_creates_from_config(self, fake_modal: FakeModal) -> None:
        backend = await started(
            image='ubuntu:22.04',
            app_name='my-app',
            create_app_if_missing=False,
            sandbox_timeout=120,
            workdir='/work',
            env={'FOO': 'bar'},
        )
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-owned')
        assert fake_modal.app_lookups[-1] == {'name': 'my-app', 'create_if_missing': False}
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.create_kwargs[-1]['timeout'] == 120
        assert fake_modal.create_kwargs[-1]['workdir'] == '/work'
        assert fake_modal.create_kwargs[-1]['env'] == {'FOO': 'bar'}

    async def test_default_app_and_image(self, fake_modal: FakeModal) -> None:
        await started()
        assert fake_modal.app_lookups[-1] == {'name': 'pydantic-ai-harness', 'create_if_missing': True}
        assert fake_modal.image_tags[-1] == 'python:3.12-slim'
        assert fake_modal.create_kwargs[-1]['env'] is None

    async def test_modal_error_becomes_a_start_failure(self, fake_modal: FakeModal) -> None:
        fake_modal.create_error = fake_modal.error_type('capacity')
        with pytest.raises(WorkspaceError, match='Could not start Modal sandbox: capacity'):
            await started()

    async def test_auth_error_is_terminal(self, fake_modal: FakeModal) -> None:
        fake_modal.create_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(WorkspaceUnavailableError, match='Modal rejected the credentials'):
            await started()

    async def test_rejects_relative_workdir(self, fake_modal: FakeModal) -> None:
        with pytest.raises(ValueError, match='workdir must be an absolute workspace path'):
            await started(workdir='repo')

    async def test_preserves_parent_segments_in_workdir(self, fake_modal: FakeModal) -> None:
        await started(workdir='/linked/../target')

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

    async def test_connect_to_an_unknown_id_fails(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_error = fake_modal.unavailable_type('not found')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-nope'"):
            await started(ref=WorkspaceRef(provider='modal', id='sb-nope'))


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
        fake_modal.sandboxes[0].fs_error = fake_modal.module.exception.SandboxFilesystemNotADirectoryError('nope')
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
        fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied')
        with pytest.raises(WorkspaceError, match='Permission denied'):
            await backend.exists('/root/x')

    async def test_a_filesystem_error_is_recoverable_while_the_sandbox_runs(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied')
        with pytest.raises(WorkspaceError, match='Permission denied') as exc:
            await backend.write_bytes('/root/x', b'data')
        assert isinstance(exc.value, WorkspaceError)
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    async def test_a_filesystem_error_on_a_dead_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        # Modal's filesystem wraps a dead sandbox as an ordinary-looking error, so the poll
        # is what keeps the model out of a retry loop against a corpse.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('request failed')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(WorkspaceUnavailableError):
            await backend.read_bytes('/x')

    async def test_a_wrapped_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('request failed')
        fake_modal.sandboxes[0].poll_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(WorkspaceUnavailableError, match='Modal rejected the credentials'):
            await backend.list_dir('/x')
