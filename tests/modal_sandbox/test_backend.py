"""Tests for `ModalSandboxBackend`, the Modal implementation of the sandbox protocol."""

from __future__ import annotations

import asyncio
import builtins
import functools
import sys
import time
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

from pydantic_ai_harness.modal_sandbox import (
    ModalSandboxAuthError,
    ModalSandboxBackend,
    ModalSandboxError,
    ModalSandboxUnavailableError,
)

from ..sandbox_conformance import (
    check_command_validation,
    check_missing_file,
    check_timeout,
)
from .fake_modal import FakeModal, FileInfo, _AioCallable


class _HangingCall(_AioCallable):
    """A teardown RPC that never returns, to prove the teardown deadline bounds it."""

    def __init__(self) -> None:
        super().__init__(lambda: None)

    async def aio(self, *args: object, **kwargs: object) -> None:
        await anyio.sleep_forever()


pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def started(**settings: Any) -> ModalSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting the property is that touch.
    """
    backend = ModalSandboxBackend(**settings)
    await backend.sandbox
    return backend


# The two ways to reach an existing sandbox, for tests that behave identically over both.
_Attach = Callable[[], Awaitable[ModalSandboxBackend]]
_CONNECT_BY_ID: _Attach = functools.partial(started, ref=SandboxRef(sandbox_id='sb-keep'))
_CONNECT_BY_NAME: _Attach = functools.partial(started, app_name='app', name='stable')


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class TestConformance:
    async def test_backend_implements_run_and_filesystem_protocols(self, fake_modal: FakeModal) -> None:
        # `isinstance` is shallow (member presence only); the signature half is pinned
        # statically by the `if TYPE_CHECKING` block in `_backend.py`.
        backend = await started()
        assert isinstance(backend, SandboxBackend)
        assert isinstance(backend, SupportsFilesystem)

    async def test_identity_is_modal_object_id(self, fake_modal: FakeModal) -> None:
        backend = await started()
        assert backend.ref == SandboxRef(sandbox_id='sb-owned')
        assert await backend.sandbox is fake_modal.sandboxes[0]

    async def test_shared_command_validation(self, fake_modal: FakeModal) -> None:
        await check_command_validation(started)

    async def test_shared_missing_file(self, fake_modal: FakeModal) -> None:
        await check_missing_file(started)

    async def test_shared_timeout(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('partial', '', -1)
        await check_timeout(started)

    async def test_shared_run_and_nonzero_result(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('', '', 2)
        backend = await started()
        result = await backend.run(['false'])
        assert result.exit_code != 0


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
        assert backend.ref == SandboxRef(sandbox_id='sb-owned')
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
        with pytest.raises(ModalSandboxError, match='Could not start Modal sandbox: capacity'):
            await started()

    async def test_auth_error_is_terminal(self, fake_modal: FakeModal) -> None:
        fake_modal.create_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(ModalSandboxAuthError, match='Modal rejected the credentials'):
            await started()

    async def test_hanging_create_does_not_hang_the_caller(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Creation is shielded, so its own deadline is the only bound between a wedged
        # control plane and a hung process.
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._CREATE_TIMEOUT', 0.05)
        monkeypatch.setattr(fake_modal.module.App, 'lookup', _HangingCall())
        with anyio.fail_after(5):
            with pytest.raises(ModalSandboxError, match='did not complete within'):
                await started()

    async def test_rejects_relative_workdir(self, fake_modal: FakeModal) -> None:
        with pytest.raises(ValueError, match='workdir must be an absolute sandbox path'):
            await started(workdir='repo')

    async def test_normalizes_parent_segments_in_workdir(self, fake_modal: FakeModal) -> None:
        await started(workdir='/linked/../target')

        assert fake_modal.create_kwargs[-1]['workdir'] == '/target'

    async def test_a_named_create_that_loses_the_race_attaches_instead(self, fake_modal: FakeModal) -> None:
        # Two workers can race to create the same named sandbox -- a durable retry is the
        # ordinary case. The loser must attach to the winner's sandbox, not fail and not
        # provision a second one.
        first = await started(name='stable')

        second = await started(name='stable')

        assert await second.sandbox is await first.sandbox
        assert len(fake_modal.sandboxes) == 1


class TestConnect:
    async def test_connects_to_a_running_sandbox(self, fake_modal: FakeModal) -> None:
        backend = await started(ref=SandboxRef(sandbox_id='sb-keep'))
        assert fake_modal.attach_ids == ['sb-keep']
        assert backend.ref == SandboxRef(sandbox_id='sb-keep')

    async def test_connect_to_a_finished_sandbox_fails(self, fake_modal: FakeModal) -> None:
        # Modal hands back a handle for a sandbox it still knows about even after it has
        # terminated, so a ref must not resolve to a dead environment.
        fake_modal.attach_poll_result = 0
        with pytest.raises(ModalSandboxUnavailableError, match='no longer running'):
            await started(ref=SandboxRef(sandbox_id='sb-gone'))

    async def test_connect_to_an_unknown_id_fails(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_error = fake_modal.unavailable_type('not found')
        with pytest.raises(ModalSandboxUnavailableError, match="'sb-nope'"):
            await started(ref=SandboxRef(sandbox_id='sb-nope'))

    @pytest.mark.parametrize('attach', [_CONNECT_BY_ID, _CONNECT_BY_NAME], ids=['by_id', 'by_name'])
    async def test_attach_auth_error_is_terminal(self, fake_modal: FakeModal, attach: _Attach) -> None:
        fake_modal.attach_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(ModalSandboxAuthError, match='Modal rejected the credentials'):
            await attach()

    @pytest.mark.parametrize('attach', [_CONNECT_BY_ID, _CONNECT_BY_NAME], ids=['by_id', 'by_name'])
    async def test_attach_sdk_error_is_translated(self, fake_modal: FakeModal, attach: _Attach) -> None:
        fake_modal.attach_error = fake_modal.error_type('transport failed')
        with pytest.raises(ModalSandboxError, match='transport failed'):
            await attach()


class TestConnectName:
    async def test_connects_to_running_named_sandbox(self, fake_modal: FakeModal) -> None:
        created = await started(name='stable')

        connected = await started(app_name='pydantic-ai-harness', name='stable')

        assert await connected.sandbox is await created.sandbox

    async def test_finished_named_sandbox_is_unavailable(self, fake_modal: FakeModal) -> None:
        await started(name='finished')
        fake_modal.sandboxes[0].poll_result = 0

        with pytest.raises(ModalSandboxUnavailableError, match='not running'):
            await started(app_name='pydantic-ai-harness', name='finished')


@pytest.mark.parametrize(
    'entry_point',
    [started, _CONNECT_BY_ID, _CONNECT_BY_NAME],
    ids=['create', 'attach_by_id', 'attach_by_name'],
)
async def test_missing_modal_package_is_named(monkeypatch: pytest.MonkeyPatch, entry_point: _Attach) -> None:
    # Every path into the sandbox guards its own lazy `import modal`.
    real_import = builtins.__import__

    def no_modal(name: str, *args: object, **kwargs: object) -> object:
        if name == 'modal':
            raise ImportError('No module named modal')
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, 'modal', raising=False)
    monkeypatch.setattr(builtins, '__import__', no_modal)
    with pytest.raises(ModalSandboxError, match="The 'modal' package is required"):
        await entry_point()


class TestClose:
    async def test_an_unused_backend_has_nothing_to_close(self, fake_modal: FakeModal) -> None:
        # Building one does no I/O, so closing it must not reach Modal either -- resolving here
        # would create the very sandbox being released.
        await ModalSandboxBackend().close(terminate=True)

        assert fake_modal.sandboxes == []
        assert fake_modal.create_kwargs == []

    async def test_attaching_twice_to_one_id_reuses_the_same_sandbox(self, fake_modal: FakeModal) -> None:
        first = await started(ref=SandboxRef(sandbox_id='sb-keep'))
        second = await started(ref=SandboxRef(sandbox_id='sb-keep'))

        assert await second.sandbox is await first.sandbox
        assert fake_modal.attach_ids == ['sb-keep', 'sb-keep']

    async def test_terminates_and_detaches_when_owned(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.close(terminate=True)
        assert fake_modal.sandboxes[0].terminated is True
        assert fake_modal.sandboxes[0].detached is True

    async def test_detaches_without_terminating(self, fake_modal: FakeModal) -> None:
        backend = await started(ref=SandboxRef(sandbox_id='sb-keep'))
        await backend.close(terminate=False)
        assert fake_modal.sandboxes[0].terminated is False
        assert fake_modal.sandboxes[0].detached is True

    async def test_terminate_failure_still_detaches(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].terminate_error = RuntimeError('terminate boom')
        with pytest.raises(ModalSandboxError, match='terminate boom'):
            await backend.close(terminate=True)
        assert fake_modal.sandboxes[0].detached is True

    async def test_already_gone_sandbox_is_not_an_error(self, fake_modal: FakeModal) -> None:
        # An owned run that outlived its `sandbox_timeout` self-terminates; the teardown
        # terminate then hits "already gone", which is success, not a failure to raise.
        backend = await started()
        fake_modal.sandboxes[0].terminate_error = fake_modal.sandbox_terminated_type('already terminated')
        await backend.close(terminate=True)
        assert fake_modal.sandboxes[0].detached is True

    async def test_detach_failure_is_visible(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].detach_error = RuntimeError('detach boom')
        with pytest.raises(ModalSandboxError, match='detach boom'):
            await backend.close(terminate=True)
        assert fake_modal.sandboxes[0].terminated is True

    async def test_first_failure_wins_when_both_calls_fail(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].terminate_error = RuntimeError('terminate boom')
        fake_modal.sandboxes[0].detach_error = RuntimeError('detach boom')
        with pytest.raises(ModalSandboxError, match='terminate boom'):
            await backend.close(terminate=True)

    async def test_auth_failure_during_close_is_typed(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].terminate_error = fake_modal.auth_type('unauthenticated')

        with pytest.raises(ModalSandboxAuthError, match='Modal rejected the credentials'):
            await backend.close(terminate=True)

        assert fake_modal.sandboxes[0].detached is True

    async def test_hanging_terminate_is_bounded_and_detach_still_runs(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        backend = await started()
        fake_modal.sandboxes[0].terminate = _HangingCall()
        with anyio.fail_after(5):
            with pytest.raises(ModalSandboxError, match='Timed out'):
                await backend.close(terminate=True)
        assert fake_modal.sandboxes[0].detached is True

    async def test_hanging_detach_is_bounded(self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch) -> None:
        # Teardown runs shielded, so a hanging detach would be uncancellable; its own
        # deadline is the only bound between a wedged control plane and a hung process.
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        backend = await started()
        fake_modal.sandboxes[0].detach = _HangingCall()
        with anyio.fail_after(5):
            with pytest.raises(ModalSandboxError, match='Timed out'):
                await backend.close(terminate=True)
        assert fake_modal.sandboxes[0].terminated is True


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

        with pytest.raises(ValueError, match='cwd must be an absolute sandbox path'):
            await backend.run(['pwd'], cwd='repo')

    async def test_normalizes_parent_segments_in_cwd(self, fake_modal: FakeModal) -> None:
        backend = await started()

        await backend.run(['pwd'], cwd='/linked/../target')

        assert fake_modal.sandboxes[0].exec_calls[-1].workdir == '/target'

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
        with pytest.raises(SandboxTimeoutError) as exc:
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
        with pytest.raises(SandboxTimeoutError):
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
        with pytest.raises(ModalSandboxError, match='Command could not run in the sandbox: transient blip') as exc:
            await backend.run(['x'])
        assert isinstance(exc.value, SandboxError)
        assert not isinstance(exc.value, SandboxUnavailableError)

    @pytest.mark.parametrize(
        ('exc_property', 'match'),
        [
            ('unavailable_type', 'no longer running'),
            ('sandbox_terminated_type', 'no longer running'),
            ('sandbox_timeout_type', 'no longer running'),
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
        with pytest.raises(SandboxUnavailableError, match=match):
            await backend.run(['x'])

    async def test_dead_sandbox_conflict_is_terminal(self, fake_modal: FakeModal) -> None:
        # A first exec on a dead sandbox surfaces as Modal's ambiguous ConflictError; the
        # poll disambiguates it from a transient abort.
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('Sandbox already finished')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(ModalSandboxUnavailableError, match='sandbox_timeout of 300s'):
            await backend.run(['x'])

    async def test_a_terminated_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        # The same ambiguous ConflictError against a sandbox this process terminated: the
        # poll finds it exited, so the failure is terminal rather than a transient abort.
        backend = await started()
        await backend.close(terminate=True)
        fake_modal.exec_error = fake_modal.conflict_type('Sandbox already finished')
        with pytest.raises(ModalSandboxUnavailableError):
            await backend.run(['x'])

    async def test_transient_conflict_stays_recoverable(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        with pytest.raises(ModalSandboxError, match='aborted') as exc:
            await backend.run(['x'])
        assert not isinstance(exc.value, SandboxUnavailableError)

    async def test_failing_poll_preserves_the_original_error(self, fake_modal: FakeModal) -> None:
        # The classifying poll can itself fail with a raw transport error; that must not
        # abort the run in place of the error we were classifying.
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        fake_modal.sandboxes[0].poll_error = ValueError('transport gone')
        with pytest.raises(ModalSandboxError, match='aborted'):
            await backend.run(['x'])

    async def test_poll_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(ModalSandboxAuthError, match='Modal rejected the credentials'):
            await backend.run(['x'])

    async def test_poll_reporting_a_missing_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.conflict_type('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.unavailable_type('gone')
        with pytest.raises(ModalSandboxUnavailableError):
            await backend.run(['x'])

    async def test_attached_sandbox_names_itself_when_gone(self, fake_modal: FakeModal) -> None:
        # A connected backend does not know the lifetime it was created with, so it points
        # at the sandbox instead of quoting a `sandbox_timeout` it never set.
        backend = await started(ref=SandboxRef(sandbox_id='sb-keep'))
        fake_modal.exec_error = fake_modal.sandbox_terminated_type('gone')
        with pytest.raises(ModalSandboxUnavailableError, match="'sb-keep' is no longer running"):
            await backend.run(['x'])

    async def test_run_wait_failure_is_a_sandbox_error(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = fake_modal.error_type('wait failed')
        backend = await started()
        with pytest.raises(ModalSandboxError, match='wait failed'):
            await backend.run(['x'])

    async def test_raw_run_wait_failure_is_a_sandbox_error(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = RuntimeError('raw wait failed')
        backend = await started()
        with pytest.raises(ModalSandboxError, match='raw wait failed'):
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
    async def test_created_workdir_needs_no_probe(self, fake_modal: FakeModal) -> None:
        backend = await started(workdir='/work')
        assert await backend.working_dir() == '/work'
        assert fake_modal.sandboxes[0].exec_calls == []

    async def test_probed_once_and_cached(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        backend = await started()
        assert await backend.working_dir() == '/srv'
        assert await backend.working_dir() == '/srv'
        assert [call.argv for call in fake_modal.sandboxes[0].exec_calls] == [['pwd']]

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
        with pytest.raises(ModalSandboxError, match='Could not determine the working directory'):
            await backend.working_dir()

    async def test_the_facade_resolves_relative_paths_against_it(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        sandbox = Sandbox(await started())
        assert await sandbox.resolve('src/main.py') == '/srv/src/main.py'


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
        with pytest.raises(ModalSandboxError, match='Permission denied'):
            await backend.exists('/root/x')

    async def test_a_filesystem_error_is_recoverable_while_the_sandbox_runs(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('Permission denied')
        with pytest.raises(ModalSandboxError, match='Permission denied') as exc:
            await backend.write_bytes('/root/x', b'data')
        assert isinstance(exc.value, SandboxError)
        assert not isinstance(exc.value, SandboxUnavailableError)

    async def test_a_filesystem_error_on_a_dead_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        # Modal's filesystem wraps a dead sandbox as an ordinary-looking error, so the poll
        # is what keeps the model out of a retry loop against a corpse.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('request failed')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(ModalSandboxUnavailableError):
            await backend.read_bytes('/x')

    async def test_a_wrapped_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.filesystem_error_type('request failed')
        fake_modal.sandboxes[0].poll_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(ModalSandboxAuthError, match='Modal rejected the credentials'):
            await backend.list_dir('/x')

    async def test_a_direct_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.auth_type('unauthenticated')
        with pytest.raises(ModalSandboxAuthError, match='Modal rejected the credentials'):
            await backend.make_dir('/x')
