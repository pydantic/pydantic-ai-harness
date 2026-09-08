"""Tests for `E2BSandboxBackend`, the E2B implementation of the sandbox protocol."""

from __future__ import annotations

import subprocess
import sys
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


async def started(**settings: Any) -> E2BSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting `sandbox` is that touch.
    """
    backend = E2BSandboxBackend(**settings)
    await backend.sandbox
    return backend


class TestConformance:
    async def test_sandbox_property_is_lazy_and_reuses_handle(self, fake_e2b: FakeE2B) -> None:
        backend = E2BSandboxBackend()
        pending = backend.sandbox
        assert not fake_e2b.sandboxes
        sandbox = await pending
        assert await backend.sandbox is sandbox

    async def test_backend_implements_run_and_filesystem_protocols(self, fake_e2b: FakeE2B) -> None:
        # Protocol inheritance also checks signatures statically.
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

    async def test_attaching_uses_the_sdk_default_lifetime(self, fake_e2b: FakeE2B) -> None:
        # The SDK owns its default connection lifetime.
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


class TestLifecycle:
    async def test_an_unused_backend_has_nothing_to_destroy_pause_or_stop(self, fake_e2b: FakeE2B) -> None:
        # Building one does no I/O, so lifecycle methods must not resolve it -- doing so would
        # create the very sandbox being released.
        backend = E2BSandboxBackend()
        await backend.destroy()
        await backend.pause()
        await backend.stop()

        assert fake_e2b.sandboxes == []
        assert fake_e2b.kill_ids == []

    async def test_destroy_kills_when_owned(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.destroy()
        assert fake_e2b.sandboxes[0].killed is True

    async def test_destroy_before_first_use_kills_by_id_without_connecting(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('sbx-keep')
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))

        await backend.destroy()

        assert fake_e2b.kill_ids == ['sbx-keep']
        assert fake_e2b.connect_calls == []
        assert fake_e2b.sandboxes[0].killed is True

    async def test_pause_before_first_use_uses_class_api_without_connecting(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('other')
        fake_e2b.new_sandbox('sbx-keep')
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))

        await backend.pause()

        assert fake_e2b.pause_ids == [('sbx-keep', True)]
        assert fake_e2b.connect_calls == []

    async def test_stop_before_first_use_drops_memory_without_connecting(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('sbx-keep')
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))

        await backend.stop()

        assert fake_e2b.pause_ids == [('sbx-keep', False)]
        assert fake_e2b.connect_calls == []

    async def test_pause_and_stop_use_attached_handle(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.pause()
        assert fake_e2b.pause_ids == []
        assert fake_e2b.pause_calls == [('sbx-1', True)]

        backend = await started()
        await backend.stop()
        assert fake_e2b.pause_calls[-1] == ('sbx-2', False)

    async def test_pause_failure_on_attached_handle_is_retryable(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.pause_error = RuntimeError('pause boom')
        with pytest.raises(E2BSandboxError, match='pause boom'):
            await backend.pause()
        fake_e2b.pause_error = None
        await backend.pause()

    async def test_stop_keeps_an_attached_sandbox_filesystem(self, fake_e2b: FakeE2B) -> None:
        backend = await started(ref=SandboxRef(sandbox_id='sbx-keep'))
        await backend.stop()
        assert fake_e2b.sandboxes[0].killed is False

    async def test_pause_clears_working_dir_and_next_operation_reconnects(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/srv\n', '', 0)
        backend = await started()
        assert await backend.working_dir() == '/srv'

        await backend.pause()

        assert await backend.working_dir() == '/srv'
        assert fake_e2b.connect_calls == [('sbx-1', None)]

    async def test_hanging_pause_is_bounded(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        backend = await started()
        fake_e2b.pause_hangs = True
        with anyio.fail_after(5):
            with pytest.raises(E2BSandboxError, match='Timed out'):
                await backend.pause()

    async def test_hanging_pause_by_id_is_bounded(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        fake_e2b.new_sandbox('sbx-keep')
        fake_e2b.pause_hangs = True
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))
        with anyio.fail_after(5):
            with pytest.raises(E2BSandboxError, match='Timed out'):
                await backend.pause()

    async def test_destroy_failure_is_visible_and_retryable(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.kill_error = RuntimeError('kill boom')
        with pytest.raises(E2BSandboxError, match='kill boom') as exc:
            await backend.destroy()
        assert exc.value.__cause__ is fake_e2b.kill_error
        fake_e2b.kill_error = None
        await backend.destroy()
        assert fake_e2b.sandboxes[0].killed is True

    async def test_already_gone_sandbox_is_not_an_error(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.kill_error = fake_e2b.sandbox_gone_type('already gone')
        await backend.destroy()

    async def test_already_gone_saved_ref_is_not_an_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('sbx-keep').killed = True
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))
        await backend.destroy()

    async def test_pause_failure_before_first_use_is_translated(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('sbx-keep')
        fake_e2b.pause_error = RuntimeError('pause boom')
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))
        with pytest.raises(E2BSandboxError, match='pause boom'):
            await backend.pause()

    async def test_pause_missing_sandbox_is_typed(self, fake_e2b: FakeE2B) -> None:
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-gone'))
        with pytest.raises(E2BSandboxUnavailableError, match="'sbx-gone'"):
            await backend.pause()

    async def test_auth_failure_during_destroy_is_typed(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.kill_error = fake_e2b.auth_type('bad key')

        with pytest.raises(E2BSandboxAuthError, match='E2B rejected the credentials'):
            await backend.destroy()

    async def test_hanging_destroy_is_bounded(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        # Teardown runs shielded, so a hanging kill would be uncancellable; its own deadline
        # is the only bound between a wedged control plane and a hung process.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        fake_e2b.new_sandbox('sbx-keep')
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))
        fake_e2b.kill_hangs = True
        with anyio.fail_after(5):
            with pytest.raises(E2BSandboxError, match='Timed out'):
                await backend.destroy()

    async def test_hanging_attached_destroy_is_bounded(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        backend = await started()
        fake_e2b.kill_hangs = True
        with anyio.fail_after(5):
            with pytest.raises(E2BSandboxError, match='Timed out'):
                await backend.destroy()

    async def test_destroy_cleanup_survives_cancellation(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('sbx-keep')
        fake_e2b.kill_gate = anyio.Event()
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))
        cancel_scope: list[anyio.CancelScope] = []
        finished = anyio.Event()
        outcomes: list[BaseException] = []

        async def destroy() -> None:
            with anyio.CancelScope() as scope:
                cancel_scope.append(scope)
                try:
                    await backend.destroy()
                except BaseException as error:
                    outcomes.append(error)
                finally:
                    finished.set()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(destroy)
            with anyio.fail_after(5):
                while not fake_e2b.kill_started:
                    await anyio.wait_all_tasks_blocked()
            cancel_scope[0].cancel()
            fake_e2b.kill_gate.set()
            with anyio.fail_after(5):
                await finished.wait()

        assert outcomes == []
        assert fake_e2b.sandboxes[0].killed is True

    async def test_destroy_cancellation_wins_over_cleanup_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.new_sandbox('sbx-keep')
        fake_e2b.kill_gate = anyio.Event()
        fake_e2b.kill_error = RuntimeError('kill boom')
        backend = E2BSandboxBackend(ref=SandboxRef(sandbox_id='sbx-keep'))
        cancel_scope: list[anyio.CancelScope] = []
        finished = anyio.Event()
        outcomes: list[BaseException] = []

        async def destroy() -> None:
            with anyio.CancelScope() as scope:
                cancel_scope.append(scope)
                try:
                    await backend.destroy()
                except BaseException as error:
                    outcomes.append(error)
                finally:
                    finished.set()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(destroy)
            with anyio.fail_after(5):
                while not fake_e2b.kill_started:
                    await anyio.wait_all_tasks_blocked()
            cancel_scope[0].cancel()
            fake_e2b.kill_gate.set()
            with anyio.fail_after(5):
                await finished.wait()

        assert len(outcomes) == 1
        assert isinstance(outcomes[0], anyio.get_cancelled_exc_class())
        assert fake_e2b.sandboxes[0].killed is False


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


async def test_command_timeout_bounds_initial_provisioning(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_hangs = True
    backend = E2BSandboxBackend()
    with anyio.fail_after(1):
        with pytest.raises(SandboxTimeoutError) as exc:
            await backend.run(['echo', 'ready'], timeout=0.01)
    assert exc.value.timeout == 0.01
    assert exc.value.stdout == ''


async def test_direct_identity_is_recorded_and_reused(fake_e2b: FakeE2B) -> None:
    first = await started(identity={'workspace': 'one'})
    second = await started(identity={'workspace': 'one'})
    assert first.ref == second.ref
    assert len(fake_e2b.create_calls) == 1


async def test_filesystem_first_use_preserves_auth_error(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_error = fake_e2b.auth_type('denied')
    with pytest.raises(E2BSandboxAuthError):
        await E2BSandboxBackend().read_bytes('/file')
