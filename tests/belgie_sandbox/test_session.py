from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from pydantic_ai_harness.belgie_sandbox import (
    BelgieSandboxError,
    BelgieSandboxExecutionError,
    BelgieSandboxSession,
    BelgieSandboxTimeoutError,
    BelgieSandboxUnavailableError,
)

from .fake_belgie import BelgieJavaScriptError, FakeBelgie

pytestmark = pytest.mark.anyio


class TestBelgieSandboxSession:
    async def test_default_session_is_restricted_and_temporary(self, fake_belgie: FakeBelgie) -> None:
        session = BelgieSandboxSession()

        async with session:
            workspace = session.workspace
            assert workspace is not None
            assert workspace.exists()
            assert session.is_open
            assert await session.run_script('export default () => ({ ok: true })') == {'ok': True}

            environment = fake_belgie.environments[0]
            assert environment.dependencies is None
            assert environment.options is not None
            assert environment.options.allow_remote is False
            assert environment.options.no_npm is True
            assert environment.install_calls == 0

            options = fake_belgie.runtimes[0].options
            assert options is not None
            assert options.max_old_generation_size_mb == 128
            assert options.permissions is not None
            assert options.permissions.kwargs == {'allow_read': [str(workspace)]}

        assert not session.is_open
        assert session.workspace is None
        assert not workspace.exists()
        assert fake_belgie.environments[0].exited
        assert fake_belgie.runtimes[0].exited

    async def test_package_imports_and_network_are_explicit(self, fake_belgie: FakeBelgie) -> None:
        session = BelgieSandboxSession(
            allow_package_imports=True,
            allow_network=True,
            max_old_generation_size_mb=None,
        )

        async with session:
            workspace = session.workspace
            assert workspace is not None
            environment = fake_belgie.environments[0]
            assert environment.dependencies is None
            assert environment.options is not None
            assert environment.options.allow_remote is True
            assert environment.options.no_npm is False
            assert environment.install_calls == 0

            options = fake_belgie.runtimes[0].options
            assert options is not None
            assert options.max_old_generation_size_mb is None
            assert options.permissions is not None
            assert options.permissions.kwargs == {
                'allow_read': [str(workspace)],
                'allow_net': [],
            }

    async def test_package_imports_do_not_enable_runtime_network(self, fake_belgie: FakeBelgie) -> None:
        async with BelgieSandboxSession(allow_package_imports=True):
            options = fake_belgie.runtimes[0].options
            assert options is not None
            assert options.permissions is not None
            assert 'allow_net' not in options.permissions.kwargs

    async def test_rendering_uses_side_channel_without_script_ffi(self, fake_belgie: FakeBelgie) -> None:
        from pydantic_ai_harness.belgie_sandbox._session import (
            DEFAULT_RENDER_SPECIFIER,
            DEFAULT_VITE_SYS_PERMISSIONS,
            RENDER_REQUEST_KEY,
        )

        fake_belgie.result = {RENDER_REQUEST_KEY: 1}
        source = 'export default () => render({ widget: null, plugins: [] })'
        async with BelgieSandboxSession(enable_rendering=True) as session:
            workspace = session.workspace
            assert workspace is not None
            assert await session.run_script(source) == '<html>rendered</html>'

            environment = fake_belgie.environments[0]
            assert environment.dependencies == {
                '@belgie/render': DEFAULT_RENDER_SPECIFIER,
                'react': 'npm:react@19.2.8',
                'react-dom': 'npm:react-dom@19.2.8',
            }
            assert environment.options is not None
            assert environment.options.allow_remote is True
            assert environment.options.no_npm is False
            assert environment.install_calls == 1

            assert len(fake_belgie.runtimes) == 2
            script_permissions = fake_belgie.runtimes[0].options
            assert script_permissions is not None
            assert script_permissions.permissions is not None
            assert script_permissions.permissions.kwargs == {'allow_read': [str(workspace)]}

            render_permissions = fake_belgie.runtimes[1].options
            assert render_permissions is not None
            assert render_permissions.permissions is not None
            assert render_permissions.permissions.kwargs == {
                'allow_read': [str(workspace)],
                'allow_net': ['localhost'],
                'allow_ffi': [str(workspace / 'node_modules')],
                'allow_sys': list(DEFAULT_VITE_SYS_PERMISSIONS),
                'allow_write': [str(workspace)],
            }
            assert fake_belgie.render_calls == [(source, (workspace / '__deno_python_inline__.tsx').resolve().as_uri())]

        assert all(runtime.exited for runtime in fake_belgie.runtimes)

    async def test_render_request_without_side_channel_is_an_execution_error(self, fake_belgie: FakeBelgie) -> None:
        from pydantic_ai_harness.belgie_sandbox._session import RENDER_REQUEST_KEY

        fake_belgie.result = {RENDER_REQUEST_KEY: 1}
        runtime = fake_belgie.module.Runtime()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        async with BelgieSandboxSession(runtime=runtime) as session:  # pyright: ignore[reportArgumentType]
            with pytest.raises(BelgieSandboxExecutionError, match='no renderer side-channel'):
                await session.run_script('export default () => render()')

    async def test_custom_runtime_is_entered_and_has_no_workspace(self, fake_belgie: FakeBelgie) -> None:
        runtime = fake_belgie.module.Runtime()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        session = BelgieSandboxSession(runtime=runtime)  # pyright: ignore[reportArgumentType]

        async with session:
            assert session.workspace is None
            assert await session.run_script('export default () => 1') == {'ok': True}

        assert runtime.entered
        assert runtime.exited
        assert fake_belgie.environments == []

    async def test_rejects_double_enter(self, fake_belgie: FakeBelgie) -> None:
        session = BelgieSandboxSession()
        async with session:
            with pytest.raises(BelgieSandboxError, match='already open'):
                await session.__aenter__()

    async def test_rejects_concurrent_enter(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.enter_started = asyncio.Event()
        fake_belgie.enter_gate = asyncio.Event()
        runtime = fake_belgie.module.Runtime()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        session = BelgieSandboxSession(runtime=runtime)  # pyright: ignore[reportArgumentType]

        first = asyncio.create_task(session.__aenter__())
        await fake_belgie.enter_started.wait()
        with pytest.raises(BelgieSandboxError, match='already open'):
            await session.__aenter__()
        fake_belgie.enter_gate.set()
        await first
        await session.close()
        assert runtime.exited

    async def test_requires_asyncio(self, fake_belgie: FakeBelgie, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_loop() -> None:
            raise RuntimeError('no loop')

        monkeypatch.setattr(asyncio, 'get_running_loop', no_loop)
        with pytest.raises(BelgieSandboxError, match='requires an asyncio'):
            await BelgieSandboxSession().__aenter__()

    async def test_missing_dependency_clears_entering_guard(
        self, fake_belgie: FakeBelgie, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, 'belgie', None)
        session = BelgieSandboxSession()
        with pytest.raises(BelgieSandboxUnavailableError, match='Python 3.12-3.14'):
            await session.__aenter__()
        with pytest.raises(BelgieSandboxUnavailableError, match='Python 3.12-3.14'):
            await session.__aenter__()

    async def test_start_failure_cleans_up(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.start_error = RuntimeError('worker failed')
        session = BelgieSandboxSession()

        with pytest.raises(BelgieSandboxUnavailableError, match='worker failed'):
            await session.__aenter__()

        assert not session.is_open
        assert session.workspace is None
        assert fake_belgie.environments[0].exited

    async def test_start_failure_retains_state_when_cleanup_fails(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.start_error = RuntimeError('worker failed')
        fake_belgie.environment_exit_error = RuntimeError('cleanup failed')
        session = BelgieSandboxSession()

        startup_error = fake_belgie.start_error
        with pytest.raises(
            BelgieSandboxUnavailableError, match='worker failed.*Cleanup also failed.*cleanup failed'
        ) as exc_info:
            await session.__aenter__()

        assert exc_info.value.__cause__ is startup_error
        # Environment exit failed, but best-effort cleanup still removed the temp workspace.
        assert session.workspace is None
        assert fake_belgie.environments[0].exit_calls == 1
        assert not fake_belgie.environments[0].exited
        with pytest.raises(BelgieSandboxError, match='pending cleanup'):
            await session.__aenter__()
        fake_belgie.environment_exit_error = None
        await session.close()
        assert session.workspace is None
        assert fake_belgie.environments[0].exit_calls == 2
        assert fake_belgie.environments[0].exited

    async def test_start_cancellation_is_preserved(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.start_error = asyncio.CancelledError()
        session = BelgieSandboxSession()

        with pytest.raises(asyncio.CancelledError):
            await session.__aenter__()

        assert session.workspace is None
        assert fake_belgie.environments[0].exited

    async def test_start_cancellation_is_preserved_when_cleanup_fails(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.start_error = asyncio.CancelledError()
        cleanup_error = RuntimeError('cleanup failed')
        fake_belgie.environment_exit_error = cleanup_error
        session = BelgieSandboxSession()

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await session.__aenter__()

        assert exc_info.value.__cause__ is cleanup_error
        assert session.workspace is None
        assert fake_belgie.environments[0].exit_calls == 1
        fake_belgie.environment_exit_error = None
        await session.close()
        assert session.workspace is None
        assert fake_belgie.environments[0].exit_calls == 2

    async def test_close_retains_state_when_cleanup_fails(self, fake_belgie: FakeBelgie) -> None:
        session = BelgieSandboxSession()
        await session.__aenter__()
        fake_belgie.runtime_exit_error = RuntimeError('cleanup failed')

        with pytest.raises(RuntimeError, match='cleanup failed'):
            await session.close()

        # Runtime exit failed and is retained for retry; later resources still closed.
        assert session.is_open
        assert session.workspace is None
        assert fake_belgie.runtimes[0].exit_calls == 1
        assert not fake_belgie.runtimes[0].exited
        assert fake_belgie.environments[0].exit_calls == 1
        assert fake_belgie.environments[0].exited
        fake_belgie.runtime_exit_error = None
        await session.close()
        assert not session.is_open
        assert session.workspace is None
        assert fake_belgie.runtimes[0].exit_calls == 2
        assert fake_belgie.runtimes[0].exited
        assert fake_belgie.environments[0].exit_calls == 1

    async def test_environment_cleanup_failure_preserves_environment_for_retry(self, fake_belgie: FakeBelgie) -> None:
        session = BelgieSandboxSession()
        await session.__aenter__()
        fake_belgie.environment_exit_error = RuntimeError('cleanup failed')

        with pytest.raises(RuntimeError, match='cleanup failed'):
            await session.close()

        assert not session.is_open
        assert session.workspace is None
        assert fake_belgie.runtimes[0].exit_calls == 1
        assert fake_belgie.runtimes[0].exited
        assert fake_belgie.environments[0].exit_calls == 1
        assert not fake_belgie.environments[0].exited
        fake_belgie.environment_exit_error = None
        await session.close()
        assert session.workspace is None
        assert fake_belgie.runtimes[0].exit_calls == 1
        assert fake_belgie.environments[0].exit_calls == 2
        assert fake_belgie.environments[0].exited

    async def test_script_error_is_normalized(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.script_error = BelgieJavaScriptError('boom')
        async with BelgieSandboxSession() as session:
            with pytest.raises(BelgieSandboxExecutionError, match='execution failed'):
                await session.run_script('throw new Error("boom")')

    async def test_invalid_json_error_is_normalized(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.script_error = TypeError('BigInt is not JSON')
        async with BelgieSandboxSession() as session:
            with pytest.raises(BelgieSandboxExecutionError, match='invalid JSON'):
                await session.run_script('export default () => 1n')

    async def test_timeout_cancels_script(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.hang = True
        async with BelgieSandboxSession() as session:
            with pytest.raises(BelgieSandboxTimeoutError, match='0.01 seconds'):
                await session.run_script('export default async () => await never', timeout=0.01)

        assert fake_belgie.cancelled

    async def test_runtime_timeout_error_is_an_execution_failure(self, fake_belgie: FakeBelgie) -> None:
        runtime_error = TimeoutError('runtime failed before the deadline')
        fake_belgie.script_error = runtime_error
        async with BelgieSandboxSession() as session:
            with pytest.raises(BelgieSandboxExecutionError, match='runtime failed before the deadline') as exc_info:
                await session.run_script('export default () => fail()', timeout=10)

        assert exc_info.value.__cause__ is runtime_error
        assert not fake_belgie.cancelled

    async def test_caller_cancellation_is_preserved(self, fake_belgie: FakeBelgie) -> None:
        fake_belgie.hang = True
        async with BelgieSandboxSession() as session:
            task = asyncio.create_task(session.run_script('export default async () => await never'))
            while not fake_belgie.scripts and not task.done():
                await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert fake_belgie.cancelled

    async def test_rejects_calls_while_closed(self, fake_belgie: FakeBelgie) -> None:
        session = BelgieSandboxSession()
        with pytest.raises(BelgieSandboxError, match='not open'):
            await session.run_script('export default () => 1')
        await session.__aexit__(None, None, None)

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'allow_package_imports': 1}, 'allow_package_imports must be a bool'),
            ({'allow_network': 1}, 'allow_network must be a bool'),
            ({'enable_rendering': 1}, 'enable_rendering must be a bool'),
            ({'max_old_generation_size_mb': 0}, 'must be a positive integer or None'),
        ],
    )
    async def test_rejects_invalid_configuration(
        self, fake_belgie: FakeBelgie, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            BelgieSandboxSession(**kwargs)  # pyright: ignore[reportArgumentType]

    async def test_runtime_rejects_owned_settings(self, fake_belgie: FakeBelgie) -> None:
        runtime = fake_belgie.module.Runtime()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        with pytest.raises(ValueError, match='cannot be combined with `runtime`'):
            BelgieSandboxSession(runtime=runtime, enable_rendering=True)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize(
        ('source', 'timeout', 'error_type', 'message'),
        [
            (1, 1.0, TypeError, 'source must be a string'),
            ('code', 0, ValueError, 'timeout must be a positive finite number'),
            ('code', True, ValueError, 'timeout must be a positive finite number'),
        ],
    )
    async def test_validates_run_arguments(
        self,
        fake_belgie: FakeBelgie,
        source: object,
        timeout: object,
        error_type: type[Exception],
        message: str,
    ) -> None:
        async with BelgieSandboxSession() as session:
            with pytest.raises(error_type, match=message):
                await session.run_script(source, timeout=timeout)  # pyright: ignore[reportArgumentType]


def test_public_session_workspace_type() -> None:
    """Keep the public workspace contract concrete for callers."""
    assert BelgieSandboxSession().workspace is None
    assert Path is not None
