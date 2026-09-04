"""Integration tests that require a real, running E2B sandbox.

The fake-backed suites already cover the harness-owned logic: deadline arithmetic, protocol
translation, path resolution math, and exception mapping. This live tier admits only
regressions a correctly written fake could not catch: real process execution, a client-owned
deadline killing a real process, one filesystem shared by E2B's file API and the shell,
create-time environment propagation, and real lifecycle state in E2B's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real E2B behavior.

Portability:
  These assert durable sandbox behaviors, not E2B-specific spellings, so if the E2B mechanism
  is later swapped for a different backend the suite retargets rather than gets rewritten.
  Everything below runs through the `SandboxBackend` protocol, so the retarget is mostly a
  change of constructor.

Gating:
  * `e2b_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_E2B_LIVE=1` opts in explicitly.
  * also requires `E2B_API_KEY`.
  * a module-scoped `anyio_backend` fixture keeps the shared E2B handle on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_E2B_LIVE=1 uv run pytest -m e2b_live tests/e2b_sandbox/test_e2b_live.py`
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest
from pydantic_ai.sandboxes import Sandbox, SandboxTimeoutError

from pydantic_ai_harness.e2b_sandbox import (
    E2BSandboxBackend,
    E2BSandboxUnavailableError,
)

_live_enabled = os.getenv('PYDANTIC_AI_HARNESS_E2B_LIVE') == '1'

pytestmark = [
    pytest.mark.e2b_live,
    pytest.mark.skipif(
        not _live_enabled or os.getenv('E2B_API_KEY') is None,
        reason='requires PYDANTIC_AI_HARNESS_E2B_LIVE=1 and E2B_API_KEY',
    ),
]


def _unique(prefix: str) -> str:
    """Return a collision-resistant path or name segment for a shared live sandbox."""
    return f'{prefix}-{uuid.uuid4().hex}'


@asynccontextmanager
async def _owned(**settings: object) -> AsyncGenerator[E2BSandboxBackend]:
    """Create a sandbox and kill it on the way out, so a live run leaves nothing behind."""
    backend = E2BSandboxBackend(**settings)  # type: ignore[arg-type]
    await backend.sandbox
    try:
        yield backend
    finally:
        await backend.close(terminate=True)


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def sandbox() -> AsyncIterator[E2BSandboxBackend]:
    """One live owned sandbox shared by command and filesystem tests.

    Each test writes under `_unique(...)` paths, so the shared microVM avoids repeated cold
    starts without coupling test state. Lifecycle tests create their own sandboxes because
    ownership, expiry, attach, and teardown are the behavior under test there.
    """
    async with _owned(sandbox_timeout=600) as live:
        yield live


class TestRealExecution:
    """Behaviors that only exist because a real process runs in a real microVM."""

    async def test_runs_a_real_process(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that stdout, stderr, and exit code match a process."""
        result = await sandbox.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=30)

        assert result.stdout.strip() == 'out'
        assert result.stderr.strip() == 'err'
        assert result.exit_code == 3

    async def test_argv_elements_stay_single_words(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that `shlex.join` survives E2B's shell.

        E2B has no argv form, so an argv sequence is quoted into one shell word string; a
        value with a space and a `$` proves the quoting holds through `/bin/bash -l -c`.
        """
        result = await sandbox.run(['printf', '%s', 'a b $HOME'], timeout=30)

        assert result.stdout == 'a b $HOME'

    async def test_timeout_kills_the_command_and_keeps_its_output(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that a client-owned deadline kills a real process.

        E2B's own command `timeout` abandons the output stream and leaves the command running,
        so the backend enforces the deadline itself and calls the per-command kill. The marker
        the command would have written after the deadline must never appear.
        """
        marker = f'/tmp/{_unique("after-deadline")}'
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await sandbox.run(f'echo DIAGNOSTIC; sleep 20; touch {marker}', shell=True, timeout=2)

        assert 'DIAGNOSTIC' in exc_info.value.stdout
        await anyio.sleep(25)
        assert await sandbox.exists(marker) is False

    async def test_a_background_child_outlives_the_kill(self, sandbox: E2BSandboxBackend) -> None:
        """Pins the documented limitation that E2B's kill signals the command's own process only.

        A process the command started in the background is not reached by that signal and runs
        until the sandbox is torn down. If E2B ever kills the whole group, this test fails and
        the limitation can be removed from the docs.
        """
        marker = f'/tmp/{_unique("orphan")}'
        with pytest.raises(SandboxTimeoutError):
            await sandbox.run(f'(sleep 5; touch {marker}) & sleep 30', shell=True, timeout=2)

        await anyio.sleep(10)
        assert await sandbox.exists(marker) is True

    async def test_a_cancelled_run_stops_the_command(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the protocol's cancellation contract against real E2B.

        A cancelled `run()` must not knowingly leave the command running; E2B has a
        per-command kill, so the marker written after the cancellation must never appear.
        """
        marker = f'/tmp/{_unique("cancelled")}'
        with anyio.move_on_after(2):
            await sandbox.run(f'sleep 15; touch {marker}', shell=True, timeout=60)

        await anyio.sleep(20)
        assert await sandbox.exists(marker) is False

    async def test_large_stderr_does_not_block_stdout(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B buffers both streams without deadlock."""
        result = await sandbox.run('seq 1 200000 1>&2; echo done', shell=True, timeout=120)

        assert result.exit_code == 0
        assert result.stdout == 'done\n'
        stderr_lines = result.stderr.splitlines()
        assert (stderr_lines[0], stderr_lines[-1], len(stderr_lines)) == ('1', '200000', 200000)

    async def test_concurrent_commands_share_one_sandbox(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that one E2B sandbox multiplexes concurrent commands."""
        results: dict[int, str] = {}

        async def run(n: int) -> None:
            out = await sandbox.run(f'echo job-{n}', shell=True, timeout=60)
            results[n] = out.stdout.strip()

        async with anyio.create_task_group() as tg:
            for n in range(8):
                tg.start_soon(run, n)

        assert results == {n: f'job-{n}' for n in range(8)}

    async def test_signal_exit_is_a_real_exit(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that a signalled death is a plain exit code, not a timeout."""
        result = await sandbox.run('kill -KILL $$', shell=True, timeout=30)

        assert result.exit_code > 128

    async def test_a_missing_binary_is_a_reported_exit(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B reports a lookup failure as an exit code.

        The SDK raises `CommandExitException` on a non-zero exit; the protocol calls that a
        normal result, so the backend must unwrap it rather than let it reach the caller.
        """
        result = await sandbox.run([_unique('definitely-not-a-real-binary')], timeout=30)

        assert result.exit_code == 127


class TestCreateConfiguration:
    """Create-time configuration reaching the real process, not only E2B create arguments."""

    async def test_env_reaches_commands(self) -> None:
        """Validates the fake-encoded assumption that create-time `env` reaches commands."""
        probe = _unique('live-value')
        async with _owned(sandbox_timeout=120, env={'HARNESS_ENV_PROBE': probe}) as backend:
            result = await backend.run('printf %s "$HARNESS_ENV_PROBE"', shell=True, timeout=30)

        assert result.stdout == probe

    async def test_working_dir_is_discovered_when_not_configured(self) -> None:
        """Validates the fake-encoded assumption that `pwd` answers for a sandbox with no configured workdir."""
        async with _owned(sandbox_timeout=120) as backend:
            working_dir = await backend.working_dir()
            printed = await backend.run(['pwd'], timeout=30)

        assert working_dir == printed.stdout.strip()

    async def test_a_configured_working_dir_applies_per_command(self) -> None:
        """Validates the fake-encoded assumption that E2B takes a working directory per command.

        E2B has no create-time workdir, so the backend supplies it on every command instead.
        """
        async with _owned(sandbox_timeout=120, working_dir='/tmp') as backend:
            result = await backend.run(['pwd'], timeout=30)

        assert result.stdout.strip() == '/tmp'

    async def test_per_command_cwd_and_env_reach_the_process(self) -> None:
        """Validates the fake-encoded assumption that E2B applies per-command `cwd` and `envs`."""
        probe = _unique('per-command')
        async with _owned(sandbox_timeout=120) as backend:
            result = await backend.run(
                'printf "%s %s" "$(pwd)" "$PROBE"', shell=True, cwd='/etc', env={'PROBE': probe}, timeout=30
            )

        assert result.stdout == f'/etc {probe}'


class TestRealFilesystem:
    """One real filesystem shared by E2B's file API and the shell."""

    async def test_shell_and_file_api_see_the_same_filesystem(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the protocol's one-environment contract against real E2B."""
        api_path = f'/tmp/{_unique("api")}.txt'
        await sandbox.write_bytes(api_path, b'from-file-api\n')
        via_shell = await sandbox.run(['cat', api_path], timeout=30)
        assert via_shell.stdout == 'from-file-api\n'

        shell_path = f'/tmp/{_unique("shell")}.txt'
        wrote = await sandbox.run(f'printf from-shell > {shell_path}', shell=True, timeout=30)
        assert wrote.exit_code == 0
        assert await sandbox.read_bytes(shell_path) == b'from-shell'

    async def test_binary_roundtrip_creating_parent_dirs(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B stores raw bytes and creates real parent dirs."""
        path = f'/tmp/{_unique("io")}/nested/deep/data.bin'
        payload = b'\x00\x01hello \xf0\x9f\x9a\x80 world'

        await sandbox.write_bytes(path, payload)

        assert await sandbox.read_bytes(path) == payload

    async def test_large_filesystem_transfer_near_read_limit(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B's file API handles a near-limit transfer."""
        path = f'/tmp/{_unique("big")}.bin'
        payload = b'A' * (4 * 1024 * 1024)

        await sandbox.write_bytes(path, payload)

        assert (await sandbox.stat(path)).size == len(payload)
        assert await sandbox.read_bytes(path) == payload

    async def test_missing_file_raises_the_builtin_error(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the protocol's contract that a missing path raises the builtin `FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            await sandbox.read_bytes(f'/tmp/{_unique("missing")}')

        assert await sandbox.exists(f'/tmp/{_unique("missing")}') is False

    async def test_list_dir_reports_basenames_and_dir_flags(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B lists entries by basename with a real dir flag."""
        root = f'/tmp/{_unique("ls")}'
        await sandbox.write_bytes(f'{root}/file.txt', b'x')
        await sandbox.write_bytes(f'{root}/sub/nested.txt', b'y')

        entries = await sandbox.list_dir(root)

        assert sorted((entry.name, entry.is_dir, entry.path) for entry in entries) == [
            ('file.txt', False, f'{root}/file.txt'),
            ('sub', True, f'{root}/sub'),
        ]

    async def test_make_dir_and_remove_are_recursive(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B's `mkdir -p` and recursive remove behave as documented."""
        root = f'/tmp/{_unique("tree")}'
        await sandbox.make_dir(f'{root}/a/b')
        await sandbox.write_bytes(f'{root}/a/b/file.txt', b'x')

        await sandbox.remove(root)

        assert await sandbox.exists(root) is False

    async def test_relative_paths_resolve_against_the_working_directory(self) -> None:
        """Validates the fake-encoded assumption that the facade's resolution matches the process cwd."""
        filename = f'{_unique("rel")}.txt'
        async with _owned(sandbox_timeout=120, working_dir='/tmp') as backend:
            facade = Sandbox(backend)
            await facade.write_text(filename, 'from-relative-path\n')
            result = await backend.run(['cat', filename], timeout=30)

        assert result.stdout == 'from-relative-path\n'


class TestRealLifecycle:
    """Teardown and attach semantics in E2B's real control plane."""

    async def test_kill_actually_destroys_the_sandbox(self) -> None:
        """Validates the fake-encoded assumption that closing an owned sandbox destroys the real one."""
        async with _owned(sandbox_timeout=120) as owner:
            ref = owner.ref
        assert ref is not None

        with pytest.raises(E2BSandboxUnavailableError):
            await E2BSandboxBackend(ref=ref).sandbox

    async def test_connect_reuses_state_and_leaves_the_sandbox_running(self) -> None:
        """Validates the fake-encoded assumption that connecting reuses state and does not take ownership."""
        marker = f'/tmp/{_unique("persist")}.txt'
        async with _owned(sandbox_timeout=120) as owner:
            await owner.write_bytes(marker, b'shared')

            attached = E2BSandboxBackend(ref=owner.ref)
            assert (await attached.sandbox).sandbox_id == (await owner.sandbox).sandbox_id
            assert await attached.read_bytes(marker) == b'shared'
            await attached.close(terminate=False)

            assert (await owner.run(['cat', marker], timeout=30)).stdout == 'shared'

    async def test_connect_resumes_a_paused_sandbox(self) -> None:
        """Pins the documented behavior that attaching to a paused sandbox restarts it.

        This is the E2B-specific half of attach mode: a paused sandbox is not gone, and the
        capability's `get_sandbox` brings it back rather than failing.
        """
        marker = f'/tmp/{_unique("paused")}.txt'
        async with _owned(sandbox_timeout=120) as owner:
            await owner.write_bytes(marker, b'before-pause')
            await (await owner.sandbox).beta_pause()

            attached = E2BSandboxBackend(ref=owner.ref)
            assert await attached.read_bytes(marker) == b'before-pause'
