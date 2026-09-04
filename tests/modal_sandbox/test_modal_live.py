"""Integration tests that require a real, running Modal container.

The fake-backed suites already cover the harness-owned logic: timeout quantization,
protocol translation, path resolution math, and exception mapping. This live tier
admits only regressions a correctly written fake could not catch: real process
execution, infra-enforced deadlines, one filesystem shared by Modal's file API and
the shell, create-time environment and workdir propagation, and real lifecycle
state in Modal's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption
  it validates against real Modal behavior.

Portability:
  These assert durable sandbox behaviors, not Modal-specific spellings, so if the Modal
  mechanism is later swapped for a different backend the suite retargets rather than gets
  rewritten. Everything below runs through the `SandboxBackend` protocol, so the retarget is
  mostly a change of constructor.

Gating:
  * `modal_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_MODAL_LIVE=1` opts in explicitly.
  * also requires `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`, or `~/.modal.toml`.
  * a module-scoped `anyio_backend` fixture keeps the shared Modal handle on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_MODAL_LIVE=1 uv run pytest -m modal_live tests/modal_sandbox/test_modal_live.py`
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from pydantic_ai.sandboxes import Sandbox, SandboxTimeoutError

from pydantic_ai_harness.modal_sandbox import (
    ModalSandboxBackend,
    ModalSandboxUnavailableError,
)


def _has_modal_credentials() -> bool:
    has_env_token = os.getenv('MODAL_TOKEN_ID') is not None and os.getenv('MODAL_TOKEN_SECRET') is not None
    return has_env_token or Path('~/.modal.toml').expanduser().exists()


_live_enabled = os.getenv('PYDANTIC_AI_HARNESS_MODAL_LIVE') == '1'

pytestmark = [
    pytest.mark.modal_live,
    pytest.mark.skipif(
        not _live_enabled or not _has_modal_credentials(),
        reason=(
            'requires PYDANTIC_AI_HARNESS_MODAL_LIVE=1 and either MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml'
        ),
    ),
]

# A small, common image keeps cold starts cheap; these tests need only a POSIX shell and coreutils.
_IMAGE = 'python:3.12-slim'


def _unique(prefix: str) -> str:
    """Return a collision-resistant path or name segment for a shared live sandbox."""
    return f'{prefix}-{uuid.uuid4().hex}'


@asynccontextmanager
async def _owned(**settings: object) -> AsyncGenerator[ModalSandboxBackend]:
    """Create a sandbox and terminate it on the way out, as the capability's hooks do."""
    backend = ModalSandboxBackend(image=_IMAGE, **settings)  # type: ignore[arg-type]
    await backend.sandbox
    try:
        yield backend
    finally:
        await backend.close(terminate=True)


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def sandbox() -> AsyncIterator[ModalSandboxBackend]:
    """One live owned sandbox shared by exec and filesystem tests.

    Each test writes under `_unique(...)` paths, so the shared container avoids repeated cold starts
    without coupling test state. Lifecycle tests create their own sandboxes because ownership,
    expiry, attach, and termination are the behavior under test there.
    """
    async with _owned(sandbox_timeout=600) as live:
        yield live


class TestRealExecution:
    """Behaviors that only exist because a real process runs on real Modal infra."""

    async def test_runs_a_real_process(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that stdout, stderr, and exit code match a process."""
        result = await sandbox.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=30)

        assert result.stdout.strip() == 'out'
        assert result.stderr.strip() == 'err'
        assert result.exit_code == 3

    async def test_timeout_preserves_pre_deadline_output(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that output printed before an infra timeout is preserved."""
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await sandbox.run('echo DIAGNOSTIC; sleep 30', shell=True, timeout=2)

        assert 'DIAGNOSTIC' in exc_info.value.stdout

    async def test_timeout_preserves_stderr(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that Modal spells a deadline kill in two ways.

        Modal reports it either as its client-side `-1` sentinel or, when the server's SIGKILL
        wins the race, as a plain 137 exit; the backend recognizes both (137 counts once the
        command consumed its whole deadline window).
        """
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await sandbox.run('echo STDERR-DIAGNOSTIC 1>&2; sleep 30', shell=True, timeout=2)

        assert 'STDERR-DIAGNOSTIC' in exc_info.value.stderr

    async def test_large_stderr_does_not_block_stdout(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that Modal buffers streams without stderr deadlock."""
        result = await sandbox.run('seq 1 300000 1>&2; echo done', shell=True, timeout=60)

        assert result.exit_code == 0
        assert result.stdout == 'done\n'
        stderr_lines = result.stderr.splitlines()
        assert stderr_lines[0] == '1'
        assert stderr_lines[-1] == '300000'
        assert len(stderr_lines) == 300000

    async def test_concurrent_commands_share_one_container(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that one Modal sandbox can multiplex concurrent commands."""
        results: dict[int, str] = {}

        async def run(n: int) -> None:
            out = await sandbox.run(f'echo job-{n}', shell=True, timeout=15)
            results[n] = out.stdout.strip()

        async with anyio.create_task_group() as tg:
            for n in range(8):
                tg.start_soon(run, n)

        assert results == {n: f'job-{n}' for n in range(8)}

    async def test_signal_exit_is_not_timeout(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that signal death is a real exit, not Modal's timeout sentinel."""
        result = await sandbox.run('kill -KILL $$', shell=True, timeout=15)

        assert result.exit_code > 128
        assert result.exit_code != -1

    async def test_nonexistent_binary_returns_modal_exit_code(self, sandbox: ModalSandboxBackend) -> None:
        """Pins Modal's current return code for an executable lookup failure."""
        result = await sandbox.run([_unique('definitely-not-a-real-binary')], timeout=15)
        assert result.exit_code == 128


class TestCreateConfiguration:
    """Create-time configuration reaching the real process, not only Modal create kwargs."""

    async def test_env_and_workdir_reach_processes(self) -> None:
        """Validates the fake-encoded assumption that create-time `env` and `workdir` reach commands."""
        probe = _unique('live-value')
        async with _owned(sandbox_timeout=120, workdir='/tmp', env={'HARNESS_ENV_PROBE': probe}) as backend:
            env_result = await backend.run('printf %s "$HARNESS_ENV_PROBE"', shell=True, timeout=15)
            working_dir = await backend.working_dir()

        assert env_result.stdout == probe
        assert working_dir == '/tmp'

    async def test_working_dir_is_discovered_when_not_configured(self) -> None:
        """Validates the fake-encoded assumption that `pwd` answers for a sandbox created without a workdir."""
        async with _owned(sandbox_timeout=120) as backend:
            working_dir = await backend.working_dir()
            printed = await backend.run(['pwd'], timeout=15)

        assert working_dir == printed.stdout.strip()

    async def test_per_command_cwd_and_env_reach_the_process(self) -> None:
        """Validates the fake-encoded assumption that Modal applies per-command `workdir` and `env`."""
        probe = _unique('per-command')
        async with _owned(sandbox_timeout=120) as backend:
            result = await backend.run('printf "%s %s" "$(pwd)" "$PROBE"', shell=True, cwd='/etc', env={'PROBE': probe})

        assert result.stdout == f'/etc {probe}'


class TestRealFilesystem:
    """One real filesystem shared by Modal's file API and the shell."""

    async def test_shell_and_file_api_see_the_same_filesystem(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the protocol's one-environment contract against real Modal."""
        api_path = f'/tmp/{_unique("api")}.txt'
        await sandbox.write_bytes(api_path, b'from-file-api\n')
        via_shell = await sandbox.run(['cat', api_path], timeout=15)
        assert via_shell.stdout == 'from-file-api\n'

        shell_path = f'/tmp/{_unique("shell")}.txt'
        wrote = await sandbox.run(f'printf from-shell > {shell_path}', shell=True, timeout=15)
        assert wrote.exit_code == 0
        assert await sandbox.read_bytes(shell_path) == b'from-shell'

    async def test_binary_roundtrip_creating_parent_dirs(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that Modal stores raw bytes and creates real parent dirs."""
        path = f'/tmp/{_unique("io")}/nested/deep/data.bin'
        payload = b'\x00\x01hello \xf0\x9f\x9a\x80 world'

        await sandbox.write_bytes(path, payload)

        assert await sandbox.read_bytes(path) == payload

    async def test_large_filesystem_transfer_near_read_limit(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that Modal's fs API handles a near-limit transfer."""
        path = f'/tmp/{_unique("big")}.bin'
        payload = b'A' * (4 * 1024 * 1024)

        await sandbox.write_bytes(path, payload)

        assert (await sandbox.stat(path)).size == len(payload)
        assert await sandbox.read_bytes(path) == payload

    async def test_missing_file_raises_the_builtin_error(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the protocol's contract that a missing path raises the builtin `FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            await sandbox.read_bytes(f'/tmp/{_unique("missing")}')

        assert await sandbox.exists(f'/tmp/{_unique("missing")}') is False

    async def test_list_dir_reports_basenames_and_dir_flags(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that Modal lists entries by basename with a real dir flag."""
        root = f'/tmp/{_unique("ls")}'
        await sandbox.write_bytes(f'{root}/file.txt', b'x')
        await sandbox.write_bytes(f'{root}/sub/nested.txt', b'y')

        entries = await sandbox.list_dir(root)

        assert sorted((entry.name, entry.is_dir, entry.path) for entry in entries) == [
            ('file.txt', False, f'{root}/file.txt'),
            ('sub', True, f'{root}/sub'),
        ]

    async def test_make_dir_and_remove_are_recursive(self, sandbox: ModalSandboxBackend) -> None:
        """Validates the fake-encoded assumption that Modal's `mkdir -p` and recursive remove behave as documented."""
        root = f'/tmp/{_unique("tree")}'
        await sandbox.make_dir(f'{root}/a/b')
        await sandbox.write_bytes(f'{root}/a/b/file.txt', b'x')

        await sandbox.remove(root)

        assert await sandbox.exists(root) is False

    async def test_relative_paths_resolve_against_the_working_directory(self) -> None:
        """Validates the fake-encoded assumption that the facade's resolution matches the process cwd."""
        filename = f'{_unique("rel")}.txt'
        async with _owned(sandbox_timeout=120, workdir='/tmp') as backend:
            facade = Sandbox(backend)
            await facade.write_text(filename, 'from-relative-path\n')
            result = await backend.run(['cat', filename], timeout=15)

        assert result.stdout == 'from-relative-path\n'


class TestRealLifecycle:
    """Teardown and attach semantics in Modal's real control plane."""

    async def test_terminate_actually_destroys_the_container(self) -> None:
        """Validates the fake-encoded assumption that closing an owned sandbox destroys the real one."""
        async with _owned(sandbox_timeout=120) as owner:
            ref = owner.ref
        assert ref is not None

        became_unavailable = False
        attempts = 8
        for attempt in range(attempts):
            try:
                await ModalSandboxBackend(ref=ref).sandbox
            except ModalSandboxUnavailableError:
                became_unavailable = True
                break
            # Modal can lag about 30s before reporting external termination to a fresh connect.
            if attempt < attempts - 1:
                await anyio.sleep(5)

        assert became_unavailable

    async def test_connect_reuses_state_and_leaves_container_running(self) -> None:
        """Validates the fake-encoded assumption that connecting reuses state and does not take ownership."""
        marker = f'/tmp/{_unique("persist")}.txt'
        async with _owned(sandbox_timeout=120) as owner:
            await owner.write_bytes(marker, b'shared')

            attached = ModalSandboxBackend(ref=owner.ref)
            await attached.sandbox
            assert attached.ref == owner.ref
            assert await attached.read_bytes(marker) == b'shared'
            await attached.close(terminate=False)

            assert (await owner.run(['cat', marker], timeout=15)).stdout == 'shared'
