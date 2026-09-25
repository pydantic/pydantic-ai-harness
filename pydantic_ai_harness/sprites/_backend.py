"""Fly.io Sprites backend for Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified 2026-09-15 against sprites-py 0.7.0 source and a local
WebSocket transport probe, with no live cloud calls:

* `AsyncSpritesClient` accepts token, base URL, and HTTP timeout; sprite creation uses the SDK's
  fixed 120-second request timeout, while `aclose` closes the local async HTTP client and control pools:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/async_client.py
* `create_sprite` and `get_sprite` raise `AuthenticationError` (401), `NotFoundError` (404),
  `NetworkError` (transport), and a plain `SpriteError` for any other HTTP failure:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/client.py
* `ControlConnection` is asyncio-based and exposes `connect`, `start_op`, `close`, and its
  WebSocket as `ws`; an operation provides `wait`, `get_stdout`, and `get_stderr`, and a failed
  handshake raises `websockets.exceptions.InvalidStatus`:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/control.py
* A control WebSocket disconnect does not kill the remote command, so the backend's RUN/CANCEL
  process supervision is required:
  https://sprites.dev/api/sprites/exec
* Provider retention is separate from local client disconnect:
  https://docs.sprites.dev/concepts/lifecycle/

Re-check these sources, the installed signatures, and the local transport probe before changing
lifecycle or command transport behavior. The integration uses the SDK's native asyncio client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import posixpath
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeVar

import anyio
from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness._workspace_provider import absolute_path, command_argv
from pydantic_ai_harness.sprites._process import CANCEL, RUN, SPAWN_FAILED, SPAWN_FAILED_EXIT

try:
    from sprites import AsyncSprite, AsyncSpritesClient
    from sprites.control import ControlConnection
    from sprites.exceptions import AuthenticationError, NetworkError, NotFoundError, SpriteError
    from sprites.exceptions import TimeoutError as SpriteTimeoutError
    from websockets.exceptions import InvalidStatus
except ImportError as exc:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[sprites]` to use SpriteWorkspace.') from exc

logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_CONTROL_TIMEOUT = 6.0
# Above the SDK's fixed 120-second creation request timeout, so the SDK's own error wins when it fires.
_ACQUIRE_TIMEOUT = 150.0
_AUTH_MESSAGE = 'Sprites rejected the credentials. Set SPRITE_TOKEN or pass token= and try again.'


async def cleanup_call(call: Callable[[], Awaitable[object]], *, timeout: float) -> Exception | None:
    """Run one teardown RPC shielded from cancellation and bounded by `timeout`.

    Returns the failure instead of raising so the caller owns translation; a bare `TimeoutError`
    means the bound expired. Shielded because teardown must still go out while a run is being
    cancelled; bounded so a wedged control plane cannot hang teardown.
    """
    error: Exception | None = None
    with anyio.move_on_after(timeout, shield=True) as scope:
        try:
            await call()
        except Exception as exc:
            error = exc
    if scope.cancel_called:
        return TimeoutError()
    return error


async def run_to_completion(call: Callable[[], Awaitable[_T]]) -> _T:
    """Await `call` in a task of its own and see it finish even if the caller is cancelled meanwhile.

    For work whose outcome must be recorded, such as a created Sprite or a closed client: the
    caller's cancellation is re-raised only once `call` has finished, and nothing outlives this
    await. `call` must be bounded; the caller waits for it.
    """
    task = asyncio.ensure_future(call())
    cancelled: asyncio.CancelledError | None = None
    # The anyio shield holds off cancel scopes; a native `Task.cancel()` (which anyio also re-sends to
    # a task awaited from a cancelled scope) still interrupts the wait, so it is caught and the wait
    # resumed. Waiting on the task never cancels it.
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.wait([task])
            except asyncio.CancelledError as error:
                cancelled = error
    if cancelled is not None:
        task.exception()  # Retrieved, so asyncio does not report it: the cancellation supersedes it.
        raise cancelled
    return task.result()


async def _close_connection(connection: ControlConnection) -> None:
    """Close a control connection; if that fails, abort its socket so nothing is left open.

    Finished even when the caller (a cancelled command) is cancelled meanwhile.
    """
    error = await run_to_completion(lambda: cleanup_call(connection.close, timeout=_CONTROL_TIMEOUT))
    if error is not None:
        logger.warning('Could not close a Sprite control connection, aborting it: %r', error)
        # `close` has nothing to fail on before `connect` opened the socket.
        if connection.ws is not None:  # pragma: no branch
            connection.ws.transport.abort()


def _map_error(error: Exception, name: str) -> WorkspaceError | None:
    """Translate a Sprites failure, or return `None` for one that propagates as is.

    Rejected credentials and a missing Sprite end the run; any other request the API refused is
    a failed operation. Transport failures (`NetworkError`, a handshake that fails with another
    status, a closed connection), rate limits, and anything unknown propagate unchanged, for durable
    engines to retry.
    """
    status = error.response.status_code if isinstance(error, InvalidStatus) else None
    if isinstance(error, AuthenticationError) or status == 401:
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if isinstance(error, NotFoundError) or status == 404:
        return WorkspaceUnavailableError(f'Sprite {name!r} no longer exists.')
    # sprites-py reports a rate limit as a plain `SpriteError` naming the HTTP status.
    if isinstance(error, (NetworkError, SpriteTimeoutError)) or '(status 429)' in str(error):
        return None
    if isinstance(error, SpriteError):
        return WorkspaceError(f'Sprites refused the request: {error}')
    return None


class SpriteWorkspaceBackend(WorkspaceBackend, SupportsCommands):
    """A Fly.io Sprite behind the Pydantic AI `WorkspaceBackend` protocol.

    Construction does no I/O. The typed `sprites.AsyncSprite` is available through `get_client()`.
    The backend does not delete the Sprite; that is the application's job, through the native
    handle. Commands run under `/bin/sh -c` with `shell=True`, in the Sprite's own environment
    plus `env`. Every command gets its own asyncio control connection, which is closed before the
    result is returned.
    """

    def __init__(
        self,
        *,
        workspace: AsyncSprite | None = None,
        client: AsyncSpritesClient | None = None,
        ref: WorkspaceRef | None = None,
        name: str | None = None,
        token: str | None = None,
        base_url: str = 'https://api.sprites.dev',
        api_timeout: float = 30.0,
        runtime: str | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'sprites':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'sprites'")
        if workspace is not None and ref is not None:
            raise ValueError('pass either `workspace` or `ref`, not both')
        self._workspace = workspace
        self._ref = ref if workspace is None else WorkspaceRef(provider='sprites', id=workspace.name)
        self._name = name or f'pydantic-ai-{uuid.uuid4().hex}'
        self._token = token
        self._base_url = base_url
        self._api_timeout = api_timeout
        self._runtime = runtime
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env or {})
        self._canonical_working_dir: str | None = None
        self._client = client
        self._owns_client = client is None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def get_client(self) -> AsyncSprite:
        """Return the typed `sprites.AsyncSprite`, creating or attaching to it on first use.

        This is the Sprite handle, not the `AsyncSpritesClient` passed as `client=`. A handle
        obtained before `aclose()` belongs to the closed client; call this again for a fresh one.

        The only place `_client` and `_workspace` are read, so nothing can reach an
        unhydrated one: both stay optional and every other method comes through here.
        The lock serializes concurrent first uses -- two callers each creating a Sprite
        would leave the loser billed and unreferenced. Attaching by `ref` to a Sprite that
        no longer exists raises `WorkspaceUnavailableError`; it does not create a replacement.
        """
        async with self._lock:
            if (workspace := self._workspace) is not None:
                return workspace

            client = self._client
            if client is None:
                token = self._token or os.getenv('SPRITE_TOKEN')
                if not token:
                    raise WorkspaceUnavailableError(_AUTH_MESSAGE)
                client = AsyncSpritesClient(token=token, base_url=self._base_url, timeout=self._api_timeout)
                self._client = client

            ref = self._ref

            async def acquire() -> AsyncSprite:
                with anyio.move_on_after(_ACQUIRE_TIMEOUT):
                    if ref is not None:
                        workspace = await client.get_sprite(ref.id)
                    else:
                        workspace = await client.create_sprite(self._name, runtime=self._runtime)
                    # Recorded as soon as the SDK returns, so a cancelled caller still leaves it named.
                    self._workspace = workspace
                    self._ref = WorkspaceRef(provider='sprites', id=workspace.name)
                    return workspace
                # Only our own bound lands here; an SDK `TimeoutError` propagates as raised. A stalled
                # control plane is a transport failure, which propagates for a retry;
                # `WorkspaceTimeoutError` is reserved for command deadlines.
                action = 'creation' if ref is None else 'connection'
                raise TimeoutError(
                    f'Sprite {action} did not complete within {_ACQUIRE_TIMEOUT:g}s; '
                    'the Sprites control plane may be unreachable.'
                )

            try:
                # Creation runs to completion even if the caller is cancelled, so a Sprite that
                # was created is never left unnamed; attaching creates nothing and stays cancellable.
                return await (acquire() if ref is not None else run_to_completion(acquire))
            except SpriteError as error:
                if (mapped := _map_error(error, self._name if ref is None else ref.id)) is None:
                    raise
                raise mapped from error

    async def aclose(self) -> None:
        """Close the `AsyncSpritesClient` this backend created, if it created one.

        The Sprite is untouched: the next operation opens a fresh client and reattaches by `ref`.
        A caller-supplied `client=` or `workspace=` handle is never closed. `SpriteWorkspace`
        calls this for the backend it supplied when each run ends. A close that fails or times out
        is logged, not raised, and the client is kept so the next `aclose()` tries again.
        """
        if not self._owns_client:
            return

        async def close() -> None:
            # Under the lock, so a Sprite still being created is recorded before its client closes.
            async with self._lock:
                client = self._client
                if client is None:
                    return
                error = await cleanup_call(client.aclose, timeout=_CONTROL_TIMEOUT)
                if error is not None:
                    # Kept, so a later `aclose()` tries again.
                    logger.warning('Could not close Sprites SDK client: %r', error)
                else:
                    self._client = None
                    self._workspace = None

        # Finished even when the caller (a run being cancelled) is cancelled meanwhile.
        await run_to_completion(close)

    async def working_dir(self) -> str:
        if self._canonical_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=30)
            directory = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(directory):
                raise WorkspaceError('Could not determine the Sprite working directory.')
            self._canonical_working_dir = directory
        return self._canonical_working_dir

    async def _cancel_remote(self, sprite: AsyncSprite, control: str) -> None:
        connection = ControlConnection(sprite)

        async def cancel() -> None:
            await connection.connect()
            operation = await connection.start_op('exec', cmd=['python3', '-I', '-c', CANCEL, control], stdin=False)
            if await operation.wait() != 0:
                raise RuntimeError('the cancellation script failed')

        error = await cleanup_call(cancel, timeout=_CONTROL_TIMEOUT)
        if error is not None:
            logger.warning('Could not confirm remote Sprite command termination: %r', error)
        await _close_connection(connection)

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f'timeout must be a positive finite number or None, got {timeout!r}.')
        directory = absolute_path('cwd', cwd) if cwd is not None else self._working_dir
        args = command_argv(command, shell)
        options = json.dumps({'args': args, 'cwd': directory, 'env': {**self._env, **(env or {})}})
        control = f'/tmp/pydantic-ai-{uuid.uuid4().hex}'

        # The deadline is measured from this call, so it covers acquiring the Sprite too.
        deadline = anyio.CancelScope(deadline=math.inf if timeout is None else anyio.current_time() + timeout)
        sprite = connection = operation = None
        code = -1
        try:
            with deadline:
                sprite = await self.get_client()
                connection = ControlConnection(sprite)
                await connection.connect()
                operation = await connection.start_op(
                    'exec', cmd=['python3', '-I', '-c', RUN, control, options], stdin=False
                )
                code = await operation.wait()
            stdout = operation.get_stdout() if operation is not None else b''
            stderr = operation.get_stderr() if operation is not None else b''
            if timeout is not None and deadline.cancelled_caught:
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds',
                    stdout=_decode(stdout),
                    stderr=_decode(stderr),
                    timeout=timeout,
                )
            # Only the deadline leaves the block before the connection exists.
            assert connection is not None
            if code == -1:
                # The SDK keeps the transport failure that closed the connection; that is what propagates.
                if connection.close_error is not None:
                    raise connection.close_error
                raise ConnectionError(
                    f'Sprite command transport closed before reporting an exit status. {_decode(stderr).strip()}'.strip()
                )
        except BaseException as error:
            if sprite is None:
                raise
            await self._cancel_remote(sprite, control)
            if connection is not None:  # pragma: no branch - created right after the Sprite
                await _close_connection(connection)
            if isinstance(error, Exception) and (mapped := _map_error(error, sprite.name)) is not None:
                raise mapped from error
            raise
        await _close_connection(connection)
        if code == SPAWN_FAILED_EXIT and stderr.startswith(SPAWN_FAILED.encode()):
            raise _spawn_error(stderr)
        return CommandResult(exit_code=code, stdout=_decode(stdout), stderr=_decode(stderr))


_SPAWN_ERRORS: dict[str, type[OSError]] = {
    error.__name__: error for error in (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)
}


def _spawn_error(stderr: bytes) -> Exception:
    """The error for a command the Sprite could not start, as `RUN` reported it.

    A path-level failure (a missing `cwd` or executable, say) is the builtin error; anything else
    is a `WorkspaceError`.
    """
    name, number, message, filename = json.loads(stderr.removeprefix(SPAWN_FAILED.encode()))
    if (error := _SPAWN_ERRORS.get(name)) is not None:
        return error(number, message, filename)
    return WorkspaceError(f'Could not start the command in the Sprite: {message}')


def _decode(data: bytes) -> str:
    return data.decode('utf-8', errors='replace')
