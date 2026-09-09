"""Fly.io Sprites backend for Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified 2026-09-08 against sprites-py 0.6.0 source and a local
WebSocket transport probe, with no live cloud calls:

* `SpritesClient` accepts token, base URL, and HTTP timeout; sprite creation uses the SDK's
  fixed 120-second request timeout, while `close` only closes the local HTTP client:
  https://github.com/superfly/sprites-py/blob/v0.6.0/src/sprites/client.py
* `ControlConnection` is asyncio-based and exposes `connect`, `start_op`, and `close`; an
  operation provides `wait`, `get_stdout`, and `get_stderr`:
  https://github.com/superfly/sprites-py/blob/v0.6.0/src/sprites/control.py
* A control WebSocket disconnect does not kill the remote command, so the backend's RUN/CANCEL
  process supervision is required:
  https://sprites.dev/api/sprites/exec
* Provider retention is separate from local client disconnect:
  https://docs.sprites.dev/concepts/lifecycle/

Re-check these sources, the installed signatures, and the local transport probe before changing
lifecycle or command transport behavior. The integration is asyncio-only, and the synchronous SDK
calls run in a worker thread that a cancelled caller cannot abort.
"""

from __future__ import annotations

import json
import logging
import math
import os
import posixpath
import uuid
from collections.abc import Awaitable, Callable, Mapping
from functools import cached_property
from typing import NoReturn, TypeVar

import anyio
import anyio.to_thread
from anyio.lowlevel import checkpoint
from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from pydantic_ai_harness._workspace_provider import absolute_path
from pydantic_ai_harness.sprites._process import CANCEL, RUN

try:
    from sprites import Sprite, SpritesClient
    from sprites.control import ControlConnection
    from sprites.exceptions import AuthenticationError, NotFoundError, SpriteError
    from websockets.exceptions import InvalidStatus
except ImportError as exc:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[sprites]` to use SpriteWorkspace.') from exc

logger = logging.getLogger(__name__)
T = TypeVar('T')
_CONTROL_TIMEOUT = 6.0
_AUTH_MESSAGE = 'Sprites rejected the credentials. Set SPRITE_TOKEN or pass token= and try again.'


async def _call(call: Callable[[], T]) -> T:
    return await anyio.to_thread.run_sync(call, abandon_on_cancel=True)


async def cleanup_call(call: Callable[[], Awaitable[object]], *, timeout: float) -> Exception | None:
    """Run one teardown RPC shielded from cancellation and bounded by `timeout`.

    Returns the failure instead of raising so the caller owns translation; a bare `TimeoutError`
    means the bound expired. Shielded because teardown must still go out while a run is being
    cancelled; bounded so a wedged control plane cannot hang teardown.
    """
    error: Exception | None = None
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(timeout) as scope:
            try:
                await call()
            except Exception as exc:
                error = exc
        if scope.cancel_called:
            return TimeoutError()
    return error


async def raise_after_cleanup(error: Exception, *, cause: Exception | None = None) -> NoReturn:
    """Deliver pending cancellation before raising a cleanup error."""
    await checkpoint()
    raise error from cause


def _command_args(command: WorkspaceCommand, shell: bool) -> list[str]:
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a command string')
        return ['/bin/bash', '--noprofile', '--norc', '-c', command]
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    args = list(command)
    if not args:
        raise TypeError('a command needs at least the program to run; the argv sequence is empty')
    return args


def _operation_error(error: Exception, context: str) -> WorkspaceError:
    if isinstance(error, AuthenticationError):
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if isinstance(error, InvalidStatus):
        if error.response.status_code == 401:
            return WorkspaceUnavailableError(_AUTH_MESSAGE)
        if error.response.status_code == 404:
            return WorkspaceUnavailableError('The requested Sprite no longer exists.')
    if isinstance(error, NotFoundError):
        return WorkspaceUnavailableError('The requested Sprite no longer exists.')
    return WorkspaceError(f'{context}: {type(error).__name__}: {error}')


class SpriteWorkspaceBackend(WorkspaceBackend):
    """A Fly.io Sprite behind the Pydantic AI `WorkspaceBackend` protocol.

    Construction does no I/O. Await `workspace` to obtain the native SDK object. Every command
    gets its own asyncio control connection, which is closed before the result is returned.
    Callers finish in-flight commands before invoking `disconnect`, which only detaches this
    backend's owned SDK client and never touches the remote Sprite. No lifecycle method is called
    automatically by core.
    """

    def __init__(
        self,
        *,
        workspace: Sprite | None = None,
        client: SpritesClient | None = None,
        ref: WorkspaceRef | None = None,
        name: str | None = None,
        token: str | None = None,
        base_url: str = 'https://api.sprites.dev',
        api_timeout: float = 30.0,
        runtime: str | None = None,
        working_dir: str | None = None,
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
        self._canonical_working_dir: str | None = None
        self._client = client
        self._owns_client = client is None

    @property
    def workspace(self) -> Awaitable[Sprite]:
        return self._get_workspace()

    async def _get_workspace(self) -> Sprite:
        if self._workspace is None:
            async with self._lock:
                if self._workspace is None:
                    self._workspace = await self._create_or_attach(self._ref)
                    self._ref = WorkspaceRef(provider='sprites', id=self._workspace.name)
        assert self._workspace is not None
        return self._workspace

    @cached_property
    def _lock(self) -> anyio.Lock:
        return anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    def _ensure_client(self) -> SpritesClient:
        if self._client is None:
            token = self._token or os.getenv('SPRITE_TOKEN')
            if not token:
                raise WorkspaceUnavailableError(_AUTH_MESSAGE)
            self._client = SpritesClient(token=token, base_url=self._base_url, timeout=self._api_timeout)
            self._owns_client = True
        return self._client

    async def _create_or_attach(self, ref: WorkspaceRef | None) -> Sprite:
        """Acquire once; a missing explicit reference never creates a replacement."""
        client = self._ensure_client()

        def acquire() -> Sprite:
            if ref is not None:
                return client.get_sprite(ref.id)
            return client.create_sprite(self._name, runtime=self._runtime)

        try:
            return await _call(acquire)
        except SpriteError as error:
            raise _operation_error(error, 'Could not acquire Sprite') from error

    async def working_dir(self) -> str:
        if self._canonical_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=30)
            directory = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(directory):
                raise WorkspaceError('Could not determine the Sprite working directory.')
            self._canonical_working_dir = directory
        return self._canonical_working_dir

    async def _cancel_remote(self, sprite: Sprite, control: str) -> Exception | None:
        connection = ControlConnection(sprite)

        async def cancel() -> None:
            await connection.connect()
            operation = await connection.start_op('exec', cmd=['python3', '-I', '-c', CANCEL, control], stdin=False)
            if await operation.wait() != 0:
                raise WorkspaceError('Sprite cancellation did not complete successfully.')

        try:
            return await cleanup_call(cancel, timeout=_CONTROL_TIMEOUT)
        finally:
            close_error = await cleanup_call(connection.close, timeout=_CONTROL_TIMEOUT)
            if close_error is not None:
                logger.warning('Could not close Sprite cancellation connection: %s', close_error)

    async def _raise_run_failure(
        self,
        error: BaseException,
        stdout: bytes,
        stderr: bytes,
        timeout: float | None,
    ) -> NoReturn:
        if isinstance(error, TimeoutError):
            raise WorkspaceTimeoutError(
                'Sprite command deadline expired.',
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                timeout=timeout,
            ) from error
        if isinstance(error, WorkspaceError):
            raise error
        if isinstance(error, Exception):
            raise _operation_error(error, 'Could not execute Sprite command') from error
        raise error

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
        args = _command_args(command, shell)

        control = f'/tmp/pydantic-ai-{uuid.uuid4().hex}'
        stdout = b''
        stderr = b''
        sprite: Sprite | None = None
        connection: ControlConnection | None = None
        operation = None
        command_error: BaseException | None = None
        close_error: Exception | None = None
        code = 0
        try:
            with anyio.fail_after(timeout):
                sprite = await self.workspace
                options = json.dumps({'args': args, 'cwd': directory, 'env': dict(env or {})})
                connection = ControlConnection(sprite)
                await connection.connect()
                operation = await connection.start_op(
                    'exec', cmd=['python3', '-I', '-c', RUN, control, options], stdin=False
                )
                code = await operation.wait()
                stdout = operation.get_stdout()
                stderr = operation.get_stderr()
                if code == -1:
                    detail = stderr.decode('utf-8', errors='replace').strip()
                    message = 'Sprite command transport closed before reporting an exit status.'
                    if detail:
                        message = f'{message} {detail}'
                    raise WorkspaceError(message) from connection.close_error
        except BaseException as error:
            command_error = error
            if operation is not None:
                stdout = operation.get_stdout()
                stderr = operation.get_stderr()
            if sprite is not None:
                cleanup_error = await self._cancel_remote(sprite, control)
                if cleanup_error is not None:
                    logger.warning('Could not confirm remote Sprite command termination: %s', cleanup_error)
        finally:
            if connection is not None:
                close_error = await cleanup_call(connection.close, timeout=_CONTROL_TIMEOUT)

        if command_error is not None:
            if close_error is not None:
                logger.warning('Could not close original Sprite command connection: %s', close_error)
            await self._raise_run_failure(command_error, stdout, stderr, timeout)

        if close_error is not None:
            if isinstance(close_error, TimeoutError):
                await raise_after_cleanup(
                    WorkspaceError('Could not close Sprite command connection within the cleanup bound.'),
                    cause=close_error,
                )
            await raise_after_cleanup(
                _operation_error(close_error, 'Could not close Sprite command connection'), cause=close_error
            )
        return CommandResult(
            exit_code=code,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
        )

    async def disconnect(self) -> None:
        """Detach local state while leaving the remote Sprite unchanged.

        A supplied SDK client is never closed. An owned client is closed with bounded cleanup;
        a successful disconnect clears local caches and later operations reattach using the saved
        ref. A failure retains the client and ref so disconnect can be retried. Finish in-flight
        commands first.
        """
        async with self._lock:
            client = self._client
            if client is None or not self._owns_client:
                self._workspace = None
                self._canonical_working_dir = None
                return
            error = await cleanup_call(lambda: _call(client.close), timeout=self._api_timeout)
            if error is not None:
                await raise_after_cleanup(
                    _operation_error(error, 'Could not disconnect from Sprite SDK client'), cause=error
                )
            self._client = None
            self._workspace = None
            self._canonical_working_dir = None
