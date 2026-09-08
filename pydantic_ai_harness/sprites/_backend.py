"""Fly.io Sprites backend for the core sandbox protocol.

The command transport uses the public asynchronous `ControlConnection` API from
sprites-py 0.6. The integration is asyncio-only because that API owns asyncio
tasks for each control connection.
"""

from __future__ import annotations

import json
import logging
import math
import os
import posixpath
import uuid
from collections.abc import Callable, Mapping
from functools import partial
from typing import NoReturn, TypeVar

import anyio
import anyio.to_thread
from pydantic_ai.sandboxes import (
    CommandResult,
    LazySandbox,
    SandboxBackend,
    SandboxCommand,
    SandboxError,
    SandboxRef,
    SandboxTimeoutError,
    SandboxUnavailableError,
)

from pydantic_ai_harness._sandbox_provider import absolute_path, cleanup_call, raise_after_cleanup
from pydantic_ai_harness.sprites._process import CANCEL, RUN

try:
    from sprites import Sprite, SpritesClient
    from sprites.control import ControlConnection
    from sprites.exceptions import AuthenticationError, NotFoundError, SpriteError
    from websockets.exceptions import InvalidStatus
except ImportError as exc:
    raise ImportError('Install `pydantic-ai-harness[sprites]` to use SpriteSandbox.') from exc

logger = logging.getLogger(__name__)
T = TypeVar('T')
_CONTROL_TIMEOUT = 6.0


class SpriteSandboxError(SandboxError):
    """A Fly.io Sprites operation failed."""


class SpriteSandboxAuthError(SpriteSandboxError, SandboxUnavailableError):
    """The Sprites token is missing or was rejected."""


class SpriteSandboxUnavailableError(SpriteSandboxError, SandboxUnavailableError):
    """The requested Sprite no longer exists."""


async def _call(call: Callable[[], T]) -> T:
    return await anyio.to_thread.run_sync(call, abandon_on_cancel=True)


def _command_args(command: SandboxCommand, shell: bool) -> list[str]:
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


class SpriteSandboxBackend(LazySandbox[Sprite], SandboxBackend):
    """A lazy Sprite with commands and core-derived filesystem operations.

    Construction does no I/O. Await `sandbox` to obtain the native SDK object. Every command
    gets its own asyncio control connection, which is closed before the result is returned.
    Callers finish in-flight commands before invoking lifecycle methods. `destroy` deletes a
    saved remote Sprite, while `disconnect` only detaches this backend's owned SDK client. Neither
    method creates a Sprite, and no lifecycle method is called automatically by core.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        ref: SandboxRef | None = None,
        name: str | None = None,
        base_url: str = 'https://api.sprites.dev',
        api_timeout: float = 30.0,
        runtime: str | None = None,
        working_dir: str | None = None,
        client: SpritesClient | None = None,
    ) -> None:
        super().__init__()
        self._token = token
        self._ref = ref
        self._name = name or f'pydantic-ai-{uuid.uuid4().hex}'
        self._base_url = base_url
        if not math.isfinite(api_timeout) or api_timeout <= 0:
            raise ValueError('api_timeout must be positive and finite.')
        self._api_timeout = api_timeout
        self._runtime = runtime
        self._working_dir = absolute_path('working_dir', working_dir)
        self._canonical_working_dir: str | None = None
        self._client = client
        self._client_owned = client is None

    @property
    def ref(self) -> SandboxRef | None:
        """Provider name, known immediately for an explicit reference."""
        return self._ref

    def _ensure_client(self) -> SpritesClient:
        if self._client is None:
            token = self._token or os.getenv('SPRITE_TOKEN')
            if not token:
                raise SpriteSandboxAuthError('Set SPRITE_TOKEN or pass token= to SpriteSandbox.')
            self._client = SpritesClient(token=token, base_url=self._base_url, timeout=self._api_timeout)
            self._client_owned = True
        return self._client

    async def create_or_attach(self) -> Sprite:
        """Acquire once; a missing explicit reference never creates a replacement."""
        client = self._ensure_client()

        def acquire() -> Sprite:
            if self._ref is not None:
                return client.get_sprite(self._ref.sandbox_id)
            try:
                return client.get_sprite(self._name)
            except NotFoundError:
                pass
            try:
                return client.create_sprite(self._name, runtime=self._runtime)
            except AuthenticationError:
                raise
            except SpriteError as creation_error:
                try:
                    return client.get_sprite(self._name)
                except NotFoundError:
                    raise creation_error

        try:
            sprite = await _call(acquire)
        except SpriteError as error:
            raise self._error(error) from error
        self._ref = SandboxRef(sandbox_id=sprite.name)
        return sprite

    @staticmethod
    def _error(error: Exception, context: str = 'Sprites operation failed') -> SpriteSandboxError:
        if isinstance(error, AuthenticationError):
            return SpriteSandboxAuthError('Sprites rejected the credentials; check SPRITE_TOKEN or token=.')
        if isinstance(error, NotFoundError):
            return SpriteSandboxUnavailableError('The requested Sprite no longer exists.')
        if isinstance(error, InvalidStatus):
            if error.response.status_code == 401:
                return SpriteSandboxAuthError('Sprites rejected the credentials; check SPRITE_TOKEN or token=.')
            if error.response.status_code == 404:
                return SpriteSandboxUnavailableError('The requested Sprite no longer exists.')
        return SpriteSandboxError(f'{context}: {type(error).__name__}: {error}')

    async def working_dir(self) -> str:
        if self._canonical_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=30)
            directory = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(directory):
                raise SpriteSandboxError('Could not determine the Sprite working directory.')
            self._canonical_working_dir = directory
        return self._canonical_working_dir

    async def _cancel_remote(self, sprite: Sprite, control: str) -> Exception | None:
        connection = ControlConnection(sprite)

        async def cancel() -> None:
            await connection.connect()
            operation = await connection.start_op('exec', cmd=['python3', '-I', '-c', CANCEL, control], stdin=False)
            if await operation.wait() != 0:
                raise SpriteSandboxError('Sprite cancellation did not complete successfully.')

        try:
            return await cleanup_call(cancel, timeout=_CONTROL_TIMEOUT)
        finally:
            close_error = await cleanup_call(connection.close, timeout=_CONTROL_TIMEOUT)
            if close_error is not None:
                logger.warning('Could not close Sprite cancellation connection: %s', close_error)

    async def _raise_run_failure(
        self,
        error: BaseException,
        sprite: Sprite | None,
        control: str,
        stdout: bytes,
        stderr: bytes,
        timeout: float | None,
    ) -> NoReturn:
        if sprite is not None:
            cleanup_error = await self._cancel_remote(sprite, control)
            if cleanup_error is not None:
                logger.warning('Could not confirm remote Sprite command termination: %s', cleanup_error)
        if isinstance(error, TimeoutError):
            raise SandboxTimeoutError(
                'Sprite command deadline expired.',
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                timeout=timeout,
            ) from error
        if isinstance(error, SandboxError):
            raise error
        if isinstance(error, Exception):
            raise self._error(error, 'Could not execute Sprite command') from error
        raise error

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('timeout must be positive and finite or None.')
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
                sprite = await self.sandbox
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
                    raise SpriteSandboxError(message) from connection.close_error
        except BaseException as error:
            command_error = error
            if operation is not None:
                stdout = operation.get_stdout()
                stderr = operation.get_stderr()
        finally:
            if connection is not None:
                close_error = await cleanup_call(connection.close, timeout=_CONTROL_TIMEOUT)

        if command_error is not None:
            await self._raise_run_failure(command_error, sprite, control, stdout, stderr, timeout)

        if close_error is not None:
            if isinstance(close_error, TimeoutError):
                await raise_after_cleanup(
                    SpriteSandboxError('Could not close Sprite command connection within the cleanup bound.'),
                    cause=close_error,
                )
            await raise_after_cleanup(
                self._error(close_error, 'Could not close Sprite command connection'), cause=close_error
            )
        return CommandResult(
            exit_code=code,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
        )

    async def destroy(self) -> None:
        """Delete the saved remote Sprite, including one attached before first use.

        With no saved ref this is a no-op and does not create an SDK client. A saved ref is
        deleted directly by ID without lookup or creation; an already missing Sprite succeeds.
        The ref is retained after failure for retry. Successful destruction clears local caches.
        Finish in-flight commands first.
        """
        async with self._lock:
            if self._ref is None:
                return
            client = self._ensure_client()
            ref = self._ref
            error = await cleanup_call(
                lambda: _call(partial(client.destroy_sprite, ref.sandbox_id)), timeout=self._api_timeout
            )
            if isinstance(error, NotFoundError):
                error = None
            if error is not None:
                await raise_after_cleanup(
                    self._error(error, f'Could not destroy Sprite {ref.sandbox_id!r}'), cause=error
                )
            self._live = None
            self._canonical_working_dir = None

    async def disconnect(self) -> None:
        """Detach local state while leaving the remote Sprite unchanged.

        An injected SDK client is never closed. An owned client is closed with bounded cleanup;
        successful disconnect clears local caches and later operations reattach using the saved ref.
        Failures retain the client and ref so disconnect can be retried. Finish in-flight commands first.
        """
        async with self._lock:
            client = self._client
            if client is None:
                self._live = None
                self._canonical_working_dir = None
                return
            if not self._client_owned:
                self._live = None
                self._canonical_working_dir = None
                return
            error = await cleanup_call(lambda: _call(client.close), timeout=self._api_timeout)
            if error is not None:
                await raise_after_cleanup(
                    self._error(error, 'Could not disconnect from Sprite SDK client'), cause=error
                )
            self._client = None
            self._live = None
            self._canonical_working_dir = None
