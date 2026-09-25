"""Fly.io Sprites backend for Pydantic AI's `WorkspaceBackend` protocol.

External assumptions last verified 2026-09-25 against sprites-py 0.7.0 source, the Sprites API
docs, and (2026-09-15) a local WebSocket transport probe, with no live cloud calls:

* `AsyncSpritesClient` requires its token as an argument and reads no environment variable; sprite
  creation uses the SDK's fixed 120-second request timeout, while `aclose` closes the local async
  HTTP client and control pools:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/async_client.py
* `create_sprite` and `get_sprite` raise `AuthenticationError` (401), `NotFoundError` (404),
  `NetworkError` (transport), and a plain `SpriteError` for any other HTTP failure:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/client.py
* Commands run over the documented exec WebSocket through the SDK's `WSCommand`, the SDK's own
  default path. It sends stdin EOF when no stdin is given, reads the exit status from the binary
  EXIT frame or the JSON `exit` message, raises `NetworkError` when the socket closes before either,
  and turns a failed handshake into a parsed `APIError` carrying the HTTP status. The working
  directory goes in the `dir` query parameter as the SDK sends it; the API page lists `dir` only
  for the HTTP exec endpoint:
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/websocket.py
* The multiplexed control protocol (`sprites.control`, used only in the SDK's opt-in control mode)
  is not used: in 0.7.0 its `op.complete` handler overwrites the exit status from the EXIT frame
  with the message's own `exitCode`, which defaults to 0, so every command reported success
  (observed against real Sprites on 2026-09-25):
  https://github.com/superfly/sprites-py/blob/v0.7.0/src/sprites/control.py
* The exec API documents that a set `env` replaces the default environment, so the backend does
  not pass `env`; it runs the command under the POSIX `env` utility instead, which adds the
  variables to the Sprite's own environment:
  https://sprites.dev/api/sprites/exec
* A disconnect does not stop a non-TTY command at once; it keeps running for
  `max_run_after_disconnect` (10 seconds by default; `0` means no limit, as the TTY default shows).
  The backend asks for one second, so closing the socket on a timeout or cancellation ends the
  command shortly after:
  https://sprites.dev/api/sprites/exec
* Provider retention is separate from local client disconnect:
  https://docs.sprites.dev/concepts/lifecycle/

Re-check these sources, the installed signatures, and the local transport probe before changing
lifecycle or command transport behavior. The integration uses the SDK's native asyncio client.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import posixpath
import re
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

try:
    from sprites import AsyncSprite, AsyncSpritesClient
    from sprites.exceptions import APIError, AuthenticationError, NetworkError, NotFoundError, SpriteError
    from sprites.exceptions import TimeoutError as SpriteTimeoutError
    from sprites.websocket import WSCommand
except ImportError as exc:  # pragma: no cover - exercised by the isolated missing-extra test
    raise ImportError('Install `pydantic-ai-harness[sprites]` to use SpritesSandbox.') from exc

logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_CLOSE_TIMEOUT = 6.0
# Above the SDK's fixed 120-second creation request timeout, so the SDK's own error wins when it fires.
_ACQUIRE_TIMEOUT = 150.0
_AUTH_MESSAGE = (
    'Sprites rejected the credentials. Set SPRITE_TOKEN, or pass a configured `AsyncSpritesClient` as `client=`.'
)


async def _cleanup_call(call: Callable[[], Awaitable[object]], *, timeout: float) -> Exception | None:
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


async def _run_to_completion(call: Callable[[], Awaitable[_T]]) -> _T:
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


class _ExecCommand(WSCommand):
    """The SDK's exec WebSocket command, asking the Sprite to end the command soon after a disconnect."""

    def _build_websocket_url(self) -> str:
        # The SDK always asks for stdin, and with it the live Sprite sent stderr on the stdout stream;
        # commands here never read stdin, so it is turned off and the stream kept plain (no TTY).
        # A non-TTY command keeps running for 10 seconds after its socket closes unless told otherwise,
        # and `0` means no limit. One second makes closing the socket on a timeout or cancellation
        # stop the command.
        url = super()._build_websocket_url()
        assert url.endswith('&stdin=true'), url
        return f'{url.removesuffix("&stdin=true")}&stdin=false&tty=false&max_run_after_disconnect=1s'


async def _close_command(command: WSCommand) -> None:
    """Close an exec WebSocket; if that fails, abort its socket so nothing is left open.

    Finished even when the caller (a cancelled command) is cancelled meanwhile.
    """
    error = await _run_to_completion(lambda: _cleanup_call(command.close, timeout=_CLOSE_TIMEOUT))
    if error is not None:
        logger.warning('Could not close a Sprite exec connection, aborting it: %r', error)
        # `close` has nothing to fail on before `start` opened the socket.
        if command.ws is not None:  # pragma: no branch
            command.ws.transport.abort()


def _map_error(error: Exception, sprite_name: str | None) -> WorkspaceError | None:
    """Translate a Sprites failure, or return `None` for one that propagates as is.

    `sprite_name` is `None` while the Sprite is being created. Rejected credentials, a refused
    creation, and a missing Sprite end the run; any other request the API refused is a failed
    operation. Transport failures (`NetworkError`, a socket that closed before the exit status),
    rate limits, server errors, and anything unknown propagate unchanged, for durable engines to retry.
    """
    # The exec handshake reports its HTTP status as an `APIError`.
    status = error.status_code if isinstance(error, APIError) else None
    if isinstance(error, AuthenticationError) or status == 401:
        return WorkspaceUnavailableError(_AUTH_MESSAGE)
    if not isinstance(error, SpriteError) or isinstance(error, (NetworkError, SpriteTimeoutError)):
        return None
    # sprites-py reports every other HTTP failure, a rate limit or a server error included, as a plain
    # `SpriteError` whose message names the status.
    if re.search(r'\(status (429|5\d\d)\)', str(error)) or (status is not None and (status == 429 or status >= 500)):
        return None
    if sprite_name is None:
        # An unknown runtime or a bad request fails the same way on every retry.
        return WorkspaceUnavailableError(f'Could not start Sprites sandbox: {error}')
    if isinstance(error, NotFoundError) or status == 404:
        return WorkspaceUnavailableError(
            f'The Sprite {sprite_name!r} no longer exists: it was deleted. '
            "Pass `workspace='new'` to start a fresh sandbox."
        )
    return WorkspaceError(f'Sprites refused the request: {error}')


class SpritesSandboxBackend(WorkspaceBackend, SupportsCommands):
    """A Fly.io Sprite behind the Pydantic AI `WorkspaceBackend` protocol.

    Construction does no I/O. The typed `sprites.AsyncSprite` is available through `get_client()`.
    Without `client=`, the backend creates an `AsyncSpritesClient` from `SPRITE_TOKEN` on first use
    and closes it in `aclose()`.
    The backend does not delete the Sprite; that is the application's job, through the native
    handle. Commands run under `/bin/sh -c` with `shell=True`, in the Sprite's own environment
    plus `env`.
    """

    def __init__(
        self,
        *,
        workspace: AsyncSprite | None = None,
        client: AsyncSpritesClient | None = None,
        ref: WorkspaceRef | None = None,
        runtime: str | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if ref is not None and ref.provider != 'sprites':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'sprites'")
        if workspace is not None and ref is not None:
            raise ValueError('pass either `workspace` or `ref`, not both')
        self._sandbox = workspace
        self._ref = ref if workspace is None else WorkspaceRef(provider='sprites', id=workspace.name)
        self._new_sprite_name = f'pydantic-ai-{uuid.uuid4().hex}'
        self._runtime = runtime
        self._working_dir = absolute_path('working_dir', working_dir)
        self._env = dict(env or {})
        # `working_dir` as the Sprite resolves it (`pwd -P`): the protocol reports a canonical absolute path.
        self._resolved_working_dir: str | None = None
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
            if (workspace := self._sandbox) is not None:
                return workspace

            client = self._client
            if client is None:
                # The SDK reads no environment variable, so the token comes from `SPRITE_TOKEN` here.
                token = os.getenv('SPRITE_TOKEN')
                if not token:
                    raise WorkspaceUnavailableError(_AUTH_MESSAGE)
                client = AsyncSpritesClient(token=token)
                self._client = client

            ref = self._ref

            async def acquire() -> AsyncSprite:
                with anyio.move_on_after(_ACQUIRE_TIMEOUT):
                    if ref is not None:
                        workspace = await client.get_sprite(ref.id)
                    else:
                        workspace = await client.create_sprite(self._new_sprite_name, runtime=self._runtime)
                    # Recorded as soon as the SDK returns, so a cancelled caller still leaves it named.
                    self._sandbox = workspace
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
                return await (acquire() if ref is not None else _run_to_completion(acquire))
            except SpriteError as error:
                if (mapped := _map_error(error, None if ref is None else ref.id)) is None:
                    raise
                raise mapped from error

    async def aclose(self) -> None:
        """Close the `AsyncSpritesClient` this backend created, if it created one.

        The Sprite is untouched: the next operation opens a fresh client and reattaches by `ref`.
        A caller-supplied `client=` or `workspace=` handle is never closed. `SpritesSandbox`
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
                error = await _cleanup_call(client.aclose, timeout=_CLOSE_TIMEOUT)
                if error is not None:
                    # Kept, so a later `aclose()` tries again.
                    logger.warning('Could not close Sprites SDK client: %r', error)
                else:
                    self._client = None
                    self._sandbox = None

        # Finished even when the caller (a run being cancelled) is cancelled meanwhile.
        await _run_to_completion(close)

    async def working_dir(self) -> str:
        if self._resolved_working_dir is None:
            result = await self.run(['pwd', '-P'], timeout=30)
            directory = result.stdout.removesuffix('\n')
            if result.exit_code != 0 or not posixpath.isabs(directory):
                raise WorkspaceError('Could not determine the Sprite working directory.')
            self._resolved_working_dir = directory
        return self._resolved_working_dir

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
        marker = f'pydantic-ai-end-{uuid.uuid4().hex}'
        # `env` runs inside the wrapper: a `sh` such as dash drops variables whose names are not shell
        # identifiers from the environment it passes on.
        args = _ending_with(marker, _with_env(command_argv(command, shell), {**self._env, **(env or {})}))

        # Acquiring the Sprite has its own bound; the deadline is the command's alone.
        sprite = await self.get_client()
        exec_command = _ExecCommand(sprite.command(*args, cwd=directory))
        deadline = anyio.CancelScope(deadline=math.inf if timeout is None else anyio.current_time() + timeout)
        code = -1
        try:
            with deadline:
                await exec_command.start()
                code = await exec_command.wait()
            if timeout is not None and deadline.cancelled_caught:
                partial = _split_output(exec_command.get_stdout(), exec_command.get_stderr(), marker)
                raise WorkspaceTimeoutError(
                    f'Command timed out after {timeout:g} seconds',
                    stdout=partial[0],
                    stderr=partial[1],
                    timeout=timeout,
                )
        except BaseException as error:
            # On a timeout or a cancellation, closing the socket is what ends the command in the Sprite.
            await _close_command(exec_command)
            if isinstance(error, Exception) and (mapped := _map_error(error, sprite.name)) is not None:
                raise mapped from error
            raise
        await _close_command(exec_command)
        stdout, stderr = _split_output(exec_command.get_stdout(), exec_command.get_stderr(), marker)
        return CommandResult(exit_code=code, stdout=stdout, stderr=stderr)


def _ending_with(marker: str, args: list[str]) -> list[str]:
    """`args` run under a `sh` that reports their stdout, a `marker` line, then their stderr, all on stdout.

    The live Sprite's stderr stream is not dependable: the same command's stderr arrived on the stderr
    stream in one run and on the stdout stream in the next, whole lines included (2026-09-25), while
    stdout arrived intact every time. So the command's stderr goes to a temporary file in the Sprite,
    printed on stdout after the marker line, and `_split_output` separates the two again. The exit
    status is the command's.
    """
    script = (
        'err=$(mktemp) || exit 125; '
        f'"$@" 2>"$err"; status=$?; printf "\\n%s\\n" {marker}; cat "$err"; rm -f "$err"; exit "$status"'
    )
    return ['sh', '-c', script, 'sh', *args]


def _split_output(stdout: bytes, stderr: bytes, marker: str) -> tuple[str, str]:
    """The command's stdout and stderr from what `_ending_with` printed.

    Without the marker line (the command was stopped before it finished), stdout is all the output
    there is, and the command's stderr stayed in the Sprite. Anything on the stderr stream itself came
    from the wrapper and is kept.
    """
    head, found, tail = _decode(stdout).partition(f'\n{marker}\n')
    return head, (tail + _decode(stderr)) if found else _decode(stderr)


def _with_env(args: list[str], env: dict[str, str]) -> list[str]:
    """`args` run under the POSIX `env` utility, which adds `env` to the Sprite's own environment."""
    if not env:
        return args
    for key, value in env.items():
        if not key or '=' in key or '\0' in key or '\0' in value:
            raise ValueError(
                f'illegal environment variable {key!r}: a name is non-empty without "=" or NUL, a value without NUL'
            )
    # `--` ends `env`'s options, so a name starting with `-` is not read as one.
    return ['env', '--', *(f'{key}={value}' for key, value in env.items()), *args]


def _decode(data: bytes) -> str:
    return data.decode('utf-8', errors='replace')
