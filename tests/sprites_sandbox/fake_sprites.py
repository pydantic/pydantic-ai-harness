"""Controllable fake for the Sprites SDK boundary.

The Sprite handles, clients, exception classes and the exec WebSocket protocol handler
(`sprites.websocket.WSCommand`) are the real ones from the installed SDK; only the network calls
are replaced. `SpriteTransport.connect` stands in for the `websockets` `connect` that `WSCommand`
calls, so the SDK builds the real exec URL, sends stdin EOF, and reads the real frames. Commands run
for real: an exec socket runs the URL's `cmd` argv in a local subprocess, in its `dir`
(`SpriteTransport.root` by default) and this process's environment, so commands share one host
directory and deadlines are real. Output streams back as STDOUT and STDERR frames, then an EXIT
frame. Closing the socket kills a command that is still running, as a positive
`max_run_after_disconnect` makes the Sprite do (the fake does it at once instead of after that time).

Deletion follows the SDK: `destroy_sprite` (and `AsyncSprite.delete()`) returns once the API accepts
the request, after which `get_sprite` raises `NotFoundError` and an exec handshake with the deleted
Sprite fails with HTTP 404 (`websockets.exceptions.InvalidStatus`, which the SDK parses into an
`APIError`).
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import IO
from urllib.parse import parse_qs, unquote, urlsplit

import anyio
from sprites import AsyncSprite, AsyncSpritesClient
from sprites.exceptions import NotFoundError
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

_STDOUT, _STDERR, _EXIT = 1, 2, 3


class FakeSocketTransport:
    def __init__(self, sprites: SpriteTransport) -> None:
        self.sprites = sprites

    def abort(self) -> None:
        self.sprites.aborted += 1


class FakeExecSocket:
    """One exec WebSocket, from the handshake to the EXIT frame."""

    def __init__(self, sprites: SpriteTransport, url: str) -> None:
        self.sprites = sprites
        self.query = parse_qs(urlsplit(url).query)
        self.transport = FakeSocketTransport(sprites)
        # Read by the SDK when the stream ends without an EXIT frame.
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.sent: list[bytes] = []
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self.process = subprocess.Popen(
            self.query['cmd'],
            cwd=self.query.get('dir', [str(sprites.root)])[0],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Its own process group, so closing the socket ends the command's children too.
            start_new_session=True,
        )
        self.thread = threading.Thread(target=self._execute)
        self.thread.start()

    def _execute(self) -> None:
        process = self.process
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=self._pump, args=(process.stdout, _STDOUT)),
            threading.Thread(target=self._pump, args=(process.stderr, _STDERR)),
        ]
        for reader in readers:
            reader.start()
        code = process.wait()
        for reader in readers:
            reader.join()
        if self.sprites.connection_dropped:
            self.close_code = 1006
            self._send(None)
        else:
            exit_code = code if self.sprites.exit_override is None else self.sprites.exit_override
            self._send(bytes([_EXIT, exit_code % 256]))

    def _pump(self, source: IO[bytes], stream: int) -> None:
        # The live Sprite delivered stderr on the stdout stream in some runs and not others; this
        # takes the worst case every time, so the backend must not rely on the stderr stream.
        del stream
        while chunk := os.read(source.fileno(), 4096):
            self._send(bytes([_STDOUT]) + chunk)
        source.close()

    def _send(self, frame: bytes | None) -> None:
        self._loop.call_soon_threadsafe(self._frames.put_nowait, frame)

    def __aiter__(self) -> FakeExecSocket:
        return self

    async def __anext__(self) -> bytes:
        frame = await self._frames.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def send(self, message: bytes) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        sprites = self.sprites
        sprites.exec_close_started.set()
        if sprites.release_exec_close is not None:
            await sprites.release_exec_close.wait()
        if sprites.exec_close_hang:
            await anyio.sleep(1)
        if sprites.exec_close_error is not None:
            raise sprites.exec_close_error
        sprites.exec_closes += 1
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
        self._frames.put_nowait(None)


class SpriteTransport:
    """SDK acquisition and exec WebSocket fake; exec commands run in local subprocesses."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.names: set[str] = set()
        self.created: list[str] = []
        self.create_started = asyncio.Event()
        self.release_create: asyncio.Event | None = None
        self.release_get: asyncio.Event | None = None
        self.clients: list[AsyncSpritesClient] = []
        self.execs: list[FakeExecSocket] = []
        self.exec_started = asyncio.Event()
        self.get_error: Exception | None = None
        self.create_error: Exception | None = None
        self.close_error: Exception | None = None
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.release_close: asyncio.Event | None = None
        self.connect_error: Exception | None = None
        # The socket closes before the command reports an exit status.
        self.connection_dropped = False
        self.exec_close_error: Exception | None = None
        self.exec_close_hang = False
        self.exec_close_started = asyncio.Event()
        self.release_exec_close: asyncio.Event | None = None
        self.exec_closes = 0
        self.aborted = 0
        self.exit_override: int | None = None

    def client(self, token: str) -> AsyncSpritesClient:
        client = AsyncSpritesClient(token=token)
        self.clients.append(client)
        return client

    async def connect(self, url: str, **kwargs: object) -> FakeExecSocket:
        if self.connect_error is not None:
            raise self.connect_error
        # /v1/sprites/{name}/exec
        name = unquote(urlsplit(url).path.split('/')[3])
        if name not in self.names:
            raise InvalidStatus(Response(404, 'Not Found', Headers()))
        socket = FakeExecSocket(self, url)
        self.execs.append(socket)
        self.exec_started.set()
        return socket

    async def get(self, client: AsyncSpritesClient, name: str) -> AsyncSprite:
        if self.get_error is not None:
            raise self.get_error
        if self.release_get is not None:
            await self.release_get.wait()
        if name not in self.names:
            raise NotFoundError(name)
        return AsyncSprite(name, client)

    async def create(self, client: AsyncSpritesClient, name: str, *, runtime: str | None) -> AsyncSprite:
        if self.create_error is not None:
            raise self.create_error
        self.names.add(name)
        self.created.append(name)
        self.create_started.set()
        if self.release_create is not None:
            await self.release_create.wait()
        return AsyncSprite(name, client)

    async def destroy(self, client: AsyncSpritesClient, name: str) -> None:
        self.names.discard(name)

    async def close(self, client: AsyncSpritesClient) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.release_close is not None:
            await self.release_close.wait()
        if self.close_error is not None:
            error = self.close_error
            self.close_error = None
            raise error
