"""Controllable fake for the Sprites SDK boundary.

The Sprite handles, clients and exception classes are the real ones from the installed SDK; only
the network calls are replaced. Commands run for real: an exec operation runs its argv in a local
subprocess, in its `dir` (`SpriteTransport.root` by default) and this process's environment, so
commands share one host directory and deadlines are real. A command that
cannot start completes without an exit status and with the error on stderr, as `op.error` does, and
`signal()` reaches only the command's own process.

Deletion follows the SDK: `destroy_sprite` (and `AsyncSprite.delete()`) returns once the API accepts
the request, after which `get_sprite` raises `NotFoundError` and a control connection to the
deleted Sprite fails its WebSocket handshake with HTTP 404 (`websockets.exceptions.InvalidStatus`).
"""

from __future__ import annotations

import asyncio
import signal
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO

import anyio
from sprites import AsyncSprite, AsyncSpritesClient
from sprites.exceptions import NotFoundError
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response


class FakeOperation:
    def __init__(self, transport: SpriteTransport, cmd: list[str], dir: str | None) -> None:
        self.transport = transport
        self.closed = False
        self.stdout = b''
        self.stderr = b''
        self.process: subprocess.Popen[bytes] | None = None
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=dir or transport.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            self.stderr = f'Error: {error}\n'.encode()
        self._task = asyncio.create_task(asyncio.to_thread(self._execute))

    def _execute(self) -> int:
        if (process := self.process) is None:
            return -1
        assert process.stdout is not None and process.stderr is not None
        stdout = bytearray()
        stderr = bytearray()

        def read_stream(source: BinaryIO, target: bytearray, output: str) -> None:
            while chunk := source.read(1):
                target.extend(chunk)
                setattr(self, output, bytes(target))
            source.close()

        readers = [
            threading.Thread(target=read_stream, args=(process.stdout, stdout, 'stdout')),
            threading.Thread(target=read_stream, args=(process.stderr, stderr, 'stderr')),
        ]
        for reader in readers:
            reader.start()
        code = process.wait(timeout=10)
        for reader in readers:
            reader.join()
        self.stdout, self.stderr = bytes(stdout), bytes(stderr)
        return code

    async def wait(self) -> int:
        code = await asyncio.shield(self._task)
        self.closed = True
        if self.transport.connection_dropped:
            return -1
        return self.transport.exit_override if self.transport.exit_override is not None else code

    async def signal(self, sig: str) -> None:
        self.transport.signals.append(sig)
        if self.transport.signal_error is not None:
            raise self.transport.signal_error
        assert self.process is not None
        self.process.send_signal(getattr(signal, f'SIG{sig}'))

    def get_stdout(self) -> bytes:
        return self.stdout

    def get_stderr(self) -> bytes:
        return self.stderr


class FakeSocketTransport:
    def __init__(self, transport: SpriteTransport) -> None:
        self.transport = transport

    def abort(self) -> None:
        self.transport.aborted += 1


class FakeSocket:
    def __init__(self, transport: SpriteTransport) -> None:
        self.transport = FakeSocketTransport(transport)


class FakeControlConnection:
    transport: SpriteTransport

    def __init__(self, sprite: AsyncSprite) -> None:
        self.sprite = sprite
        # The SDK's read loop stores the exception that ended the connection here.
        self.close_error = self.transport.connection_lost
        self.closed = self.transport.connection_dropped
        self.ws: FakeSocket | None = None

    async def connect(self) -> None:
        if self.transport.connect_error is not None:
            raise self.transport.connect_error
        if self.sprite.name not in self.transport.names:
            raise InvalidStatus(Response(404, 'Not Found', Headers()))
        self.ws = FakeSocket(self.transport)

    async def start_op(self, op: str, *, cmd: list[str], dir: str | None, stdin: bool) -> FakeOperation:
        assert op == 'exec'
        assert stdin is False
        operation = FakeOperation(self.transport, cmd, dir)
        self.transport.operations.append(operation)
        self.transport.exec_started.set()
        return operation

    async def close(self) -> None:
        self.transport.control_close_started.set()
        if self.transport.release_control_close is not None:
            await self.transport.release_control_close.wait()
        if self.transport.control_close_hang:
            await anyio.sleep(1)
        if self.transport.control_close_error is not None:
            raise self.transport.control_close_error
        self.closed = True
        self.transport.control_closes += 1


class SpriteTransport:
    """SDK acquisition fake; exec operations run in local subprocesses."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.names: set[str] = set()
        self.created: list[str] = []
        self.create_started = asyncio.Event()
        self.release_create: asyncio.Event | None = None
        self.release_get: asyncio.Event | None = None
        self.clients: list[AsyncSpritesClient] = []
        self.operations: list[FakeOperation] = []
        self.exec_started = asyncio.Event()
        self.get_error: Exception | None = None
        self.close_error: Exception | None = None
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.release_close: asyncio.Event | None = None
        self.connect_error: Exception | None = None
        self.control_close_error: Exception | None = None
        self.connection_lost: Exception | None = None
        # The connection closes before the command reports an exit status.
        self.connection_dropped = False
        self.control_close_hang = False
        self.control_close_started = asyncio.Event()
        self.release_control_close: asyncio.Event | None = None
        self.control_closes = 0
        self.aborted = 0
        self.exit_override: int | None = None
        self.signals: list[str] = []
        self.signal_error: Exception | None = None

    def client(self, token: str, base_url: str, timeout: float) -> AsyncSpritesClient:
        client = AsyncSpritesClient(token=token, base_url=base_url, timeout=timeout)
        self.clients.append(client)
        return client

    async def get(self, client: AsyncSpritesClient, name: str) -> AsyncSprite:
        if self.get_error is not None:
            raise self.get_error
        if self.release_get is not None:
            await self.release_get.wait()
        if name not in self.names:
            raise NotFoundError(name)
        return AsyncSprite(name, client)

    async def create(self, client: AsyncSpritesClient, name: str, *, runtime: str | None) -> AsyncSprite:
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
