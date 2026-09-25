"""Browser authorization-code logins: a one-shot `127.0.0.1` listener raced against a pasted callback URL.

OpenRouter and Google share this; each supplies its authorization URL and code exchange.
"""

import asyncio
import base64
import hashlib
import secrets
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
from urllib.parse import parse_qs, urlparse

from anyio import fail_after
from pydantic_ai.exceptions import UserError
from rich.console import Console

from . import theme
from .auth import ReadLine, read_line

ResultT = TypeVar('ResultT')


def pkce() -> tuple[str, str]:
    """A fresh S256 code verifier and its challenge."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).rstrip(b'=').decode()
    return verifier, challenge


@dataclass(frozen=True, kw_only=True)
class Provider:
    """What one login names and checks."""

    name: str
    retry: str
    """How to try again, appended to the timeout message."""
    fallback: str
    """What to do instead when no listener can start."""
    state: str | None = None
    """The OAuth `state` a callback URL must carry; bare pasted codes cannot carry one."""


def authorization_code(*, text: str, provider: Provider) -> str:
    """Accept a pasted callback URL or a bare authorization code without echoing it."""
    parsed = urlparse(text)
    if parsed.scheme or parsed.netloc or text.startswith('/'):
        params = parse_qs(parsed.query)
        if provider.state is not None and params.get('state') != [provider.state]:
            raise UserError(f'That URL belongs to a different {provider.name} login attempt.')
        if 'error' in params:
            raise UserError(f'{provider.name} authorization was denied. Try connecting again.')
        codes = params.get('code', [])
        if len(codes) != 1 or not codes[0].strip():
            raise UserError('The callback URL must contain one authorization code.')
        return codes[0]
    if not text.strip():
        raise UserError('An authorization code is required.')
    return text.strip()


class LoopbackLogin:
    """Own one browser login, including its listener and terminal prompt."""

    def __init__(
        self,
        provider: Provider,
        *,
        console: Console,
        read_line: ReadLine = read_line,
        open_browser: Callable[[str], bool] = webbrowser.open,
        timeout: float = 300,
    ) -> None:
        """Inject the terminal boundaries without opening a listener yet."""
        self.provider = provider
        self.console = console
        self.read_line = read_line
        self.open_browser = open_browser
        self.timeout = timeout

    async def run(
        self, *, authorize_url: Callable[[str], str], exchange: Callable[[str, str], Awaitable[ResultT]]
    ) -> ResultT:
        """Show `authorize_url(redirect_uri)`, then `exchange(code, redirect_uri)` the first code to arrive.

        The listener stays up during the exchange so late browser requests get an answer.
        Cancellation or failure leaves existing credentials alone; nothing is saved here.
        """
        name = self.provider.name
        code: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        handlers: set[asyncio.Task[None]] = set()

        def connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.create_task(self._callback(reader=reader, writer=writer, code=code))
            handlers.add(task)
            task.add_done_callback(handlers.discard)

        try:
            server = await asyncio.start_server(connected, '127.0.0.1', 0)
        except OSError:
            raise UserError(f'Could not start the {name} callback listener. {self.provider.fallback}') from None
        redirect_uri = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/callback'
        url = authorize_url(redirect_uri)
        paste: asyncio.Task[str] | None = None
        try:
            async with server:
                with fail_after(self.timeout):
                    self.console.print(
                        f'Sign in to {name} in your browser. Waiting up to five minutes.',
                        style=theme.color(theme.INFO),
                    )
                    self.console.print(url, markup=False, highlight=False)
                    try:
                        opened = await asyncio.to_thread(self.open_browser, url)
                    except webbrowser.Error:
                        opened = False
                    if not opened:
                        self.console.print('Open the URL above manually.', style=theme.color(theme.WARNING))
                    paste = asyncio.create_task(self._paste())
                    done, _ = await asyncio.wait({code, paste}, return_when=asyncio.FIRST_COMPLETED)
                    received = code.result() if code in done else paste.result()
                    return await exchange(received, redirect_uri)
        except TimeoutError:
            raise UserError(f'{name} login timed out. {self.provider.retry}') from None
        finally:
            code.cancel()
            tasks: list[asyncio.Task[None] | asyncio.Task[str]] = [*handlers]
            if paste is not None:
                tasks.append(paste)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
            if paste is not None:
                await asyncio.gather(paste, return_exceptions=True)
            await asyncio.gather(code, return_exceptions=True)

    async def _paste(self) -> str:
        try:
            while not (
                text := (await self.read_line('Finish in the browser, or paste its callback URL or code: ')).strip()
            ):
                pass
        except (EOFError, KeyboardInterrupt):
            raise UserError(f'{self.provider.name} login cancelled.') from None
        return authorization_code(text=text, provider=self.provider)

    def _denied(self, target: str) -> bool:
        """An error callback ends the login only when it carries this login's state."""
        params = parse_qs(urlparse(target).query)
        state = self.provider.state
        return 'error' in params and (state is None or params.get('state') == [state])

    async def _callback(
        self, *, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, code: asyncio.Future[str]
    ) -> None:
        try:
            with fail_after(10):
                line = (await reader.readline()).decode('ascii', errors='replace').split()
                status, message = '404 Not Found', 'Callback endpoint not found.'
                if len(line) == 3 and line[0] == 'GET' and urlparse(line[1]).path == '/callback':
                    try:
                        received = authorization_code(text=line[1], provider=self.provider)
                    except UserError as exc:
                        if self._denied(line[1]) and not code.done():
                            code.set_exception(exc)
                        status, message = '400 Bad Request', 'Authorization failed. Return to CLAI and try again.'
                    else:
                        status, message = '200 OK', 'Authorization received. Return to CLAI to finish connecting.'
                        if not code.done():
                            code.set_result(received)
                body = message.encode()
                writer.write(
                    (
                        f'HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\n'
                        'Connection: close\r\n\r\n'
                    ).encode()
                    + body
                )
                await writer.drain()
        except (TimeoutError, ConnectionError, ValueError):
            pass  # Malformed or disconnected local clients must not end a login.
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
