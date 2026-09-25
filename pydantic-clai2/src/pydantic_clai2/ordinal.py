"""The built-in `ordinal` plugin: harness `Ordinal`, signed in with `ORDINAL_ACCESS_TOKEN` or through the browser.

Harness `Ordinal(auth='oauth')` keeps its tokens in memory, so every launch would sign in again. Without a token,
this plugin gives `Ordinal` its own transport whose OAuth tokens live in CLAI's keyring, as `/mcp` servers' do.
"""

import os
import sys
from typing import Generic

from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.ordinal import Ordinal

from . import theme
from .commands import Command
from .mcp import TokenStore, http_client, sign_in
from .plugins import DepsT, PluginHost

URL = 'https://app.tryordinal.com/mcp'
"""Harness `Ordinal`'s endpoint, which this plugin needs to build its own signed-in transport."""
TOKEN_ENV = 'ORDINAL_ACCESS_TOKEN'
TOKENS = 'plugin_ordinal'
"""The keyring entry (`mcp-plugin_ordinal`). `/mcp` server names cannot contain `_`, so none shares it."""
USAGE = 'Usage: /ordinal [logout]'


class BrowserSignIn(Generic[DepsT]):
    """Hands each run the `Ordinal` for the current browser sign-in.

    Once connected, FastMCP's `OAuth` keeps the access token in memory, so clearing the keyring alone would leave
    this session signed in. `logout` therefore replaces the whole `Ordinal`, transport and sign-in handler included.
    """

    def __init__(self, tokens: TokenStore) -> None:
        """Connect through `tokens`' keyring entry."""
        self.tokens = tokens
        self.ordinal = self._connect()

    def __call__(self, ctx: RunContext[DepsT]) -> Ordinal[DepsT]:
        """The capability for this run."""
        return self.ordinal

    def logout(self) -> None:
        """Forget the saved tokens and the in-memory ones, so the next run opens the browser."""
        self.tokens.forget()
        self.ordinal = self._connect()

    def _connect(self) -> Ordinal[DepsT]:
        auth = sign_in(self.tokens.name)
        return Ordinal[DepsT](client=StreamableHttpTransport(url=URL, auth=auth, httpx_client_factory=http_client))


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Ordinal`, or refuse to load when no token, saved sign-in, or terminal for a browser sign-in exists."""
    tokens = TokenStore(TOKENS)
    browser: BrowserSignIn[DepsT] | None = None
    if os.environ.get(TOKEN_ENV):
        host.add(Ordinal[DepsT]())
    else:
        signed_in = tokens.signed_in()
        if not signed_in and not sys.stdin.isatty():
            raise UserError(f'Set `{TOKEN_ENV}`, or start clai2 in a terminal once to sign in to Ordinal.')
        browser = BrowserSignIn[DepsT](tokens)
        host.add(browser)
        if not signed_in:
            host.console.print(
                'Ordinal: not signed in; your browser opens to sign in on first use.', style=theme.color(theme.MUTED)
            )

    def command(args: list[str]) -> str:
        match args:
            case []:
                return status(tokens)
            case ['logout']:
                if browser is None:
                    tokens.forget()
                    return f'Forgot any saved Ordinal sign-in; runs still use `{TOKEN_ENV}`.'
                browser.logout()
                return 'Signed out of Ordinal; the next run that uses it opens the browser to sign in.'
            case _:
                return USAGE

    host.commands.register(
        Command(
            name='ordinal',
            description='Show how Ordinal signs in, or sign out (/ordinal logout).',
            handler=command,
            complete=lambda args: ['logout'] if len(args) <= 1 else [],
        )
    )


def status(tokens: TokenStore) -> str:
    """One line on which Ordinal account runs use."""
    if os.environ.get(TOKEN_ENV):
        return f'Ordinal uses `{TOKEN_ENV}`.'
    return {
        True: 'Ordinal: signed in through the browser.',
        False: 'Ordinal: not signed in; the browser opens on first use.',
        None: 'Ordinal: sign-in unknown; the keyring could not be read.',
    }[tokens.signed_in()]
