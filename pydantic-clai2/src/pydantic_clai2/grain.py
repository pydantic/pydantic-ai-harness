"""The built-in `grain` plugin: harness's `Grain` capability, signed in once per machine.

`GRAIN_ACCESS_TOKEN` wins when it is set. Otherwise the first prompt that connects opens the browser to sign in
to Grain, and the tokens go to the OS keyring the way `/mcp` OAuth servers keep theirs, so later sessions refresh
them instead of signing in again. A sign-in needs someone at the terminal: in headless mode (`clai2 -p`) with no
saved sign-in, connecting fails with a message saying how to sign in.
"""

import os
from functools import partial

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.grain import Grain

from . import theme
from .commands import Command
from .mcp import OAUTH_TIMEOUT, SignIn, TokenStore, http_client
from .plugins import PluginHost

GRAIN_MCP_URL = 'https://api.grain.com/_/mcp'
"""Grain's hosted MCP endpoint, the one `Grain` connects to when it is given a token rather than a client."""

TOKEN_ACCOUNT = 'plugin_grain'
"""The `TokenStore` name. `/mcp` server names cannot contain `_`, so no `/mcp` server shares these tokens."""

_ENV = 'GRAIN_ACCESS_TOKEN'


class GrainSettings(BaseModel):
    """Plugin settings, given with `/plugins add grain pydantic_clai2.grain JSON`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    read_only: bool = True
    """Offer only the tools Grain marks read-only; `false` also lets the agent create clips and tag meetings."""


class GrainSignIn(SignIn):
    """Say where the sign-in happens before FastMCP opens the browser, and refuse when no one can sign in."""

    def __init__(self, host: PluginHost[None]) -> None:
        """Tokens persist under `TOKEN_ACCOUNT`."""
        # Grain's client registration rejects a `127.0.0.1` redirect URI with `invalid_redirect_uri`
        # and accepts `localhost` (checked 2026-09-25).
        super().__init__(TOKEN_ACCOUNT, callback_host='localhost')
        self._host = host

    async def redirect_handler(self, authorization_url: str) -> None:
        """Print the URL too, for a browser that does not open (for example over SSH)."""
        try:
            async with self._host.full_screen():
                self._host.console.print(
                    f'Signing in to Grain in your browser. If it does not open, visit:\n{authorization_url}',
                    style=theme.color(theme.INFO),
                    markup=False,
                )
        except RuntimeError as exc:
            # Headless mode binds a screen that refuses interaction, since no one is there to sign in.
            raise UserError(
                f'Grain needs a browser sign-in. Run clai2 interactively once to sign in, or set {_ENV}.'
            ) from exc
        await super().redirect_handler(authorization_url)


def activate(host: PluginHost[None]) -> None:
    """Add `Grain`, authenticated by `GRAIN_ACCESS_TOKEN` or by a keyring-backed browser sign-in."""
    settings = host.settings(GrainSettings)
    uses_token = bool(os.environ.get(_ENV))
    if uses_token:
        host.add(Grain(read_only=settings.read_only))
    else:
        transport = StreamableHttpTransport(GRAIN_MCP_URL, auth=GrainSignIn(host), httpx_client_factory=http_client)
        # The default 5 second handshake timeout would end a browser sign-in before the user finishes it.
        client = Client(transport, init_timeout=OAUTH_TIMEOUT)
        host.add(Grain(client=client, read_only=settings.read_only))
    host.commands.register(
        Command(
            name='grain',
            description='Show how CLAI signs in to Grain, or sign out (/grain logout).',
            handler=partial(grain_command, uses_token=uses_token),
            complete=lambda args: ('logout',) if len(args) <= 1 else (),
        )
    )


async def grain_command(args: list[str], *, uses_token: bool) -> str:
    """Report how this session authenticates, or forget the saved sign-in."""
    if args == ['logout']:
        await to_thread.run_sync(TokenStore(TOKEN_ACCOUNT).forget)
        # The loaded client keeps its tokens in memory until the plugin is loaded again.
        return 'Signed out of Grain. After /plugins reload grain or a restart, the next prompt signs in again.'
    if args:
        raise ValueError('Usage: /grain [logout]')
    if uses_token:
        return f'Grain uses {_ENV}.'
    signed_in = await to_thread.run_sync(TokenStore(TOKEN_ACCOUNT).signed_in)
    return {
        True: 'Signed in to Grain; the tokens are in the OS keyring. /grain logout signs out.',
        False: 'Not signed in to Grain; the next prompt opens the browser to sign in.',
        None: 'Unknown: the keyring could not be read.',
    }[signed_in]
