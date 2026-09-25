"""The built-in `ordinal` plugin: harness `Ordinal`, with no secret in plugin settings, which are plaintext SQLite.

Each run authenticates with the first of:

1. A named key from `/keys`, chosen when the plugin is enabled or with `/ordinal key`. Only the key's name is
   saved (in the credential store, beside the `vllm` and `openrouter` connections), and it is resolved on every
   run: replacing the key in `/keys` applies on the next run, and deleting it fails the run closed.
2. `ORDINAL_ACCESS_TOKEN`, the environment variable harness `Ordinal` documents.
3. A browser sign-in whose tokens go to the OS keyring, as `/mcp` OAuth servers' do. Harness
   `Ordinal(auth='oauth')` would keep them in memory, so every launch would sign in again.
"""

import os
import sys
from functools import partial
from typing import Generic

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.ordinal import Ordinal

from . import theme
from .api_keys import KeyReference, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import delete_credentials, load_codex_credentials
from .mcp import OAUTH_TIMEOUT, TokenStore, http_client, sign_in
from .plugins import DepsT, PluginHost, SessionStart

URL = 'https://app.tryordinal.com/mcp'
"""Harness `Ordinal`'s endpoint, which this plugin needs to build its own signed-in client."""
KEY_NAME = 'ORDINAL_ACCESS_TOKEN'
"""The variable harness `Ordinal` reads, and the `/keys` label a token typed into CLAI is saved under."""
KEY_ACCOUNT = 'ordinal'
"""The credential account holding the chosen `/keys` entry's name, never the token."""
TOKENS = 'plugin_ordinal'
"""The keyring entry for browser tokens (`mcp-plugin_ordinal`). `/mcp` server names cannot contain `_`."""
USAGE = 'Usage: /ordinal [key | logout]'


class KeyChoice(BaseModel):
    """The `/keys` entry Ordinal uses, by name. `key_users` reads this so `/keys` cannot rename it away."""

    token: KeyReference


def saved_key() -> KeyReference | None:
    """The `/keys` entry chosen for Ordinal, or `None` when there is none."""
    raw = load_codex_credentials(account=KEY_ACCOUNT)
    if raw is None:
        return None
    try:
        return KeyChoice.model_validate_json(raw).token
    except ValidationError:
        raise UserError('The saved Ordinal key choice is invalid. Choose a key again with /ordinal key.') from None


def ready(tokens: TokenStore) -> bool:
    """Whether a run can authenticate without asking: a chosen key, the environment, or a saved sign-in."""
    return saved_key() is not None or bool(os.environ.get(KEY_NAME)) or bool(tokens.signed_in())


class OrdinalAuth(Generic[DepsT]):
    """Hands each run an `Ordinal` for the current credential, so a change applies without a reload.

    Once connected, FastMCP's `OAuth` keeps the access token in memory, so clearing the keyring alone would leave
    this session signed in. `logout` therefore replaces the browser `Ordinal`, client and sign-in handler included.
    """

    def __init__(self, tokens: TokenStore) -> None:
        """Browser tokens persist through `tokens`."""
        self.tokens = tokens
        self.browser = self._browser()

    async def __call__(self, ctx: RunContext[DepsT]) -> Ordinal[DepsT]:
        """The capability for this run; a chosen `/keys` entry that was deleted raises instead of connecting."""
        reference = await to_thread.run_sync(saved_key)
        if reference is not None:
            return Ordinal[DepsT](auth=await to_thread.run_sync(partial(resolve_key, token=reference)))
        if os.environ.get(KEY_NAME):
            return Ordinal[DepsT]()
        return self.browser

    def logout(self) -> None:
        """Forget the saved browser tokens and the in-memory ones, so the next browser run signs in again."""
        self.tokens.forget()
        self.browser = self._browser()

    def _browser(self) -> Ordinal[DepsT]:
        transport = StreamableHttpTransport(url=URL, auth=sign_in(self.tokens.name), httpx_client_factory=http_client)
        # A bare transport gets `MCPToolset`'s 5 second handshake timeout, which would end a browser sign-in early.
        return Ordinal[DepsT](client=Client(transport, init_timeout=OAUTH_TIMEOUT))


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Ordinal`, or refuse to load when no key, token, saved sign-in, or terminal to sign in from exists."""
    tokens = TokenStore(TOKENS)
    if not sys.stdin.isatty() and not ready(tokens):
        raise UserError(f'Choose a /keys entry with /ordinal key, set `{KEY_NAME}`, or sign in from a terminal.')
    auth = OrdinalAuth[DepsT](tokens)
    host.add(auth)

    @host.on('session_start')
    async def offer_key(_: SessionStart) -> None:  # pyright: ignore[reportUnusedFunction]
        if host.console.is_terminal and sys.stdin.isatty() and not await to_thread.run_sync(ready, tokens):
            host.console.print(await choose_key(), style=theme.color(theme.MUTED), markup=False, highlight=False)

    async def command(args: list[str]) -> str:
        match args:
            case []:
                return await to_thread.run_sync(status, tokens)
            case ['key']:
                return await choose_key()
            case ['logout']:
                return await to_thread.run_sync(logout, auth)
            case _:
                return USAGE

    host.commands.register(
        Command(
            name='ordinal',
            description='Show how Ordinal authenticates, choose a /keys entry (/ordinal key), or sign out.',
            handler=command,
            complete=lambda args: ['key', 'logout'] if len(args) <= 1 else [],
        )
    )


async def choose_key() -> str:
    """Pick a `/keys` entry or type a token (saved to `/keys` as `ORDINAL_ACCESS_TOKEN`); only the name is kept."""
    prompt: PromptSession[str] = PromptSession()
    label = f'Ordinal access token (saved in /keys as {KEY_NAME}; Enter or "No API key" signs in through the browser): '
    token = await prompt_api_key(prompt=prompt, label=label, optional=True)
    if token is None:
        return 'Ordinal key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            await to_thread.run_sync(partial(delete_credentials, account=KEY_ACCOUNT))
            return f'Ordinal uses no /keys entry; runs use `{KEY_NAME}` if set, or sign in through the browser.'
        if KEY_NAME in await to_thread.run_sync(load_keys):
            try:
                answer = await prompt.prompt_async(
                    f'Replace {KEY_NAME} in /keys for every connection using it? [y/N]: '
                )
            except (EOFError, KeyboardInterrupt):
                return 'Ordinal key unchanged.'
            if answer.strip().lower() != 'y':
                return 'Ordinal key unchanged.'
        await to_thread.run_sync(partial(save_key, name=KEY_NAME, value=token))
        token = KeyReference(name=KEY_NAME)
    choice = KeyChoice(token=token).model_dump_json()
    await to_thread.run_sync(partial(save_key_connection, account=KEY_ACCOUNT, token=token, value=choice))
    return f'Ordinal uses {token.name} from /keys, starting with the next run.'


def status(tokens: TokenStore) -> str:
    """One line on which credential the next run uses."""
    reference = saved_key()
    if reference is not None:
        return f'Ordinal uses {reference.name} from /keys. /ordinal key chooses another.'
    if os.environ.get(KEY_NAME):
        return f'Ordinal uses `{KEY_NAME}` from the environment.'
    return {
        True: 'Ordinal: signed in through the browser.',
        False: 'Ordinal: not signed in; the browser opens on first use.',
        None: 'Ordinal: sign-in unknown; the keyring could not be read.',
    }[tokens.signed_in()]


def logout(auth: OrdinalAuth[DepsT]) -> str:
    """Sign out of the browser session; a `/keys` entry or environment variable is not ours to revoke."""
    auth.logout()
    reference = saved_key()
    if reference is not None:
        return f'Signed out of the browser session; runs still use {reference.name} from /keys (/ordinal key to stop).'
    if os.environ.get(KEY_NAME):
        return f'Signed out of the browser session; runs still use `{KEY_NAME}`.'
    return 'Signed out of Ordinal; the next run that uses it opens the browser to sign in.'
