"""The built-in `linear` plugin: harness `Linear`, with its key named in `/keys` rather than stored in settings.

Plugin settings are plaintext SQLite, so they carry no credential. `/linear key` picks a saved key or saves a
new one under `LINEAR_API_KEY`, and only that name is stored. Each run resolves the name, so replacing the key
in `/keys` reaches Linear on the next run and a deleted key fails the run instead of connecting without it.
"""

from collections.abc import Awaitable, Callable

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.linear import Linear

from . import theme
from .api_keys import KeyReference, SecretPrompt, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import load_codex_credentials
from .mcp import HTTPServer, TokenStore, http_client, oauth
from .plugins import DepsT, PluginHost, SessionStart

KEY_NAME = 'LINEAR_API_KEY'
"""The `/keys` label a new Linear key is saved under, and the one used before `/linear key` picks another."""
ACCOUNT = 'linear'
"""Credential-store account holding the chosen key's name. `api_keys.key_users` checks it before a rename."""
TOKEN_ACCOUNT = 'linear_plugin'
"""`TokenStore` name for OAuth tokens. `/mcp` server names cannot contain `_`, so no `/mcp` server shares it."""
# The endpoints harness `Linear` connects to; with `client`, the plugin picks the URL itself.
_URL = 'https://mcp.linear.app/mcp'
_READ_ONLY_URL = 'https://mcp.linear.app/mcp/readonly'
_RECONFIGURE = '/linear key'


class LinearSettings(BaseModel):
    """The JSON a `linear` declaration may carry. Credentials live in `/keys` or the keyring, never here."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(default=True, description="Connect to Linear's read-only endpoint.")
    oauth: bool = Field(default=False, description='Sign in through the browser; tokens are kept in the keyring.')


class _Connection(BaseModel):
    """The saved choice: a key name, in the shape `api_keys.key_users` reads."""

    token: KeyReference


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Linear` and `/linear`; a missing key is reported now and fails each run until one is chosen."""
    settings = host.settings(LinearSettings)
    if settings.oauth:
        host.add(Linear[DepsT](client=_oauth_client(settings.read_only)))
        store = TokenStore(TOKEN_ACCOUNT)

        async def logout(args: list[str]) -> str:
            if args != ['logout']:
                raise ValueError('Usage: /linear logout')
            await to_thread.run_sync(store.forget)
            return 'Signed out of Linear. The next run opens the browser to sign in again.'

        _register(host, 'Sign out of Linear (/linear logout).', logout, 'logout')
        return

    def token(_: RunContext[DepsT]) -> str:
        return resolve_key(token=reference(), reconfigure=_RECONFIGURE)

    host.add(Linear[DepsT](auth=token, read_only=settings.read_only))

    async def configure(args: list[str]) -> str:
        if args != ['key']:
            raise ValueError('Usage: /linear key')
        prompt: PromptSession[str] = PromptSession()
        return await choose_key(prompt)

    _register(host, 'Choose the /keys entry Linear uses (/linear key).', configure, 'key')

    @host.on('session_start')
    async def check(_: SessionStart) -> None:  # pyright: ignore[reportUnusedFunction]
        try:
            await to_thread.run_sync(lambda: resolve_key(token=reference(), reconfigure=_RECONFIGURE))
        except UserError as exc:
            host.console.print(f'Linear: {exc}', style=theme.color(theme.WARNING), markup=False)


def reference() -> KeyReference:
    """The key Linear uses: the one `/linear key` chose, else `LINEAR_API_KEY`."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return KeyReference(name=KEY_NAME)
    try:
        return _Connection.model_validate_json(raw).token
    except ValidationError:
        raise UserError(f'The saved Linear key choice is invalid. Choose again with {_RECONFIGURE}.') from None


async def choose_key(prompt: SecretPrompt) -> str:
    """Pick a `/keys` entry or enter a new key masked; save the name, and the value only in `/keys`."""
    choice = await prompt_api_key(prompt=prompt, label=f'Linear API key (saved in /keys as {KEY_NAME}): ')
    if choice is None:
        return 'Linear key unchanged.'
    saved = ''
    if isinstance(choice, KeyReference):
        key = choice
    else:
        value = choice.strip()
        if not value:
            raise ValueError('A Linear API key is required.')
        if KEY_NAME in await to_thread.run_sync(load_keys):
            try:
                answer = await prompt.prompt_async(f'Replace {KEY_NAME} in /keys for every plugin using it? [y/N]: ')
            except (EOFError, KeyboardInterrupt):
                answer = ''
            if answer.strip().lower() != 'y':
                return 'Linear key unchanged.'
        saved = await to_thread.run_sync(lambda: save_key(name=KEY_NAME, value=value)) + ' '
        key = KeyReference(name=KEY_NAME)
    value_json = _Connection(token=key).model_dump_json()
    await to_thread.run_sync(lambda: save_key_connection(account=ACCOUNT, token=key, value=value_json))
    return f'{saved}Linear uses {key.name} from /keys from the next run.'


def _register(
    host: PluginHost[DepsT], description: str, handler: Callable[[list[str]], Awaitable[str]], action: str
) -> None:
    host.commands.register(
        Command(name='linear', description=description, handler=handler, complete=lambda _: (action,))
    )


def _oauth_client(read_only: bool) -> Client[StreamableHttpTransport]:
    """Connect the way `/mcp` connects an OAuth server: keyring tokens, no redirects, time for a browser sign-in.

    Plain `Linear(auth='oauth')` keeps tokens in memory and allows the 5-second default for `initialize`, which
    a browser sign-in does not fit in. The URL carries `read_only` here: `Linear`'s own `read_only` with a
    `client` filters on tool annotations instead, which would be a second, different boundary.
    """
    server = HTTPServer.model_validate({'type': 'http', 'url': _READ_ONLY_URL if read_only else _URL, 'auth': 'oauth'})
    transport = StreamableHttpTransport(
        url=str(server.url), auth=oauth(TOKEN_ACCOUNT, server), httpx_client_factory=http_client
    )
    return Client(transport, init_timeout=server.init_timeout())
