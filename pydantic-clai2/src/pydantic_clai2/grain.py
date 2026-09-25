"""The built-in `grain` plugin: harness's `Grain` capability, with no secret in plugin settings.

The token comes from, in order: the `GRAIN_ACCESS_TOKEN` environment variable; a named key from `/keys` chosen
with `/grain key` (only the key's name is saved, and it is resolved on every run, so replacing the key in `/keys`
applies and deleting it fails closed); or a browser sign-in whose tokens go to the OS keyring the way `/mcp`
OAuth servers keep theirs. A sign-in needs someone at the terminal: in headless mode (`clai2 -p`) with no saved
sign-in, connecting fails with a message saying how to sign in.
"""

import os
from functools import partial

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.grain import Grain

from . import theme
from .api_keys import KeyReference, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import delete_credentials, load_codex_credentials
from .mcp import OAUTH_TIMEOUT, SignIn, http_client
from .plugins import PluginHost

GRAIN_MCP_URL = 'https://api.grain.com/_/mcp'
"""Grain's hosted MCP endpoint, the one `Grain` connects to when it is given a token rather than a client."""

TOKEN_ACCOUNT = 'plugin_grain'
"""The `TokenStore` name. `/mcp` server names cannot contain `_`, so no `/mcp` server shares these tokens."""

KEY_NAME = 'GRAIN_ACCESS_TOKEN'
"""The environment variable harness's `Grain` reads, and the `/keys` label a token typed into `/grain key` gets."""

KEY_ACCOUNT = 'grain'
"""The credential-store account holding the name of the chosen `/keys` entry, never the token."""


class KeyChoice(BaseModel):
    """The `/keys` entry Grain uses, by name. `key_users` reads this to stop renaming a key still in use."""

    token: KeyReference


def saved_key() -> KeyReference | None:
    """The `/keys` entry chosen with `/grain key`, or `None` when there is none."""
    raw = load_codex_credentials(account=KEY_ACCOUNT)
    if raw is None:
        return None
    try:
        return KeyChoice.model_validate_json(raw).token
    except ValidationError:
        raise UserError('The saved Grain key choice is invalid. Choose a key again with /grain key.') from None


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
                f'Grain needs a browser sign-in. Run clai2 interactively once to sign in, or set {KEY_NAME}.'
            ) from exc
        await super().redirect_handler(authorization_url)

    async def forget(self) -> None:
        """Sign out now: drop the saved sign-in and the tokens this session holds in memory."""
        await to_thread.run_sync(self.tokens.forget)
        self.context.clear_tokens()


def activate(host: PluginHost[None]) -> None:
    """Add `Grain`, authenticated by `GRAIN_ACCESS_TOKEN`, a named `/keys` entry, or a browser sign-in."""
    settings = host.settings(GrainSettings)
    auth: GrainSignIn | KeyReference | None = None
    if os.environ.get(KEY_NAME):
        host.add(Grain(read_only=settings.read_only))
    elif (reference := saved_key()) is not None:
        auth = reference
        host.add(Grain(auth=partial(_resolve, reference), read_only=settings.read_only))
    else:
        auth = GrainSignIn(host)
        transport = StreamableHttpTransport(GRAIN_MCP_URL, auth=auth, httpx_client_factory=http_client)
        # The default 5 second handshake timeout would end a browser sign-in before the user finishes it.
        client = Client(transport, init_timeout=OAUTH_TIMEOUT)
        host.add(Grain(client=client, read_only=settings.read_only))
    host.commands.register(
        Command(
            name='grain',
            description='Show how CLAI authenticates to Grain, choose a /keys token (/grain key), or sign out.',
            handler=partial(grain_command, auth=auth),
            complete=lambda args: ('key', 'logout') if len(args) <= 1 else (),
        )
    )


def _resolve(reference: KeyReference, _ctx: RunContext[None]) -> str:
    # Per run, so a key replaced in /keys applies and a deleted one fails closed.
    return resolve_key(token=reference)


async def grain_command(args: list[str], *, auth: GrainSignIn | KeyReference | None) -> str:
    """Report how this session authenticates, choose a `/keys` token, or sign out; `None` means the environment."""
    if args == ['key']:
        return await choose_key()
    if args not in ([], ['logout']):
        raise ValueError('Usage: /grain [key | logout]')
    if auth is None:
        if args:
            return f'Grain uses {KEY_NAME}, which /grain logout cannot revoke. Unset it, then /plugins reload grain.'
        return f'Grain uses the {KEY_NAME} environment variable.'
    if isinstance(auth, KeyReference):
        if args:
            return f'Grain uses the /keys entry {auth.name}. Choose "No API key" in /grain key to stop using it.'
        return f'Grain uses the /keys entry {auth.name}.'
    if args:
        await auth.forget()
        return 'Signed out of Grain. The next prompt that uses Grain opens the browser to sign in.'
    signed_in = await to_thread.run_sync(auth.tokens.signed_in)
    return {
        True: 'Signed in to Grain; the tokens are in the OS keyring. /grain logout signs out.',
        False: 'Not signed in to Grain; the next prompt opens the browser to sign in.',
        None: 'Unknown: the keyring could not be read.',
    }[signed_in]


async def choose_key() -> str:
    """Pick a `/keys` entry or type a token, saved to `/keys` as `GRAIN_ACCESS_TOKEN`; only the name is kept here."""
    prompt: PromptSession[str] = PromptSession()
    label = f'Grain access token (saved in /keys as {KEY_NAME}; Enter for none): '
    token = await prompt_api_key(prompt=prompt, label=label, optional=True)
    if token is None:
        return 'Grain key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            await to_thread.run_sync(partial(delete_credentials, account=KEY_ACCOUNT))
            return 'Grain uses no /keys entry. After /plugins reload grain, it signs in through the browser.'
        await to_thread.run_sync(partial(save_key, name=KEY_NAME, value=token))
        token = KeyReference(name=KEY_NAME)
    choice = KeyChoice(token=token).model_dump_json()
    await to_thread.run_sync(partial(save_key_connection, account=KEY_ACCOUNT, token=token, value=choice))
    return f'Grain uses the /keys entry {token.name}. /plugins reload grain applies it.'
