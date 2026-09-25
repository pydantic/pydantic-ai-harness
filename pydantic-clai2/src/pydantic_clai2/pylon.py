"""The built-in `pylon` plugin: Pylon's support issues, accounts, and contacts through harness `Pylon`.

The token is never kept in plugin settings, which are plaintext SQLite. By default the plugin connects with a
named key from `/keys`: enabling it (or `/pylon key`) runs the shared key picker, and only the key's name is
saved. The key is resolved on every run, so replacing it in `/keys` takes effect on the next run, and a
deleted key fails the run instead of connecting without it.

With `"auth": "browser"`, CLAI signs in through the browser the way `/mcp` OAuth servers do, keeping the
tokens in the keyring. Harness `Pylon(auth='oauth')` would keep them in memory and give the browser the
5-second connect timeout, so CLAI builds that client itself.
"""

import asyncio
import json
from typing import Literal

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.pylon import Pylon

from .api_keys import KeyReference, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import load_codex_credentials
from .mcp import OAUTH_TIMEOUT, http_client, sign_in
from .plugins import DepsT, PluginHost, SessionStart

PYLON_MCP_URL = 'https://mcp.usepylon.com'
"""Harness `Pylon`'s endpoint, repeated here because a custom client owns its URL."""

KEY_NAME = 'PYLON_ACCESS_TOKEN'
"""The `/keys` label for a new token: harness `Pylon`'s documented variable name, used as a label only."""

ACCOUNT = 'pylon'
"""The credential account holding the key reference, beside the `vllm` and `openrouter` connections."""

TOKEN_ACCOUNT = 'plugin_pylon'
"""Browser tokens are stored as `mcp-plugin_pylon`; `/mcp` server names cannot contain `_`, so none shares them."""

_HELP = 'Usage: /pylon (show the saved key) or /pylon key (choose a key from /keys or enter a new one)'


class PylonSettings(BaseModel):
    """The JSON a `pylon` declaration may carry. Nothing here is secret."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Literal['key', 'browser'] = Field(
        default='key', description='Connect with a named key from /keys, or sign in through the browser.'
    )
    read_only: bool = Field(default=False, description="Keep only the tools Pylon's server labels read-only.")


class _Saved(BaseModel):
    token: KeyReference


def saved_key() -> KeyReference | None:
    """The key Pylon connects with, or `None` before one is chosen."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return None
    try:
        return _Saved.model_validate_json(raw).token
    except ValidationError:
        raise UserError('The saved Pylon key reference is invalid. Choose a key again with /pylon key.') from None


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Pylon` with a named key resolved per run, or with browser sign-in."""
    settings = host.settings(PylonSettings)
    if settings.auth == 'browser':
        host.add(Pylon[DepsT](client=_browser_client(), read_only=settings.read_only))
        return
    host.add(Pylon[DepsT](auth=_token, read_only=settings.read_only))

    @host.on('session_start')
    async def choose_on_enable(_: SessionStart) -> None:  # pyright: ignore[reportUnusedFunction]
        if host.console.is_terminal and await asyncio.to_thread(saved_key) is None:
            host.console.print(await choose_key(), markup=False)

    async def command(args: list[str]) -> str:
        if args == ['key']:
            return await choose_key()
        if args:
            raise ValueError(_HELP)
        reference = await asyncio.to_thread(saved_key)
        if reference is None:
            return 'Pylon has no key yet, so runs get no Pylon tools. Choose one with /pylon key.'
        return f'Pylon connects with {reference.name} from /keys. /pylon key chooses another.'

    host.commands.register(
        Command(
            name='pylon',
            description='Show or choose the /keys entry Pylon connects with (/pylon key).',
            handler=command,
            complete=lambda args: ['key'] if len(args) <= 1 else [],
        )
    )


async def choose_key() -> str:
    """Pick a saved key or enter a new masked one; only the key's name is saved for Pylon."""
    prompt: PromptSession[str] = PromptSession()
    label = f'{KEY_NAME} (a Pylon OAuth access token; Pylon API keys are not accepted): '
    token = await prompt_api_key(prompt=prompt, label=label)
    if token is None:
        return 'Pylon key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            raise ValueError('A Pylon access token is required.')
        if KEY_NAME in await asyncio.to_thread(load_keys):
            try:
                answer = await prompt.prompt_async(
                    f'Replace {KEY_NAME} in /keys for every connection using it? [y/N]: '
                )
            except (EOFError, KeyboardInterrupt):
                return 'Pylon key unchanged.'
            if answer.strip().lower() != 'y':
                return 'Pylon key unchanged.'
        await asyncio.to_thread(save_key, name=KEY_NAME, value=token)
        token = KeyReference(name=KEY_NAME)
    saved = json.dumps({'token': token.model_dump()})
    await asyncio.to_thread(save_key_connection, account=ACCOUNT, token=token, value=saved)
    return f'Pylon connects with {token.name} from /keys.'


def _token(_: RunContext[DepsT]) -> str | None:
    """Resolve the named key for this run; no key chosen means no Pylon tools, a deleted key raises."""
    reference = saved_key()
    return None if reference is None else resolve_key(token=reference)


def _browser_client() -> Client[StreamableHttpTransport]:
    """A Pylon connection that signs in on first use and allows the browser as long as `/mcp` does."""
    transport = StreamableHttpTransport(
        url=PYLON_MCP_URL, auth=sign_in(TOKEN_ACCOUNT), httpx_client_factory=http_client
    )
    return Client(transport, init_timeout=OAUTH_TIMEOUT)
