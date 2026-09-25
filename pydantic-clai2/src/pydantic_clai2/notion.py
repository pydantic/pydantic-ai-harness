"""The built-in `notion` plugin: harness `Notion`, connected with a named key from `/keys` or a browser sign-in.

Plugin settings are plaintext SQLite, so they never hold a secret. `/notion key` picks or enters a key in the
named keystore and saves only its name, in CLAI's credential store; each run resolves it again, so replacing
the key in `/keys` reaches every plugin that shares it, and a deleted key fails the run rather than connecting.
"""

import asyncio
from collections.abc import Iterable
from typing import Literal

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.notion import Notion

from .api_keys import KeyReference, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import delete_credentials, load_codex_credentials
from .mcp import OAUTH_TIMEOUT, TokenStore, browser_sign_in, http_client
from .plugins import DepsT, PluginHost

NOTION_MCP_URL = 'https://mcp.notion.com/mcp'
KEY_NAME = 'NOTION_API_KEY'
"""The `/keys` label a newly entered token is saved under, so other Notion consumers can share it."""
ACCOUNT = 'notion'
"""The credential account holding the selected key's name, never its value."""
TOKENS = TokenStore('plugin_notion')
"""Browser sign-in tokens. `/mcp` server names cannot contain `_`, so this account never collides with one."""
USAGE = 'Usage: /notion key | /notion logout'


class NotionSettings(BaseModel):
    """The JSON a `notion` declaration may carry. Secrets are not accepted here; they live in `/keys`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Literal['key', 'oauth'] | None = Field(
        default=None,
        description='`key` uses the key chosen with `/notion key` and never opens a browser; `oauth` always signs '
        'in through the browser; unset uses the chosen key when there is one.',
    )
    read_only: bool = Field(default=False, description='Keep only the tools the server marks as read-only.')


class _Selection(BaseModel):
    token: KeyReference


def selected_key() -> KeyReference | None:
    """The key `/notion key` chose, or `None`. Only the name is stored."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return None
    try:
        return _Selection.model_validate_json(raw).token
    except ValidationError:
        raise UserError('The saved Notion key selection is invalid. Choose one again with /notion key.') from None


def activate(host: PluginHost[DepsT]) -> None:
    """Choose the connection per run, so `/notion key` and `/notion logout` apply without a reload."""
    settings = host.settings(NotionSettings)
    if settings.auth == 'key' and selected_key() is None:
        host.console.print('Notion: no key selected, so runs fail until you choose one with /notion key.', markup=False)

    async def connect(_: RunContext[DepsT]) -> Notion[DepsT]:
        reference = None if settings.auth == 'oauth' else await asyncio.to_thread(selected_key)
        if reference is not None:
            token = await asyncio.to_thread(resolve_key, token=reference)
            return Notion[DepsT](auth=token, read_only=settings.read_only)
        if settings.auth == 'key':
            raise UserError('No Notion key is selected. Choose one from /keys with /notion key.')
        # Harness `auth='oauth'` keeps tokens in memory behind a 5-second handshake; this keeps them in the
        # keyring and allows the browser round trip, like an OAuth server added through `/mcp`.
        transport = StreamableHttpTransport(
            NOTION_MCP_URL, auth=browser_sign_in(TOKENS), httpx_client_factory=http_client
        )
        return Notion[DepsT](client=Client(transport, init_timeout=OAUTH_TIMEOUT), read_only=settings.read_only)

    host.add(connect)
    host.commands.register(
        Command(
            name='notion',
            description='Choose the Notion key from /keys, or sign out (/notion key, /notion logout).',
            handler=_command,
            complete=_complete,
        )
    )


async def _command(args: list[str]) -> str:
    if args == ['key']:
        return await choose_key()
    if args == ['logout']:
        await asyncio.to_thread(TOKENS.forget)
        await asyncio.to_thread(delete_credentials, account=ACCOUNT)
        return 'Signed out of Notion and cleared the selected key. The key itself stays in /keys.'
    raise ValueError(USAGE)


async def choose_key() -> str:
    """Pick a saved key or enter a new masked one; only its name is kept for Notion."""
    prompt: PromptSession[str] = PromptSession()
    choice = await prompt_api_key(prompt=prompt, label=f'Notion access token (saved in /keys as {KEY_NAME}): ')
    if choice is None:
        return 'Notion key unchanged.'
    if isinstance(choice, KeyReference):
        reference, saved = choice, ''
    else:
        if not choice.strip():
            raise ValueError('A Notion access token is required.')
        if KEY_NAME in await asyncio.to_thread(load_keys):
            try:
                answer = await prompt.prompt_async(f'Replace {KEY_NAME} for everything that uses it? [y/N]: ')
            except (EOFError, KeyboardInterrupt):
                answer = ''
            if answer.strip().lower() != 'y':
                return 'Notion key unchanged.'
        saved = await asyncio.to_thread(save_key, name=KEY_NAME, value=choice) + ' '
        reference = KeyReference(name=KEY_NAME)
    value = _Selection(token=reference).model_dump_json()
    await asyncio.to_thread(save_key_connection, account=ACCOUNT, token=reference, value=value)
    return f'{saved}Notion uses {reference.name}; manage it in /keys.'


def _complete(args: list[str]) -> Iterable[str]:
    return ('key', 'logout') if len(args) <= 1 else ()
