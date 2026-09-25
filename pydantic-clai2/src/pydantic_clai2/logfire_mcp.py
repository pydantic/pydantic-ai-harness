"""The built-in `logfire_mcp` plugin: harness `LogfireMCP` with CLAI's named keys and keyring-backed OAuth.

Secrets never go in plugin settings, which are plaintext SQLite. A key lives in `/keys`; this plugin stores only
the name it uses, in the credential store, and reads the value at the start of each run.
"""

import asyncio
import os
import webbrowser
from collections.abc import Callable
from typing import Literal

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP

from .api_keys import KeyReference, SecretPrompt, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import delete_credentials, load_codex_credentials
from .mcp import HTTPServer, TokenStore, http_client, oauth
from .plugins import PluginHost

KEY_NAME = 'LOGFIRE_API_KEY'
"""The conventional `/keys` label, matching the variable `LogfireMCP` reads, so other tools can share one key."""
ACCOUNT = 'logfire_mcp'
"""Credential account for the chosen key's name, and `TokenStore` name for OAuth tokens (stored as `mcp-logfire_mcp`).

`/mcp` server names cannot contain `_`, so no server shares the token account.
"""
HELP = """\
/logfire_mcp key      Choose a saved key from /keys, or enter one to save as LOGFIRE_API_KEY
/logfire_mcp logout   Forget the chosen key and the OAuth sign-in; keys in /keys are kept"""

_URLS = {'us': LOGFIRE_US_MCP_URL, 'eu': LOGFIRE_EU_MCP_URL}


class LogfireMCPSettings(BaseModel):
    """Non-secret options. Keys are managed in `/keys` and chosen with `/logfire_mcp key`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    oauth: bool = True
    """Sign in through the browser when no key is chosen, set, or saved under `LOGFIRE_API_KEY`."""
    region: Literal['us', 'eu'] = 'us'
    read_only: bool = True
    """Offer only the tools the server marks read-only, so the agent cannot change Logfire resources."""


class ChosenKey(BaseModel):
    """The name-only reference `/logfire_mcp key` saves; resolved like `vllm` and `openrouter` references."""

    token: KeyReference


def activate(host: PluginHost[None]) -> None:
    """Add `LogfireMCP`, or refuse to load when no credential could connect it."""
    host.add(_capability(settings=host.settings(LogfireMCPSettings)))
    host.commands.register(
        Command(
            name='logfire_mcp',
            description='Choose the Logfire MCP key from /keys, or sign out (/logfire_mcp help).',
            handler=command,
            complete=lambda _: ('key', 'logout', 'help'),
        )
    )


def _capability(*, settings: LogfireMCPSettings) -> LogfireMCP[None]:
    url = _URLS[settings.region]
    reference = chosen_key()
    if reference is None and not os.environ.get(KEY_NAME) and KEY_NAME in load_keys():
        reference = KeyReference(name=KEY_NAME)
    if reference is not None:
        return LogfireMCP[None](auth=_per_run(reference=reference), url=url, read_only=settings.read_only)
    if os.environ.get(KEY_NAME):
        return LogfireMCP[None](url=url, read_only=settings.read_only)
    if settings.oauth:
        return LogfireMCP[None](client=_oauth_client(url=url), read_only=settings.read_only)
    raise UserError(f'Save a key named {KEY_NAME} in /keys, set {KEY_NAME}, or turn `oauth` back on.')


def chosen_key() -> KeyReference | None:
    """The key picked with `/logfire_mcp key`, or `None` when none was picked."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return None
    try:
        return ChosenKey.model_validate_json(raw).token
    except ValidationError:
        raise UserError(
            'The saved Logfire key choice is invalid. Run /logfire_mcp logout, then choose again.'
        ) from None


def _per_run(*, reference: KeyReference) -> Callable[[RunContext[None]], str]:
    if reference.name not in load_keys():
        raise UserError(f'Saved API key {reference.name} is missing. Restore it in /keys or run /logfire_mcp key.')
    # Resolved per run, like model connections, so a replaced key applies next turn and a deleted one fails closed.
    return lambda _ctx: resolve_key(token=reference)


def _oauth_client(*, url: str) -> Client[StreamableHttpTransport]:
    server = HTTPServer.model_validate({'type': 'http', 'url': url, 'auth': 'oauth'})
    if not TokenStore(ACCOUNT).signed_in():
        try:
            webbrowser.get()
        except webbrowser.Error:
            raise UserError(
                f'Logfire sign-in needs a browser. Save a key named {KEY_NAME} in /keys, or set {KEY_NAME}.'
            ) from None
    transport = StreamableHttpTransport(url, auth=oauth(ACCOUNT, server), httpx_client_factory=http_client)
    return Client(transport, init_timeout=server.init_timeout())


async def command(args: list[str]) -> str:
    """`/logfire_mcp key`, `logout`, or `help`."""
    match args:
        case ['key']:
            prompt: PromptSession[str] = PromptSession()
            return await choose_key(prompt=prompt)
        case ['logout']:
            await asyncio.to_thread(logout)
            return 'Forgot the Logfire key choice and OAuth sign-in. Run /plugins reload logfire_mcp.'
        case ['help']:
            return HELP
        case _:
            raise ValueError(f'Usage:\n{HELP}')


async def choose_key(*, prompt: SecretPrompt) -> str:
    """Pick a saved key or enter a new one; only its name is stored here, never the value."""
    token = await prompt_api_key(prompt=prompt, label=f'Logfire API key (saved to /keys as {KEY_NAME}): ')
    if token is None:
        return 'Logfire key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            raise ValueError('A Logfire API key is required.')
        if KEY_NAME in await asyncio.to_thread(load_keys):
            try:
                answer = await prompt.prompt_async(f'Replace {KEY_NAME}, which other tools may share? [y/N]: ')
            except (EOFError, KeyboardInterrupt):
                answer = ''
            if answer.strip().lower() != 'y':
                return 'Logfire key unchanged. Save it under another name in /keys, then choose it here.'
        await asyncio.to_thread(save_key, name=KEY_NAME, value=token)
        token = KeyReference(name=KEY_NAME)
    await asyncio.to_thread(
        save_key_connection, account=ACCOUNT, token=token, value=ChosenKey(token=token).model_dump_json()
    )
    return f'Logfire MCP uses {token.name}. Run /plugins reload logfire_mcp to connect with it.'


def logout() -> None:
    """Forget the key choice and OAuth tokens; the keys themselves stay in `/keys`."""
    delete_credentials(account=ACCOUNT)
    TokenStore(ACCOUNT).forget()
