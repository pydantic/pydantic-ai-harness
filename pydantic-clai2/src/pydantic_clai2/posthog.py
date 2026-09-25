"""The built-in `posthog` plugin: PostHog's hosted MCP server through harness `PostHog`.

The personal API key is never kept in plugin settings, which are plaintext SQLite. It lives in `/keys`, and the
plugin saves only the key's name, in the credential store beside the `vllm` and `openrouter` connections. Enabling
the plugin (or `/posthog key`) runs the shared key picker. The name is resolved on every run, so replacing the key
in `/keys` reaches the next run, and a deleted key fails the run instead of connecting without it.

With `"auth": "browser"`, CLAI signs in through the browser the way `/mcp` OAuth servers do. Harness
`PostHog(auth='oauth')` would keep the tokens in memory, sign in again on every run, and give the browser a
5-second connect timeout, so this plugin builds that client itself and keeps the tokens in the keyring.
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
from pydantic_ai_harness.posthog import PostHog

from . import theme
from .api_keys import KeyReference, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import load_codex_credentials
from .mcp import OAUTH_TIMEOUT, TokenStore, http_client, sign_in
from .plugins import PluginHost, SessionStart

KEY_NAME = 'POSTHOG_PERSONAL_API_KEY'
"""The `/keys` label for a new key: harness `PostHog`'s documented variable name, used as a label only."""

ACCOUNT = 'posthog'
"""The credential account holding the key reference, never the key."""

TOKENS = 'posthog_plugin'
"""Browser tokens are the `mcp-posthog_plugin` credential. `/mcp` server names cannot contain `_`, so none shares it."""

POSTHOG_MCP_URL = 'https://mcp.posthog.com/mcp'
"""Harness `PostHog`'s endpoint, repeated here because a custom client owns its URL."""

_SIGN_IN_STATES: dict[bool | None, str] = {
    True: 'signed in through the browser',
    False: 'not signed in; the first prompt that uses it opens the browser',
    None: 'in an unknown sign-in state: the keyring cannot be read',
}


class PostHogSettings(BaseModel):
    """The JSON a `posthog` declaration may carry. Nothing here is secret."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(default=True, description='Ask PostHog to serve only the tools it marks read-only.')
    auth: Literal['key', 'browser'] = Field(
        default='key', description='Connect with a named key from /keys, or sign in through the browser.'
    )


class _Saved(BaseModel):
    token: KeyReference


def saved_key() -> KeyReference | None:
    """The key PostHog connects with, or `None` before one is chosen."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return None
    try:
        return _Saved.model_validate_json(raw).token
    except ValidationError:
        raise UserError('The saved PostHog key reference is invalid. Choose a key again with /posthog key.') from None


def activate(host: PluginHost[None]) -> None:
    """Add `PostHog` with a named key resolved per run, or with browser sign-in kept in the keyring."""
    settings = host.settings(PostHogSettings)
    mode = 'read-only' if settings.read_only else 'read-write'
    if settings.auth == 'browser':
        _activate_browser(host, read_only=settings.read_only, mode=mode)
        return
    host.add(PostHog[None](auth=_token, read_only=settings.read_only))

    @host.on('session_start')
    async def choose_on_enable(_: SessionStart) -> None:  # pyright: ignore[reportUnusedFunction]
        if await asyncio.to_thread(saved_key) is not None:
            return
        if host.console.is_terminal:
            host.console.print(await choose_key(), markup=False)
        if await asyncio.to_thread(saved_key) is None:
            host.console.print(
                'PostHog has no key, so runs get no PostHog tools. Choose one with /posthog key.',
                style=theme.color(theme.WARNING),
                markup=False,
            )

    async def command(args: list[str]) -> str:
        if args == ['key']:
            return await choose_key()
        if args:
            raise ValueError('Usage: /posthog (show the saved key) or /posthog key (choose or enter one)')
        reference = await asyncio.to_thread(saved_key)
        if reference is None:
            return 'PostHog has no key yet, so runs get no PostHog tools. Choose one with /posthog key.'
        return f'PostHog ({mode}) connects with {reference.name} from /keys. /posthog key chooses another.'

    host.commands.register(
        Command(
            name='posthog',
            description='Show or choose the /keys entry PostHog connects with (/posthog key).',
            handler=command,
            complete=lambda args: ['key'] if len(args) <= 1 else [],
        )
    )


def _activate_browser(host: PluginHost[None], *, read_only: bool, mode: str) -> None:
    capability = PostHog[None](client=_browser_client(read_only))
    host.add(capability)
    tokens = TokenStore(TOKENS)

    async def command(args: list[str]) -> str:
        if args == ['logout']:
            await asyncio.to_thread(tokens.forget)
            # The live sign-in still holds the tokens it loaded, so later runs need a fresh one.
            capability.client = _browser_client(read_only)
            return 'Signed out of PostHog. The next prompt that uses it opens the browser to sign in again.'
        if args:
            raise ValueError('Usage: /posthog [logout]')
        return f'PostHog ({mode}) is {_SIGN_IN_STATES[await asyncio.to_thread(tokens.signed_in)]}.'

    host.commands.register(
        Command(
            name='posthog',
            description='Show the PostHog browser sign-in, or sign out (/posthog logout).',
            handler=command,
            complete=lambda args: ['logout'] if len(args) <= 1 else [],
        )
    )


async def choose_key() -> str:
    """Pick a saved key or enter a new masked one; only the key's name is saved for PostHog."""
    prompt: PromptSession[str] = PromptSession()
    label = f'{KEY_NAME} (a PostHog personal API key with the MCP Server preset): '
    token = await prompt_api_key(prompt=prompt, label=label)
    if token is None:
        return 'PostHog key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            raise ValueError('A PostHog personal API key is required.')
        if KEY_NAME in await asyncio.to_thread(load_keys):
            try:
                answer = await prompt.prompt_async(
                    f'Replace {KEY_NAME} in /keys for every connection using it? [y/N]: '
                )
            except (EOFError, KeyboardInterrupt):
                return 'PostHog key unchanged.'
            if answer.strip().lower() != 'y':
                return 'PostHog key unchanged.'
        await asyncio.to_thread(save_key, name=KEY_NAME, value=token)
        token = KeyReference(name=KEY_NAME)
    saved = json.dumps({'token': token.model_dump()})
    await asyncio.to_thread(save_key_connection, account=ACCOUNT, token=token, value=saved)
    return f'PostHog connects with {token.name} from /keys.'


def _token(_: RunContext[None]) -> str | None:
    """Resolve the named key for this run; no key chosen means no PostHog tools, a deleted key raises."""
    reference = saved_key()
    return None if reference is None else resolve_key(token=reference)


def _browser_client(read_only: bool) -> Client[StreamableHttpTransport]:
    """A PostHog connection that signs in on first use and allows the browser as long as `/mcp` does.

    With a custom client harness `PostHog` cannot send the read-only header, and its own `read_only` would filter
    out PostHog's single `posthog` tool, so the header is set here.
    """
    transport = StreamableHttpTransport(
        POSTHOG_MCP_URL,
        headers={'x-posthog-read-only': 'true'} if read_only else {},
        auth=sign_in(TOKENS),
        httpx_client_factory=http_client,
    )
    return Client(transport, init_timeout=OAUTH_TIMEOUT)
