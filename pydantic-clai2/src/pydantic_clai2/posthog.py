"""The built-in `posthog` plugin: PostHog's hosted MCP server through harness `PostHog`, signed in for CLAI.

Harness `PostHog` builds a new connection for every run, so its `auth='oauth'` would keep its tokens in memory and
open the browser on every turn. This plugin hands `PostHog` one transport instead, whose OAuth tokens live in the
keyring like `/mcp`'s, so signing in once lasts across turns and launches. With its own transport `PostHog` cannot
send the read-only header or pick the URL, so both are set here, from the provider contract in
`pydantic_ai_harness.posthog`.
"""

import os
from typing import Literal

from anyio import to_thread
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.posthog import PostHog

from .api_keys import load_keys
from .commands import Command
from .mcp import TokenStore, http_client, sign_in
from .plugins import PluginHost

KEY_NAME = 'POSTHOG_PERSONAL_API_KEY'
"""The environment variable harness `PostHog` reads, and the `/keys` name CLAI looks up when it is unset."""

TOKENS = 'posthog_plugin'
"""OAuth tokens are the `mcp-posthog_plugin` credential. `/mcp` server names cannot contain `_`, so none shares it."""

_URL = 'https://mcp.posthog.com/mcp'
_SIGN_IN_STATES: dict[bool | None, str] = {
    True: 'signed in through the browser',
    False: 'not signed in; the first prompt that uses it opens the browser',
    None: 'in an unknown sign-in state: the keyring cannot be read',
}


class PostHogSettings(BaseModel):
    """The JSON a `posthog` declaration may carry."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(default=True, description='Ask PostHog to serve only the tools it marks read-only.')
    auth: Literal['api_key', 'oauth'] | None = Field(
        default=None,
        description=f'`api_key` requires `{KEY_NAME}`; `oauth` signs in through the browser; unset tries the key first.',
    )


def activate(host: PluginHost[None]) -> None:
    """Add `PostHog` over a key from the environment or `/keys`, else a browser sign-in kept in the keyring."""
    settings = host.settings(PostHogSettings)
    key, source = _api_key() if settings.auth != 'oauth' else (None, None)
    if key is None and settings.auth == 'api_key':
        raise UserError(f'Set {KEY_NAME} or save it in /keys, or remove `"auth": "api_key"` to sign in with a browser.')
    capability = PostHog[None](client=_transport(key, read_only=settings.read_only))
    host.add(capability)
    mode = 'read-only' if settings.read_only else 'read-write'

    async def command(args: list[str]) -> str:
        if args not in ([], ['logout']):
            raise ValueError('Usage: /posthog [logout]')
        if source is not None:
            if args:
                return f'PostHog uses {source}; unset or delete it to stop using it.'
            return f'PostHog ({mode}) uses {source}.'
        tokens = TokenStore(TOKENS)
        if args:
            await to_thread.run_sync(tokens.forget)
            # The live sign-in still holds the tokens it loaded, so later runs need a fresh one.
            capability.client = _transport(None, read_only=settings.read_only)
            return 'Signed out of PostHog. The next prompt that uses it opens the browser to sign in again.'
        state = _SIGN_IN_STATES[await to_thread.run_sync(tokens.signed_in)]
        return f'PostHog ({mode}) is {state}.'

    host.commands.register(
        Command(
            name='posthog',
            description='Show how PostHog is signed in, or sign out of it (/posthog logout).',
            handler=command,
            complete=lambda args: ['logout'] if len(args) <= 1 else [],
        )
    )


def _transport(key: str | None, *, read_only: bool) -> StreamableHttpTransport:
    return StreamableHttpTransport(
        _URL,
        headers={'x-posthog-read-only': 'true'} if read_only else {},
        auth=key if key is not None else sign_in(TOKENS),
        httpx_client_factory=http_client,
    )


def _api_key() -> tuple[str, str] | tuple[None, None]:
    if key := os.environ.get(KEY_NAME):
        return key, f'{KEY_NAME} from the environment'
    if saved := load_keys().get(KEY_NAME):
        return saved.get_secret_value(), f'{KEY_NAME} from /keys'
    return None, None
