"""The built-in `notion` plugin: harness `Notion`, connected with `NOTION_ACCESS_TOKEN` or a browser sign-in."""

import os
from collections.abc import Iterable
from functools import partial
from typing import Literal

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.notion import Notion

from .commands import Command
from .mcp import OAUTH_TIMEOUT, TokenStore, browser_sign_in, http_client
from .plugins import DepsT, PluginHost

NOTION_MCP_URL = 'https://mcp.notion.com/mcp'
TOKEN_ENV = 'NOTION_ACCESS_TOKEN'
TOKENS = TokenStore('plugin_notion')
"""Browser sign-in tokens. `/mcp` server names cannot contain `_`, so this account never collides with one."""


class NotionSettings(BaseModel):
    """The JSON a `notion` declaration may carry. Tokens are not accepted here; they stay in the environment."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Literal['token', 'oauth'] | None = Field(
        default=None,
        description=f'`token` reads `{TOKEN_ENV}`, `oauth` signs in through the browser; unset picks `token` when '
        f'`{TOKEN_ENV}` is set.',
    )
    read_only: bool = Field(default=False, description='Keep only the tools the server marks as read-only.')


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Notion`, refusing to load in `token` mode without a token rather than failing on the first run."""
    settings = host.settings(NotionSettings)
    token = os.environ.get(TOKEN_ENV)
    if settings.auth == 'token' and not token:
        raise UserError(
            f'Set {TOKEN_ENV} to a Notion OAuth access token, or drop `"auth": "token"` to sign in through the browser.'
        )
    read_only = settings.read_only
    uses_token = bool(token) and settings.auth != 'oauth'
    if uses_token:
        host.add(Notion[DepsT](auth=token, read_only=read_only))
    else:

        def sign_in(_: RunContext[DepsT]) -> Notion[DepsT]:
            # Harness `auth='oauth'` keeps tokens in memory behind a 5-second handshake; this keeps them in the
            # keyring and allows the browser round trip, like an OAuth server added through `/mcp`. A client per
            # run reloads them from the keyring, so `/notion logout` applies from the next run.
            transport = StreamableHttpTransport(
                NOTION_MCP_URL, auth=browser_sign_in(TOKENS), httpx_client_factory=http_client
            )
            return Notion[DepsT](client=Client(transport, init_timeout=OAUTH_TIMEOUT), read_only=read_only)

        host.add(sign_in)
    host.commands.register(
        Command(
            name='notion',
            description='Sign out of Notion (/notion logout).',
            handler=partial(_command, uses_token=uses_token),
            complete=_complete,
        )
    )


async def _command(args: list[str], *, uses_token: bool) -> str:
    if args != ['logout']:
        raise ValueError('Usage: /notion logout')
    await to_thread.run_sync(TOKENS.forget)
    if uses_token:
        return (
            f'Cleared the saved browser sign-in, but this session connects with {TOKEN_ENV}, which /notion logout '
            'cannot revoke. /plugins disable notion stops using it.'
        )
    return 'Signed out of Notion. The next run opens the browser to sign in.'


def _complete(args: list[str]) -> Iterable[str]:
    return ('logout',) if len(args) <= 1 else ()
