"""The built-in `notion` plugin: harness `Notion`, connected with `NOTION_ACCESS_TOKEN` or a browser sign-in."""

import os
from collections.abc import Iterable
from typing import Literal

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field
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
    if token and settings.auth != 'oauth':
        host.add(Notion[DepsT](auth=token, read_only=settings.read_only))
    else:
        # Harness `auth='oauth'` keeps tokens in memory behind a 5-second handshake; CLAI keeps them in the
        # keyring and allows the browser round trip, the same as an OAuth server added through `/mcp`.
        transport = StreamableHttpTransport(
            NOTION_MCP_URL, auth=browser_sign_in(TOKENS), httpx_client_factory=http_client
        )
        host.add(Notion[DepsT](client=Client(transport, init_timeout=OAUTH_TIMEOUT), read_only=settings.read_only))
    host.commands.register(
        Command(
            name='notion',
            description='Sign out of Notion (/notion logout).',
            handler=_command,
            complete=_complete,
        )
    )


def _command(args: list[str]) -> str:
    if args != ['logout']:
        raise ValueError('Usage: /notion logout')
    TOKENS.forget()
    return 'Signed out of Notion. The next browser sign-in asks for your account again.'


def _complete(args: list[str]) -> Iterable[str]:
    return ('logout',) if len(args) <= 1 else ()
