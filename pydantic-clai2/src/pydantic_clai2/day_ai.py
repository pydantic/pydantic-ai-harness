"""The built-in `day_ai` plugin: harness `DayAI`, signed in with `DAY_AI_ACCESS_TOKEN` or through the browser.

Day AI issues access tokens only through OAuth. Without the variable, the plugin signs in the way `/mcp` does for an
OAuth server: FastMCP's browser flow, with tokens kept in the keyring under `mcp-day_ai`. `/mcp` server names cannot
contain underscores, so that credential never belongs to one of your servers.
"""

import os

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.day_ai import DayAI

from . import theme
from .mcp import TokenStore, browser_sign_in, http_client
from .plugins import DepsT, PluginHost, SessionStart

DAY_AI_MCP_URL = 'https://day.ai/api/mcp'
"""The hosted MCP endpoint harness `DayAI` connects to when it is given a token."""

TOKEN_ENV = 'DAY_AI_ACCESS_TOKEN'
TOKEN_ACCOUNT = 'day_ai'
"""The `/mcp` token store name, so the keyring credential is `mcp-day_ai`."""


class DayAISettings(BaseModel):
    """The JSON a `day_ai` declaration may carry. Tokens are not accepted here."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    oauth: bool = Field(default=True, description=f'Sign in through the browser when `{TOKEN_ENV}` is unset.')


def activate(host: PluginHost[DepsT]) -> None:
    """Add `DayAI` with the environment token, or sign in through the browser before the plugin counts as loaded."""
    settings = host.settings(DayAISettings)
    if os.environ.get(TOKEN_ENV):
        host.add(DayAI[DepsT]())
        return
    if not settings.oauth:
        raise UserError(f'Set {TOKEN_ENV} to connect to Day AI, or remove `"oauth": false` to sign in in the browser.')
    host.add(DayAI[DepsT](client=_transport()))

    @host.on('session_start')
    async def sign_in(_: SessionStart) -> None:  # pyright: ignore[reportUnusedFunction]
        if await to_thread.run_sync(TokenStore(TOKEN_ACCOUNT).signed_in):
            return
        if not host.console.is_terminal:
            raise UserError(f'Set {TOKEN_ENV}, or sign in to Day AI from an interactive CLAI session first.')
        host.console.print('Opening your browser to sign in to Day AI.', style=theme.color(theme.MUTED))
        # A throwaway connection runs the sign-in now, so a failure fails the load rather than the next prompt.
        async with Client(_transport()):
            pass


def _transport() -> StreamableHttpTransport:
    return StreamableHttpTransport(
        DAY_AI_MCP_URL, auth=browser_sign_in(TOKEN_ACCOUNT), httpx_client_factory=http_client
    )
