"""The built-in `pylon` plugin: Pylon's support issues, accounts, and contacts through harness `Pylon`.

`PYLON_ACCESS_TOKEN` wins when it is set. Otherwise CLAI signs in through the browser the way `/mcp`
OAuth servers do, keeping the tokens in the keyring so a restart does not mean signing in again.
Harness `Pylon(auth='oauth')` would keep them in memory and give the browser the 5-second connect
timeout, so CLAI builds the client itself.
"""

import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.pylon import Pylon

from .mcp import OAUTH_TIMEOUT, http_client, sign_in
from .plugins import DepsT, PluginHost

PYLON_MCP_URL = 'https://mcp.usepylon.com'
"""Harness `Pylon`'s endpoint, repeated here because a custom client owns its URL."""

TOKEN_ACCOUNT = 'plugin_pylon'
"""Stored as `mcp-plugin_pylon`. `/mcp` server names cannot contain `_`, so no server shares these tokens."""


class PylonSettings(BaseModel):
    """The JSON a `pylon` declaration may carry."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    browser_sign_in: bool = Field(
        default=True, description='Sign in through the browser when `PYLON_ACCESS_TOKEN` is unset.'
    )
    read_only: bool = Field(default=False, description="Keep only the tools Pylon's server labels read-only.")


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Pylon` with the environment token, else browser sign-in; refuse to load without either."""
    settings = host.settings(PylonSettings)
    if os.environ.get('PYLON_ACCESS_TOKEN'):
        host.add(Pylon[DepsT](read_only=settings.read_only))
    elif settings.browser_sign_in:
        host.add(Pylon[DepsT](client=_browser_client(), read_only=settings.read_only))
    else:
        raise UserError('Set `PYLON_ACCESS_TOKEN`, or turn `browser_sign_in` back on, to connect to Pylon.')


def _browser_client() -> Client[StreamableHttpTransport]:
    """A Pylon connection that signs in on first use and allows the browser as long as `/mcp` does."""
    transport = StreamableHttpTransport(
        url=PYLON_MCP_URL, auth=sign_in(TOKEN_ACCOUNT), httpx_client_factory=http_client
    )
    return Client(transport, init_timeout=OAUTH_TIMEOUT)
