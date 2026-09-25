"""The built-in `logfire_mcp` plugin: harness `LogfireMCP` with CLAI's key store and keyring-backed OAuth."""

import os
import webbrowser
from collections.abc import Callable
from typing import Annotated, Literal

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import AfterValidator, BaseModel, ConfigDict
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP

from .api_keys import KeyReference, load_keys, normalize_name, resolve_key
from .mcp import HTTPServer, TokenStore, http_client, oauth
from .plugins import PluginHost

TOKEN_ACCOUNT = 'logfire_mcp'
"""OAuth tokens live under this `TokenStore` name. `/mcp` server names cannot contain `_`, so none shares it."""

_URLS = {'us': LOGFIRE_US_MCP_URL, 'eu': LOGFIRE_EU_MCP_URL}


class LogfireMCPSettings(BaseModel):
    """Non-secret options. Keys come from the environment or `/keys`, never from plugin settings."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    key: Annotated[str, AfterValidator(lambda name: normalize_name(name=name))] | None = None
    """The name of a key saved with `/keys`, resolved at the start of each run."""
    oauth: bool = True
    """Sign in through the browser when neither `key` nor `LOGFIRE_API_KEY` is set."""
    region: Literal['us', 'eu'] = 'us'
    read_only: bool = True
    """Offer only the tools the server marks read-only, so the agent cannot change Logfire resources."""


def activate(host: PluginHost[None]) -> None:
    """Add `LogfireMCP`, or refuse to load when no credential could connect it."""
    host.add(_capability(settings=host.settings(LogfireMCPSettings)))


def _capability(*, settings: LogfireMCPSettings) -> LogfireMCP[None]:
    url = _URLS[settings.region]
    if settings.key is not None:
        return LogfireMCP[None](auth=_saved_key(name=settings.key), url=url, read_only=settings.read_only)
    if os.environ.get('LOGFIRE_API_KEY'):
        return LogfireMCP[None](url=url, read_only=settings.read_only)
    if settings.oauth:
        return LogfireMCP[None](client=_oauth_client(url=url), read_only=settings.read_only)
    raise UserError('Set `LOGFIRE_API_KEY`, save a key with /keys and set `key`, or turn `oauth` back on.')


def _saved_key(*, name: str) -> Callable[[RunContext[None]], str]:
    if name not in load_keys():
        raise UserError(f'No saved API key is named {name}. Add it with /keys.')
    reference = KeyReference(name=name)
    # Resolved per run, like model connections, so a replaced key applies next turn and a deleted one fails closed.
    return lambda _ctx: resolve_key(token=reference)


def _oauth_client(*, url: str) -> Client[StreamableHttpTransport]:
    server = HTTPServer.model_validate({'type': 'http', 'url': url, 'auth': 'oauth'})
    if not TokenStore(TOKEN_ACCOUNT).signed_in():
        try:
            webbrowser.get()
        except webbrowser.Error:
            raise UserError(
                'Logfire sign-in needs a browser. Set `LOGFIRE_API_KEY`, or save a key with /keys and set `key`.'
            ) from None
    transport = StreamableHttpTransport(url, auth=oauth(TOKEN_ACCOUNT, server), httpx_client_factory=http_client)
    return Client(transport, init_timeout=server.init_timeout())
