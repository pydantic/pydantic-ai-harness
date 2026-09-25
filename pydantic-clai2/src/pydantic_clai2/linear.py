"""The built-in `linear` plugin: harness `Linear`, authenticated the way CLAI stores other secrets."""

import os

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.linear import Linear

from .api_keys import KeyReference, resolve_key
from .commands import Command
from .mcp import HTTPServer, TokenStore, http_client, oauth
from .plugins import DepsT, PluginHost

TOKEN_ENV = 'LINEAR_ACCESS_TOKEN'
TOKEN_ACCOUNT = 'linear_plugin'
"""`TokenStore` name for OAuth tokens. `/mcp` server names cannot contain `_`, so no `/mcp` server shares it."""
# The endpoints harness `Linear` connects to; with `client`, the plugin picks the URL itself.
_URL = 'https://mcp.linear.app/mcp'
_READ_ONLY_URL = 'https://mcp.linear.app/mcp/readonly'


class LinearSettings(BaseModel):
    """The JSON a `linear` declaration may carry. Secrets stay in the environment, `/keys`, or the keyring."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(default=True, description="Connect to Linear's read-only endpoint.")
    oauth: bool = Field(default=False, description='Sign in through the browser; tokens are kept in the keyring.')
    api_key: str | None = Field(
        default=None,
        min_length=1,
        description=f'Name of a key saved with `/set api_key`, used instead of `{TOKEN_ENV}`.',
    )

    @model_validator(mode='after')
    def _one_credential(self) -> 'LinearSettings':
        if self.oauth and self.api_key is not None:
            raise ValueError('Choose `oauth` or `api_key`, not both.')
        return self


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Linear`, or raise before adding anything when no credential is available."""
    settings = host.settings(LinearSettings)
    if settings.oauth:
        host.add(Linear[DepsT](client=_oauth_client(settings.read_only)))
        store = TokenStore(TOKEN_ACCOUNT)

        async def logout(args: list[str]) -> str:
            if args != ['logout']:
                raise ValueError('Usage: /linear logout')
            await to_thread.run_sync(store.forget)
            return 'Signed out of Linear. The next run opens the browser to sign in again.'

        host.commands.register(
            Command(name='linear', description='Sign out of Linear (/linear logout).', handler=logout)
        )
        return
    if settings.api_key is not None:
        token = resolve_key(token=KeyReference(name=settings.api_key))
    elif not (token := os.environ.get(TOKEN_ENV, '')):
        raise UserError(
            f'Linear needs a credential: set {TOKEN_ENV}, or save a key with /set api_key and declare '
            '`{"api_key": "NAME"}`, or declare `{"oauth": true}` to sign in through the browser. '
            'See PLUGINS.md "Linear".'
        )
    host.add(Linear[DepsT](auth=token, read_only=settings.read_only))


def _oauth_client(read_only: bool) -> Client[StreamableHttpTransport]:
    """Connect the way `/mcp` connects an OAuth server: keyring tokens, no redirects, time for a browser sign-in.

    Plain `Linear(auth='oauth')` keeps tokens in memory and allows the 5-second default for `initialize`, which
    a browser sign-in does not fit in. The URL carries `read_only` here: `Linear`'s own `read_only` with a
    `client` filters on tool annotations instead, which would be a second, different boundary.
    """
    server = HTTPServer.model_validate({'type': 'http', 'url': _READ_ONLY_URL if read_only else _URL, 'auth': 'oauth'})
    transport = StreamableHttpTransport(
        url=str(server.url), auth=oauth(TOKEN_ACCOUNT, server), httpx_client_factory=http_client
    )
    return Client(transport, init_timeout=server.init_timeout())
