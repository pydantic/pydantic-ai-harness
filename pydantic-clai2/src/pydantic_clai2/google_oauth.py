"""Sign in with Google for the `google_workspace` plugin: your own Desktop OAuth client, PKCE, and a refresh token.

Protocol: https://developers.google.com/identity/protocols/oauth2/native-app. Google has no
dynamic client registration, so the user brings a client ID and secret. Only key names and
non-secret facts leave /keys; access tokens stay in memory.
"""

import asyncio
import secrets
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.google_workspace import GoogleWorkspaceService
from rich.console import Console
from typing_extensions import Self

from .api_keys import KeyReference, load_keys, save_key
from .auth import ReadLine, read_line
from .oauth_loopback import LoopbackLogin, Provider, pkce

AUTHORIZE_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
TOKEN_URL = 'https://oauth2.googleapis.com/token'
REFRESH_LABEL = 'GOOGLE_REFRESH_TOKEN'
SECRET_LABEL = 'GOOGLE_CLIENT_SECRET'
AGAIN = 'Run /google_workspace and sign in with Google again.'

_AUTH = 'https://www.googleapis.com/auth/'
_DRIVE = ('drive.readonly', 'drive.file')
SCOPES: Mapping[GoogleWorkspaceService, tuple[str, ...]] = {
    'gmail': ('gmail.readonly', 'gmail.compose'),
    'drive': _DRIVE,
    'docs': (*_DRIVE, 'documents.readonly', 'documents'),
    'sheets': (*_DRIVE, 'spreadsheets.readonly', 'spreadsheets'),
    'slides': (*_DRIVE, 'presentations.readonly', 'presentations'),
    'calendar': ('calendar.calendarlist.readonly', 'calendar.events.freebusy', 'calendar.events.readonly'),
    'chat': (
        'chat.spaces.readonly',
        'chat.memberships.readonly',
        'chat.messages.readonly',
        'chat.messages.create',
        'chat.users.readstate',
    ),
    'people': ('directory.readonly', 'userinfo.profile', 'contacts.readonly'),
}
"""Per-server scopes from https://developers.google.com/workspace/guides/configure-mcp-servers."""


def scopes_for(services: Sequence[GoogleWorkspaceService]) -> list[str]:
    """The full scope URLs the selected products' MCP servers need."""
    return sorted({_AUTH + scope for service in services for scope in SCOPES[service]})


class SignedIn(BaseModel):
    """What a sign-in leaves in the credential store: key names and non-secret facts, never a token."""

    model_config = ConfigDict(extra='forbid')
    client_id: str = Field(min_length=1)
    client_secret: KeyReference
    refresh_token: KeyReference
    scopes: list[str]
    """What Google granted, which can be less than what was asked for."""

    @model_validator(mode='after')
    def _separate_keys(self) -> Self:
        # Saving a rotated refresh token would otherwise overwrite the client secret.
        if self.client_secret.name == self.refresh_token.name:
            raise ValueError('The client secret and refresh token need different /keys entries.')
        return self

    def problem(self, *, client_id: str, services: Sequence[GoogleWorkspaceService]) -> str | None:
        """Why this sign-in no longer matches the settings, or `None` when it still does."""
        if client_id != self.client_id:
            return f'The OAuth client ID changed since you signed in. {AGAIN}'
        if set(scopes_for(services)) - set(self.scopes):
            return f'The selected products need Google permissions this sign-in does not have. {AGAIN}'
        return None


class Tokens(BaseModel):
    """The fields CLAI uses from Google's token endpoint."""

    access_token: SecretStr = Field(min_length=1)
    expires_in: int = Field(gt=0)
    refresh_token: SecretStr | None = Field(default=None, min_length=1)
    scope: str = ''


class SignInTokens(BaseModel):
    """A sign-in's tokens, which must include the refresh token that makes it last."""

    tokens: Tokens
    refresh_token: SecretStr


class GoogleOAuth:
    """Browser sign-in plus an in-memory cache of the access tokens a refresh token mints."""

    def __init__(
        self,
        *,
        console: Console,
        read_line: ReadLine = read_line,
        open_browser: Callable[[str], bool] = webbrowser.open,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Inject terminal, HTTP, and clock boundaries; nothing is read or opened yet."""
        self.console = console
        self.read_line = read_line
        self.open_browser = open_browser
        self.transport = transport
        self.timeout = timeout
        self.clock = clock
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}

    async def sign_in(self, *, client_id: str, client_secret: str, scopes: Sequence[str]) -> SignInTokens:
        """Run the browser flow and return tokens without saving anything."""
        verifier, challenge = pkce()
        state = secrets.token_urlsafe(24)
        provider = Provider(
            name='Google',
            retry=AGAIN,
            fallback='Try again, or choose an access token key in /google_workspace.',
            state=state,
        )

        def authorize_url(redirect_uri: str) -> str:
            return f'{AUTHORIZE_URL}?' + urlencode(
                {
                    'client_id': client_id,
                    'redirect_uri': redirect_uri,
                    'response_type': 'code',
                    'scope': ' '.join(scopes),
                    'code_challenge': challenge,
                    'code_challenge_method': 'S256',
                    'state': state,
                    # Offline access plus forced consent is what makes Google return a refresh token.
                    'access_type': 'offline',
                    'prompt': 'consent',
                }
            )

        async def exchange(code: str, redirect_uri: str) -> Tokens:
            form = {
                'grant_type': 'authorization_code',
                'code': code,
                'code_verifier': verifier,
                'redirect_uri': redirect_uri,
                'client_id': client_id,
                'client_secret': client_secret,
            }
            failed = 'Google sign-in failed. Check the OAuth client ID and secret, then sign in again.'
            return await self._request(form, rejected=failed, failed=failed)

        login = LoopbackLogin(
            provider,
            console=self.console,
            read_line=self.read_line,
            open_browser=self.open_browser,
            timeout=self.timeout,
        )
        tokens = await login.run(authorize_url=authorize_url, exchange=exchange)
        if tokens.refresh_token is None:
            raise UserError(f'Google did not return a refresh token. {AGAIN}')
        return SignInTokens(tokens=tokens, refresh_token=tokens.refresh_token)

    def remember(self, signed_in: SignedIn, *, refresh_token: str, tokens: Tokens) -> None:
        """Cache an access token until a minute before Google says it expires."""
        key = (signed_in.client_id, refresh_token)
        self._cache[key] = (tokens.access_token.get_secret_value(), self.clock() + tokens.expires_in - 60)

    async def access_token(self, signed_in: SignedIn) -> str:
        """A current access token, refreshed when the cached one is about to expire.

        Keys are read on every call, so replacing or deleting them in /keys applies to the next turn.
        """
        keys = await asyncio.to_thread(load_keys)
        for reference in (signed_in.client_secret, signed_in.refresh_token):
            if reference.name not in keys:
                raise UserError(f'Saved key {reference.name} is missing. {AGAIN}')
        refresh_token = keys[signed_in.refresh_token.name].get_secret_value()
        cached = self._cache.get((signed_in.client_id, refresh_token))
        if cached is not None and cached[1] > self.clock():
            return cached[0]
        form = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'client_id': signed_in.client_id,
            'client_secret': keys[signed_in.client_secret.name].get_secret_value(),
        }
        tokens = await self._request(
            form,
            rejected=f'Google rejected the saved sign-in; it may have expired or been revoked. {AGAIN}',
            failed='Could not refresh the Google access token. Check your connection and try again.',
        )
        if tokens.refresh_token is not None:
            refresh_token = tokens.refresh_token.get_secret_value()
            await asyncio.to_thread(save_key, name=signed_in.refresh_token.name, value=refresh_token)
        self.remember(signed_in, refresh_token=refresh_token, tokens=tokens)
        return tokens.access_token.get_secret_value()

    async def _request(self, form: dict[str, str], *, rejected: str, failed: str) -> Tokens:
        """Post to Google's token endpoint; never expose response bodies or follow redirects."""
        async with httpx.AsyncClient(transport=self.transport, timeout=30, follow_redirects=False) as client:
            try:
                response = await client.post(TOKEN_URL, data=form)
            except httpx.HTTPError:
                raise UserError(failed) from None
        if response.status_code in (400, 401):
            raise UserError(rejected)
        try:
            response.raise_for_status()
            return Tokens.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            raise UserError(failed) from None
