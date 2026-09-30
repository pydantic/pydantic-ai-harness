"""OpenRouter's browser authorization-code flow with S256 PKCE.

Protocol: https://openrouter.ai/docs/use-cases/oauth-pkce. OpenRouter returns
an API key, not refresh tokens, and uses PKCE rather than an OAuth state field.
"""

import webbrowser
from collections.abc import Callable
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_ai.exceptions import UserError
from rich.console import Console

from . import oauth_loopback
from .auth import ReadLine, read_line
from .oauth_loopback import LoopbackLogin, Provider, pkce

_PROVIDER = Provider(
    name='OpenRouter',
    retry='Connect again through /add_model > openrouter.',
    fallback='Try again or enter an API key.',
)


class KeyResponse(BaseModel):
    """Only accept a nonempty key from the exchange endpoint."""

    key: SecretStr = Field(min_length=1)


def authorization_code(*, text: str) -> str:
    """Accept a pasted callback URL or a bare authorization code without echoing it."""
    return oauth_loopback.authorization_code(text=text, provider=_PROVIDER)


class OpenRouterAuth:
    """Own one browser login, including its listener and terminal prompt."""

    def __init__(
        self,
        *,
        console: Console,
        read_line: ReadLine = read_line,
        open_browser: Callable[[str], bool] = webbrowser.open,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 300,
    ) -> None:
        """Inject terminal and HTTP boundaries without opening a listener yet."""
        self.login_flow = LoopbackLogin(
            _PROVIDER, console=console, read_line=read_line, open_browser=open_browser, timeout=timeout
        )
        self.transport = transport

    async def login(self) -> SecretStr:
        """Return a key without saving it; cancellation leaves existing credentials alone."""
        verifier, challenge = pkce()

        def authorize_url(redirect_uri: str) -> str:
            return 'https://openrouter.ai/auth?' + urlencode(
                {'callback_url': redirect_uri, 'code_challenge': challenge, 'code_challenge_method': 'S256'}
            )

        async def exchange(code: str, redirect_uri: str) -> SecretStr:
            return await self.exchange(code=code, verifier=verifier)

        return await self.login_flow.run(authorize_url=authorize_url, exchange=exchange)

    async def exchange(self, *, code: str, verifier: str) -> SecretStr:
        """Exchange only at OpenRouter; never expose response bodies or follow redirects."""
        async with httpx.AsyncClient(transport=self.transport, timeout=30, follow_redirects=False) as client:
            try:
                response = await client.post(
                    'https://openrouter.ai/api/v1/auth/keys',
                    json={'code': code, 'code_verifier': verifier, 'code_challenge_method': 'S256'},
                )
                response.raise_for_status()
                return KeyResponse.model_validate_json(response.content).key
            except (httpx.HTTPError, ValidationError):
                raise UserError(
                    'OpenRouter key exchange failed. Connect again through /add_model > openrouter.'
                ) from None
