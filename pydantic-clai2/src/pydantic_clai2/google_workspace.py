"""The built-in `google_workspace` plugin: Google's hosted Workspace MCP servers, through harness `GoogleWorkspace`."""

import os

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.google_workspace import GoogleWorkspace, GoogleWorkspaceService

from .api_keys import load_keys
from .plugins import DepsT, PluginHost

TOKEN_NAME = 'GOOGLE_ACCESS_TOKEN'
"""The saved `/keys` name checked first, then the environment variable of the same name."""

MISSING_TOKEN = (
    f'Google Workspace needs a Google OAuth access token. Save it in /keys as {TOKEN_NAME} '
    f'or set the {TOKEN_NAME} environment variable, then enable the plugin again.'
)


class GoogleWorkspaceSettings(BaseModel):
    """The JSON a `google_workspace` declaration may carry. The token never goes here."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    services: list[GoogleWorkspaceService] = Field(
        default_factory=lambda: list[GoogleWorkspaceService](['gmail', 'calendar', 'drive']),
        min_length=1,
        description='Workspace products to connect; the token must carry their scopes.',
    )
    read_only: bool = Field(
        default=True, description='Keep only the tools Google marks as read-only; CLAI runs tools without approval.'
    )


def access_token() -> str | None:
    """Read the token at use time, so replacing the saved key reaches the next turn without a reload."""
    saved = load_keys().get(TOKEN_NAME)
    return (saved.get_secret_value() if saved is not None else None) or os.environ.get(TOKEN_NAME) or None


def activate(host: PluginHost[DepsT]) -> None:
    """Refuse to load without a token rather than register a capability whose every run fails."""
    settings = host.settings(GoogleWorkspaceSettings)
    if access_token() is None:
        raise UserError(MISSING_TOKEN)

    def token(ctx: RunContext[DepsT]) -> str:
        # An access token expires after about an hour, so each run reads the current one.
        # `GoogleWorkspace` drops the tools for a run whose token is empty; say why instead.
        value = access_token()
        if value is None:
            raise UserError(MISSING_TOKEN)
        return value

    host.add(GoogleWorkspace[DepsT](services=settings.services, auth=token, read_only=settings.read_only))
