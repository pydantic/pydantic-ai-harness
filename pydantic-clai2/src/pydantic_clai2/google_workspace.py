"""The built-in `google_workspace` plugin: Google's hosted Workspace MCP servers, through harness `GoogleWorkspace`."""

import asyncio

from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.google_workspace import GoogleWorkspace, GoogleWorkspaceService

from . import theme
from .api_keys import KeyReference, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import load_codex_credentials
from .plugins import DepsT, PluginHost

TOKEN_LABEL = 'GOOGLE_ACCESS_TOKEN'
"""The `/keys` name used until `/google_workspace` picks another; a label, not an environment variable."""

ACCOUNT = 'google-workspace'
"""Credential-store account holding the chosen key's name, never its value."""


class GoogleWorkspaceSettings(BaseModel):
    """The JSON a `google_workspace` declaration may carry. Plain SQLite, so the token never goes here."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    services: list[GoogleWorkspaceService] = Field(
        default_factory=lambda: list[GoogleWorkspaceService](['gmail', 'calendar', 'drive']),
        min_length=1,
        description='Workspace products to connect; the token must carry their scopes.',
    )
    read_only: bool = Field(
        default=True, description='Keep only the tools Google marks as read-only; CLAI runs tools without approval.'
    )


class Connection(BaseModel):
    """Which saved `/keys` entry the plugin authenticates with."""

    token: KeyReference


def load_connection() -> Connection:
    """The chosen key reference, defaulting to the conventional label so an existing key needs no setup."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return Connection(token=KeyReference(name=TOKEN_LABEL))
    try:
        return Connection.model_validate_json(raw)
    except ValidationError:
        raise UserError('The saved Google Workspace key choice is invalid. Run /google_workspace again.') from None


def missing(name: str) -> str:
    """Explain how to supply the token, naming the key the plugin is looking for."""
    return (
        f'Google Workspace needs a Google OAuth access token in /keys as {name}. '
        'Run /google_workspace to choose a saved key or enter one.'
    )


def access_token() -> str:
    """Resolve at use time: replacing the key in /keys reaches the next turn, and a deleted key fails closed."""
    reference = load_connection().token
    if reference.name not in load_keys():
        raise UserError(missing(reference.name))
    return resolve_key(token=reference)


async def configure(args: list[str]) -> str:
    """Pick a saved key or enter one; only the key's name is remembered outside /keys."""
    if args:
        raise ValueError('Usage: /google_workspace (the token is chosen or entered privately)')
    prompt: PromptSession[str] = PromptSession()
    token = await prompt_api_key(prompt=prompt, label=f'Google OAuth access token (saved in /keys as {TOKEN_LABEL}): ')
    if token is None:
        return 'Google Workspace key unchanged.'
    if not isinstance(token, KeyReference):
        await asyncio.to_thread(save_key, name=TOKEN_LABEL, value=token)
        token = KeyReference(name=TOKEN_LABEL)
    connection = Connection(token=token)
    await asyncio.to_thread(save_key_connection, account=ACCOUNT, token=token, value=connection.model_dump_json())
    return f'Google Workspace uses the saved key {token.name} from the next turn.'


def activate(host: PluginHost[DepsT]) -> None:
    """Load without a token so `/google_workspace` is available to supply one; every run needs it."""
    settings = host.settings(GoogleWorkspaceSettings)
    host.commands.register(
        Command(
            name='google_workspace',
            description='Choose or enter the Google Workspace access token, kept in /keys',
            handler=configure,
        )
    )
    try:
        access_token()
    except UserError as exc:
        host.console.print(str(exc), style=theme.color(theme.WARNING), markup=False)

    def token(ctx: RunContext[DepsT]) -> str:
        # `GoogleWorkspace` drops the tools for a run whose token is empty; failing says why instead.
        return access_token()

    host.add(GoogleWorkspace[DepsT](services=settings.services, auth=token, read_only=settings.read_only))
