"""The built-in `google_workspace` plugin: Google's hosted Workspace MCP servers, through harness `GoogleWorkspace`."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Generic, get_args

from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.google_workspace import GoogleWorkspace, GoogleWorkspaceService
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from ._rendering import markdown_style
from .api_keys import KeyReference, load_keys, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import delete_credentials, load_codex_credentials
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, run_flow
from .google_oauth import REFRESH_LABEL, SECRET_LABEL, GoogleOAuth, SignedIn, scopes_for
from .menu_worker import menu_key, run_worker
from .plugins import DepsT, PluginHost

TOKEN_LABEL = 'GOOGLE_ACCESS_TOKEN'
"""The `/keys` name used until `/google_workspace` picks another; a label, not an environment variable."""

ACCOUNT = 'google-workspace'
"""Credential-store account holding the chosen key names or sign-in, never a secret value."""

SERVICES: tuple[GoogleWorkspaceService, ...] = get_args(GoogleWorkspaceService)
_CLIENT_SUFFIX = '.apps.googleusercontent.com'
_UNSET = '(not set)'


class GoogleWorkspaceSettings(BaseModel):
    """The JSON a `google_workspace` declaration may carry. Plain SQLite, so no secret goes here."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    services: list[GoogleWorkspaceService] = Field(
        default_factory=lambda: list[GoogleWorkspaceService](['gmail', 'calendar', 'drive']),
        min_length=1,
        description='Workspace products to connect; the token must carry their scopes.',
    )
    read_only: bool = Field(
        default=True, description='Keep only the tools Google marks as read-only; CLAI runs tools without approval.'
    )
    include_instructions: bool = Field(
        default=True, description="Pass the Google servers' own instructions to the agent."
    )
    client_id: str = Field(
        default='', description='OAuth client ID used by Sign in with Google; the client secret stays in /keys.'
    )


class Connection(BaseModel):
    """Which saved `/keys` entry holds a ready-made access token."""

    model_config = ConfigDict(extra='forbid')
    token: KeyReference


Choice = Connection | SignedIn
_CHOICE: TypeAdapter[Choice] = TypeAdapter(Choice)


def load_connection() -> Choice:
    """The saved choice, defaulting to the conventional access-token label so an existing key needs no setup."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return Connection(token=KeyReference(name=TOKEN_LABEL))
    try:
        return _CHOICE.validate_json(raw)
    except ValidationError:
        raise UserError('The saved Google Workspace key choice is invalid. Run /google_workspace again.') from None


def missing(name: str) -> str:
    """Explain how to supply the token, naming the key the plugin is looking for."""
    return (
        f'Google Workspace needs a Google OAuth access token in /keys as {name}. '
        'Run /google_workspace to sign in with Google, or to choose a saved key or enter one.'
    )


def setup_problem(settings: GoogleWorkspaceSettings) -> str | None:
    """What stops the next turn, checked without network access; `None` when nothing is known to."""
    try:
        choice = load_connection()
    except UserError as exc:
        return str(exc)
    if isinstance(choice, Connection):
        return None if choice.token.name in load_keys() else missing(choice.token.name)
    return choice.problem(client_id=settings.client_id, services=settings.services)


async def access_token(settings: GoogleWorkspaceSettings, oauth: GoogleOAuth) -> str:
    """Resolve at use time: /keys changes reach the next turn, and anything missing fails closed."""
    choice = await asyncio.to_thread(load_connection)
    if isinstance(choice, Connection):
        if choice.token.name not in await asyncio.to_thread(load_keys):
            raise UserError(missing(choice.token.name))
        return await asyncio.to_thread(resolve_key, token=choice.token)
    if problem := choice.problem(client_id=settings.client_id, services=settings.services):
        raise UserError(problem)
    return await oauth.access_token(choice)


async def _ask(label: str) -> str | KeyReference | None:
    """Pick a saved key or enter a value, which is not saved yet; `None` when cancelled."""
    prompt: PromptSession[str] = PromptSession()
    return await prompt_api_key(prompt=prompt, label=label)


async def _stored(secret: str | KeyReference, *, name: str) -> KeyReference:
    """Save an entered value under `name`; a picked key is already stored."""
    if isinstance(secret, KeyReference):
        return secret
    await asyncio.to_thread(save_key, name=name, value=secret)
    return KeyReference(name=name)


async def choose_key() -> str:
    """Use a ready-made access token from /keys; only the key's name is remembered outside /keys."""
    answer = await _ask(f'Google OAuth access token (saved in /keys as {TOKEN_LABEL}): ')
    if answer is None:
        return 'Google Workspace key unchanged.'
    token = await _stored(answer, name=TOKEN_LABEL)
    connection = Connection(token=token)
    await asyncio.to_thread(save_key_connection, account=ACCOUNT, token=token, value=connection.model_dump_json())
    return f'Google Workspace uses the saved key {token.name} from the next turn.'


async def sign_in(host: PluginHost[DepsT], oauth: GoogleOAuth) -> str:
    """Sign in through the browser and keep the refresh token in /keys; replaces any access-token choice.

    Nothing is saved until Google has issued the tokens, so a denied or failed sign-in leaves an
    earlier one, including its client secret, working.
    """
    settings = host.settings(GoogleWorkspaceSettings)
    if not settings.client_id:
        return 'Set the OAuth client ID first, then sign in with Google.'
    answer = await _ask(f'Google OAuth client secret (saved in /keys as {SECRET_LABEL}): ')
    if answer is None:
        return 'Google sign-in cancelled.'
    if isinstance(answer, KeyReference):
        if answer.name == REFRESH_LABEL:
            raise UserError(f'{REFRESH_LABEL} holds the refresh token. Choose another key for the client secret.')
        client_secret = await asyncio.to_thread(resolve_key, token=answer)
    elif not (client_secret := answer.strip()):
        raise UserError('A client secret is required.')
    wanted = scopes_for(settings.services)
    result = await oauth.sign_in(client_id=settings.client_id, client_secret=client_secret, scopes=wanted)
    granted = result.tokens.scope.split()
    if refused := set(wanted) - set(granted):
        raise UserError(
            f'Google did not grant {len(refused)} of the requested permissions. Sign in again and allow them all, '
            'or turn off the products that need them.'
        )
    secret = await _stored(answer if isinstance(answer, KeyReference) else client_secret, name=SECRET_LABEL)
    refresh_token = result.refresh_token.get_secret_value()
    await asyncio.to_thread(save_key, name=REFRESH_LABEL, value=refresh_token)
    signed_in = SignedIn(
        client_id=settings.client_id,
        client_secret=secret,
        refresh_token=KeyReference(name=REFRESH_LABEL),
        scopes=sorted(granted),
    )
    await asyncio.to_thread(
        save_key_connection,
        account=ACCOUNT,
        token=signed_in.refresh_token,
        references=[secret],
        value=signed_in.model_dump_json(),
    )
    oauth.remember(signed_in, refresh_token=refresh_token, tokens=result.tokens)
    return f'Signed in with Google. The refresh token is saved in /keys as {REFRESH_LABEL}.'


class _Leave(Exception):
    """Leave the menu worker so `action` can prompt or open a browser on the event loop, then reopen the menu."""

    def __init__(self, action: Callable[[], Awaitable[str]]) -> None:
        self.action = action
        super().__init__()


def _leaving(action: Callable[[], Awaitable[str]]) -> Callable[[], list[str]]:
    def leave() -> list[str]:
        raise _Leave(action)

    return leave


_BOOLEAN = ('true', 'false')


class SettingsSource(Generic[DepsT]):
    """The `/google_workspace` rows. Every edit is saved to the plugin declaration as it is made."""

    title = 'Google Workspace'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Read and write through `host`, so edits reach the next run without reloading."""
        self._settings = partial(host.settings, GoogleWorkspaceSettings)
        self._save_settings = host.save_settings
        self.log: list[str] = []
        """What changed, in order; kept here because leaving for a prompt restarts the field menu."""

    def rows(self) -> list[FieldRow]:
        """How the token is obtained first, then the capability's non-secret options."""
        defaults = GoogleWorkspaceSettings()
        return [
            FieldRow(
                key='sign_in',
                label='Sign in with Google',
                default='not signed in',
                description='Enter opens Google in your browser and keeps a refresh token in /keys as '
                f'{REFRESH_LABEL}, so the access token renews itself. Needs the OAuth client ID below and asks '
                f'for the client secret, kept in /keys as {SECRET_LABEL}. r signs out.',
                note='browser',
            ),
            FieldRow(
                key='client_id',
                label='OAuth client ID',
                default=_UNSET,
                description='The client ID of a Desktop app OAuth client in your Google Cloud project, ending in '
                f'{_CLIENT_SUFFIX}. Not secret; stored in plugin settings.',
            ),
            FieldRow(
                key='token',
                label='Access token key',
                default=TOKEN_LABEL,
                description='Instead of signing in: which /keys entry holds a ready-made Google OAuth access '
                'token. Enter lists saved key names or asks for a new token without echoing it; only the name is '
                f'stored here. r goes back to {TOKEN_LABEL}.',
                note='name in /keys',
            ),
            FieldRow(
                key='services',
                label='Products',
                default=', '.join(defaults.services),
                description='Workspace products to connect. Enter opens a searchable checklist. Adding a product '
                'after signing in needs a new sign-in for its permissions.',
            ),
            FieldRow(
                key='read_only',
                label='Read-only tools',
                default='true',
                choices=_BOOLEAN,
                allow_custom=False,
                description='Keep only the tools Google marks as read-only. CLAI runs tools without asking, so '
                'false lets the agent send, change, and delete.',
            ),
            FieldRow(
                key='include_instructions',
                label='Server instructions',
                default='true',
                choices=_BOOLEAN,
                allow_custom=False,
                description="Pass the Google servers' own instructions to the agent.",
            ),
        ]

    def current(self, row: FieldRow) -> str:
        """The value as the menu shows it; token rows show only key names or sign-in state."""
        if row.key in ('sign_in', 'token'):
            try:
                choice = load_connection()
            except UserError:
                return '(invalid; Enter to choose again)'
            if row.key == 'sign_in':
                return 'signed in' if isinstance(choice, SignedIn) else 'not signed in'
            return '(signed in with Google)' if isinstance(choice, SignedIn) else choice.token.name
        value = getattr(self._settings(), row.key)
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, str):
            return value or _UNSET
        return ', '.join(value)

    def problem(self, row: FieldRow, text: str) -> str | None:
        """The client ID is checked for Google's shape; the true/false rows take only their choices."""
        if row.key == 'client_id':
            return None if text.endswith(_CLIENT_SUFFIX) else f'Google client IDs end with {_CLIENT_SUFFIX}.'
        return None if text in _BOOLEAN else 'Choose true or false.'

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save one typed or chosen option."""
        if row.key == 'client_id':
            return self._save(f'Saved {row.label}.', client_id=raw)
        return self._save(f'Saved {row.label}: {raw}.', **{row.key: raw == 'true'})

    def reset(self, row: FieldRow) -> str:
        """Return one option to its default; either token row forgets the saved choice or sign-in."""
        if row.key in ('sign_in', 'token'):
            delete_credentials(account=ACCOUNT)
            message = f'Google Workspace uses the saved key {TOKEN_LABEL} again.'
            self.log.append(message)
            return message
        default = getattr(GoogleWorkspaceSettings(), row.key)
        return self._save(f'Reset {row.label}.', **{row.key: default})

    def edit_services(self, runners: Runners) -> list[str]:
        """Toggle products until Esc; each toggle is saved, and the last product cannot be removed."""
        cursor = 0
        while True:
            chosen = self._settings().services
            menu = (
                MenuBuilder('Google Workspace products')
                .style(markdown_style())
                .items([MenuItem(f'[{"x" if name in chosen else " "}] {name}', value=name) for name in SERVICES])
                .searchable()
                .initial_index(cursor)
                .footer_hint('type to filter - Enter toggle - Esc back')
                .key_source(menu_key)
                .build()
            )
            result = runners.run_choice(menu)
            service = result.item.value if result.item is not None else None
            if result.cancelled or service not in SERVICES:
                return []
            cursor = SERVICES.index(service)
            if service in chosen and len(chosen) == 1:
                self.log.append('Google Workspace needs at least one product.')
                continue
            services = [name for name in SERVICES if (name in chosen) != (name == service)]
            self._save(f'Products: {", ".join(services)}.', services=services)

    def _save(self, message: str, **changes: object) -> str:
        current = self._settings().model_dump()
        self._save_settings(GoogleWorkspaceSettings.model_validate({**current, **changes}))
        self.log.append(message)
        return message


async def configure(
    host: PluginHost[DepsT], args: list[str], *, oauth: GoogleOAuth, runners: Runners = TERMINAL
) -> str:
    """Open the settings menu; Esc closes it with every edit already saved."""
    if args:
        raise ValueError('Usage: /google_workspace (opens the settings menu)')
    source = SettingsSource(host)
    submenus = {
        'services': partial(source.edit_services, runners),
        'token': _leaving(choose_key),
        'sign_in': _leaving(partial(sign_in, host, oauth)),
    }

    def flow() -> list[str]:
        return run_flow(FieldMenu(source, searchable=False), runners, submenus=submenus)

    while True:
        try:
            await run_worker(flow)
        except _Leave as leave:
            try:
                source.log.append(await leave.action())
            except UserError as exc:
                source.log.append(str(exc))
            continue
        return '\n'.join(source.log) or 'No changes.'


def activate(host: PluginHost[DepsT]) -> None:
    """Load without a token so `/google_workspace` is available to supply one; every run needs it."""
    settings = host.settings(GoogleWorkspaceSettings)
    oauth = GoogleOAuth(console=host.console)
    host.commands.register(
        Command(
            name='google_workspace',
            description='Google Workspace settings: sign-in or /keys token, products, and read-only tools',
            handler=partial(configure, host, oauth=oauth),
        )
    )
    if problem := setup_problem(settings):
        host.console.print(problem, style=theme.color(theme.WARNING), markup=False)

    async def workspace(ctx: RunContext[DepsT]) -> GoogleWorkspace[DepsT]:
        # Built per run so settings edits and a fresh access token apply to the next turn without a reload.
        settings = host.settings(GoogleWorkspaceSettings)
        token = await access_token(settings, oauth)

        def auth(run: RunContext[DepsT]) -> str:
            return token

        return GoogleWorkspace[DepsT](
            services=settings.services,
            auth=auth,
            read_only=settings.read_only,
            include_instructions=settings.include_instructions,
        )

    host.add(workspace)
