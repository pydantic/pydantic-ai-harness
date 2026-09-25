"""The built-in `slack` plugin: harness `Slack`, set up in the settings menu that `/plugins configure slack` opens.

The user token lives in `/keys`; the credential store keeps only its name, so `/keys` refuses to rename it while
Slack uses it. The plugin's settings hold only harness `Slack`'s non-secret options.
"""

import asyncio
import os
from dataclasses import replace
from typing import Generic

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.slack import Slack

from . import theme
from .api_keys import KeyReference, load_keys, resolve_key, save_key_connection
from .credential_store import delete_credentials, load_codex_credentials
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from .menu_worker import run_worker
from .plugin_keys import choose_key, on_loop
from .plugins import DepsT, PluginHost, SessionStart, TurnStart

TOKEN_NAME = 'SLACK_USER_TOKEN'
"""The `/keys` label a newly entered token is saved under: harness `Slack`'s documented name, not an exported variable."""
SETUP = 'Run /plugins configure slack to choose its user token from /keys.'
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""

_ACCOUNT = 'slack'


class SlackSettings(BaseModel):
    """The JSON a `slack` declaration may carry. Settings are stored in plaintext, so the token is not one of them."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(
        default=True,
        description='Keep only the tools Slack marks read-only, so the agent cannot post or edit as you.',
    )
    include_instructions: bool = Field(default=True, description="Forward the Slack server's instructions.")


class SlackConnection(BaseModel):
    """What the menu saves in the credential store: the name of a `/keys` entry, never its value."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    token: KeyReference


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Slack` with a token resolved from `/keys` before each turn; without one, the turn has no Slack tools."""
    settings = host.settings(SlackSettings)
    token: str | None = None

    def current_token(ctx: RunContext[DepsT]) -> str | None:
        return token

    async def refresh() -> None:
        nonlocal token
        try:
            # `/keys` takes a cross-process lock that can wait up to 20 seconds; keep it off the event loop.
            token = await to_thread.run_sync(connected_token)
        except UserError as exc:
            token = None
            host.console.print(f'Slack tools are off. {exc}', style=theme.color(theme.WARNING), markup=False)

    @host.on('session_start')
    async def start(event: SessionStart) -> None:
        await refresh()

    @host.on('turn_start')
    async def turn(event: TurnStart) -> None:
        await refresh()

    @host.configure
    async def configure() -> str:
        message = await configure_menu(SlackSource(host))
        # A token change needs no reload; the loader reloads for settings changes, which refreshes again.
        await refresh()
        return message

    host.add(
        Slack[DepsT](
            auth=current_token, read_only=settings.read_only, include_instructions=settings.include_instructions
        )
    )


def connected_token() -> str:
    """The chosen key's current value, so replacing it in `/keys` reaches the next turn and deleting it fails closed."""
    connection = saved_connection()
    if connection is None:
        if os.environ.get(TOKEN_NAME):
            raise UserError(f'CLAI does not read {TOKEN_NAME} from the environment. {SETUP}')
        raise UserError(SETUP)
    return resolve_key(token=connection.token, configure='/plugins configure slack')


def saved_connection() -> SlackConnection | None:
    """The saved key reference, or `None` when no key was chosen."""
    raw = load_codex_credentials(account=_ACCOUNT)
    if raw is None:
        return None
    try:
        return SlackConnection.model_validate_json(raw)
    except ValidationError:
        raise UserError(f'The saved Slack connection is invalid. {SETUP}') from None


def save_connection(reference: KeyReference) -> None:
    """Persist the key's name; `save_key_connection` checks it still exists under the `/keys` lock."""
    connection = SlackConnection(token=reference)
    save_key_connection(account=_ACCOUNT, token=reference, value=connection.model_dump_json())


def user_token_only(value: str) -> None:
    """Slack's MCP server acts as a user; it rejects bot tokens, so catch one before saving it."""
    if value.startswith('xoxb-'):
        raise ValueError("That is a bot token (xoxb-). Slack's MCP server accepts only user tokens (xoxp-).")


_TOKEN = FieldRow(
    key='token',
    label='User token',
    description=(
        'The /keys entry Slack connects with. Enter picks a saved key or saves a new user token (xoxp-) as '
        f'{TOKEN_NAME}; only the name is saved for Slack. The token decides the workspace and user, so pick '
        "another key to use another workspace. Slack's MCP server rejects bot tokens (xoxb-). R forgets the choice."
    ),
    default='(not chosen)',
)
_ROWS = (
    FieldRow(
        key='read_only',
        label='Tools',
        description='Read-only keeps only the tools Slack marks read-only, so the agent cannot post or edit as you.',
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'read-only', 'false': 'read and write'},
        allow_custom=False,
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Whether the Slack server's own instructions reach the agent.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'forwarded', 'false': 'left out'},
        allow_custom=False,
    ),
)


class SlackSource(Generic[DepsT]):
    """The settings menu's rows: the token reference from the credential store, the rest from plugin settings."""

    title = 'Slack'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Option edits go through `host.save_settings`; token choices go to the credential store."""
        self._host = host

    @property
    def settings(self) -> SlackSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(SlackSettings)

    def rows(self) -> list[FieldRow]:
        """Every option, with the token marked when it is unset, invalid, or gone from `/keys`."""
        return [replace(_TOKEN, note=self._token_note()), *_ROWS]

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it."""
        if row.key == 'token':
            try:
                connection = saved_connection()
            except UserError:
                return '(invalid)'
            return connection.token.name if connection else row.default
        return str(getattr(self.settings, row.key)).lower()

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Validate against the whole settings model, as saving would."""
        try:
            self._updated(row, text)
        except ValidationError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save immediately; the loader loads the plugin again when the menu closes."""
        self._host.save_settings(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        """Restore an option's default, or forget the chosen key, which turns Slack's tools off."""
        if row.key == 'token':
            delete_credentials(account=_ACCOUNT)
            return 'Slack no longer uses a key from /keys; its tools are off until you choose one.'
        data = self.settings.model_dump()
        del data[row.key]
        self._host.save_settings(SlackSettings.model_validate(data))
        return f'Reset {row.label}.'

    def _updated(self, row: FieldRow, raw: str) -> SlackSettings:
        data: dict[str, object] = self.settings.model_dump()
        data[row.key] = {'true': True, 'false': False}.get(raw, raw)
        return SlackSettings.model_validate(data)

    def _token_note(self) -> str:
        try:
            connection = saved_connection()
        except UserError:
            return 'invalid; choose again'
        if connection is None:
            return 'choose a key'
        return '' if connection.token.name in load_keys() else 'missing from /keys'


async def configure_menu(source: SlackSource[DepsT]) -> str:
    """The settings menu: Enter edits a row, Esc closes, and every change is saved as it is made."""
    loop = asyncio.get_running_loop()

    def pick_token() -> list[str]:
        name = TOKEN_NAME
        label = f'Slack user token (xoxp-); a new one is saved in /keys as {name}'
        try:
            reference = on_loop(
                lambda: choose_key(name=name, label=label, runners=RUNNERS, check=user_token_only), loop
            )
            if reference is None:
                return []
            save_connection(reference)
        except (ValueError, UserError) as exc:
            return [str(exc)]
        return [f'Slack uses the saved key {reference.name}. Manage it in /keys.']

    menu = FieldMenu(source)
    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus={'token': pick_token}))
    return '\n'.join(messages) or 'Slack settings unchanged.'
