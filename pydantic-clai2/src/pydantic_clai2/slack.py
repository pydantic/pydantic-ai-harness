"""The built-in `slack` plugin: harness `Slack`, with its user token chosen from `/keys` by `/slack`."""

from functools import partial

from anyio import to_thread
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.slack import Slack

from . import theme
from .api_keys import KeyReference, SecretPrompt, prompt_api_key, resolve_key, save_key, save_key_connection
from .commands import Command
from .credential_store import load_codex_credentials
from .plugins import DepsT, PluginHost, SessionStart, TurnStart

TOKEN_NAME = 'SLACK_USER_TOKEN'
"""The `/keys` label a newly entered token is saved under: harness `Slack`'s documented name, not an exported variable."""

_ACCOUNT = 'slack'


class SlackSettings(BaseModel):
    """The JSON a `slack` declaration may carry. Settings are stored in plaintext, so the token is not one of them."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    read_only: bool = Field(
        default=True,
        description='Keep only the tools Slack marks read-only, so the agent cannot post or edit as you.',
    )


class SlackConnection(BaseModel):
    """What `/slack` saves in the credential store: the name of a `/keys` entry, never its value."""

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

    async def configure(args: list[str]) -> str:
        if args:
            raise ValueError('Usage: /slack (choose the Slack user token from /keys, or enter one privately)')
        prompt: PromptSession[str] = PromptSession()
        reference = await choose_token(prompt)
        if reference is None:
            return 'Slack connection cancelled.'
        await to_thread.run_sync(save_connection, reference)
        await refresh()
        return f'Slack now uses the {reference.name} key from /keys.'

    host.commands.register(
        Command(name='slack', description='Choose the Slack user token from /keys', handler=configure)
    )
    host.add(Slack[DepsT](auth=current_token, read_only=settings.read_only))


async def choose_token(prompt: SecretPrompt) -> KeyReference | None:
    """Pick a saved key, or save a newly entered one under `TOKEN_NAME`; `None` means cancelled."""
    token = await prompt_api_key(prompt=prompt, label='Slack user token (xoxp-): ')
    if token is None or isinstance(token, KeyReference):
        return token
    # A different value under an existing name would silently change it for every plugin sharing that key.
    await to_thread.run_sync(partial(save_key, name=TOKEN_NAME, value=token, replace=False))
    return KeyReference(name=TOKEN_NAME)


def save_connection(reference: KeyReference) -> None:
    """Persist the key's name; `save_key_connection` checks it still exists under the `/keys` lock."""
    connection = SlackConnection(token=reference)
    save_key_connection(account=_ACCOUNT, token=reference, value=connection.model_dump_json())


def connected_token() -> str:
    """The current value of the chosen key, so replacing it in `/keys` reaches the next turn and deleting it fails closed."""
    raw = load_codex_credentials(account=_ACCOUNT)
    if raw is None:
        raise UserError('Run /slack to choose its user token from /keys or enter one privately.')
    try:
        connection = SlackConnection.model_validate_json(raw)
    except ValidationError:
        raise UserError('The saved Slack connection is invalid. Run /slack to choose the token again.') from None
    return resolve_key(token=connection.token, configure='/slack')
