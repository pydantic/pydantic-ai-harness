"""The built-in `github` plugin: GitHub's hosted MCP tools, through harness `GitHub`.

The token lives in `/keys`. The plugin's settings hold only its name plus the non-secret `GitHub`
options, all edited in the settings menu that `/plugins configure github` opens.
"""

import asyncio
import concurrent.futures
import re
from dataclasses import replace
from typing import Generic
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator
from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from ._rendering import markdown_style
from .api_keys import KeyExistsError, KeyReference, SavedKey, load_keys, prompt_api_key, save_key
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from .menu_worker import menu_key, run_worker, worker_stopping
from .plugins import DepsT, PluginHost, SessionStart

KEY_NAME = 'GITHUB_TOKEN'
"""The conventional `/keys` label, shared by every plugin that uses a GitHub token. Not read from the environment."""
SETUP = 'Run /plugins configure github to choose or enter a token.'
ENTERPRISE = 'enterprise'
"""The host choice that asks for a GitHub Enterprise Cloud URL."""
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""
_GROUP = re.compile(r'[a-z][a-z0-9_]*')


class GitHubSettings(BaseModel):
    """The JSON a `github` declaration may carry: `GitHub`'s non-secret options and the name of its token."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    token: KeyReference = Field(
        default_factory=lambda: KeyReference(name=KEY_NAME), description='The saved API key in /keys to connect with.'
    )
    url: str = Field(default=GITHUB_MCP_URL, description="GitHub's MCP endpoint, or a ghe.com one.")
    read_only: bool = Field(default=True, description="Offer only GitHub's read tools.")
    toolsets: list[str] | None = Field(default=None, description="GitHub's tool groups; `None` keeps its defaults.")
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")

    @field_validator('url')
    @classmethod
    def _https(cls, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme != 'https' or not parts.hostname:
            raise ValueError('Use an https:// URL.')
        return url

    @field_validator('toolsets')
    @classmethod
    def _groups(cls, groups: list[str] | None) -> list[str] | None:
        if groups is not None and not groups:
            raise ValueError('Name at least one tool group, or use the server defaults.')
        if groups is not None and not all(_GROUP.fullmatch(group) for group in groups):
            raise ValueError('Tool groups are lowercase names such as repos, separated by commas.')
        return groups


def activate(host: PluginHost[DepsT]) -> None:
    """Add `GitHub` with a token resolved from `/keys` on every run, and offer the settings menu."""
    settings = host.settings(GitHubSettings)
    host.add(
        GitHub[DepsT](
            auth=SavedKey(name=settings.token.name, setup=SETUP),
            url=settings.url,
            read_only=settings.read_only,
            toolsets=settings.toolsets,
            include_instructions=settings.include_instructions,
        )
    )

    @host.configure
    async def configure() -> str:
        return await _configure(GitHubSource(host))

    @host.on('session_start')
    async def warn_without_token(event: SessionStart) -> None:
        # Loading anyway keeps the settings menu available; each run fails closed until a token is saved.
        # A worker thread, because the `/keys` lock can wait for another CLAI process.
        if settings.token.name not in await asyncio.to_thread(load_keys):
            host.console.print(
                f'GitHub has no token: {settings.token.name} is not in /keys. {SETUP}',
                style=theme.color(theme.WARNING),
                markup=False,
            )


def enterprise_url(text: str) -> str:
    """The MCP endpoint for a ghe.com host such as `octocorp.ghe.com`; any other URL is kept as typed."""
    text = text.strip()
    parts = urlsplit(text if '://' in text else f'https://{text}')
    host = parts.hostname or ''
    if host.endswith('.ghe.com') and not host.startswith('copilot-api.'):
        return f'https://copilot-api.{host}/mcp'
    if host.startswith('copilot-api.') and parts.path in ('', '/'):
        return f'https://{host}/mcp'
    return parts.geturl()


_TOKEN = FieldRow(
    key='token',
    label='Token',
    description=(
        'The saved API key in /keys that GitHub connects with. Enter picks a saved key or saves a new one '
        'there; plugin settings keep only its name. Any plugin naming the same key shares it.'
    ),
    default=KEY_NAME,
)
_HOST = FieldRow(
    key='url',
    label='GitHub host',
    description='github.com, or GitHub Enterprise Cloud with data residency (ghe.com). Enterprise Server has no hosted MCP.',
    default=GITHUB_MCP_URL,
    choices=(GITHUB_MCP_URL, ENTERPRISE),
    choice_labels={GITHUB_MCP_URL: 'github.com', ENTERPRISE: 'GitHub Enterprise Cloud (ghe.com)'},
    allow_custom=False,
)
_ENTERPRISE_URL = FieldRow(
    key='enterprise_url',
    label='Enterprise URL',
    description='Your ghe.com host, such as octocorp.ghe.com, or the full MCP URL.',
    default=GITHUB_MCP_URL,
)
_ROWS = (
    _TOKEN,
    _HOST,
    FieldRow(
        key='read_only',
        label='Tools',
        description='Read-only keeps the agent from changing repositories, issues, or pull requests.',
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'read-only', 'false': 'read and write'},
        allow_custom=False,
    ),
    FieldRow(
        key='toolsets',
        label='Tool groups',
        description=(
            'Comma-separated GitHub tool groups, such as repos, issues, pull_requests, actions, code_security, '
            'discussions, gists, notifications, orgs, projects, or users.'
        ),
        default='default',
        choices=('default', 'all', 'repos,issues,pull_requests', 'context,repos,issues,pull_requests,users'),
        choice_labels={'default': 'server defaults', 'all': 'every group'},
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Whether the GitHub server's own instructions reach the agent.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'forwarded', 'false': 'left out'},
        allow_custom=False,
    ),
)
_FIELDS = {'token': 'token', 'url': 'url', 'enterprise_url': 'url'}


class GitHubSource(Generic[DepsT]):
    """The settings menu's rows, read from and saved straight to the plugin's settings."""

    title = 'GitHub'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Every edit goes through `host.save_settings`."""
        self._host = host

    @property
    def settings(self) -> GitHubSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(GitHubSettings)

    def rows(self) -> list[FieldRow]:
        """Every option, with the token marked when its key is gone from `/keys`."""
        missing = self.settings.token.name not in load_keys()
        return [replace(_TOKEN, note='missing from /keys') if missing else _TOKEN, *_ROWS[1:]]

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it."""
        settings = self.settings
        if row.key == 'token':
            return settings.token.name
        if row.key == 'toolsets':
            return ','.join(settings.toolsets) if settings.toolsets else 'default'
        value: object = getattr(settings, _FIELDS.get(row.key, row.key))
        return str(value).lower() if isinstance(value, bool) else str(value)

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Validate against the whole settings model, as saving would."""
        try:
            self._updated(row, text)
        except ValidationError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save immediately; the loader loads the plugin again when the menu closes."""
        self.save(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        """Restore one option's default."""
        data = self.settings.model_dump(mode='json')
        del data[_FIELDS.get(row.key, row.key)]
        self.save(GitHubSettings.model_validate(data))
        return f'Reset {row.label}.'

    def save(self, settings: GitHubSettings) -> None:
        """Persist to the plugin's declaration."""
        self._host.save_settings(settings)

    def _updated(self, row: FieldRow, raw: str) -> GitHubSettings:
        data = self.settings.model_dump(mode='json')
        value: JsonValue = raw
        if row.key in ('read_only', 'include_instructions'):
            value = raw == 'true' if raw in ('true', 'false') else raw
        elif row.key == 'toolsets':
            value = None if raw == 'default' else [group.strip() for group in raw.split(',')]
        elif row.key == 'enterprise_url':
            value = enterprise_url(raw)
        data[_FIELDS.get(row.key, row.key)] = value
        return GitHubSettings.model_validate(data)


async def _configure(source: GitHubSource[DepsT]) -> str:
    loop = asyncio.get_running_loop()

    def pick_token() -> list[str]:
        # The key picker is async, so the menu's thread hands it back to the event loop. Its widgets
        # watch their own stop signal, so cancelling this worker must cancel the picker explicitly.
        name = source.settings.token.name
        label = f'GitHub token (saved in /keys as {name})'
        picking = asyncio.run_coroutine_threadsafe(prompt_api_key(prompt=_MaskedPrompt(), label=label), loop)
        while not (picking.done() or worker_stopping()):
            concurrent.futures.wait([picking], timeout=0.05)
        if not picking.done():
            picking.cancel()
            return []
        # Saving happens here, after the cancellable picker, so a cancelled menu saves nothing.
        reference = _saved(name, picking.result())
        if reference is None:
            return []
        source.save(source.settings.model_copy(update={'token': reference}))
        return [f'GitHub uses the saved key {reference.name}. Manage it in /keys.']

    menu = FieldMenu(source)
    submenus = {'token': pick_token, 'url': lambda: _pick_host(menu, source)}
    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus=submenus))
    return '\n'.join(messages) or 'GitHub settings unchanged.'


def _pick_host(menu: FieldMenu, source: GitHubSource[DepsT]) -> list[str]:
    pick = RUNNERS.run_choice(menu.build_choices(_HOST))
    if pick.cancelled or pick.item is None:
        return []
    if pick.item.value != ENTERPRISE:
        return [source.apply(_HOST, GITHUB_MCP_URL)]
    typed = RUNNERS.run_text(menu.build_editor(_ENTERPRISE_URL))
    if typed.cancelled or not isinstance(typed.value, str) or not typed.value.strip():
        return []
    return [source.apply(_ENTERPRISE_URL, typed.value.strip())]


class _MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked termflow input, matching the settings menu."""

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Token: ')
            .placeholder('Paste a GitHub token; it is saved in /keys')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: RUNNERS.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError  # `prompt_api_key` reads this as cancellation.
        return result.value


def _saved(name: str, choice: str | KeyReference | None) -> KeyReference | None:
    """A picked key as is, or a typed token saved under `name`; `None` means cancelled."""
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        return None
    try:
        save_key(name=name, value=value, replace=False)
    except KeyExistsError:
        if not _confirm_replace(name):
            return None
        save_key(name=name, value=value)
    return KeyReference(name=name)


def _confirm_replace(name: str) -> bool:
    menu = (
        MenuBuilder(f'{name} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved token', value=False),
                MenuItem(f'Replace {name} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = RUNNERS.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True
