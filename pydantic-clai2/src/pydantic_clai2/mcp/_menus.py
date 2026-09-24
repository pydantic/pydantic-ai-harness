"""Full-screen `/mcp install` browser and `/mcp edit` form, like Code Puppy's install menu and server form.

Everything here is synchronous and runs in a menu worker thread; saving only touches the user file.
"""

import shlex
import shutil
from collections.abc import Sequence

from pydantic import JsonValue, TypeAdapter, ValidationError
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInput  # pyright: ignore[reportMissingTypeStubs]

from .._rendering import markdown_style
from ..field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from ..menu_worker import menu_key
from ._catalog import CATALOG, CatalogEntry
from ._settings import Server, ServerName, StdioServer, missing
from ._store import MCPStore

CUSTOM = 'custom'
_NAME: TypeAdapter[str] = TypeAdapter(ServerName)
_SERVER: TypeAdapter[Server] = TypeAdapter(Server)


def check_name(store: MCPStore, name: str) -> str:
    """A valid, unused server name, or a `ValueError` explaining why not."""
    try:
        _NAME.validate_python(name)
    except ValidationError:
        raise ValueError(f'Invalid server name {name!r}: start with a letter, then letters and digits only.') from None
    if name in store.load().servers:
        raise ValueError(f'{name} already exists. Use /mcp edit {name}, or /mcp remove {name} first.')
    return name


def custom_server(target: str, args: Sequence[str] = ()) -> Server:
    """A URL becomes a Streamable HTTP server; anything else is a program run without a shell."""
    if target.startswith(('http://', 'https://')):
        if args:
            raise ValueError('An HTTP server takes a URL only; add headers with /mcp edit.')
        return _SERVER.validate_python({'transport': 'http', 'url': target})
    return StdioServer(transport='stdio', command=target, args=list(args))


def install(store: MCPStore, entry: CatalogEntry, name: str, values: dict[str, str]) -> str:
    """Save a catalog server and say what the user still has to provide."""
    store.put(check_name(store, name), entry.build(values))
    return installed_message(name, entry.server, entry.requires)


def installed_message(name: str, server: Server, requires: Sequence[str] = ()) -> str:
    """Next steps after adding a server."""
    lines = [f'Installed {name}. The agent can use it on your next prompt; /mcp start {name} connects now.']
    absent = [program for program in requires if shutil.which(program) is None]
    if absent:
        lines.append(f'Not found on PATH: {", ".join(absent)}. Install it before starting {name}.')
    unset = missing(server)
    if unset:
        lines.append(f'Set {", ".join(unset)} in your environment; the saved file only holds the reference.')
    return '\n'.join(lines)


def catalog_details(item: MenuItem) -> str:
    """The install menu's right-hand panel."""
    entry = item.value
    if not isinstance(entry, CatalogEntry):
        return 'Add a server that is not in the catalog: a program to run, or a Streamable HTTP URL.'
    lines = [
        entry.title + (' (popular)' if entry.popular else ''),
        '',
        entry.description,
        '',
        f'id        {entry.id}',
        f'category  {entry.category}',
        f'type      {entry.server.transport}',
    ]
    if entry.requires:
        lines.append(f'needs     {", ".join(entry.requires)}')
    if entry.env_vars():
        lines.append(f'env       {", ".join(entry.env_vars())}')
    if entry.tags:
        lines.append(f'tags      {", ".join(entry.tags)}')
    return '\n'.join(lines)


def catalog_menu(installed: Sequence[str]) -> Menu:
    """Every catalog entry plus a custom row, searchable, with details on the right."""
    items = [
        MenuItem(
            f'{"*" if entry.popular else " "} {entry.id:<20} {entry.category}'
            + ('  (installed)' if entry.id in installed else ''),
            value=entry,
        )
        for entry in CATALOG
    ]
    items.append(MenuItem('+ Custom server...', value=CUSTOM))
    return (
        MenuBuilder('Install an MCP server')
        .style(markdown_style())
        .items(items)
        .searchable(True)
        .preview(catalog_details)
        .footer_hint('type to filter - Enter install - Esc close')
        .key_source(menu_key)
        .build()
    )


def _ask(title: str, *, initial: str = '', placeholder: str = '') -> TextInput:
    builder = (
        TextInputBuilder(title)
        .style(markdown_style())
        .prompt('> ')
        .placeholder(placeholder)
        .footer_hint('Enter confirm - Esc cancel')
        .key_source(menu_key)
    )
    if initial:
        builder.initial(initial)
    return builder.build()


class _Cancelled(Exception):
    """Esc on any install prompt."""


def _text(runners: Runners, widget: TextInput) -> str:
    result = runners.run_text(widget)
    if result.cancelled or not isinstance(result.value, str):
        raise _Cancelled
    return result.value.strip()


def install_menu(store: MCPStore, runners: Runners = TERMINAL) -> str:
    """Browse the catalog, answer its questions, and save. Esc at any step cancels."""
    picked = runners.run_list(catalog_menu(list(store.load().servers)))
    if picked.cancelled or picked.item is None:
        return ''
    try:
        if isinstance(picked.item.value, CatalogEntry):
            entry = picked.item.value
            name = _text(runners, _ask(f'Name for {entry.title}', initial=entry.id))
            values = {
                arg.name: _text(runners, _ask(arg.prompt, initial=arg.default, placeholder=arg.default))
                for arg in entry.args
            }
            return install(store, entry, name, values)
        name = check_name(store, _text(runners, _ask('Server name', placeholder='letters and digits, e.g. mytools')))
        line = _text(runners, _ask('Command line or URL', placeholder='uvx my-mcp-server --flag, or https://...'))
        words = shlex.split(line)
        if not words:
            return 'Nothing to install.'
        server = custom_server(words[0], words[1:])
        store.put(name, server)
        return installed_message(name, server) + f'\nAdd env, headers, or a working directory with /mcp edit {name}.'
    except _Cancelled:
        return 'Install cancelled.'


def _pairs(text: str) -> dict[str, JsonValue] | None:
    words = shlex.split(text)
    if not words:
        return None
    if any('=' not in word for word in words):
        raise ValueError('Use KEY=VALUE pairs separated by spaces; quote values with spaces.')
    pairs: dict[str, JsonValue] = {}
    for word in words:
        key, value = word.split('=', 1)
        pairs[key] = value
    return pairs


def _unpairs(values: dict[str, str] | None) -> str:
    return shlex.join(f'{key}={value}' for key, value in (values or {}).items())


class ServerForm:
    """The `/mcp edit NAME` fields for one user server, as a `FieldSource`."""

    def __init__(self, store: MCPStore, name: str) -> None:
        """Edits save straight to the user file."""
        self._store = store
        self._name = name

    @property
    def server(self) -> Server:
        """The saved server, read fresh so every row reflects the last edit."""
        return self._store.load().servers[self._name]

    @property
    def title(self) -> str:
        """Menu title."""
        return f'Edit MCP server {self._name}'

    def rows(self) -> list[FieldRow]:
        """Stdio and HTTP servers expose different fields."""
        enabled = FieldRow(
            key='enabled',
            description='Offer its tools to the agent.',
            default='true',
            choices=('true', 'false'),
            allow_custom=False,
        )
        if isinstance(self.server, StdioServer):
            return [
                FieldRow(key='command', description='Program to run, without a shell.', default=''),
                FieldRow(key='args', description='Arguments, shell-quoted.', default=''),
                FieldRow(
                    key='env',
                    description='KEY=VALUE pairs. Use $VAR to read a secret from your environment.',
                    default='',
                ),
                FieldRow(key='cwd', description="Working directory; empty inherits CLAI's.", default=''),
                enabled,
            ]
        return [
            FieldRow(key='url', description='Streamable HTTP endpoint. Redirects are not followed.', default=''),
            FieldRow(
                key='headers',
                description='KEY=VALUE pairs. Use $VAR to read a secret from your environment.',
                default='',
            ),
            FieldRow(
                key='auth',
                description='oauth signs in through your browser on connect.',
                default='none',
                choices=('none', 'oauth'),
                allow_custom=False,
            ),
            enabled,
        ]

    def current(self, row: FieldRow) -> str:
        """The saved value as the user would type it."""
        server = self.server
        values: dict[str, str] = {'enabled': str(server.enabled).lower()}
        if isinstance(server, StdioServer):
            values |= {
                'command': server.command,
                'args': shlex.join(server.args),
                'env': _unpairs(server.env),
                'cwd': server.cwd or '',
            }
        else:
            values |= {'url': str(server.url), 'headers': _unpairs(server.headers), 'auth': server.auth or 'none'}
        return values[row.key]

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Validate against the server model without saving."""
        try:
            self._updated(row.key, text)
        except ValidationError as exc:
            return first_error(exc)
        except ValueError as exc:
            return str(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save one field."""
        self._store.put(self._name, self._updated(row.key, raw))
        return f'{self._name}: saved {row.key}.'

    def reset(self, row: FieldRow) -> str:
        """Clear an optional field; required ones keep their value."""
        if row.key in ('command', 'url'):
            return f'{self._name}: {row.key} is required; type a new value instead.'
        self._store.put(self._name, self._updated(row.key, row.default))
        return f'{self._name}: reset {row.key}.'

    def _updated(self, key: str, text: str) -> Server:
        data: dict[str, JsonValue] = self.server.model_dump(mode='json')
        converted: JsonValue
        if key == 'args':
            converted = list(shlex.split(text))
        elif key in ('env', 'headers'):
            converted = _pairs(text)
        elif key == 'enabled':
            converted = text == 'true'
        elif key in ('cwd', 'auth'):
            converted = None if text in ('', 'none') else text
        else:
            converted = text
        return _SERVER.validate_python({**data, key: converted})


def edit_menu(store: MCPStore, name: str, runners: Runners = TERMINAL) -> list[str]:
    """Edit fields until Esc; returns what changed."""
    return run_flow(FieldMenu(ServerForm(store, name), searchable=False), runners)
