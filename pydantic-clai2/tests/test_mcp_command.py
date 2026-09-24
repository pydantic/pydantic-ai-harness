"""The `/mcp` command surface: catalog, install and edit menus, trust, help, and completion."""

import io
import sys
from pathlib import Path

import pytest
from menu_script import Script, pick, typed
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInputResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.commands import Commands
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.mcp import (
    CATALOG,
    HELP,
    PROJECT_MCP_FILE,
    CatalogArg,
    CatalogEntry,
    HTTPServer,
    MCPCommand,
    MCPServers,
    MCPStore,
    ServerForm,
    StdioServer,
    activate,
    catalog_details,
    catalog_menu,
)
from pydantic_clai2.plugins import PluginHost


def make(tmp_path: Path, script: Script | None = None) -> tuple[MCPCommand, MCPStore]:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path / 'repo')
    (tmp_path / 'repo' / '.git').mkdir(parents=True)
    servers = MCPServers(store)
    return (MCPCommand(servers=servers, runners=script.runners) if script else MCPCommand(servers=servers)), store


def entry(entry_id: str) -> MenuResult:
    return pick(next(item for item in CATALOG if item.id == entry_id))


async def test_help_errors_and_usage(tmp_path: Path) -> None:
    command, _ = make(tmp_path)
    assert await command(['help']) == HELP
    for sub in ('install', 'start', 'stop', 'restart', 'status', 'logs', 'edit', 'remove', 'trust', 'search'):
        assert f'/mcp {sub}' in HELP
    with pytest.raises(ValueError, match='Unknown MCP subcommand: nope'):
        await command(['nope'])
    with pytest.raises(ValueError, match='Usage: /mcp start NAME'):
        await command(['start'])
    with pytest.raises(ValueError, match='Unknown MCP server: ghost. Known: none'):
        await command(['status', 'ghost'])
    assert await command(['status']) == await command([])
    assert await command(['start-all']) == 'No MCP servers to start.'
    assert await command(['stop-all']) == 'No MCP servers to stop.'


async def test_search(tmp_path: Path) -> None:
    command, _ = make(tmp_path)
    everything = await command(['search'])
    assert all(item.id in everything for item in CATALOG)
    database = await command(['search', 'database'])
    assert 'sqlite' in database and 'postgres' in database and 'github' not in database
    await command(['install', 'sqlite'])
    assert 'sqlite' in (await command(['search', 'SQL'])) and '(installed)' in await command(['search', 'sqlite'])
    assert 'No catalog servers match' in await command(['search', 'zzz'])


async def test_catalog_install_variants(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    message = await command(['install', 'sqlite', 'mydb', 'db_path=./app.db'])
    assert message.startswith('Installed mydb.')
    saved = store.load().servers['mydb']
    assert isinstance(saved, StdioServer) and saved.args == ['mcp-server-sqlite', '--db-path', './app.db']
    await command(['install', 'filesystem'])
    fs = store.load().servers['filesystem']
    assert isinstance(fs, StdioServer) and fs.args[-1] == '.', 'defaults fill unanswered questions'
    await command(['install', 'context7'])
    assert isinstance(store.load().servers['context7'], HTTPServer)
    await command(['install', 'thinking'])
    assert 'sequentialthinking' in store.load().servers, 'a unique search hit installs'
    with pytest.raises(ValueError, match='already exists'):
        await command(['install', 'sqlite', 'mydb'])
    with pytest.raises(ValueError, match='Matches: '):
        await command(['install', 'database'])
    with pytest.raises(ValueError, match='Try /mcp search'):
        await command(['install', 'zzz'])
    with pytest.raises(ValueError, match='Invalid server name'):
        await command(['install', 'git', 'my-git'])
    required = CatalogEntry(
        id='needy',
        title='Needy',
        description='Has a question without a default',
        category='Test',
        server=StdioServer(transport='stdio', command='x', args=['${path}']),
        args=(CatalogArg(name='path', prompt='Path'),),
    )
    with pytest.raises(ValueError, match='needy needs path=VALUE'):
        required.build({})
    assert required.build({'path': 'p'}).model_dump()['args'] == ['p']


async def test_missing_program_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', str(tmp_path / 'empty'))
    command, _ = make(tmp_path)
    assert 'Not found on PATH: uvx' in await command(['install', 'fetch'])


async def test_custom_install(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    await command(['install', 'custom', 'docs', 'https://example.com/mcp'])
    await command(['install', 'custom', 'local', 'uvx', 'my-server', '--flag'])
    servers = store.load().servers
    assert isinstance(servers['docs'], HTTPServer)
    assert servers['local'] == StdioServer(transport='stdio', command='uvx', args=['my-server', '--flag'])
    with pytest.raises(ValueError, match='URL only'):
        await command(['install', 'custom', 'web', 'https://example.com/mcp', 'extra'])
    with pytest.raises(ValueError, match='Usage: /mcp install custom'):
        await command(['install', 'custom', 'web'])
    dashboard = await command([])
    assert 'docs' in dashboard and 'local' in dashboard and '0/2 running, 2 available' in dashboard
    assert 'Removed local.' == await command(['remove', 'local'])


async def test_install_menu(tmp_path: Path) -> None:
    script = Script(
        lists=[entry('sqlite'), MenuResult(item=MenuItem('+ Custom', value='custom')), MenuResult(cancelled=True)],
        choices=[],
        texts=[typed('db'), typed('./x.db'), typed('mine'), typed(f'{sys.executable} -m server --verbose')],
    )
    command, store = make(tmp_path, script)
    assert (await command(['install'])).startswith('Installed db.')
    custom = await command(['install'])
    assert custom.startswith('Installed mine.') and '/mcp edit mine' in custom
    assert await command(['install']) == ''
    servers = store.load().servers
    assert isinstance(servers['db'], StdioServer) and servers['db'].args[-1] == './x.db'
    assert servers['mine'] == StdioServer(transport='stdio', command=sys.executable, args=['-m', 'server', '--verbose'])


async def test_install_menu_cancel_and_empty(tmp_path: Path) -> None:
    custom = MenuResult(item=MenuItem('+ Custom', value='custom'))
    script = Script(
        lists=[entry('git'), custom],
        choices=[],
        texts=[TextInputResult(cancelled=True), typed('blank'), typed('  ')],
    )
    command, store = make(tmp_path, script)
    assert await command(['install']) == 'Install cancelled.'
    assert await command(['install']) == 'Nothing to install.'
    assert store.load().servers == {}


def test_catalog_menu_preview() -> None:
    assert catalog_menu(['github']) is not None
    preview = catalog_details
    github = next(item for item in CATALOG if item.id == 'github')
    text = preview(MenuItem('github', value=github))
    assert 'GITHUB_TOKEN' in text and 'type      http' in text and '(popular)' in text
    sqlite = next(item for item in CATALOG if item.id == 'sqlite')
    assert 'needs     uvx' in preview(MenuItem('sqlite', value=sqlite))
    assert 'not in the catalog' in preview(MenuItem('custom', value='custom'))
    bare = CatalogEntry(id='bare', title='Bare', description='d', category='c', server=sqlite.server)
    assert 'tags' not in preview(MenuItem('bare', value=bare))


async def test_edit_form_and_command(tmp_path: Path) -> None:
    script = Script(
        lists=[pick('args'), pick('env'), pick('enabled'), MenuResult(cancelled=True), MenuResult(cancelled=True)],
        choices=[pick('false')],
        texts=[typed('-m server --port "8 0"'), typed('TOKEN=$MY_TOKEN MODE=dev')],
    )
    command, store = make(tmp_path, script)
    await command(['install', 'custom', 'local', 'python'])
    result = await command(['edit', 'local'])
    assert result.splitlines() == ['local: saved args.', 'local: saved env.', 'local: saved enabled.']
    saved = store.load().servers['local']
    assert isinstance(saved, StdioServer)
    assert saved.args == ['-m', 'server', '--port', '8 0']
    assert saved.env == {'TOKEN': '$MY_TOKEN', 'MODE': 'dev'}
    assert not saved.enabled
    assert await command(['edit', 'local']) == 'No changes.'

    form = ServerForm(store, 'local')
    rows = {row.key: row for row in form.rows()}
    assert list(rows) == ['command', 'args', 'env', 'cwd', 'enabled']
    assert form.current(rows['args']) == "-m server --port '8 0'"
    assert form.problem(rows['env'], 'novalue') is not None
    assert form.problem(rows['command'], '') is not None
    assert form.problem(rows['cwd'], '/tmp') is None
    assert 'required' in form.reset(rows['command'])
    assert form.reset(rows['env']) == 'local: reset env.'
    assert form.apply(rows['cwd'], '/tmp') == 'local: saved cwd.'
    assert form.current(rows['cwd']) == '/tmp'
    assert FieldMenu(form, searchable=False).items()

    await command(['install', 'custom', 'web', 'https://example.com/mcp'])
    web = ServerForm(store, 'web')
    web_rows = {row.key: row for row in web.rows()}
    assert list(web_rows) == ['url', 'headers', 'auth', 'enabled']
    assert web.current(web_rows['auth']) == 'none'
    web.apply(web_rows['auth'], 'oauth')
    web.apply(web_rows['headers'], 'Authorization="Bearer $API_KEY"')
    web_saved = store.load().servers['web']
    assert isinstance(web_saved, HTTPServer) and web_saved.auth == 'oauth'
    assert web.current(web_rows['headers']) == "'Authorization=Bearer $API_KEY'"
    assert web.problem(web_rows['url'], 'not a url') is not None
    assert web.current(web_rows['url']) == 'https://example.com/mcp'


async def test_edit_restarts_a_running_server(tmp_path: Path) -> None:
    server = tmp_path / 'server.py'
    server.write_text(
        'from mcp.server.fastmcp import FastMCP\nserver = FastMCP("t")\n'
        '@server.tool()\ndef ping() -> str:\n    return "pong"\nserver.run()\n'
    )
    script = Script(lists=[pick('cwd'), MenuResult(cancelled=True)], choices=[], texts=[typed(str(tmp_path))])
    command, _ = make(tmp_path, script)
    await command(['install', 'custom', 'local', sys.executable, str(server)])
    await command(['start', 'local'])
    result = await command(['edit', 'local'])
    assert result.splitlines() == [
        'local: saved cwd.',
        'Started local with 1 tools. The agent can use them on your next prompt.',
    ]
    await command.servers.close()


async def test_project_file_trust(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    assert 'No .clai/mcp_servers.json' in await command(['trust'])
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.write_text('{"servers": {"team": {"transport": "stdio", "command": "team-server"}}}')

    dashboard = await command([])
    assert 'not trusted, so its servers are not loaded' in dashboard and 'team' not in dashboard.split('\n\n')[0]
    assert (await command(['trust'])).endswith('untrusted')
    assert 'Servers loaded: team' in await command(['trust', 'accept'])
    assert 'project' in await command([])
    assert 'team' in await command(['status', 'team'])
    with pytest.raises(ValueError, match='change that file instead'):
        await command(['edit', 'team'])
    assert 'Stopped team' in await command(['stop', 'team'])
    assert '- team' in await command([]), 'project servers stop for the session only'
    assert 'team-server' in project.read_text()

    project.write_text('{"servers": {"team": {"transport": "stdio", "command": "other"}}}')
    assert 'changed since you trusted it' in await command([])
    assert (await command(['trust', 'status'])).endswith('changed')
    assert 'Revoked trust' in await command(['trust', 'revoke'])
    assert 'was not trusted' in await command(['trust', 'revoke'])
    with pytest.raises(ValueError, match='Usage: /mcp trust'):
        await command(['trust', 'maybe'])

    project.write_text('{"servers": {"bad-name": {}}}')
    store.trust(project)
    with pytest.raises(ValueError, match='mcp_servers.json'):
        await command([])
    project.unlink()
    project.mkdir()
    assert store.trust_state(project) == 'changed', 'an unreadable file fails closed'


def test_user_servers_shadow_project_servers(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    project = tmp_path / 'repo' / PROJECT_MCP_FILE
    project.parent.mkdir()
    project.write_text('{"servers": {"shared": {"transport": "stdio", "command": "project"}}}')
    store.trust(project)
    store.put('shared', StdioServer(transport='stdio', command='mine'))
    [only] = command.servers.entries()
    assert only.source == 'user' and isinstance(only.server, StdioServer) and only.server.command == 'mine'


async def test_completion(tmp_path: Path) -> None:
    command, store = make(tmp_path)
    await command(['install', 'custom', 'local', 'python'])
    assert 'install' in command.complete([]) and 'start-all' in command.complete([''])
    assert list(command.complete(['start', ''])) == ['local']
    assert 'github' in command.complete(['install', ''])
    assert 'custom' in command.complete(['install', ''])
    assert tuple(command.complete(['trust', ''])) == ('status', 'accept', 'revoke')
    assert tuple(command.complete(['search', 'x', ''])) == ()
    store.path.write_text('not json')
    assert tuple(command.complete(['start', ''])) == ()


async def test_registered_completion_through_the_command_registry(tmp_path: Path) -> None:
    host: PluginHost[None] = PluginHost(name='mcp', console=Console(file=io.StringIO()), settings={})
    activate(host, store=MCPStore(tmp_path / 'config', workspace=tmp_path))
    registry = Commands()
    registry.register_many(host.commands)
    await registry.execute_async('/mcp install custom local python')
    [mcp] = list(registry)
    assert 'local' in mcp.complete(['logs', ''])
