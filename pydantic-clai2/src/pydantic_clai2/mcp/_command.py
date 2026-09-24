"""The `/mcp` command family, routed like Code Puppy's `MCPCommandHandler`."""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from ..field_menu import TERMINAL, Runners
from ..menu_worker import run_worker
from ._catalog import CATALOG, find, search
from ._menus import check_name, custom_server, edit_menu, install, install_menu, installed_message
from ._runtime import MCPServers, ServerEntry, State, not_owned
from ._settings import references, target

_GLYPHS: dict[State, str] = {'running': '+', 'ready': 'o', 'stopped': '-', 'error': '!'}
SERVER_SUBCOMMANDS = ('start', 'stop', 'restart', 'status', 'logs', 'edit', 'remove', 'tools')
SUBCOMMANDS = ('list', 'install', 'search', 'start-all', 'stop-all', 'trust', 'help', *SERVER_SUBCOMMANDS)

HELP = """MCP server management

Registry
  /mcp search [QUERY]            Search the catalog
  /mcp install                   Browse the catalog, or add a custom server
  /mcp install ID [NAME] [K=V]   Install a catalog server; K=V answers its questions
  /mcp install custom NAME CMD|URL [ARGS...]
                                 Add a program (run without a shell) or a Streamable HTTP URL

Servers
  /mcp                           Status dashboard (also /mcp list, /mcp status)
  /mcp start NAME                Enable and connect now
  /mcp stop NAME                 Disconnect and disable
  /mcp restart NAME              Reconnect, picking up config and environment changes
  /mcp start-all | stop-all      Every server at once
  /mcp status NAME               Details: target, env references, tools, last error
  /mcp tools NAME                Connect and list the tools the agent sees
  /mcp logs NAME [LINES]         Server stderr and lifecycle events (default 20 lines)
  /mcp edit NAME                 Edit a saved server
  /mcp remove NAME               Stop and forget a saved server
  /mcp trust [status|accept|revoke]
                                 Load this repository's .clai/mcp_servers.json

States:  + running (connected)  o ready (connects on your next prompt)  - stopped  ! error

Examples
  /mcp search database
  /mcp install sqlite db_path=./app.db
  /mcp install custom docs https://example.com/mcp
  /mcp start sqlite"""


def _uptime(seconds: float | None) -> str:
    if seconds is None:
        return '-'
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes}m' if hours else f'{minutes}m {secs}s' if minutes else f'{secs}s'


@dataclass(kw_only=True)
class MCPCommand:
    """Parse `/mcp` arguments and dispatch. Menus run through `runners` so tests can script them."""

    servers: MCPServers
    runners: Runners = TERMINAL

    async def __call__(self, args: list[str]) -> str:
        """Bare `/mcp` shows the dashboard, like Code Puppy."""
        await self.servers.sync()
        if not args or (args[0] in ('list', 'status') and len(args) == 1):
            return self.dashboard()
        action, rest = args[0].lower(), args[1:]
        simple: dict[str, Callable[[list[str]], Awaitable[str]]] = {
            'install': self._install,
            'search': self._search,
            'start-all': self._start_all,
            'stop-all': self._stop_all,
            'trust': self._trust,
        }
        if action in simple:
            return await simple[action](rest)
        if action == 'help':
            return HELP
        if action in SERVER_SUBCOMMANDS:
            if not rest:
                raise ValueError(f'Usage: /mcp {action} NAME')
            return await self._server_action(action, rest[0], rest[1:])
        raise ValueError(f'Unknown MCP subcommand: {action}. Type /mcp help for available commands.')

    def complete(self, args: list[str]) -> Iterable[str]:
        """Subcommands, then server names, catalog ids, or trust actions."""
        if len(args) <= 1:
            return SUBCOMMANDS
        if len(args) == 2 and args[0] in SERVER_SUBCOMMANDS:
            try:
                return [entry.name for entry in self.servers.entries()]
            except ValueError:
                return ()
        if len(args) == 2 and args[0] == 'install':
            return ('custom', *(entry.id for entry in CATALOG))
        if len(args) == 2 and args[0] == 'trust':
            return ('status', 'accept', 'revoke')
        return ()

    def dashboard(self) -> str:
        """One row per server with state, source, uptime, and what it needs."""
        entries = self.servers.entries()
        trust = self._trust_notice()
        if not entries:
            lines = [
                'No MCP servers yet.',
                '  /mcp install              browse the catalog',
                '  /mcp install custom NAME COMMAND|URL   add your own',
            ]
            return '\n'.join([*lines, *([trust] if trust else [])])
        width = max(len(entry.name) for entry in entries)
        lines = ['MCP servers', f'  {"NAME":<{width}}  TYPE   STATE    SOURCE   UPTIME   STATUS']
        for entry in entries:
            state = self.servers.state(entry)
            lines.append(
                f'{_GLYPHS[state]} {entry.name:<{width}}  {entry.server.transport:<5}  {state:<7}  '
                f'{entry.source:<7}  {_uptime(self.servers.uptime(entry)):<7}  {self._summary(entry, state)}'
            )
        running = sum(self.servers.state(entry) == 'running' for entry in entries)
        usable = sum(self.servers.state(entry) in ('running', 'ready') for entry in entries)
        lines += ['', f'{running}/{len(entries)} running, {usable} available to the agent. /mcp help lists commands.']
        return '\n'.join([*lines, *([trust] if trust else [])])

    def details(self, entry: ServerEntry) -> str:
        """`/mcp status NAME`."""
        state = self.servers.state(entry)
        source = {'user': str(self.servers.store.path), 'project': str(self.servers.store.project_file())}.get(
            entry.source, '/plugins settings for mcp'
        )
        variables = references(entry.server)
        tools = self.servers.tools(entry)
        return '\n'.join(
            [
                f'{_GLYPHS[state]} {entry.name}',
                f'  state    {state}' + (f' for {_uptime(self.servers.uptime(entry))}' if state == 'running' else ''),
                f'  type     {entry.server.transport}',
                f'  target   {target(entry.server)}',
                f'  source   {entry.source} ({source})',
                f'  env      {", ".join(variables) if variables else "none referenced"}',
                f'  tools    {", ".join(tools) if tools else "listed after /mcp start"}',
                f'  error    {self.servers.problem(entry) or "none"}',
                f'  log      {self.servers.log_path(entry.name)}',
            ]
        )

    def _summary(self, entry: ServerEntry, state: State) -> str:
        if state == 'running':
            return f'{len(self.servers.tools(entry))} tools'
        if state == 'error':
            return self.servers.problem(entry) or 'error'
        if state == 'ready':
            return 'connects on next prompt'
        return f'/mcp start {entry.name}'

    def _trust_notice(self) -> str | None:
        path = self.servers.store.project_file()
        if path is None:
            return None
        state = self.servers.store.trust_state(path)
        if state == 'trusted':
            return None
        reason = 'changed since you trusted it' if state == 'changed' else 'not trusted'
        return f'\n{path} is {reason}, so its servers are not loaded. Review it, then /mcp trust accept.'

    async def _server_action(self, action: str, name: str, extra: list[str]) -> str:
        entry = self.servers.get(name)
        if action == 'status':
            return self.details(entry)
        if action == 'logs':
            return self._logs(name, extra)
        if action == 'tools':
            return '\n'.join(await self.servers.list_tools(name)) or f'No tools provided by {name}.'
        if action == 'remove':
            await self.servers.remove(name)
            return f'Removed {name}.'
        if action == 'edit':
            reason = not_owned(entry, self.servers.store)
            if reason:
                raise ValueError(reason)
            messages = await run_worker(lambda: edit_menu(self.servers.store, name, self.runners))
            restarted = [await self.servers.restart(name)] if self.servers.state(entry) == 'running' else []
            return '\n'.join([*messages, *restarted]) or 'No changes.'
        methods = {'start': self.servers.start, 'stop': self.servers.stop, 'restart': self.servers.restart}
        return await methods[action](name)

    def _logs(self, name: str, extra: list[str]) -> str:
        if extra and not extra[0].isdigit():
            raise ValueError('Usage: /mcp logs NAME [LINES]')
        limit = int(extra[0]) if extra else 20
        path = self.servers.log_path(name)
        lines = path.read_text(errors='replace').splitlines() if path.exists() else []
        if not lines:
            return f'No log entries for {name} yet.'
        return '\n'.join([f'{path} (last {min(limit, len(lines))} of {len(lines)} lines)', *lines[-limit:]])

    async def _install(self, args: list[str]) -> str:
        store = self.servers.store
        if not args:
            return await run_worker(lambda: install_menu(store, self.runners))
        if args[0] == 'custom':
            if len(args) < 3:
                raise ValueError('Usage: /mcp install custom NAME COMMAND|URL [ARGS...]')
            server = custom_server(args[2], args[3:])
            store.put(check_name(store, args[1]), server)
            return installed_message(args[1], server)
        entry = find(args[0])
        if entry is None:
            matches = search(args[0])
            if len(matches) != 1:
                ids = ', '.join(match.id for match in matches[:8])
                hint = f' Matches: {ids}.' if ids else ' Try /mcp search.'
                raise ValueError(f'No catalog server with id {args[0]!r}.{hint}')
            entry = matches[0]
        values = dict(word.split('=', 1) for word in args[1:] if '=' in word)
        names = [word for word in args[1:] if '=' not in word]
        return install(store, entry, names[0] if names else entry.id, values)

    async def _search(self, args: list[str]) -> str:
        installed = {entry.name for entry in self.servers.entries()}
        results = search(' '.join(args))
        if not results:
            return f'No catalog servers match {" ".join(args)!r}. /mcp install custom adds any server.'
        width = max(len(entry.id) for entry in results)
        lines = [
            f'{"*" if entry.popular else " "} {entry.id:<{width}}  {entry.description}'
            + (' (installed)' if entry.id in installed else '')
            for entry in results
        ]
        return '\n'.join([*lines, '', '* popular. Install with /mcp install ID.'])

    async def _start_all(self, _: list[str]) -> str:
        entries = self.servers.entries()
        return '\n'.join([await self.servers.start(entry.name) for entry in entries]) or 'No MCP servers to start.'

    async def _stop_all(self, _: list[str]) -> str:
        entries = self.servers.entries()
        return '\n'.join([await self.servers.stop(entry.name) for entry in entries]) or 'No MCP servers to stop.'

    async def _trust(self, args: list[str]) -> str:
        store = self.servers.store
        path = store.project_file()
        action = args[0] if args else 'status'
        if action not in ('status', 'accept', 'revoke'):
            raise ValueError('Usage: /mcp trust [status|accept|revoke]')
        if path is None:
            return 'No .clai/mcp_servers.json between here and the repository root.'
        if action == 'accept':
            store.trust(path)
            names = ', '.join(store.project_servers()) or 'none'
            return f'Trusted {path}. Servers loaded: {names}. Any edit to the file requires trusting it again.'
        if action == 'revoke':
            await self.servers.sync()
            return f'Revoked trust in {path}.' if store.revoke(path) else f'{path} was not trusted.'
        return f'{path}: {store.trust_state(path)}'
