"""Servers `/mcp install` and `/mcp search` offer, modelled on Code Puppy's registry catalog.

Code Puppy's catalog is larger; this one keeps servers whose packages or endpoints are published
by their maintainers. `${name}` in `args` is filled from `CatalogArg` answers at install time.
`$VAR` in `env` or `headers` stays in the saved file and is read from the environment on connect.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from string import Template

from ._settings import HTTPServer, Server, StdioServer, references


@dataclass(frozen=True, kw_only=True)
class CatalogArg:
    """A value the user supplies when installing, substituted into `args`."""

    name: str
    prompt: str
    default: str = ''


@dataclass(frozen=True, kw_only=True)
class CatalogEntry:
    """One installable server."""

    id: str
    title: str
    description: str
    category: str
    tags: tuple[str, ...] = ()
    server: Server
    args: tuple[CatalogArg, ...] = ()
    requires: tuple[str, ...] = ()
    """Programs that must be on `PATH`, such as `npx` or `uvx`."""
    popular: bool = False

    def env_vars(self) -> list[str]:
        """Environment variables the server reads through `$VAR` references."""
        return references(self.server)

    def build(self, values: Mapping[str, str]) -> Server:
        """The server to save, with every `${name}` in `args` replaced."""
        if not isinstance(self.server, StdioServer):
            return self.server
        answers = {arg.name: values.get(arg.name) or arg.default for arg in self.args}
        empty = [name for name, value in answers.items() if not value]
        if empty:
            raise ValueError(f'{self.id} needs {", ".join(f"{name}=VALUE" for name in empty)}')
        args = [Template(arg).substitute(answers) for arg in self.server.args]
        return self.server.model_copy(update={'args': args})

    def matches(self, query: str) -> bool:
        """Case-insensitive match on id, title, description, category, or tags."""
        haystack = ' '.join((self.id, self.title, self.description, self.category, *self.tags)).lower()
        return all(word in haystack for word in query.lower().split())


def _npx(package: str, *args: str) -> StdioServer:
    return StdioServer(transport='stdio', command='npx', args=['-y', package, *args])


def _uvx(package: str, *args: str) -> StdioServer:
    return StdioServer(transport='stdio', command='uvx', args=[package, *args])


def _http(url: str, headers: dict[str, str] | None = None) -> HTTPServer:
    return HTTPServer.model_validate({'transport': 'http', 'url': url, 'headers': headers})


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        id='filesystem',
        title='Filesystem',
        description='Read, write, and search files under one directory',
        category='Storage',
        tags=('files', 'directory', 'io'),
        server=_npx('@modelcontextprotocol/server-filesystem', '${path}'),
        args=(CatalogArg(name='path', prompt='Directory the server may access', default='.'),),
        requires=('npx',),
        popular=True,
    ),
    CatalogEntry(
        id='git',
        title='Git',
        description='Read, search, and change a local Git repository',
        category='Development',
        tags=('git', 'version-control', 'repository'),
        server=_uvx('mcp-server-git', '--repository', '${repository}'),
        args=(CatalogArg(name='repository', prompt='Repository path', default='.'),),
        requires=('uvx', 'git'),
        popular=True,
    ),
    CatalogEntry(
        id='github',
        title='GitHub',
        description="GitHub's hosted server: repositories, issues, pull requests, and Actions",
        category='Development',
        tags=('github', 'issues', 'pull-requests', 'remote'),
        server=_http('https://api.githubcopilot.com/mcp/', {'Authorization': 'Bearer $GITHUB_TOKEN'}),
        popular=True,
    ),
    CatalogEntry(
        id='context7',
        title='Context7',
        description='Up-to-date library documentation and code examples',
        category='Documentation',
        tags=('docs', 'documentation', 'libraries', 'remote'),
        server=_http('https://mcp.context7.com/mcp'),
        popular=True,
    ),
    CatalogEntry(
        id='deepwiki',
        title='DeepWiki',
        description='Ask questions about public GitHub repositories',
        category='Documentation',
        tags=('docs', 'github', 'wiki', 'remote'),
        server=_http('https://mcp.deepwiki.com/mcp'),
    ),
    CatalogEntry(
        id='fetch',
        title='Fetch',
        description='Fetch web pages and convert them to Markdown',
        category='Web',
        tags=('web', 'http', 'fetch', 'markdown'),
        server=_uvx('mcp-server-fetch'),
        requires=('uvx',),
        popular=True,
    ),
    CatalogEntry(
        id='playwright',
        title='Playwright',
        description='Drive a browser through accessibility snapshots',
        category='Web',
        tags=('browser', 'automation', 'testing', 'web'),
        server=_npx('@playwright/mcp@latest'),
        requires=('npx',),
        popular=True,
    ),
    CatalogEntry(
        id='sqlite',
        title='SQLite',
        description='Query and change a SQLite database',
        category='Database',
        tags=('database', 'sql', 'sqlite'),
        server=_uvx('mcp-server-sqlite', '--db-path', '${db_path}'),
        args=(CatalogArg(name='db_path', prompt='Database file', default='./database.db'),),
        requires=('uvx',),
    ),
    CatalogEntry(
        id='postgres',
        title='PostgreSQL',
        description='Read-only queries against a PostgreSQL database',
        category='Database',
        tags=('database', 'sql', 'postgres', 'postgresql'),
        server=_npx('@modelcontextprotocol/server-postgres', '${url}'),
        args=(CatalogArg(name='url', prompt='Connection URL', default='postgresql://localhost/postgres'),),
        requires=('npx',),
    ),
    CatalogEntry(
        id='memory',
        title='Memory',
        description='A knowledge graph the agent can store and recall facts in',
        category='Memory',
        tags=('memory', 'knowledge-graph'),
        server=_npx('@modelcontextprotocol/server-memory'),
        requires=('npx',),
    ),
    CatalogEntry(
        id='sequentialthinking',
        title='Sequential Thinking',
        description='A tool for step-by-step problem solving',
        category='Reasoning',
        tags=('thinking', 'planning', 'reasoning'),
        server=_npx('@modelcontextprotocol/server-sequential-thinking'),
        requires=('npx',),
    ),
    CatalogEntry(
        id='time',
        title='Time',
        description='Current time and time zone conversion',
        category='Utilities',
        tags=('time', 'timezone', 'date'),
        server=_uvx('mcp-server-time'),
        requires=('uvx',),
    ),
    CatalogEntry(
        id='serena',
        title='Serena',
        description='Semantic code navigation and editing through language servers',
        category='Development',
        tags=('code', 'lsp', 'refactoring'),
        server=_uvx('--from', 'git+https://github.com/oraios/serena', 'serena', 'start-mcp-server'),
        requires=('uvx',),
    ),
)


def find(entry_id: str) -> CatalogEntry | None:
    """The entry with exactly this id."""
    return next((entry for entry in CATALOG if entry.id == entry_id), None)


def search(query: str) -> list[CatalogEntry]:
    """Entries matching every word of `query`, popular ones first; all entries for an empty query."""
    return sorted((entry for entry in CATALOG if entry.matches(query)), key=lambda entry: not entry.popular)
