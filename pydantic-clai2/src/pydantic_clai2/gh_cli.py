"""Sign in to GitHub through the GitHub CLI, and read its token on every run.

GitHub's hosted MCP server offers no dynamic client registration, so CLAI cannot run its own
OAuth flow without a registered OAuth App. `gh auth login --web` already has one: it shows a
one-time code, the user approves it in the browser, and `gh` keeps the token in the OS keyring.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import IO
from urllib.parse import urlsplit

from pydantic_ai.exceptions import UserError

INSTALL = 'Install the GitHub CLI from https://cli.github.com, or use a token saved in /keys.'
_TOKEN_VARIABLES = ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_ENTERPRISE_TOKEN', 'GITHUB_ENTERPRISE_TOKEN')
_CODE = re.compile(r'\b([A-Z0-9]{4}-[A-Z0-9]{4})\b')
_URL = re.compile(r'https://\S+')
_ACCOUNT = re.compile(r'Logged in as (\S+)')


def gh_command() -> list[str] | None:
    """How to run `gh`, or `None` when it is not installed."""
    path = shutil.which('gh')
    return None if path is None else [path]


def _gh() -> list[str]:
    command = gh_command()
    if command is None:
        raise UserError(f'The GitHub CLI (gh) is not installed. {INSTALL}')
    return command


def gh_host(url: str) -> str:
    """The `gh` hostname for an MCP URL: `github.com`, or the ghe.com host behind `copilot-api.`."""
    host = urlsplit(url).hostname or ''
    if host == 'api.githubcopilot.com':
        return 'github.com'
    return host.removeprefix('copilot-api.')


def _environment() -> dict[str, str]:
    # Token variables would make `gh` report them instead of its login, and refuse `auth login`.
    return {name: value for name, value in os.environ.items() if name not in _TOKEN_VARIABLES}


def gh_token(hostname: str) -> str | None:
    """The token `gh` holds for `hostname`, or `None` when it has no login there; `UserError` without `gh`."""
    result = subprocess.run(
        [*_gh(), 'auth', 'token', '--hostname', hostname],
        capture_output=True,
        text=True,
        env=_environment(),
        stdin=subprocess.DEVNULL,
        check=False,
    )
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


@dataclass(frozen=True, kw_only=True)
class GhToken:
    """A `GitHub` `auth` callable that asks `gh` on every run, so signing in again applies next turn."""

    hostname: str
    setup: str

    def __call__(self, ctx: object, /) -> str:
        """The current token; fails the run when `gh` has none."""
        token = gh_token(self.hostname)
        if token is None:
            raise UserError(f'The GitHub CLI has no login for {self.hostname}. {self.setup}')
        return token


@dataclass
class GhLogin:
    """A running `gh auth login --web`: the code to enter, where to enter it, and the process."""

    process: subprocess.Popen[str]
    code: str
    url: str

    def finish(self) -> str:
        """Wait for `gh` and describe the result."""
        output = _drain(self.process)
        if self.process.wait() != 0:
            lines = [line for line in output.splitlines() if line.strip()]
            return f'gh auth login failed: {lines[-1] if lines else "no output"}'
        account = _ACCOUNT.search(output)
        return f'Signed in to GitHub as {account.group(1)}.' if account else 'Signed in to GitHub.'

    def cancel(self) -> None:
        """Stop waiting for the browser."""
        self.process.terminate()
        self.process.wait()


def start_login(hostname: str) -> GhLogin | str:
    """Start the browser sign-in, or explain why `gh` could not; `UserError` without `gh`."""
    process = subprocess.Popen(
        [*_gh(), 'auth', 'login', '--hostname', hostname, '--web', '--clipboard'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_environment(),
    )
    stream = _stream(process)
    code = ''
    seen: list[str] = []
    for line in stream:
        seen.append(line)
        if found := _CODE.search(line):
            code = found.group(1)
        if code and (url := _URL.search(line)):
            return GhLogin(process=process, code=code, url=url.group(0))
    process.wait()
    lines = [line.strip() for line in seen if line.strip()]
    return f'gh auth login failed: {lines[-1] if lines else "no output"}'


def _stream(process: subprocess.Popen[str]) -> IO[str]:
    assert process.stdout is not None  # Popen was given stdout=PIPE.
    return process.stdout


def _drain(process: subprocess.Popen[str]) -> str:
    return _stream(process).read()
