"""Where `/mcp` keeps servers: a user file, plus a project file that loads only once trusted.

A stdio server runs a program, so a cloned repository must not be able to start one by
shipping `.clai/mcp_servers.json`. Trust is recorded on the user side, keyed by the file's
resolved path and a SHA-256 of its bytes: any edit makes the file untrusted again, and a
repository cannot trust itself. This follows Code Puppy's `/mcp trust`.
"""

import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..project_settings import find_project_file
from ..settings_store import config_dir
from ._settings import HTTPServer, Server, Servers, StdioServer

PROJECT_MCP_FILE = Path('.clai') / 'mcp_servers.json'
TrustState = Literal['trusted', 'changed', 'untrusted']


class UserFile(BaseModel):
    """The user's `mcp.json`: servers and accepted project files."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    servers: Servers = Field(default_factory=dict[str, StdioServer | HTTPServer])
    trusted_projects: dict[str, str] = Field(default_factory=dict[str, str])
    """Resolved project file path to the SHA-256 accepted by `/mcp trust`."""


class ProjectFile(BaseModel):
    """A repository's `.clai/mcp_servers.json`."""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    servers: Servers = Field(default_factory=dict[str, StdioServer | HTTPServer])


class MCPStore:
    """Read and write the user file; find and gate the project file."""

    def __init__(self, directory: Path | None = None, *, workspace: Path | None = None) -> None:
        """`directory` defaults to the CLAI config folder, `workspace` to the current directory."""
        directory = directory or config_dir()
        self.path = directory / 'mcp.json'
        self.logs = directory / 'mcp_logs'
        self._workspace = workspace

    def load(self) -> UserFile:
        """The saved file, or an empty one. A malformed file fails loudly with its path."""
        try:
            return UserFile.model_validate_json(self.path.read_bytes())
        except FileNotFoundError:
            return UserFile()
        except ValidationError as exc:
            raise ValueError(f'{self.path}: {exc}') from exc

    def save(self, data: UserFile) -> None:
        """Replace the file atomically, readable only by the user since headers may be sensitive."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staging = self.path.with_suffix('.tmp')
        staging.write_text(data.model_dump_json(indent=2, exclude_defaults=True) + '\n')
        staging.chmod(0o600)
        os.replace(staging, self.path)

    def put(self, name: str, server: Server) -> None:
        """Add or replace one user server."""
        data = self.load()
        self.save(data.model_copy(update={'servers': {**data.servers, name: server}}))

    def delete(self, name: str) -> bool:
        """Forget one user server; `False` when there was none."""
        data = self.load()
        if name not in data.servers:
            return False
        self.save(data.model_copy(update={'servers': {k: v for k, v in data.servers.items() if k != name}}))
        return True

    def project_file(self) -> Path | None:
        """The nearest `.clai/mcp_servers.json` between the workspace and the git root."""
        return find_project_file(self._workspace or Path.cwd(), PROJECT_MCP_FILE)

    def trust_state(self, path: Path) -> TrustState:
        """Whether the file's current bytes are the ones the user accepted."""
        accepted = self.load().trusted_projects.get(str(path.resolve()))
        if accepted is None:
            return 'untrusted'
        try:
            current = _digest(path)
        except OSError:
            return 'changed'
        return 'trusted' if accepted == current else 'changed'

    def trust(self, path: Path) -> None:
        """Accept the file's current bytes."""
        data = self.load()
        trusted = {**data.trusted_projects, str(path.resolve()): _digest(path)}
        self.save(data.model_copy(update={'trusted_projects': trusted}))

    def revoke(self, path: Path) -> bool:
        """Withdraw acceptance; `False` when it was never given."""
        data = self.load()
        key = str(path.resolve())
        if key not in data.trusted_projects:
            return False
        trusted = {k: v for k, v in data.trusted_projects.items() if k != key}
        self.save(data.model_copy(update={'trusted_projects': trusted}))
        return True

    def project_servers(self) -> Servers:
        """The project file's servers when trusted; empty when absent or not trusted."""
        path = self.project_file()
        if path is None or self.trust_state(path) != 'trusted':
            return {}
        try:
            return ProjectFile.model_validate_json(path.read_bytes()).servers
        except (OSError, ValidationError) as exc:
            raise ValueError(f'{path}: {exc}') from exc


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
