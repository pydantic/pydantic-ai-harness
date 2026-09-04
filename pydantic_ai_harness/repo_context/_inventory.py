"""Locate (not parse) a repo's coding-assistant CE assets."""

from __future__ import annotations

import posixpath
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai.sandboxes import Sandbox, SandboxFileEntry

_ROOT_NOTES = {
    '.codex': 'Codex uses TOML config; assets are derived from the .claude/.agents setup.',
    '.grok': 'Grok setup is derived from the .claude/.agents setup.',
}

# Sandbox directory entries do not say whether a directory is a symlink, so a walk cannot
# detect a symlink cycle; this bound stops one. Real skill trees are two levels deep.
_MAX_SKILL_DEPTH = 8


class AssetRoot(BaseModel):
    """Where CE assets live under a single root directory (e.g. `.claude`)."""

    root: str = Field(description='The root directory name, relative to the workspace, e.g. ".claude".')
    exists: bool = Field(description='Whether the root directory is present in the workspace.')
    skills: list[str] = Field(default_factory=list, description='Paths to SKILL.md files found under skills/.')
    agents: list[str] = Field(default_factory=list, description='Paths to agent .md files found under agents/.')
    settings: str | None = Field(default=None, description='Path to settings.json (hooks), if present.')
    notes: str | None = Field(default=None, description='Format or derivation notes for this root, if any.')


class AgentContextInventory(BaseModel):
    """A map of where a repo's CE assets live, for an orchestrator to read or translate."""

    roots: list[AssetRoot] = Field(default_factory=list[AssetRoot], description='One entry per scanned root directory.')


async def scan_assets(sandbox: Sandbox, workspace_dir: Path, asset_roots: Sequence[str]) -> AgentContextInventory:
    """Scan `asset_roots` under `workspace_dir`, locating skills, agents, and hooks.

    This locates assets only; it does not open or parse SKILL.md, agent `.md`, or
    `settings.json` contents.
    """
    workspace = await sandbox.resolve(workspace_dir.as_posix())
    roots: list[AssetRoot] = []
    for name in asset_roots:
        if posixpath.isabs(name) or posixpath.normpath(name).startswith('../'):
            raise ValueError(f'asset root must be relative to the workspace, got {name!r}.')
        directory = posixpath.normpath(posixpath.join(workspace, name))
        notes = _ROOT_NOTES.get(name)
        try:
            entry = await sandbox.stat(directory)
        except FileNotFoundError:
            roots.append(AssetRoot(root=name, exists=False, notes=notes))
            continue
        if not entry.is_dir:
            roots.append(AssetRoot(root=name, exists=False, notes=notes))
            continue

        skills = await _scan_skills(sandbox, posixpath.join(directory, 'skills'), workspace)
        agents = await _scan_agents(sandbox, posixpath.join(directory, 'agents'), workspace)
        settings_path = posixpath.join(directory, 'settings.json')
        settings_entry = await _stat(sandbox, settings_path)
        settings = (
            _relative(settings_path, workspace) if settings_entry is not None and not settings_entry.is_dir else None
        )
        skills.sort()
        agents.sort()
        roots.append(AssetRoot(root=name, exists=True, skills=skills, agents=agents, settings=settings, notes=notes))
    return AgentContextInventory(roots=roots)


async def _scan_skills(sandbox: Sandbox, skills_root: str, workspace: str) -> list[str]:
    root = await _stat(sandbox, skills_root)
    if root is None or not root.is_dir:
        return []

    found: list[str] = []
    pending = deque([(skills_root, 0)])
    while pending:
        directory, depth = pending.popleft()
        # Defensive race: the directory may disappear after `_stat`.
        try:
            entries = await sandbox.list_dir(directory)
        except FileNotFoundError:  # pragma: no cover
            continue
        for entry in entries:
            if entry.is_dir:
                if depth < _MAX_SKILL_DEPTH:
                    pending.append((entry.path, depth + 1))
            elif entry.name == 'SKILL.md':
                found.append(_relative(entry.path, workspace))
    return found


async def _scan_agents(sandbox: Sandbox, agents_root: str, workspace: str) -> list[str]:
    root = await _stat(sandbox, agents_root)
    if root is None or not root.is_dir:
        return []
    # Defensive race: the directory may disappear after `_stat`.
    try:
        entries = await sandbox.list_dir(agents_root)
    except FileNotFoundError:  # pragma: no cover
        return []
    return [_relative(entry.path, workspace) for entry in entries if not entry.is_dir and entry.name.endswith('.md')]


async def _stat(sandbox: Sandbox, path: str) -> SandboxFileEntry | None:
    try:
        return await sandbox.stat(path)
    except FileNotFoundError:
        return None


def _relative(path: str, workspace: str) -> str:
    return posixpath.relpath(path, workspace)
