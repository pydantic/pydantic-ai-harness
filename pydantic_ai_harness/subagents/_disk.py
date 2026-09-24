"""Load sub-agent definitions from markdown files: the project's through the run workspace, the rest from the host.

A definition is a markdown file with optional YAML-style frontmatter:

```markdown
---
name: researcher
description: Researches a topic and reports findings
tools: Read, Grep
---
You research topics. Report findings with sources.
```

The frontmatter is parsed by a small, dependency-free reader limited to the keys
coding assistants write (`name`, `description`, `model`, `color`, and `tools` or
`allowed-tools`); `pyyaml` is not a runtime dependency of harness. The body after
the frontmatter is the agent's instructions. `model` and `color` are ignored: the
model is inherited from the parent (overridable via `SubAgents.agent_overrides`),
and `color` has no pyai equivalent.
"""

from __future__ import annotations

import posixpath
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.workspaces import Workspace


@dataclass(frozen=True)
class AgentOverride:
    """Per-agent override for a disk-loaded sub-agent, keyed by the agent's name.

    Both fields are optional. An unset `model` inherits the parent run's model; an
    unset `effort` runs at the capability's minimum effort floor (see
    `clamp_effort`).
    """

    model: Model | KnownModelName | str | None = None
    """Model to run this disk agent with, in place of inheriting the parent's."""

    effort: ThinkingLevel | None = None
    """Thinking/reasoning level for this disk agent. Raised to at least the floor."""


@dataclass(frozen=True)
class ParsedAgent:
    """One parsed agent definition: frontmatter fields plus the markdown body."""

    name: str | None
    description: str | None
    tools: tuple[str, ...]
    body: str


def _strip_quotes(value: str) -> str:
    """Drop a single layer of matching single or double quotes from a scalar."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _parse_frontmatter(lines: Sequence[str]) -> dict[str, str | list[str]]:
    """Parse `key: value` and block-list (`- item`) frontmatter lines.

    Only the shape coding assistants emit is supported: scalar values and, for
    list keys, either a `key: a, b` inline form (handled by callers) or a block
    list of `- item` lines under a key with an empty value.
    """
    result: dict[str, str | list[str]] = {}
    current_list_key: str | None = None
    for raw in lines:
        if not raw.strip():
            continue
        stripped = raw.lstrip()
        if current_list_key is not None and stripped.startswith('- '):
            item = stripped[2:].strip()
            existing = result[current_list_key]
            if isinstance(existing, list) and item:
                existing.append(item)
            continue
        if ':' not in raw:
            current_list_key = None
            continue
        key, _, value = raw.partition(':')
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = _strip_quotes(value)
            current_list_key = None
        else:
            result[key] = []
            current_list_key = key
    return result


def _parse_tools(fields: dict[str, str | list[str]]) -> tuple[str, ...]:
    """Read the `tools` or `allowed-tools` key as a tuple of tool-name strings."""
    raw = fields.get('tools')
    if raw is None:
        raw = fields.get('allowed-tools')
    if raw is None:
        return ()
    if isinstance(raw, list):
        return tuple(item for item in raw if item)
    return tuple(name.strip() for name in raw.split(',') if name.strip())


def parse_agent_markdown(text: str) -> ParsedAgent:
    """Parse a markdown agent definition into frontmatter fields and a body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != '---':
        return ParsedAgent(None, None, (), text.strip())
    closing: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == '---':
            closing = index
            break
    if closing is None:
        return ParsedAgent(None, None, (), text.strip())
    fields = _parse_frontmatter(lines[1:closing])
    body = '\n'.join(lines[closing + 1 :]).strip()
    name = fields.get('name')
    description = fields.get('description')
    return ParsedAgent(
        name=name if isinstance(name, str) else None,
        description=description if isinstance(description, str) else None,
        tools=_parse_tools(fields),
        body=body,
    )


@dataclass(frozen=True)
class DiskDefinition:
    """One loaded definition file: the delegate name it resolves to, plus its parsed contents.

    Hashable, so equal definitions share one built `SubAgent`, and a home definition identical to
    the project's (the project root *is* the home root) is dropped without a shadowing warning.
    """

    name: str
    parsed: ParsedAgent


def _definition(text: str, stem: str) -> DiskDefinition:
    parsed = parse_agent_markdown(text)
    return DiskDefinition(parsed.name or stem, parsed)


def _warn_unreadable(path: str, exc: Exception) -> None:
    warnings.warn(f'Skipping unreadable disk sub-agent file {path!r}: {exc}', stacklevel=3)


def _convention_folder(root: Path, leaf: str) -> Path:
    """`<root>/.agents/<leaf>` when `.agents/` exists, else the `.claude/` equivalent."""
    if (root / '.agents').is_dir():
        return root / '.agents' / leaf
    return root / '.claude' / leaf


def host_folders(agent_folders: str | Sequence[Path], home: Path) -> list[Path]:
    """The host folders to load from, in precedence order.

    - a `str`: the home convention `<home>/.agents/<str>/` (or `.claude/<str>/`). The project
      convention folder is read per run through the workspace instead; see `load_workspace_definitions`.
    - a sequence of paths: those folders verbatim, in order, deduped by absolute path (keeping the first).
    """
    if isinstance(agent_folders, str):
        return [_convention_folder(home, agent_folders)]
    seen: dict[Path, Path] = {}
    for folder in agent_folders:
        seen.setdefault(folder.resolve(), folder)
    return list(seen.values())


def load_host_definitions(folders: Sequence[Path]) -> list[DiskDefinition]:
    """Load every `*.md` definition in `folders` from the host, in sorted name order per folder."""
    result: list[DiskDefinition] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.md')):
            try:
                text = path.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError) as exc:
                _warn_unreadable(str(path), exc)
                continue
            result.append(_definition(text, path.stem))
    return result


async def _is_dir(workspace: Workspace, path: str) -> bool:
    try:
        return (await workspace.stat(path)).is_dir
    except (FileNotFoundError, NotADirectoryError):
        return False


async def load_workspace_definitions(workspace: Workspace, leaf: str) -> list[DiskDefinition]:
    """Load the project convention folder's definitions through the run's workspace.

    The folder is `.agents/<leaf>/` under the workspace's working directory, falling back to
    `.claude/<leaf>/` when `.agents/` is absent. Files load in sorted name order, so the roster --
    and so the prompt listing -- is the same for every run over the same files.
    """
    root = '.agents' if await _is_dir(workspace, '.agents') else '.claude'
    folder = posixpath.join(root, leaf)
    if not await _is_dir(workspace, folder):
        return []
    result: list[DiskDefinition] = []
    entries = sorted(await workspace.list_dir(folder), key=lambda entry: entry.name)
    for entry in entries:
        if entry.is_dir or not entry.name.endswith('.md'):
            continue
        try:
            text = await workspace.read_text(entry.path)
        except (OSError, UnicodeDecodeError) as exc:
            _warn_unreadable(entry.path, exc)
            continue
        result.append(_definition(text, posixpath.splitext(entry.name)[0]))
    return result
