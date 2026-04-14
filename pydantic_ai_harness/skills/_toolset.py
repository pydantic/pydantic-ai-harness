"""Skill dataclass, markdown parsing, and directory loading."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.toolsets.function import FunctionToolset


@dataclass
class Skill:
    """A self-contained skill that an agent can discover and load on demand.

    Args:
        name: Short, unique identifier (lowercase, hyphens allowed).
        description: One-line summary shown in search results.
        tools: Callables (or :class:`~pydantic_ai.FunctionToolset`) whose
            tools become available when the skill is loaded.
        instructions: Optional long-form guidance included in the system
            prompt when the skill is loaded.
    """

    name: str
    description: str
    tools: Sequence[Callable[..., Any]] | FunctionToolset[Any] = field(
        default_factory=lambda: list[Callable[..., Any]]()
    )
    instructions: str | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if not re.fullmatch(r'[a-z0-9]([a-z0-9-]*[a-z0-9])?', self.name):
            raise ValueError(f'Skill name must be lowercase alphanumeric with optional hyphens, got {self.name!r}')

    def tool_names(self) -> list[str]:
        """Return the names of all tools provided by this skill."""
        if isinstance(self.tools, FunctionToolset):
            return list(self.tools.tools.keys())
        return [_func_name(fn) for fn in self.tools]


def _func_name(fn: Callable[..., Any]) -> str:
    """Best-effort name extraction from a callable."""
    return getattr(fn, '__name__', None) or getattr(fn, '__qualname__', str(fn))


def load_skills_from_directory(directory: str | Path) -> list[Skill]:
    """Load skills from markdown files in *directory*.

    Each ``.md`` file is parsed as a skill definition: YAML frontmatter
    provides ``name`` and ``description``, and the body becomes the
    ``instructions``.  Frontmatter is delimited by ``---`` lines.

    Example file ``my-skill.md``::

        ---
        name: my-skill
        description: Does something useful
        ---
        Detailed instructions for the agent...

    Skills loaded from markdown carry no tools -- they are pure
    knowledge packages.  Pair them with Python-defined skills or
    attach tools separately.

    Args:
        directory: Path to scan for ``.md`` files (non-recursive).

    Returns:
        List of :class:`Skill` instances, one per file.

    Raises:
        ValueError: If a file has invalid or missing frontmatter.
    """
    dirpath = Path(directory)
    skills: list[Skill] = []
    for md_file in sorted(dirpath.glob('*.md')):
        text = md_file.read_text(encoding='utf-8')
        skill = _parse_skill_markdown(text, source=str(md_file))
        skills.append(skill)
    return skills


def _parse_skill_markdown(text: str, *, source: str = '<string>') -> Skill:
    """Parse a markdown string with YAML frontmatter into a :class:`Skill`.

    Raises:
        ValueError: If frontmatter is missing or incomplete.
    """
    stripped = text.strip()
    if not stripped.startswith('---'):
        raise ValueError(f'Missing YAML frontmatter in {source}')

    # Find closing delimiter
    end = stripped.find('---', 3)
    if end == -1:
        raise ValueError(f'Unclosed YAML frontmatter in {source}')

    frontmatter_text = stripped[3:end].strip()
    body = stripped[end + 3 :].strip() or None

    # Minimal YAML-like parsing (key: value lines) to avoid a hard
    # dependency on PyYAML for this simple case.
    # Unknown keys (e.g. from agentskills.io: tools, dependencies, etc.)
    # are silently ignored so that external skill catalogs stay compatible.
    fm: dict[str, str] = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        fm[key.strip()] = value.strip()

    name = fm.get('name')
    description = fm.get('description')
    if not name:
        raise ValueError(f'Frontmatter missing required "name" field in {source}')
    if not description:
        raise ValueError(f'Frontmatter missing required "description" field in {source}')

    return Skill(name=name, description=description, instructions=body)
