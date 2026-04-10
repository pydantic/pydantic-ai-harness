"""Repository context injection capability.

Provides the [`RepoContextInjection`][pydantic_harness.RepoContextInjection]
capability, which automatically discovers and injects repository convention
files (``AGENTS.md``, ``CLAUDE.md``, ``.cursorrules``, etc.) into the agent's
system prompt.

Example usage::

    from pydantic_ai import Agent
    from pydantic_harness import RepoContextInjection

    agent = Agent(
        'openai:gpt-4o',
        capabilities=[RepoContextInjection(root_dir='/path/to/repo')],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT

DEFAULT_FILE_PATTERNS: tuple[str, ...] = (
    'AGENTS.md',
    'CLAUDE.md',
    '.cursorrules',
    '.github/copilot-instructions.md',
    'CONVENTIONS.md',
    'CODING_GUIDELINES.md',
)
"""Default file names to search for when discovering repository context."""

_DEFAULT_MAX_TOTAL_CHARS = 100_000
"""Default limit on total injected context characters."""


@dataclass(frozen=True)
class _DiscoveredFile:
    """A context file found during discovery."""

    path: Path
    content: str


def _discover_files(
    root_dir: Path,
    file_patterns: tuple[str, ...],
    max_total_chars: int,
) -> tuple[_DiscoveredFile, ...]:
    """Walk from *root_dir* up to the filesystem root, collecting context files.

    Files are discovered bottom-up (deepest first).  If a file is a symlink
    that resolves to another discovered file, it is skipped to avoid
    duplicating content.  Discovery stops once *max_total_chars* would be
    exceeded.
    """
    found: list[_DiscoveredFile] = []
    seen_resolved: set[Path] = set()
    total_chars = 0

    current = root_dir.resolve()
    while True:
        for pattern in file_patterns:
            candidate = current / pattern
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen_resolved:
                continue
            try:
                content = candidate.read_text(encoding='utf-8')
            except OSError:
                continue
            if total_chars + len(content) > max_total_chars:
                continue
            seen_resolved.add(resolved)
            total_chars += len(content)
            found.append(_DiscoveredFile(path=candidate, content=content))

        parent = current.parent
        if parent == current:
            break
        current = parent

    return tuple(found)


def _format_context(files: tuple[_DiscoveredFile, ...]) -> str:
    """Format discovered files into a single instruction string."""
    sections: list[str] = []
    for f in files:
        sections.append(f'## Context from {f.path}\n\n{f.content}')
    return '\n\n'.join(sections)


@dataclass
class RepoContextInjection(AbstractCapability[AgentDepsT]):
    """Automatically discover and inject repository context files into agent instructions.

    Walks from ``root_dir`` up to the filesystem root, looking for convention
    files matching ``file_patterns``.  Discovered content is cached after the
    first scan and injected via
    [`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions].

    Symlinks that resolve to an already-discovered file are deduplicated, so a
    repo with ``CLAUDE.md -> AGENTS.md`` won't inject the same content twice.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness import RepoContextInjection

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[RepoContextInjection(root_dir='/path/to/repo')],
        )

    Example with custom patterns::

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[
                RepoContextInjection(
                    root_dir='/path/to/repo',
                    file_patterns=('AGENTS.md', '.custom-rules'),
                    max_total_chars=50_000,
                ),
            ],
        )
    """

    root_dir: str | Path
    """Root directory to start searching from (typically the repository root)."""

    file_patterns: tuple[str, ...] = DEFAULT_FILE_PATTERNS
    """File names to look for at each directory level."""

    max_total_chars: int = _DEFAULT_MAX_TOTAL_CHARS
    """Maximum total characters of context to inject.

    Files that would push the total beyond this limit are silently skipped.
    """

    _cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:  # noqa: D105
        if not self.file_patterns:
            raise ValueError('file_patterns must not be empty.')
        if self.max_total_chars < 1:
            raise ValueError('max_total_chars must be positive.')

    def _get_context(self) -> str:
        """Return the formatted context string, scanning on first access."""
        if self._cache is not None:
            return self._cache
        files = _discover_files(
            root_dir=Path(self.root_dir),
            file_patterns=self.file_patterns,
            max_total_chars=self.max_total_chars,
        )
        # Use object.__setattr__ is not needed since frozen=False on this dataclass.
        self._cache = _format_context(files) if files else ''
        return self._cache

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return discovered repository context as system prompt text, or None if empty."""
        context = self._get_context()
        return context if context else None

    @classmethod
    def from_spec(
        cls, *args: str | Path, **kwargs: str | Path | tuple[str, ...] | int
    ) -> RepoContextInjection[AgentDepsT]:  # type: ignore[override]
        """Create from spec arguments, coercing ``root_dir`` to a ``Path``."""
        return cls(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    'DEFAULT_FILE_PATTERNS',
    'RepoContextInjection',
]
