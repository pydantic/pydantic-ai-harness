"""Load Agent Skill instructions from the run's workspace, on demand."""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from pydantic_ai._utils import replace_no_init  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.workspaces import Workspace, WorkspaceBackend

from pydantic_ai_harness._workspace import require_workspace, secondary_workspace
from pydantic_ai_harness.skills._loader import SkillDefinition, load_skill_libraries

_MAX_DESCRIPTION_LENGTH = 1024

LOAD_SKILL_TOOL_NAME = 'load_skill'

_CATALOG_PREFIX = (
    f'The following skills hold specialized instructions. When a task matches one, call `{LOAD_SKILL_TOOL_NAME}` '
    'with its name and follow the instructions it returns:'
)


@dataclass(frozen=True)
class _SkillSource:
    """One `Skills(...)` configuration: libraries, the selection applied to them, and where they live."""

    directories: tuple[str | Path, ...]
    include: frozenset[str] | None
    exclude: frozenset[str]
    workspace: Workspace | None
    """The `workspace=` the libraries are read from, or `None` for the run's workspace."""


@dataclass(init=False, repr=False)
class Skills(AbstractCapability[AgentDepsT]):
    """Offer Agent Skill instructions from the run's workspace, loaded on demand.

    Skill libraries are directories in the run's workspace (`ctx.workspace`), read at the
    start of every run; relative paths resolve against its working directory. Attach
    `LocalWorkspace` to read directories on this machine, or pass `workspace=` to read them
    from a workspace of their own. A run with neither fails at its start.

    Each selected immediate child containing `SKILL.md` is listed by name and description in
    the instructions, and the model calls `load_skill` to receive its Markdown body. Bundled
    files are not loaded or executed. Descriptions longer than the Agent Skills limit are
    preserved and emit a warning.

    Two `Skills` on one agent combine: every library either names stays reachable through
    one `load_skill` tool.
    """

    directories: tuple[str | Path, ...]
    """Skill-library paths in the workspace, read at the start of each run."""

    include: frozenset[str] | None
    """Exact skill names to expose, or `None` to expose all discovered skills."""

    exclude: frozenset[str]
    """Exact skill names to omit from the catalog."""

    workspace: WorkspaceBackend | None
    """Where the libraries live, when not in the run's workspace; see `__init__`."""

    id: str | None = 'skills'
    """One per agent: two `Skills` combine into one catalog behind one `load_skill` tool."""

    _sources: tuple[_SkillSource, ...] = field(default=(), init=False, repr=False, compare=False)
    """Every configuration this instance serves: its own, plus those of any `Skills` combined into it."""

    _skills: tuple[SkillDefinition, ...] = field(default=(), init=False, repr=False, compare=False)
    """This run's selected skills, read in `before_run` on the per-run copy made by `for_run`."""

    _toolset: FunctionToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """This run's `load_skill` toolset, built once `before_run` has found skills."""

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str],
        exclude: None = None,
        workspace: WorkspaceBackend | None = None,
    ) -> None: ...

    @overload
    def __init__(  # pragma: no cover - overload is enforced by static type checking
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: None = None,
        exclude: Collection[str] | None = None,
        workspace: WorkspaceBackend | None = None,
    ) -> None: ...

    def __init__(
        self,
        directories: str | Path | Sequence[str | Path],
        *,
        include: Collection[str] | None = None,
        exclude: Collection[str] | None = None,
        workspace: WorkspaceBackend | None = None,
    ) -> None:
        """Configure the skill libraries to read at the start of each run.

        Args:
            directories: One skill-library path or a sequence of paths in the workspace.
            include: Exact names to expose. Omit to expose all discovered skills.
            exclude: Exact names to omit. Cannot be combined with `include`.
            workspace: A workspace backend to read the libraries from instead of the run's, such as
                `LocalWorkspaceBackend('/app')` for skills shipped with the code while the agent
                works in a sandbox. It is read in-process only in this release: a durable engine
                does not route it through its workflow machinery.
        """
        if include is not None and exclude is not None:
            raise ValueError('include and exclude cannot be used together.')

        self.directories = self._normalize_directories(directories)
        self.include = self._normalize_selection('include', include) if include is not None else None
        self.exclude = self._normalize_selection('exclude', exclude) if exclude is not None else frozenset()
        self.workspace = workspace
        own = secondary_workspace(workspace, 'Skills')
        self._sources = (_SkillSource(self.directories, self.include, self.exclude, own),)

    def __repr__(self) -> str:
        """Show only the `Skills` configuration that callers control."""
        return (
            f'{type(self).__name__}('
            f'directories={self.directories!r}, include={self.include!r}, exclude={self.exclude!r})'
        )

    @staticmethod
    def _normalize_directories(
        directories: str | Path | Sequence[str | Path],
    ) -> tuple[str | Path, ...]:
        if isinstance(directories, (str, Path)):
            return (directories,)
        normalized = tuple(directories)
        if not normalized:
            raise ValueError('Skills requires at least one skill-library directory.')
        return normalized

    @staticmethod
    def _normalize_selection(name: str, values: Collection[object]) -> frozenset[str]:
        if isinstance(values, str):
            raise TypeError(f'{name} must be a collection of skill names, not a string.')
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                raise TypeError(f'{name} must contain only skill names as strings.')
            normalized.add(value)
        return frozenset(normalized)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Serve every combined configuration's libraries through one catalog and one `load_skill` tool.

        The field-by-field default would keep only the last configuration's directories, dropping the
        other libraries. A skill name selected by two configurations must name the same `SKILL.md`.
        """
        first = capabilities[0]
        assert isinstance(first, cls)
        sources: list[_SkillSource] = []
        for capability in capabilities:
            assert isinstance(capability, cls)
            sources.extend(source for source in capability._sources if source not in sources)
        merged = replace_no_init(first)
        merged._sources = tuple(sources)
        return merged

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Skills[AgentDepsT]:
        """A per-run copy; `before_run` fills it from the run's workspace."""
        return replace_no_init(self)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Read the selected skills, from each configuration's `workspace=` or else the run's workspace.

        Raises `UserError` when a configuration without `workspace=` meets a run without a workspace.
        """
        if any(source.workspace is None for source in self._sources):
            require_workspace(ctx.workspace, 'Skills')
        self._skills = await self._load(ctx.workspace)
        if self._skills:
            self._toolset = self._make_toolset()

    async def _load(self, run_workspace: Workspace) -> tuple[SkillDefinition, ...]:
        by_name: dict[str, tuple[Workspace, SkillDefinition]] = {}
        for source in self._sources:
            workspace = source.workspace or run_workspace
            for skill in await load_skill_libraries(
                workspace, source.directories, include=source.include, exclude=source.exclude
            ):
                previous_workspace, previous = by_name.setdefault(skill.name, (workspace, skill))
                if previous_workspace is not workspace or previous.path != skill.path:
                    raise ValueError(f'Duplicate skill name {skill.name!r}: {previous.path} and {skill.path}.')
        definitions = tuple(skill for _, skill in by_name.values())
        overlong_descriptions = [
            f'{skill.name} ({len(skill.description):,} characters)'
            for skill in definitions
            if len(skill.description) > _MAX_DESCRIPTION_LENGTH
        ]
        if overlong_descriptions:
            warnings.warn(
                f'Agent Skill descriptions exceed the {_MAX_DESCRIPTION_LENGTH:,}-character limit: '
                + '; '.join(overlong_descriptions),
                UserWarning,
                stacklevel=3,
            )
        ignored = [
            f'{skill.name}: {", ".join(skill.ignored_behavioral_fields)}'
            for skill in definitions
            if skill.ignored_behavioral_fields
        ]
        if ignored:
            warnings.warn(
                'Ignoring unsupported Agent Skill behavioral frontmatter fields: ' + '; '.join(ignored),
                UserWarning,
                stacklevel=3,
            )
        return definitions

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """The skill catalog, rendered after `before_run` has read the workspace.

        The same text on every step, and on every run over the same files, so it stays in the cached prefix.
        """
        return lambda _ctx: self._render_catalog()

    def _render_catalog(self) -> str | None:
        if not self._skills:
            return None
        # Continuation lines of a multiline description are indented so they don't read as separate entries.
        entries = '\n'.join(f'- {skill.name}: ' + skill.description.replace('\n', '\n  ') for skill in self._skills)
        return f'{_CATALOG_PREFIX}\n{entries}'

    def get_toolset(self) -> Callable[[RunContext[AgentDepsT]], AbstractToolset[AgentDepsT] | None]:
        """The `load_skill` toolset `before_run` built, or `None` when the run has no skills."""
        return lambda _ctx: self._toolset

    def _make_toolset(self) -> FunctionToolset[AgentDepsT]:
        toolset = FunctionToolset[AgentDepsT]()
        toolset.add_function(self._load_skill, takes_ctx=False, name=LOAD_SKILL_TOOL_NAME)
        return toolset

    def _load_skill(self, name: str) -> str:
        """Load a listed skill's instructions.

        Args:
            name: The skill's name, as listed in the instructions.
        """
        normalized = unicodedata.normalize('NFKC', name)
        skill = next((skill for skill in self._skills if skill.name == normalized), None)
        if skill is None:
            available = ', '.join(skill.name for skill in self._skills)
            raise ModelRetry(f'Unknown skill {name!r}. Available skills: {available}.')
        return f'# Skill: {skill.name}\n\n{skill.body}' if skill.body else f'# Skill: {skill.name}'
