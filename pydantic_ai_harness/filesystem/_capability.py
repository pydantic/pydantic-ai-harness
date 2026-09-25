"""Filesystem capability that provides sandboxed file system access."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai._utils import replace_no_init  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FilteredToolset

from pydantic_ai_harness._warn import WORKING_DIR_IS_THE_WORKSPACES, warn_argument_ignored
from pydantic_ai_harness._workspace import require_workspace
from pydantic_ai_harness.filesystem._toolset import (
    DEFAULT_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    FileSystemToolset,
    root_spelling,
)

_DEFAULT_PROTECTED: list[str] = [
    '.git/*',
    '.env',
    '.env.*',
    '*.pem',
    '*.key',
    '**/secrets*',
]


@dataclass
class FileSystem(AbstractCapability[AgentDepsT]):
    """File system access to the run's workspace, scoped to a root directory.

    Every operation goes through `ctx.workspace`, so attach a workspace to the
    run: `LocalWorkspace(...)` for a local checkout, or a sandbox provider's
    capability. A run without one fails at its start. Relative paths resolve
    from the workspace's working directory.

    `root_dir` bounds the model's file tools: before each operation, the target
    must be inside it both as written and once the workspace has resolved its
    symlinks, and `protected_patterns` guard what may be written. This is a
    guardrail checked before each operation, not isolation: a symlink swapped in
    between the check and the use is not caught, and `Shell` commands are not
    bounded at all. The workspace is the isolation boundary. Listings show
    entries by name: a symlink whose target is outside `root_dir` is listed but
    refused when read or written.
    """

    root_dir: str | Path | None = None
    """The containment boundary for all file operations, as a workspace path.

    `None` (the default) is the workspace's working directory; a relative path
    resolves against it. Set it higher to let the model reach beyond the
    working directory, e.g. a parent holding sibling projects. The working
    directory must be inside it: a relative path below it (e.g. `'src'`) is
    rejected here, and an absolute one that does not contain it fails the run on
    its first file operation. `'/'` turns the containment checks off, while any
    access patterns still match a symlink's target.
    """

    cwd: str | Path | None = None
    """Deprecated and ignored: relative paths resolve from the workspace's working directory.

    Set the working directory on the workspace instead, e.g. `LocalWorkspace('./repo')`.
    """

    allowed_patterns: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only paths matching at least one glob pattern are accessible."""

    denied_patterns: Sequence[str] = field(default_factory=list[str])
    """Paths matching any of these glob patterns are rejected."""

    protected_patterns: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_PROTECTED))
    """Paths matching these patterns are read-only (writes are rejected).

    Defaults to protecting `.git/`, `.env`, key files, and secrets.
    Set to an empty list to disable protection.
    """

    max_read_lines: int = 2000
    """Maximum number of lines returned by a single `read_file` call."""

    max_read_chars: int | None = None
    """Maximum characters in a single `read_file` result, header and hint included.

    The window ends on the last complete line that fits, and the continuation
    hint names the first line not shown, so a caller paging by `offset` cannot
    skip content. Set this at or below any downstream tool-output cap; a cap
    applied after the fact cuts mid-line and drops or strands the hint. `None`
    leaves only `max_read_lines` in force.
    """

    max_list_results: int = 1000
    """Maximum number of entries returned by `list_directory`."""

    max_search_results: int = 1000
    """Maximum number of matches returned by `search_files`."""

    max_find_results: int = 1000
    """Maximum number of matches returned by `find_files`."""

    read_only: bool = False
    """Whether to expose only the tools in `READ_ONLY_TOOL_NAMES`.

    A read-only workspace (`Workspace.read_only`) narrows the tools the same way for that run.
    """

    content_hashes: bool = True
    """Whether tool results report content hashes and `write_file`/`edit_file` accept `expected_hash`.

    The hashes give a model optimistic concurrency control over a workspace
    that something else may also be editing. Turn them off for a single-writer
    coding agent, where they only add tokens to every read and write.
    """

    tools: Sequence[str] = DEFAULT_TOOL_NAMES
    """Which tools to register, from `FILE_SYSTEM_TOOL_NAMES`.

    The default is every tool that needs only the workspace's filesystem. Name
    `list_files` and `grep` to add the ripgrep-backed listing and search tools,
    which run the `rg` executable inside the workspace (the `coder` extra
    installs it for a local workspace) and respect `.gitignore`.
    `read_only` further narrows the selection to `READ_ONLY_TOOL_NAMES`.
    """

    _run_toolset: FileSystemToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """This run's toolset, which resolves the boundary on its first operation; `None` outside a run."""

    def __post_init__(self) -> None:
        if self.cwd is not None:
            warn_argument_ignored('FileSystem', 'cwd', WORKING_DIR_IS_THE_WORKSPACES)
        root_spelling(None if self.root_dir is None else Path(self.root_dir))
        # Runtime validation: dataclass field annotations are advisory, not enforced.
        # A config-driven caller could pass a string that would otherwise propagate.
        values: dict[str, Any] = {
            'max_read_lines': self.max_read_lines,
            'max_list_results': self.max_list_results,
            'max_search_results': self.max_search_results,
            'max_find_results': self.max_find_results,
        }
        if self.max_read_chars is not None:
            values['max_read_chars'] = self.max_read_chars
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> FileSystem[AgentDepsT]:
        """A per-run copy with its own toolset, so the boundary it resolves stays with this run."""
        run = replace_no_init(self)
        run._run_toolset = run._make_toolset()
        return run

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail without a workspace, without touching it: the boundary waits for the first file operation."""
        require_workspace(ctx.workspace, 'FileSystem')

    def get_toolset(self) -> FileSystemToolset[AgentDepsT] | FilteredToolset[AgentDepsT]:
        """The filesystem toolset: this run's, once `for_run` has made one."""
        toolset = self._run_toolset or self._make_toolset()
        if self.read_only:
            return FilteredToolset(toolset, lambda ctx, tool: tool.name in READ_ONLY_TOOL_NAMES)
        return toolset

    def _make_toolset(self) -> FileSystemToolset[AgentDepsT]:
        return FileSystemToolset[AgentDepsT](
            root_dir=None if self.root_dir is None else Path(self.root_dir),
            allowed_patterns=self.allowed_patterns,
            denied_patterns=self.denied_patterns,
            protected_patterns=self.protected_patterns,
            max_read_lines=self.max_read_lines,
            max_read_chars=self.max_read_chars,
            max_list_results=self.max_list_results,
            max_search_results=self.max_search_results,
            max_find_results=self.max_find_results,
            id=self.id or 'file_system',
            content_hashes=self.content_hashes,
            tools=self.tools,
        )
