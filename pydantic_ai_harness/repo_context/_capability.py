"""RepoContext: discover and load a repo's accumulated context engineering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.sandboxes import Sandbox
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness._sandbox import sandbox_path
from pydantic_ai_harness.repo_context._loader import (
    ContextFile,
    discover_instruction_files,
    find_dir_context_file,
    render_context_file,
    render_context_files,
)
from pydantic_ai_harness.repo_context._toolset import RepoContextToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INVENTORY_HINT = (
    'Call `{tool_name}` to map where this repo keeps its coding-assistant setup '
    '(instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.'
)


@dataclass
class RepoContext(AbstractCapability[AgentDepsT]):
    """Discover and load a repo's accumulated coding-assistant context engineering.

    Three strategies, each independently toggleable:

    1. Walk-up instruction autoload (`autoload_instructions`, on by default):
       load `CLAUDE.md`/`AGENTS.md` from `workspace_dir` and every ancestor up
       to `home_dir`, deduped, ancestor-first. These are read once at run start
       and injected as **static system instructions** via `get_instructions`, so
       they stay in the cached prefix and never re-read per turn.

    2. Asset inventory (`expose_inventory_tool`, on by default): a tool that
       reports where the repo's CE assets live (`.claude`/`.agents`/`.codex`/
       `.grok` and their `skills/`, `agents/`, `settings.json`). It locates
       assets; it does not parse them.

    3. Nested-on-traversal (`nested_traversal`, off by default): when the model
       lists or reads a directory (via a tool named in `traversal_tool_names`),
       surface that directory's `CLAUDE.md`/`AGENTS.md`. The note is appended to
       the **tool result** (message tail), not to system instructions, so it
       does not invalidate the cached prefix. `nested_inject='pointer'` (default)
       appends a one-line pointer; `'contents'` inlines the file body.

    Cache note: injecting file contents into the system prompt costs prompt-cache
    stability. Strategy 1 is safe because its files are static; the volatile
    Strategy 3 content rides in the message tail instead.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai_harness.repo_context import RepoContext

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[RepoContext(workspace_dir=Path('/workspace'), home_dir=Path('/home/agent'))],
    )
    ```
    """

    workspace_dir: Path
    """The deepest directory the agent works in inside the run sandbox. Relative
    paths use the sandbox working directory. The walk-up and asset scan are anchored here."""

    home_dir: Path | None = None
    """The shallowest sandbox directory to stop the walk-up at, inclusive. `None`
    (the default) scans only `workspace_dir` -- no walk-up."""

    filenames: Sequence[str] = ('CLAUDE.md', 'AGENTS.md')
    """Instruction filenames to look for, in within-directory precedence order."""

    autoload_instructions: bool = True
    """Strategy 1: load instruction files into the system prompt."""

    expose_inventory_tool: bool = True
    """Strategy 2: expose the asset-inventory tool."""

    inventory_tool_name: str = 'inventory_agent_context'
    """Name of the inventory tool exposed to the model."""

    nested_traversal: bool = False
    """Strategy 3: surface a directory's instruction file when the model lists or
    reads that directory. Off by default -- it couples to the list/read tools."""

    nested_inject: Literal['pointer', 'contents'] = 'pointer'
    """For Strategy 3: append a one-line `pointer`, or inline the file `contents`."""

    traversal_tool_names: frozenset[str] = frozenset({'list_directory', 'read_file'})
    """Tool names that trigger Strategy 3. Override to match the host's list/read
    tools (e.g. `frozenset({'list_dir', 'read_file'})`)."""

    traversal_path_arg: str = 'path'
    """The tool argument key holding the listed/read path."""

    asset_roots: Sequence[str] = ('.claude', '.agents', '.codex', '.grok')
    """Root directories the inventory tool scans, relative to `workspace_dir`."""

    _context_files: list[ContextFile] | None = field(default=None, init=False, repr=False, compare=False)
    """Walk-up result for this run, loaded once in `before_run` via `ctx.sandbox`."""

    _seen_dirs: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Run-scoped set of directories already surfaced by Strategy 3."""

    _resolved_workspace_dir: Path | None = field(default=None, init=False, repr=False, compare=False)
    """Absolute sandbox path used for this run."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> RepoContext[AgentDepsT]:
        """Return a fresh per-run instance with isolated traversal/cache state."""
        return replace(self)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Load walk-up instruction files through `ctx.sandbox` so `get_instructions` is sync."""
        if not self.autoload_instructions:
            return
        sandbox = ctx.sandbox
        workspace = await self._workspace(sandbox, path=sandbox_path(self.workspace_dir))
        home = Path(await sandbox.resolve(sandbox_path(self.home_dir))) if self.home_dir is not None else None
        self._context_files = await discover_instruction_files(sandbox, workspace, home, self.filenames)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Cache-stable instructions resolved after `before_run` loads sandbox files."""
        if not self.autoload_instructions:
            return _INVENTORY_HINT.format(tool_name=self.inventory_tool_name) if self.expose_inventory_tool else None

        def instructions(_ctx: RunContext[AgentDepsT]) -> str | None:
            return self._render_instructions()

        return instructions

    def _render_instructions(self) -> str | None:
        parts: list[str] = []
        if self._context_files:
            parts.append(
                render_context_files(
                    self._context_files, relative_to=self._resolved_workspace_dir or self.workspace_dir
                )
            )
        if self.expose_inventory_tool:
            parts.append(_INVENTORY_HINT.format(tool_name=self.inventory_tool_name))
        return '\n\n'.join(parts) or None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """The asset-inventory toolset, or `None` when the tool is disabled."""
        if not self.expose_inventory_tool:
            return None
        return RepoContextToolset[AgentDepsT](self.workspace_dir, self.asset_roots, self.inventory_tool_name)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Strategy 3: append a directory's instruction file to a list/read result."""
        if not self.nested_traversal or call.tool_name not in self.traversal_tool_names:
            return result
        raw_path = args.get(self.traversal_path_arg)
        if not isinstance(raw_path, str) or not isinstance(result, str):
            return result
        sandbox = ctx.sandbox
        directory = await self._resolve_directory(sandbox, raw_path)
        key = str(directory)
        if key in self._seen_dirs:
            return result
        context_file = await find_dir_context_file(sandbox, directory, self.filenames)
        # Parallel tool calls may both probe an unseen directory; the re-check after the
        # await keeps the note single.
        if context_file is None or key in self._seen_dirs:
            return result
        self._seen_dirs.add(key)
        note = self._render_note(context_file)
        return f'{result}\n\n{note}'

    async def _resolve_directory(self, sandbox: Sandbox, raw_path: str) -> Path:
        workspace = await self._workspace(sandbox)
        text = await sandbox.resolve(raw_path, base=workspace.as_posix())
        candidate = Path(text)
        try:
            entry = await sandbox.stat(text)
        except (FileNotFoundError, NotADirectoryError):
            return candidate
        return candidate.parent if not entry.is_dir else candidate

    def _render_note(self, context_file: ContextFile) -> str:
        label = self._label(context_file.path)
        if self.nested_inject == 'contents':
            return render_context_file(context_file, label=label)
        return (
            f'<repo-context>This directory has {context_file.path.name} ({label}). '
            f'Read it if relevant to your task.</repo-context>'
        )

    def _label(self, path: Path) -> str:
        try:
            return path.relative_to(self._resolved_workspace_dir or self.workspace_dir).as_posix()
        except ValueError:
            return path.as_posix()

    async def _workspace(self, sandbox: Sandbox, *, path: str | None = None) -> Path:
        if self._resolved_workspace_dir is None:
            self._resolved_workspace_dir = Path(await sandbox.resolve(path or self.workspace_dir.as_posix()))
        return self._resolved_workspace_dir

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'RepoContext'
