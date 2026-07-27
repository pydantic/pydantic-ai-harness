"""`Checkpoints`: shadow-git file snapshots taken before mutating tool calls.

Before a tool whose name is in `mutating_tools` runs, the capability snapshots the
project's files into a shadow git repository (see `_shadow.CheckpointStore`). Any
file damage a later tool does is then restorable with `restore`. This is the
files-only slice of undo; conversation rewind/fork lives with the branchable
session-history track (harness issue #321).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import anyio.to_thread
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset

from pydantic_ai_harness.checkpoints._shadow import (
    Checkpoint,
    CheckpointError,
    CheckpointStore,
    shadow_dir_for,
)
from pydantic_ai_harness.checkpoints._toolset import RestoreCheckpointToolset

DEFAULT_MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        # pydantic-ai-harness filesystem capability
        'write_file',
        'edit_file',
        'create_directory',
        # names common across coding harnesses / MCP filesystem servers
        'write',
        'edit',
        'multi_edit',
        'apply_patch',
        'patch',
        'str_replace',
        'str_replace_editor',
        'create',
        'create_file',
        'delete',
        'delete_file',
        'remove_file',
        'move',
        'move_file',
        'rename',
        'rename_file',
        'notebook_edit',
    }
)
"""Tool names that mutate files, used as the default `mutating_tools` set.

Covers the harness `FileSystem` toolset plus write/edit/patch/create/delete/move
names common to other coding harnesses and MCP filesystem servers. Override it for
a custom tool vocabulary.
"""

DEFAULT_BASH_TOOLS: frozenset[str] = frozenset({'run_command', 'bash', 'shell', 'run', 'execute'})
"""Shell tool names treated as mutating when `snapshot_before_bash` is on."""


class CheckpointWarning(UserWarning):
    """A snapshot could not be taken, so no checkpoint was recorded before a tool ran.

    Checkpoints are best-effort: a shadow-git failure warns and lets the run
    continue rather than aborting the agent. Escalate to an error in dev/CI with
    the stdlib `warnings` machinery::

        import warnings
        from pydantic_ai_harness.checkpoints import CheckpointWarning

        warnings.filterwarnings('error', category=CheckpointWarning)
    """


@dataclass
class Checkpoints(AbstractCapability[AgentDepsT]):
    """Snapshot project files before mutating tools run, so agent damage is restorable.

    Attach it to any coding agent. Before a tool whose name is in `mutating_tools`
    executes, the capability commits the current work tree to a shadow git repo that
    is separate from the user's own `.git`. Restore a snapshot with `restore`, or
    list them with `checkpoints`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.checkpoints import Checkpoints
    from pydantic_ai_harness.filesystem import FileSystem

    checkpoints = Checkpoints(project_root='.')
    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[FileSystem(), checkpoints])
    await agent.run('Refactor the auth module')

    for cp in checkpoints.checkpoints():
        print(cp.id, cp.tool_name, cp.files_changed)
    checkpoints.restore(checkpoints.checkpoints()[0].id)  # undo everything
    ```

    Snapshots respect the project's `.gitignore` and never touch the user's `.git`;
    they work in projects that are not git repositories. See `CheckpointStore` for
    the shadow-git mechanics.
    """

    project_root: str | Path = '.'
    """Project directory to snapshot. All snapshots and restores are scoped to it."""

    state_dir: str | Path | None = None
    """Base directory for shadow repos. Defaults to `~/.pydantic-ai-harness`.

    The shadow repo for a project lives at `<state_dir>/checkpoints/<project-slug>/`,
    where the slug is the project directory name plus a hash of its absolute path.
    """

    mutating_tools: Collection[str] = field(default_factory=lambda: set(DEFAULT_MUTATING_TOOLS))
    """Tool names that trigger a snapshot before they run. Defaults to `DEFAULT_MUTATING_TOOLS`."""

    snapshot_before_bash: bool = False
    """Also snapshot before shell tools (names in `bash_tools`).

    Off by default: shell commands are frequently read-only, so snapshotting before
    each one adds commits with no file change (deduplicated by the debounce, but
    still extra work). Turn it on when the agent runs file-mutating shell commands.
    """

    bash_tools: Collection[str] = field(default_factory=lambda: set(DEFAULT_BASH_TOOLS))
    """Shell tool names covered by `snapshot_before_bash`. Defaults to `DEFAULT_BASH_TOOLS`."""

    committer_name: str = 'pydantic-ai-harness checkpoints'
    """Author/committer name recorded on shadow commits."""

    committer_email: str = 'noreply@pydantic.dev'
    """Author/committer email recorded on shadow commits."""

    expose_tool: bool = False
    """Expose `restore_checkpoint` and `list_checkpoints` tools to the model.

    Off by default: restoring is usually a human or application action, not
    something the model should drive mid-run.
    """

    def _store(self) -> CheckpointStore:
        root = Path(self.project_root).resolve()
        base = Path(self.state_dir) if self.state_dir is not None else Path.home() / '.pydantic-ai-harness'
        return CheckpointStore(
            project_root=root,
            shadow_dir=shadow_dir_for(root, base),
            committer_name=self.committer_name,
            committer_email=self.committer_email,
        )

    def _should_snapshot(self, tool_name: str) -> bool:
        if tool_name in self.mutating_tools:
            return True
        return self.snapshot_before_bash and tool_name in self.bash_tools

    def checkpoints(self) -> list[Checkpoint]:
        """List every checkpoint for this project, oldest first."""
        return self._store().list_checkpoints()

    def restore(self, checkpoint_id: str, *, paths: Sequence[str] | None = None) -> None:
        """Restore the project's files from a checkpoint.

        With `paths=None` the whole snapshot is restored; pass project-relative
        paths to restore only those. See `CheckpointStore.restore` for the exact
        `git checkout` semantics (created-since files are not removed).
        """
        self._store().restore(checkpoint_id, paths=paths)

    def snapshot(self, *, tool_name: str | None = None) -> Checkpoint:
        """Take a checkpoint now, outside the tool-call flow (e.g. a manual save point)."""
        return self._store().snapshot(tool_name=tool_name)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Expose restore/list tools when `expose_tool` is set, else no tools."""
        if not self.expose_tool:
            return None
        return RestoreCheckpointToolset[AgentDepsT](self._store())

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Snapshot the project before a mutating tool runs. Best-effort; never blocks the run."""
        if self._should_snapshot(call.tool_name):
            store = self._store()
            take = functools.partial(store.snapshot, tool_name=call.tool_name, run_id=ctx.run_id)
            try:
                await anyio.to_thread.run_sync(take)
            except CheckpointError as exc:
                warnings.warn(
                    f'Could not snapshot before `{call.tool_name}`: {exc}. The run continues without a '
                    f'checkpoint for this tool call.',
                    CheckpointWarning,
                    stacklevel=2,
                )
        return args
