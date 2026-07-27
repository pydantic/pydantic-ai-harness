"""Optional model-facing tools for listing and restoring checkpoints.

Only registered when `Checkpoints(expose_tool=True)`. Restoring is usually a human
or application action, so these tools are off by default.
"""

from __future__ import annotations

import anyio.to_thread
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.checkpoints._shadow import CheckpointStore


class RestoreCheckpointToolset(FunctionToolset[AgentDepsT]):
    """Exposes `list_checkpoints` and `restore_checkpoint` over a `CheckpointStore`."""

    def __init__(self, store: CheckpointStore) -> None:
        super().__init__()
        self._store = store
        self.add_function(self.list_checkpoints, name='list_checkpoints')
        self.add_function(self.restore_checkpoint, name='restore_checkpoint')

    async def list_checkpoints(self) -> str:
        """List the available file checkpoints, oldest first, with the tool and files each captured."""
        checkpoints = await anyio.to_thread.run_sync(self._store.list_checkpoints)
        if not checkpoints:
            return 'No checkpoints recorded yet.'
        lines = [
            f'{cp.id}  {cp.time.isoformat()}  before={cp.tool_name or "-"}  files={", ".join(cp.files_changed) or "-"}'
            for cp in checkpoints
        ]
        return '\n'.join(lines)

    async def restore_checkpoint(self, checkpoint_id: str, paths: list[str] | None = None) -> str:
        """Restore project files from a checkpoint.

        Args:
            checkpoint_id: The id of a checkpoint from `list_checkpoints`.
            paths: Project-relative paths to restore. Omit to restore every file the
                checkpoint captured.
        """
        await anyio.to_thread.run_sync(lambda: self._store.restore(checkpoint_id, paths=paths))
        scope = ', '.join(paths) if paths else 'all files'
        return f'Restored {scope} from checkpoint {checkpoint_id}.'
