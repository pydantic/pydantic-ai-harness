"""Private helpers for capabilities adopting the run sandbox."""

from pathlib import Path
from typing import NoReturn

from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.workspaces import (
    SupportsCommands,
    UnavailableWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceReadOnlyError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    WrapperWorkspace,
)

READ_ONLY_FAILURE = 'The workspace is read-only; the change was refused.'
"""What the model sees when a tool's mutation reaches a read-only workspace."""


def workspace_path(path: Path) -> str:
    """Return the sandbox spelling of a configured path; `~` is not expanded."""
    if path.parts and path.parts[0].startswith('~'):
        raise UserError(
            f'Workspace paths do not expand `~`: {path!s}. '
            'Use an absolute path inside the sandbox or a path relative to its working directory.'
        )
    return path.as_posix()


def raise_tool_failure(error: WorkspaceError) -> NoReturn:
    """Report a deliberate workspace failure to the model as a failed tool call.

    A retry cannot fix a read-only workspace, an expired deadline, or a backend that refused the
    operation, so these become `ToolFailed` rather than `ModelRetry`. `WorkspaceUnavailableError`
    means the environment is gone and is re-raised to end the run.

    Callers catch `WorkspaceError` before any `PermissionError` or `OSError` handler:
    `WorkspaceReadOnlyError` is also a `PermissionError` and `WorkspaceTimeoutError` is also a
    `TimeoutError`, so a broader handler placed first would give them the wrong treatment.
    """
    if isinstance(error, WorkspaceUnavailableError):
        raise error
    if isinstance(error, WorkspaceReadOnlyError):
        raise ToolFailed(READ_ONLY_FAILURE) from error
    if isinstance(error, WorkspaceTimeoutError):
        deadline = '' if error.timeout is None else f' after {error.timeout:g}s'
        raise ToolFailed(f'The workspace operation timed out{deadline}.') from error
    raise ToolFailed(str(error) or f'The workspace operation failed ({type(error).__name__}).') from error


def supports_commands(workspace: Workspace) -> bool:
    """Whether `workspace.run` can succeed: the innermost backend executes commands and no policy refuses them.

    Wrappers are unwrapped through `wrapped`, not `backend`: a durable workspace refuses
    `backend` in workflow code, where tool registration runs.
    """
    if workspace.read_only:
        return False
    current: WorkspaceBackend = workspace
    while isinstance(current, Workspace):
        current = current.wrapped if isinstance(current, WrapperWorkspace) else current.backend
    return isinstance(current, SupportsCommands)


def workspace_attached(workspace: Workspace) -> bool:
    """Whether the run has a real workspace, rather than core's placeholder for a run without one.

    Wrappers are unwrapped through `wrapped`, not `backend`: a durable workspace refuses `backend`
    in workflow code.
    """
    current: WorkspaceBackend = workspace
    while isinstance(current, Workspace):
        current = current.wrapped if isinstance(current, WrapperWorkspace) else current.backend
    return not isinstance(current, UnavailableWorkspace)
