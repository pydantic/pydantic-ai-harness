"""Private helpers for capabilities adopting the run sandbox."""

import posixpath
from pathlib import Path
from typing import NoReturn

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.workspaces import (
    SupportsCommands,
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
    """Whether `workspace.run` can succeed: a workspace is attached, its innermost backend executes commands, and no policy refuses them."""
    if not workspace.attached or workspace.read_only:
        return False
    return isinstance(innermost_backend(workspace), SupportsCommands)


def innermost_backend(workspace: Workspace) -> WorkspaceBackend:
    """The backend at the bottom of `workspace`'s facades and wrappers.

    Wrappers are unwrapped through `wrapped`, not `backend`: a durable workspace refuses
    `backend` in workflow code, where tool registration runs.
    """
    current: WorkspaceBackend = workspace
    while isinstance(current, Workspace):
        current = current.wrapped if isinstance(current, WrapperWorkspace) else current.backend
    return current


def require_workspace(workspace: Workspace, owner: str) -> None:
    """Raise `UserError` when the run has no workspace, naming `owner` and how to attach one.

    Called from `before_run`, so a missing workspace fails the run at its start rather than on
    the first tool call.
    """
    if not workspace.attached:
        raise UserError(
            f'`{owner}` needs a workspace, but none is attached to this run. '
            "Add `LocalWorkspace('.')` (this machine) or a sandbox capability such as `ModalSandbox()` "
            "to the agent's capabilities, or pass `workspace=` to the run. "
            'See https://pydantic.dev/docs/ai/workspace/'
        )


def secondary_workspace(value: WorkspaceBackend | None, owner: str) -> Workspace | None:
    """Validate a capability's own `workspace=` argument, for storage or resources kept outside the run's workspace.

    It takes a backend (or a `Workspace` facade), never a workspace capability: the capability
    supplies a run's workspace, not one to read from on the side.
    """
    if value is None:
        return None
    if isinstance(value, AbstractCapability):
        raise TypeError(
            f'`{owner}(workspace=...)` takes a workspace backend, not the `{type(value).__name__}` capability. '
            "Pass a backend such as `LocalWorkspaceBackend('/app')`."
        )
    return value if isinstance(value, Workspace) else Workspace(value)


METADATA_DIR = '.pydantic-ai-harness'
"""The directory, below a workspace's working directory, that holds files harness capabilities keep for themselves."""


async def metadata_dir(workspace: Workspace, name: str) -> str:
    """Return `<working_dir>/.pydantic-ai-harness/<name>`, creating it.

    `.pydantic-ai-harness` gets a `.gitignore` holding `*` whenever it has none, including a
    directory that already existed, so none of it shows up in `git status`. Only filesystem operations are used, so a workspace that
    cannot run commands works too. A path in the way raises `WorkspaceError`: it is no tool
    argument's fault, so a retry cannot fix it.
    """
    root = posixpath.join(await workspace.working_dir(), METADATA_DIR)
    directory = posixpath.join(root, name)
    gitignore = posixpath.join(root, '.gitignore')
    try:
        await workspace.make_dir(directory)
        if not await workspace.exists(gitignore):
            await workspace.write_text(gitignore, '*\n')
    except WorkspaceError:
        raise
    except OSError as error:
        raise WorkspaceError(f'Cannot create `{METADATA_DIR}/{name}` in the workspace: {error.strerror}') from error
    return directory
