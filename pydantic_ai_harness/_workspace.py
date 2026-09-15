"""Private helpers for capabilities adopting the run workspace."""

from pathlib import Path

from pydantic_ai.exceptions import UserError


def workspace_path(path: Path) -> str:
    """Return the workspace spelling of a configured path; `~` is not expanded."""
    if path.parts and path.parts[0].startswith('~'):
        raise UserError(
            f'Workspace paths do not expand `~`: {path!s}. '
            'Use an absolute path inside the workspace or a path relative to its working directory.'
        )
    return path.as_posix()
