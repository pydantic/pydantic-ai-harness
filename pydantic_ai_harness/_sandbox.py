"""Private helpers for capabilities adopting the run sandbox."""

from pathlib import Path

from pydantic_ai.exceptions import UserError


def sandbox_path(path: Path) -> str:
    """Return the sandbox spelling of a configured path; `~` is not expanded."""
    if path.parts and path.parts[0].startswith('~'):
        raise UserError(
            f'Sandbox paths do not expand `~`: {path!s}. '
            'Use an absolute path inside the sandbox or a path relative to its working directory.'
        )
    return path.as_posix()
