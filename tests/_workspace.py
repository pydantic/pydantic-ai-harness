"""Local workspaces for tests that run real commands in them."""

import os
from pathlib import Path

from pydantic_ai.workspaces import LocalWorkspaceBackend

HOST_ENV = {'PATH': os.environ['PATH']}
"""A local workspace inherits nothing from this process, so commands get `PATH` to find `rg`, `git`, and `python`."""


def local_workspace(working_dir: str | Path) -> LocalWorkspaceBackend:
    """A `LocalWorkspaceBackend` at `working_dir` whose commands find the host's programs."""
    return LocalWorkspaceBackend(working_dir, env=HOST_ENV)
