"""Shared fixtures for the `logfire_variables` test package.

The directory is named `logfire_variables`, not `logfire`, on purpose: pyright scopes test-only
report overrides with `executionEnvironments = [{ root = 'tests' }]`, which makes `tests/` an
import root -- so a `tests/logfire/` directory would shadow the third-party `logfire` package for
every test file's `import logfire`.

All resolution runs against the code default (no Logfire provider is configured) unless a test
installs one via `variables_provider`, which is exactly the safety-net behavior the managed-variable
capabilities rely on. Each test uses a unique variable name because the default Logfire instance
keeps its variable registry across `configure()` calls.
"""

from __future__ import annotations

import logfire
import pytest


@pytest.fixture(autouse=True, scope='module')
def _configure_logfire() -> None:
    """Configure Logfire once so variable resolution does not warn (warnings are errors)."""
    logfire.configure(send_to_logfire=False, console=False)


@pytest.fixture
def anyio_backend() -> str:
    # Pin to asyncio: some tests use `asyncio` primitives directly, and the resolution behavior is
    # backend-agnostic, so running the trio leg too would only duplicate work.
    return 'asyncio'
