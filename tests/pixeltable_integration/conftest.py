"""Shared setup for the Pixeltable integration tests.

The folder is not named `tests/pixeltable`: Pyright treats `tests` as a root, and a
`pixeltable` folder there would shadow the installed package.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest

# Tests run in their own Pixeltable catalog, never `~/.pixeltable`. The variable must be set
# before any test module imports pixeltable. The home is stable per pytest-xdist worker, so
# later runs reuse the embedded Postgres server rather than each starting a new one (each
# holds a shared-memory segment until stopped).
os.environ['PIXELTABLE_HOME'] = os.path.join(
    tempfile.gettempdir(), f'harness-pixeltable-tests-{os.environ.get("PYTEST_XDIST_WORKER", "main")}'
)

# The `pixeltable` dependency is gated on the `pixeltable` extra (and needs Python 3.11+), so
# slim CI runs can't import these modules. Ignore them at collection. A conditional expression
# rather than an `if` statement: branch coverage traces statement arcs, and no single
# environment can take both arms of an install-dependent branch.
collect_ignore = (
    ['test_capability.py', 'test_memory.py', 'test_store.py', 'test_toolset.py']
    if importlib.util.find_spec('pixeltable') is None
    else []
)


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'
