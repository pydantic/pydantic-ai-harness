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

from collections.abc import Iterator

import logfire
import pytest

_LOGFIRE_CREDENTIAL_VARS = ('LOGFIRE_TOKEN', 'LOGFIRE_API_KEY')


@pytest.fixture(autouse=True, scope='package')
def _scrub_logfire_credentials() -> Iterator[None]:
    """Remove ambient Logfire credentials while this package's tests run.

    Logfire lazily creates a `LogfireRemoteVariableProvider` that issues requests to the Logfire API
    whenever a credential such as `LOGFIRE_API_KEY` is present in `os.environ`, even with
    `send_to_logfire=False`. A credential inherited from the developer's shell, or from a `.env`
    file loaded into the process, would therefore make these tests write to the real Logfire API
    from background publish threads. Package scope orders this ahead of every `logfire.configure()`
    below and in the test modules, which capture `api_key` at configure time.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in _LOGFIRE_CREDENTIAL_VARS:
            monkeypatch.delenv(name, raising=False)
        yield


@pytest.fixture(autouse=True, scope='module')
def _configure_logfire() -> None:
    """Configure Logfire once so variable resolution does not warn (warnings are errors)."""
    logfire.configure(send_to_logfire=False, console=False)


@pytest.fixture
def anyio_backend() -> str:
    # Pin to asyncio: some tests use `asyncio` primitives directly, and the resolution behavior is
    # backend-agnostic, so running the trio leg too would only duplicate work.
    return 'asyncio'
