from pathlib import Path

import pytest

# Termflow is 3.11+ only, and so is the CLI that renders with it.
pytest.importorskip('termflow')


@pytest.fixture
def anyio_backend() -> str:
    # The CLI drives runs with `asyncio.run`, and `Coder`'s members use asyncio tasks.
    return 'asyncio'


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty home, so `Config.default_path()` never reaches the developer's own config file."""
    monkeypatch.setenv('HOME', str(tmp_path))
    return tmp_path
