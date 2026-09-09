import pytest

# Termflow is 3.11+ only, and so is the CLI that renders with it.
pytest.importorskip('termflow')


@pytest.fixture
def anyio_backend() -> str:
    # The CLI drives runs with `asyncio.run`, and `Coder`'s members use asyncio tasks.
    return 'asyncio'
