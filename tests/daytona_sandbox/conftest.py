from __future__ import annotations

import pytest

from .fake_daytona import FakeDaytona


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def fake_daytona(monkeypatch: pytest.MonkeyPatch) -> FakeDaytona:
    import daytona

    fake = FakeDaytona()
    monkeypatch.setattr(daytona, 'AsyncDaytona', fake.client)
    return fake
