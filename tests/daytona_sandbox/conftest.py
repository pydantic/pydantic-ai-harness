from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

_HAS_DAYTONA = importlib.util.find_spec('daytona') is not None
collect_ignore = [] if _HAS_DAYTONA else ['test_backend_protocol.py', 'test_daytona_sandbox.py']

if TYPE_CHECKING or _HAS_DAYTONA:  # pragma: no branch - installed and slim jobs take opposite branches
    import daytona

    from .fake_daytona import FakeDaytona


class _PoisonedDaytona(types.ModuleType):
    """A `daytona` stand-in that fails if a unit test reaches the real SDK."""

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A daytona_sandbox unit test touched the real `daytona` package. Use the `fake_daytona` fixture.'
        )


@pytest.fixture(autouse=True)
def _no_real_daytona(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison lazy `daytona` imports unless a test explicitly installs the fake."""
    monkeypatch.setitem(sys.modules, 'daytona', _PoisonedDaytona('daytona'))
    yield


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


if _HAS_DAYTONA:  # pragma: no branch - the fixture requires the SDK-backed fake

    @pytest.fixture
    def fake_daytona(monkeypatch: pytest.MonkeyPatch) -> FakeDaytona:
        fake = FakeDaytona()
        monkeypatch.setitem(sys.modules, 'daytona', daytona)
        monkeypatch.setattr(daytona, 'AsyncDaytona', fake.client)
        return fake
