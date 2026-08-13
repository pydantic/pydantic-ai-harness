"""Shared safety and fake-SDK fixtures for Islo sandbox tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from .fake_islo import FakeIslo


class _PoisonedIslo(types.ModuleType):
    """Prevent unit tests from creating a billed sandbox through the real SDK."""

    def __getattr__(self, name: str) -> object:  # pragma: no cover - safety tripwire
        raise AssertionError(
            'An islo_sandbox unit test touched the real `islo` package. '
            'Use the `fake_islo` fixture, or mark the test `islo_live`.'
        )


@pytest.fixture
def anyio_backend() -> str:
    """The real Islo token provider currently requires asyncio."""
    return 'asyncio'


@pytest.fixture(autouse=True)
def _no_real_islo(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison Islo imports except in the explicitly selected live tier."""
    if 'islo_live' in request.keywords:  # pragma: no cover - live tier is opt-in
        yield
        return
    monkeypatch.setitem(sys.modules, 'islo', _PoisonedIslo('islo'))
    yield


@pytest.fixture
def fake_islo(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeIslo]:
    """Install the fake Islo module tree and yield its control surface."""
    control = FakeIslo()
    modules: dict[str, object] = {}
    control.install(modules)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    yield control
