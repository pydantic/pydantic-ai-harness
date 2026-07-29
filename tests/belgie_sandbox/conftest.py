"""Shared fixtures for Belgie Sandbox tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from .fake_belgie import FakeBelgie


class _PoisonedBelgie(types.ModuleType):
    def __getattr__(self, name: str) -> object:  # pragma: no cover - only hit by an unsafe test
        raise AssertionError(
            'A Belgie Sandbox unit test touched the real `belgie` package. '
            'Use the `fake_belgie` fixture, or mark the test `belgie_live`.'
        )


@pytest.fixture
def anyio_backend() -> str:
    """Belgie's Python async bindings require asyncio."""
    return 'asyncio'


@pytest.fixture(autouse=True)
def _no_real_belgie(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison Belgie except for the real-runtime smoke tier."""
    if 'belgie_live' in request.keywords:
        yield
        return
    poisoned = _PoisonedBelgie('belgie')
    monkeypatch.setitem(sys.modules, 'belgie', poisoned)
    monkeypatch.setitem(sys.modules, 'belgie.errors', poisoned)
    yield


@pytest.fixture
def fake_belgie(monkeypatch: pytest.MonkeyPatch) -> FakeBelgie:
    """Install and return the controllable fake Belgie modules."""
    control = FakeBelgie()
    monkeypatch.setitem(sys.modules, 'belgie', control.module)
    monkeypatch.setitem(sys.modules, 'belgie.errors', control.errors_module)
    return control
