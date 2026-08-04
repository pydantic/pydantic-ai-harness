"""Fixtures that keep E2B sandbox unit tests off the real service."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from .fake_e2b import FakeE2B


class _PoisonedE2B(types.ModuleType):
    def __getattr__(self, name: str) -> object:  # pragma: no cover - unit-test tripwire
        raise AssertionError('An E2B unit test touched the real SDK. Use the `fake_e2b` fixture.')


@pytest.fixture(autouse=True)
def _no_real_e2b(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, 'e2b', _PoisonedE2B('e2b'))
    yield


@pytest.fixture
def fake_e2b(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeE2B]:
    control = FakeE2B()
    monkeypatch.setitem(sys.modules, 'e2b', control.module)
    yield control
