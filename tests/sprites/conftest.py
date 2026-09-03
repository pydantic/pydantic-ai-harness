"""Shared fixtures that prevent unit tests from reaching the real Sprites API."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from .fake_sprites import FakeSprites


class _PoisonedSprites(types.ModuleType):
    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire
        raise AssertionError('A Sprites unit test touched the real SDK. Use the `fake_sprites` fixture.')


@pytest.fixture(autouse=True)
def _no_real_sprites(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if 'sprites_live' in request.keywords:  # pragma: no cover - live tier
        yield
        return
    monkeypatch.setitem(sys.modules, 'sprites', _PoisonedSprites('sprites'))
    monkeypatch.delitem(sys.modules, 'sprites.exceptions', raising=False)
    yield


@pytest.fixture
def fake_sprites(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSprites]:
    control = FakeSprites()
    monkeypatch.setitem(sys.modules, 'sprites', control.module)
    monkeypatch.setitem(sys.modules, 'sprites.exceptions', control.exceptions_module)
    yield control
