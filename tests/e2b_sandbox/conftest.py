"""Shared fixtures for E2BSandbox tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

_HAS_E2B = importlib.util.find_spec('e2b') is not None
collect_ignore = [] if _HAS_E2B else ['test_backend.py', 'test_e2b_live.py', 'test_e2b_sandbox.py']

if TYPE_CHECKING or _HAS_E2B:  # pragma: no branch - the SDK-installed and slim jobs take opposite branches
    from .fake_e2b import FakeE2B


class _PoisonedE2B(types.ModuleType):
    """An `e2b` stand-in that fails loudly on any attribute access.

    Real e2b is installed in the dev venv for the live tier, so a unit test that forgets the
    `fake_e2b` fixture would otherwise reach the real SDK and, with a developer API key
    configured, create real billed sandboxes. This is the `ALLOW_MODEL_REQUESTS = False` of
    the E2B seam.
    """

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'An e2b_sandbox unit test touched the real `e2b` package. '
            'Use the `fake_e2b` fixture, or mark the test `e2b_live`.'
        )


@pytest.fixture(autouse=True)
def _no_real_e2b(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison `e2b` for every test here except the opt-in live tier."""
    if 'e2b_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        yield
        return
    monkeypatch.setitem(sys.modules, 'e2b', _PoisonedE2B('e2b'))
    yield


if _HAS_E2B:  # pragma: no branch - the fixture cannot be defined without its SDK-backed fake

    @pytest.fixture
    def fake_e2b(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeE2B]:
        """Inject a fake `e2b` module and yield its control surface."""
        control = FakeE2B()
        monkeypatch.setitem(sys.modules, 'e2b', control.module)
        yield control
