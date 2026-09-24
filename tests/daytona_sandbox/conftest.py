from __future__ import annotations

import importlib.util
import os
import types
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

_HAS_DAYTONA = importlib.util.find_spec('daytona') is not None
collect_ignore = (
    []
    if _HAS_DAYTONA
    else ['test_backend.py', 'test_conformance.py', 'test_daytona_live.py', 'test_daytona_sandbox.py']
)

if TYPE_CHECKING or _HAS_DAYTONA:  # pragma: no branch - installed and slim jobs take opposite branches
    import daytona

    from .fake_daytona import FakeDaytona


def require_live_credentials() -> None:
    """Skip a live test without a non-empty `DAYTONA_API_KEY`, or fail it where `DAYTONA_REQUIRE_LIVE` is set.

    CI sets `DAYTONA_REQUIRE_LIVE`, so a missing or empty secret turns the live job red instead of
    reporting green having tested nothing.
    """
    if os.environ.get('DAYTONA_API_KEY'):
        return
    message = 'the Daytona live tier requires a non-empty DAYTONA_API_KEY'
    if os.environ.get('DAYTONA_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}:
        pytest.fail(message)
    pytest.skip(message)


class _PoisonedDaytona(types.ModuleType):
    """A `daytona` stand-in that fails if a unit test reaches the real SDK."""

    def __getattr__(self, name: str) -> object:  # pragma: no cover - tripwire, hit only by a misbehaving test
        raise AssertionError(
            'A daytona_sandbox unit test touched the real `daytona` package. Use the `fake_daytona` fixture.'
        )


@pytest.fixture(autouse=True)
def _no_real_daytona(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Poison lazy `daytona` imports unless a test installs the fake or is in the opt-in live tier."""
    if 'daytona_live' in request.keywords:  # pragma: no cover - live tier runs without coverage
        require_live_credentials()
        yield
        return
    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend.daytona', _PoisonedDaytona('daytona'))
    yield


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


if _HAS_DAYTONA:  # pragma: no branch - the fixture requires the SDK-backed fake

    @pytest.fixture
    def fake_daytona(monkeypatch: pytest.MonkeyPatch) -> FakeDaytona:
        fake = FakeDaytona()
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend.daytona', daytona)
        monkeypatch.setattr(daytona, 'AsyncDaytona', fake.client)
        return fake
