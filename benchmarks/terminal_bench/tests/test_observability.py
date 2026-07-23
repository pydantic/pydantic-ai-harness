"""Tests for the optional Logfire wiring.

These exercise only the no-token path (the default). With no `LOGFIRE_TOKEN`,
observability must stay off and `trial_span` must be a usable no-op, so the
keyless smoke path and any token-less run are unchanged. The token path pulls in
`logfire`, which is not a package dependency, so it is left to the live CI job.
"""

from __future__ import annotations

import contextlib

import pytest

from pydantic_ai_harness_terminal_bench import observability


@pytest.fixture(autouse=True)
def _reset_observability_cache() -> None:
    # The module caches its on/off decision for the process; reset it per test.
    observability._active = None


def test_configure_returns_false_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(observability.LOGFIRE_TOKEN_ENV, raising=False)
    assert observability.configure_observability() is False
    # Cached, and still off on a second call.
    assert observability.configure_observability() is False


def test_trial_span_is_a_noop_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(observability.LOGFIRE_TOKEN_ENV, raising=False)
    span = observability.trial_span('fix-git', 'fix-git__abc')
    assert isinstance(span, contextlib.nullcontext)
    with span:
        pass
