"""The Absurd SDK is installed through the package's optional `absurd` extra."""

from __future__ import annotations

import importlib.metadata


def _requires_dist() -> list[str]:
    return importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []


def test_absurd_extra_is_advertised() -> None:
    provides = importlib.metadata.metadata('pydantic-ai-harness').get_all('Provides-Extra') or []
    assert 'absurd' in provides


def test_absurd_sdk_is_an_optional_absurd_dependency() -> None:
    absurd_requirements = [req for req in _requires_dist() if req.startswith('absurd-sdk')]
    assert absurd_requirements, 'absurd-sdk must be declared'
    assert all('extra ==' in req and 'absurd' in req for req in absurd_requirements)


def test_absurd_extra_does_not_include_testcontainers() -> None:
    assert not any(req.startswith('testcontainers') for req in _requires_dist())
