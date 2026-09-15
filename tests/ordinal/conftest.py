"""Shared collection rules for the Ordinal capability tests."""

from __future__ import annotations

import importlib.util

import pytest

# `mcp` is gated on the `ordinal` extra, so slim CI runs (no extras) can't import
# these modules. Ignore them at collection; `test_packaging.py` stays collected
# because it checks package metadata, which holds on base installs too.
# A conditional expression rather than an `if` statement: branch coverage traces
# statement arcs, and no single environment can take both arms of an
# install-dependent branch.
collect_ignore = ['test_ordinal.py'] if importlib.util.find_spec('mcp') is None else []


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
