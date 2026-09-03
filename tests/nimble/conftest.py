"""Shared collection rules for the Nimble capability tests."""

from __future__ import annotations

import importlib.util

# The `nimble_python` dependency is gated on the `nimble` extra, so slim CI runs
# (no extras) can't import these modules. Ignore them at collection.
collect_ignore = ['test_nimble.py'] if importlib.util.find_spec('nimble_python') is None else []
