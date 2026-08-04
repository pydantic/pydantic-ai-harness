"""Collection rules and public-surface helpers for scheduling tests."""

from __future__ import annotations

import importlib.util

collect_ignore = ['test_cron.py'] if importlib.util.find_spec('cronsim') is None else []
