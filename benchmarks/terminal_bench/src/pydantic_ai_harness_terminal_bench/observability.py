"""Optional Logfire wiring for the live-model benchmark run.

Everything here is a no-op unless `LOGFIRE_TOKEN` is set, so the keyless smoke
path and any run without a token behave exactly as before. When the token is
present, `configure_observability` turns on Logfire and instruments Pydantic AI
once per process, and `trial_span` wraps each task in a span tagged with the
Harbor task/trial ids (VStorm's `tb.task` / `tb.trial` convention).

`logfire` is an optional, run-time-only dependency: it is installed in the live
CI job, not in the shipped package, so its imports are lazy and guarded.
"""

from __future__ import annotations

import contextlib
import os
from contextlib import AbstractContextManager

LOGFIRE_TOKEN_ENV = 'LOGFIRE_TOKEN'
"""Presence of this env var is what turns observability on."""

_SERVICE_NAME = 'terminal-bench'

# Tri-state process cache: None = not yet decided, then True/False once resolved.
_active: bool | None = None


def configure_observability() -> bool:
    """Configure Logfire and instrument Pydantic AI when `LOGFIRE_TOKEN` is set.

    Idempotent: the first call decides and caches whether observability is on;
    later calls return that decision without reconfiguring. Returns `True` when
    Logfire is active, `False` when no token is set.
    """
    global _active
    if _active is not None:
        return _active

    if not os.environ.get(LOGFIRE_TOKEN_ENV):
        _active = False
        return False

    import logfire  # pyright: ignore[reportMissingImports]

    logfire.configure(service_name=_SERVICE_NAME, console=False)
    logfire.instrument_pydantic_ai()
    _active = True
    return True


def trial_span(task_id: str, trial_id: str) -> AbstractContextManager[object]:
    """A Logfire span tagged with the task/trial ids, or a no-op when inactive.

    Returns a null context manager when observability is off, so callers can
    always `with trial_span(...):` regardless of whether a token is set.
    """
    if not configure_observability():
        return contextlib.nullcontext()

    import logfire  # pyright: ignore[reportMissingImports]

    return logfire.span(
        'terminal_bench trial {tb.task}',
        **{'tb.task': task_id, 'tb.trial': trial_id},
    )
