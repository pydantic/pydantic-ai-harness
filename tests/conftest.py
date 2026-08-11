from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic_ai.models
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# `dirty-equals` matchers are typed as `DirtyEquals[T]`, not `T`, so passing
# them where pydantic-ai expects concrete `str`/`datetime`/etc. fails pyright
# strict. Following pydantic-ai's own conftest, re-export with TYPE_CHECKING
# stubs that pretend the matchers return the concrete type. Tests should
# `from tests.conftest import IsStr, IsDatetime, ...` instead of importing
# from `dirty_equals` directly.
if TYPE_CHECKING:

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsPartialDict(*args: Any, **kwargs: Any) -> dict[Any, Any]: ...
else:
    from dirty_equals import IsDatetime, IsNow, IsPartialDict, IsStr

__all__ = ('EXTRA_MODULES', 'IsDatetime', 'IsNow', 'IsPartialDict', 'IsStr', 'is_missing_optional_extra')

# Top-level modules that this repo's own optional extras provide, from
# `[project.optional-dependencies]` in `pyproject.toml`. A capability package fails to
# import in the `slim` and `lowest-versions` CI jobs only because one of these is absent.
EXTRA_MODULES = frozenset(
    {
        'acp',  # agent-client-protocol -- acp
        'browser_use',  # browser-use -- browser-use
        'exa_py',  # exa-py -- exa
        'fastmcp',  # fastmcp -- stackone
        'logfire',  # logfire -- logfire
        'mcp',  # pydantic-ai-slim[mcp] -- stackone
        'modal',  # modal -- modal
        'pydantic_monty',  # pydantic-monty -- code-mode, dynamic-workflow
        'pymongo',  # pymongo -- mongodb
        'yaml',  # pyyaml -- skills
    }
)


def is_missing_optional_extra(error: ImportError) -> bool:
    """True when an import failed only because one of this repo's optional extras is absent.

    Deliberately not "the import failed". A capability whose annotations name something
    only the type checker has, or that imports a third party nothing declares, has to stay
    visible -- tolerating any `ImportError` would let both disappear from
    `tests/test_capability_specs.py`'s sweep and from the doc-snippet check.

    Three things have to hold. The failure must bottom out in a `ModuleNotFoundError`, which
    a capability's import gate hides: `browser_use`, `exa` and `stackone` catch it and
    re-raise their own `ImportError` carrying an install hint, so the module name lives on
    the `__cause__` chain rather than on the exception raised (`stackone` nests two deep).
    The module it names must be one an extra declared in `pyproject.toml` provides. And that
    module has to be genuinely absent.

    The last check is what separates "the extra is not installed" from "something inside an
    installed extra is missing". `browser_use.browser` going missing while `browser_use`
    imports is a version skew or a broken install -- a real failure, and one that hits
    exactly the capabilities this tolerance covers, so it has to be reported rather than
    skipped. Matching on the top-level name alone cannot tell the two apart; asking whether
    that top-level module resolves can.
    """
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, ModuleNotFoundError) and cause.name is not None:
            top, _, _ = cause.name.partition('.')
            return top in EXTRA_MODULES and importlib.util.find_spec(top) is None
        cause = cause.__cause__
    return False


# Prevent accidental real model requests during tests.
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def test_model() -> TestModel:
    """A fresh ``TestModel`` instance for each test."""
    return TestModel()


@pytest.fixture
def test_agent(test_model: TestModel) -> Agent[None, str]:
    """A minimal agent wired to ``TestModel`` for capability tests."""
    return Agent(test_model, name='test-agent')


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Convenience alias for ``tmp_path`` (useful for store / session tests)."""
    return tmp_path


@pytest.fixture
def allow_model_requests() -> Iterator[None]:
    """Temporarily allow real model requests within a test."""
    with pydantic_ai.models.override_allow_model_requests(True):
        yield
