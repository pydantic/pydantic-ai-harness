"""Regression test: the audit_log capability's doc examples must compile, not just parse.

`tests/test_doc_snippets.py` at the repo root checks every capability README/docs
snippet by calling `ast.parse` on it. A bare `await` at module scope is not a
grammar error there -- `ast.parse('await foo()')` succeeds -- it is only a
`compile()`-time error (`SyntaxError: 'await' outside function`), which is what
actually runs when a reader pastes the snippet into a file. That gap is exactly how
the Quick start example on this capability's three documentation surfaces (the
README, the mirrored docs site page, and the `AuditLog` class docstring, which
feeds the docs site's API-reference block) shipped with a top-level `await` and
still passed the repo-wide check.

This test compiles (never executes) every fenced Python example on those three
surfaces so a bare top-level `await` -- or any other error `ast.parse` misses --
cannot silently return.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from pytest_examples import CodeExample, find_examples

_ROOT = Path(__file__).resolve().parent.parent.parent
_SURFACES = (
    _ROOT / 'pydantic_ai_harness' / 'audit_log' / 'README.md',
    _ROOT / 'pydantic_ai_harness' / 'audit_log' / '_capability.py',
    _ROOT / 'docs' / 'audit-log.md',
)


def _examples() -> list[ParameterSet]:
    return [pytest.param(ex, id=f'{ex.path}:{ex.start_line}') for ex in find_examples(*_SURFACES)]


@pytest.mark.parametrize('example', _examples())
def test_example_compiles(example: CodeExample) -> None:
    """`ast.parse` (what the repo-wide check uses) accepts top-level `await`; `compile()` does not."""
    compile(example.source, str(example.path), 'exec')


def test_examples_are_discovered() -> None:
    # Guard against a discovery break (a renamed file, an unmatched fence) silently
    # making the check above vacuous.
    assert len(_examples()) >= 3
