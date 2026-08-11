"""Static validation of the code snippets shown in the docs.

Every Python snippet in a capability `README.md` (GitHub/PyPI) and in the flat
`docs/<capability>.md` pages (the unified docs site) is checked for the two
failure modes a reader hits immediately:

- **it does not parse** -- a syntax error means the snippet cannot run at all;
- **it imports a harness symbol that does not exist** -- a stale module path or a
  renamed/removed name (e.g. a snippet still importing from
  `pydantic_ai_harness.experimental.<graduated>`).

This is the *static* half of doc-snippet testing. It deliberately does not
execute the snippets -- most build an `Agent` and call `.run()`, which needs a
model -- so it stays fast and needs no mocking. Running snippets against a mocked
model is a separate concern (see `test_readme_quick_start.py` for that shape).

Illustrative signature blocks (API-reference pseudo-code with type annotations or
a bare `*`, which is not runnable Python) opt out with a `{test="skip"}` fence
directive.
"""

from __future__ import annotations as _annotations

import ast
import importlib
import importlib.util
import os
import warnings
from collections.abc import Iterable
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from pytest_examples import CodeExample, find_examples

from tests.conftest import is_missing_optional_extra  # pyright: ignore[reportMissingTypeStubs]

_ROOT = Path(__file__).parent.parent
_HARNESS = 'pydantic_ai_harness'


def _harness_import_targets(tree: ast.AST) -> Iterable[tuple[str, str | None]]:
    """`(module, name)` for every `pydantic_ai_harness` symbol a snippet imports.

    `name` is `None` for a plain `import pydantic_ai_harness.x` or a star import,
    where only the module's existence can be checked.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if module == _HARNESS or module.startswith(f'{_HARNESS}.'):
                for alias in node.names:
                    yield module, None if alias.name == '*' else alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _HARNESS or alias.name.startswith(f'{_HARNESS}.'):
                    yield alias.name, None


def _snippet_problem(source: str) -> str | None:
    """Return why a snippet is invalid, or `None` if it parses and its harness imports resolve."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f'does not parse: {exc.msg} (line {exc.lineno})'

    for module, name in _harness_import_targets(tree):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # a deprecated shim path still resolves; existence is what we check
                imported = importlib.import_module(module)
        except ImportError as error:
            if is_missing_optional_extra(error):
                continue  # the harness module exists; this environment lacks its declared extra
            return f'imports `{module}`, which does not resolve: {error}'
        if name is not None and not hasattr(imported, name):
            return f'imports `{name}` from `{module}`, but that name does not exist'
    return None


def _doc_snippets() -> Iterable[ParameterSet]:
    # `find_examples` yields only Python fenced blocks and wants paths relative to
    # the cwd, so pin it to the repo root (matches `test_skill_examples.py`).
    os.chdir(_ROOT)
    readmes = sorted(str(p.relative_to(_ROOT)) for p in _ROOT.glob(f'{_HARNESS}/**/README.md'))
    for ex in find_examples(*readmes, 'docs'):
        yield pytest.param(ex, id=f'{ex.path}:{ex.start_line}')


@pytest.mark.parametrize('example', _doc_snippets())
def test_doc_snippet_valid(example: CodeExample) -> None:
    if example.prefix_settings().get('test', '').startswith('skip'):
        pytest.skip('illustrative signature block; not runnable Python')
    problem = _snippet_problem(example.source)
    assert problem is None, (
        f'{example.path}:{example.start_line} {problem}. '
        'Fix the snippet, or mark the fence `{test="skip"}` if it is illustrative signature pseudo-code.'
    )


def test_doc_snippets_discovered() -> None:
    # Guard against a discovery break silently making the check vacuous.
    assert sum(1 for _ in _doc_snippets()) >= 100


def test_snippet_problem_detects_each_failure_mode() -> None:
    # Valid: harness imports that resolve, star imports, plain imports, and non-harness imports.
    assert _snippet_problem('from pydantic_ai_harness import CodeMode') is None
    assert _snippet_problem('import pydantic_ai_harness.code_mode') is None
    assert _snippet_problem('from pydantic_ai_harness.tool_output_limits import *') is None
    assert _snippet_problem('from os import path\nimport sys') is None
    # Invalid: syntax, a module that does not exist, and a name that does not exist.
    assert 'does not parse' in (_snippet_problem('def (:') or '')
    assert 'does not resolve' in (_snippet_problem('from pydantic_ai_harness.nope import X') or '')
    assert 'does not exist' in (_snippet_problem('from pydantic_ai_harness import NoSuchCapability') or '')


def test_missing_optional_extra_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the `slim` environment: the harness module exists, but importing it
    # fails because its third-party extra is absent. That is not a broken snippet.
    # `acp` is installed here, so its absence has to be simulated too -- the tolerance
    # asks whether the module resolves, not only what the exception names.
    def _extra_missing(module: str) -> object:
        raise ModuleNotFoundError("No module named 'acp'", name='acp')

    monkeypatch.setattr(importlib, 'import_module', _extra_missing)

    def _nothing_resolves(name: str) -> ModuleSpec | None:
        return None

    monkeypatch.setattr(importlib.util, 'find_spec', _nothing_resolves)
    assert _snippet_problem('from pydantic_ai_harness.experimental.acp import run_acp_stdio') is None


def test_a_third_party_nothing_declares_is_still_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # An undeclared import is a defect, not an uninstalled extra -- see
    # `is_missing_optional_extra`. Tolerating it would hide the module from every check.
    def _undeclared_missing(module: str) -> object:
        raise ModuleNotFoundError("No module named 'requests'", name='requests')

    monkeypatch.setattr(importlib, 'import_module', _undeclared_missing)
    assert 'does not resolve' in (_snippet_problem('from pydantic_ai_harness.shell import Shell') or '')
