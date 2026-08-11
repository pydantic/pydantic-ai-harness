"""Keep every capability's configurable fields visible in the `AgentSpec` JSON schema.

A spec file carries a `# yaml-language-server: $schema=` line, so each `capabilities:` block
is validated against the generated schema. A capability whose fields never reach that schema
still loads at runtime -- it only marks every configured block invalid in an editor and fails
any CI that validates specs. Nothing in a normal test run notices, which is why this defect
has now shipped from three separate authors (#537, #538, and the entries below).

Core builds a capability's schema entry from exactly one signature: `from_spec` when the
capability overrides it, `__init__` otherwise (`pydantic_ai/agent/spec.py`
`_get_schema_target`, then `pydantic_ai/_spec.py` `build_schema_types`). Four ways that
signature yields nothing usable, all of them silent:

1. `from_spec(*args, **kwargs)` -- `build_schema_types` drops variadic parameters, leaving
   no hints at all.
2. A name in the read signature's annotations is not resolvable at runtime (a
   `TYPE_CHECKING`-only import, or a core alias like `ModelSelection`, which is the string
   `'Model | KnownModelName | str'` and is resolved against the *harness* module's globals).
   `_get_schema_target` catches the `NameError` and falls back to `AbstractCapability`'s
   variadic `from_spec`, i.e. straight to case 1.
3. A field type with no JSON schema. `filter_serializable_type` only strips `TypeVar` and
   `Callable`, so an arbitrary class survives into a TypedDict built with
   `arbitrary_types_allowed=True`. A bare `X` or a nullable `X | None` then fails schema
   generation and pydantic drops the whole `spec_<Name>` member from the capabilities union,
   erasing every other field on that capability. (`X | <representable>` is the benign case:
   only `X` is dropped.)
4. Schema generation raises outright, so no capability in the spec gets a schema.

Cases 2 and 3 are silent because core swallows them: `_get_schema_target` catches the
`NameError` and falls back, and pydantic's union handling drops a member it cannot
schematize instead of raising. Proposed upstream in
<https://github.com/pydantic/pydantic-ai/issues/7180>. This file is the downstream half --
if either path stops being silent, cases 2 and 3 arrive as a real error and layers 1 and 2
change character.

The opt-out is in the code, not in a list here: `get_capability_registry` keys on
`get_serialization_name()`, so returning `None` removes a capability from the spec system
entirely and the sweep below skips it by construction. `SubAgents`, the guardrails and
`DynamicWorkflow` do exactly that.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import pkgutil
import types
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeGuard

import pytest
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability

from tests.conftest import is_missing_optional_extra  # pyright: ignore[reportMissingTypeStubs]

_ROOT = Path(__file__).parent.parent
_PACKAGE = _ROOT / 'pydantic_ai_harness'
_HARNESS = 'pydantic_ai_harness'

# `AbstractCapability`'s own fields. All three are plain JSON types, so any capability that
# publishes an entry at all can publish these -- which is what lets this check run with no
# allowlist. A `from_spec` override that forgets them silently removes the ability to set
# `id` or `defer_loading` on that capability from a spec.
_BASE_FIELDS = frozenset({'id', 'description', 'defer_loading'})

_REPO = 'https://github.com/pydantic/pydantic-ai-harness'

# Tracking issue per root cause -- one line to change when an issue is opened or closed.
# Every issue carries its own per-capability breakdown; the maps below only say which
# capability belongs to which. Fixing a capability means deleting its entry here too, since
# the xfails are strict and an unexpected pass fails the build.
_TRACKERS = {
    'variadic-from-spec': f'{_REPO}/issues/537',
    'unresolvable-annotation': f'{_REPO}/issues/552',
    'unrepresentable-field': f'{_REPO}/issues/553',
    'missing-base-fields': f'{_REPO}/issues/554',
}

# Capabilities that publish no schema entry at all, by root cause. An entry leaves this map
# when its capability starts publishing one; the map is expected to shrink to nothing.
_NO_SCHEMA_ENTRY = {
    # `from_spec(*args, **kwargs)`. PR #543 is in review for this one.
    'StepPersistence': 'variadic-from-spec',
    # `model: ModelSelection`, whose `'Model | ...'` alias resolves against this module.
    'Advisor': 'unresolvable-annotation',
    'SummarizingCompaction': 'unresolvable-annotation',
    # `logfire_instance: Logfire | None`, imported under `TYPE_CHECKING`. Resolving the name
    # is not enough here -- `Logfire | None` then lands in cause 3 (see #553).
    'ManagedPrompt': 'unresolvable-annotation',
    # Inherits `local_docs_path: Path | None`, but `@dataclass` rebuilds `__init__` in
    # `_deprecated.py`, which does not import `Path` at runtime.
    'PyaiDocs': 'unresolvable-annotation',
    # `os_access: AbstractOS | None`
    'CodeMode': 'unrepresentable-field',
    # `source: HistorySource`
    'ConversationSearch': 'unrepresentable-field',
    # `session: ModalSandboxSession | None`
    'ModalSandbox': 'unrepresentable-field',
}

# Capabilities whose entry exists but cannot express the `AbstractCapability` base fields.
_NO_BASE_FIELDS = {
    'BrowserUse': 'missing-base-fields',
    'ExaAgent': 'missing-base-fields',
    'ExaSearch': 'missing-base-fields',
    'Memory': 'missing-base-fields',
    'Planning': 'missing-base-fields',
    # Tracked under #553, not #554: `include`/`exclude` are `Collection[str] | None`, so the
    # long form is dropped and only the one-parameter short form survives. Naming the base
    # fields in `from_spec` would not bring it back.
    'Skills': 'unrepresentable-field',
}


def _import_capability_module(module: str) -> ModuleType | ImportError:
    """Import a capability package, returning the failure rather than raising it.

    Discovery runs at module import, so raising here is a collection error that takes every
    check in this file down with it. Returned, an unimportable package is one named test.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # deprecation shims and `experimental` warn on import
            return importlib.import_module(module)
    except ImportError as error:  # pragma: no cover -- only reached without an extra installed
        return error


def _capability_modules() -> list[str]:
    """Every capability package: the top level plus `experimental` (matches `test_docs_parity`)."""
    modules: list[str] = []
    for path, prefix in ((_PACKAGE, _HARNESS), (_PACKAGE / 'experimental', f'{_HARNESS}.experimental')):
        modules.extend(info.name for info in pkgutil.iter_modules([str(path)], f'{prefix}.') if info.ispkg)
    return sorted(modules)


def _is_capability_type(obj: object) -> TypeGuard[type[AbstractCapability[Any]]]:
    """`vars(module)` is untyped, so narrow before reaching for a capability's classmethods.

    A subscripted generic (`Callable[[str], int]`, exported by `media`, `spend`, `subagents`
    and others) satisfies `isinstance(obj, type)` on Python 3.10 but is not a class, so
    `issubclass` raises `TypeError` on it there while returning False everywhere else.
    """
    return isinstance(obj, type) and not isinstance(obj, types.GenericAlias) and issubclass(obj, AbstractCapability)


def _discover() -> tuple[list[type[AbstractCapability[Any]]], dict[str, ImportError]]:
    """Every spec-registerable capability, plus the packages that would not import.

    Capabilities are keyed by serialization name to dedupe re-exports. Only failures that
    are not an absent extra are reported: those are the ones a test has to fail on.
    """
    found: dict[str, type[AbstractCapability[Any]]] = {}
    unimportable: dict[str, ImportError] = {}
    for module_name in _capability_modules():
        module = _import_capability_module(module_name)
        if isinstance(module, ImportError):  # pragma: no cover -- only reached without an extra installed
            if not is_missing_optional_extra(module):
                unimportable[module_name] = module
            continue
        for obj in vars(module).values():
            # `AbstractCapability` has no abstract methods, so `inspect.isabstract` does not
            # exclude it; a package re-exporting it (or a core capability) would otherwise be
            # swept as if harness owned its schema.
            if not _is_capability_type(obj) or not obj.__module__.startswith(_HARNESS):
                continue
            name = obj.get_serialization_name()
            if name is not None:
                found[name] = obj
    return [found[name] for name in sorted(found)], unimportable


_CAPABILITY_MODULES = _capability_modules()
_CAPABILITIES, _UNIMPORTABLE = _discover()

if TYPE_CHECKING:

    class _DeferredOnly:
        """Never exists at runtime, standing in for `Model` in `tool_output_limits/_bands.py`."""


@dataclass(frozen=True)
class _Nested:
    """A nested dataclass whose own annotation names something only the type checker has."""

    value: _DeferredOnly | None = None


@dataclass
class _Unschematizable(AbstractCapability[Any]):
    """`Sequence[_Nested]` resolves, so `__init__` is read and pydantic then fails on `_Nested`."""

    nested: Sequence[_Nested] = ()


def _name(capability: type[AbstractCapability[Any]]) -> str:
    serialization_name = capability.get_serialization_name()
    assert serialization_name is not None  # `_discover_capabilities` filters out the opt-outs
    return serialization_name


def _cases(known_gaps: dict[str, str]) -> list[Any]:
    """One parameter per capability, xfailing the ones a tracked issue already covers.

    The xfails are strict, so a capability that starts passing fails until its entry is
    removed. That is what keeps the maps above shrinking rather than becoming a permanent
    exemption list.
    """
    cases: list[Any] = []
    for capability in _CAPABILITIES:
        name = _name(capability)
        cause = known_gaps.get(name)
        marks = (
            ()
            if cause is None
            else (pytest.mark.xfail(strict=True, reason=f'{name}: {cause}, tracked in {_TRACKERS[cause]}'),)
        )
        cases.append(pytest.param(capability, marks=marks, id=name))
    return cases


@functools.cache
def _schema(capability: type[AbstractCapability[Any]]) -> dict[str, Any]:
    return AgentSpec.model_json_schema_with_capabilities([capability])


def _defs(capability: type[AbstractCapability[Any]]) -> dict[str, Any]:
    defs: dict[str, Any] = _schema(capability).get('$defs', {})
    return defs


def test_capabilities_discovered() -> None:
    # Half of the anti-vacuity guard: a moved package root, or an environment where most
    # capabilities skipped, cannot make the parametrized checks below pass on nothing. The
    # `slim` CI job lacks the `browser-use`, `exa` and `stackone` extras and still finds 27.
    # The other half is `test_capability_package_imports`, which keeps the only reason a
    # capability may go missing to an extra this repo declares.
    assert len(_CAPABILITIES) >= 25
    assert len(_CAPABILITY_MODULES) >= 25


@pytest.mark.parametrize('module', _CAPABILITY_MODULES, ids=lambda m: m.removeprefix(f'{_HARNESS}.'))
def test_capability_package_imports(module: str) -> None:
    error = _UNIMPORTABLE.get(module)
    assert error is None, (
        f"{module} does not import, and the failure is not one of this repo's optional extras "
        f'({type(error).__name__}: {error}).\n'
        'Add the missing module to `EXTRA_MODULES` in `tests/conftest.py` if a new extra provides '
        'it; otherwise this is a defect, and tolerating it here would drop the capability out of '
        'the schema sweep below without anything saying so.'
    )


def test_a_subscripted_generic_is_not_mistaken_for_a_capability() -> None:
    """`isinstance(Callable[...], type)` is True on Python 3.10, where `issubclass` then raises."""
    assert _is_capability_type(Callable[[str], int]) is False
    assert _is_capability_type(_Unschematizable) is True


def _nothing_resolves(name: str) -> ModuleSpec | None:
    """Stand-in for `importlib.util.find_spec` in an environment where nothing is installed."""
    return None


def test_an_absent_declared_extra_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The predicate reads two things: the exception, and whether the module resolves.

    Every extra is installed in some environment that runs these tests and absent in
    another, so the second input is pinned here rather than left to the environment.
    """
    monkeypatch.setattr(importlib.util, 'find_spec', _nothing_resolves)
    assert is_missing_optional_extra(ModuleNotFoundError('no modal', name='modal')) is True
    # `browser_use`, `exa` and `stackone` re-raise a bare `ImportError` naming their extra.
    re_raised = ImportError('browser-use is required for BrowserUse')
    re_raised.__cause__ = ModuleNotFoundError('no browser_use', name='browser_use')
    assert is_missing_optional_extra(re_raised) is True
    # `stackone` nests two `ImportError`s over the `ModuleNotFoundError`.
    nested = ImportError('stackone needs fastmcp')
    nested.__cause__ = re_raised
    assert is_missing_optional_extra(nested) is True


def test_a_missing_submodule_inside_an_installed_extra_is_reported() -> None:
    """A version skew or a broken install is a real failure, not an absent extra.

    `logfire` is in the `dev` group, so it is present in every environment that runs these
    tests. Reducing `logfire.variables` to its top-level name would tolerate this, and the
    capability would drop out of the sweep exactly when its dependency is broken.
    """
    assert importlib.util.find_spec('logfire') is not None
    assert is_missing_optional_extra(ModuleNotFoundError('gone', name='logfire.variables')) is False
    gated = ImportError('logfire is required for ManagedPrompt')
    gated.__cause__ = ModuleNotFoundError('gone', name='logfire.variables')
    assert is_missing_optional_extra(gated) is False


def test_anything_other_than_a_declared_extra_is_reported() -> None:
    """`is_missing_optional_extra` decides what may vanish from the sweep, so pin its edges."""
    # A harness module that no longer exists, and a third party nothing declares, are defects.
    assert is_missing_optional_extra(ModuleNotFoundError('gone', name='pydantic_ai_harness.exa')) is False
    assert is_missing_optional_extra(ModuleNotFoundError('gone', name='requests')) is False
    # A name that vanished from a module never reaches a `ModuleNotFoundError` at all.
    assert is_missing_optional_extra(ImportError('cannot import name X')) is False
    nameless_cause = ImportError('wrapped')
    nameless_cause.__cause__ = ModuleNotFoundError('no name on this one')
    assert is_missing_optional_extra(nameless_cause) is False


def _generation_problem(capability: type[AbstractCapability[Any]]) -> str | None:
    """Why this capability cannot be schematized at all, or `None` when it can."""
    try:
        _schema(capability)
    except Exception as exc:
        return (
            f'Generating the AgentSpec schema for {_name(capability)} raises {type(exc).__name__}: {exc}\n'
            'One capability that cannot be schematized breaks the schema for every capability in the '
            'spec, not only its own entry. Give it a `from_spec` naming only the parameters a spec can '
            'express (see `SpendLimits.from_spec` and `ToolOutputLimits.from_spec`), keeping a '
            '`**unsupported: Any` catch-all so the runtime-only fields are rejected by name.'
        )
    return None


@pytest.mark.parametrize('capability', _cases({}))
def test_capability_spec_schema_generates(capability: type[AbstractCapability[Any]]) -> None:
    problem = _generation_problem(capability)
    assert problem is None, problem


def test_the_generation_check_catches_a_capability_that_cannot_be_schematized() -> None:
    """`_Unschematizable` reproduces the shape `ToolOutputLimits` had before its `from_spec`.

    The nested dataclass resolves fine, so `_get_schema_target` reads `__init__` and hands
    pydantic a type it cannot build a schema for. That raises rather than being dropped, which
    is what made one capability able to break the schema for a whole spec.
    """
    problem = _generation_problem(_Unschematizable)

    assert problem is not None
    assert 'from_spec' in problem


@pytest.mark.parametrize('capability', _cases(_NO_SCHEMA_ENTRY))
def test_capability_publishes_a_spec_entry(capability: type[AbstractCapability[Any]]) -> None:
    name = _name(capability)
    defs = _defs(capability)
    assert f'spec_{name}' in defs or f'short_spec_{name}' in defs, (
        f'{name} publishes no fields to the AgentSpec schema, so every `{name}:` block a user writes '
        'is invalid against the schema the spec file points at, while still loading at runtime.\n'
        'Give it a `from_spec` naming every spec-expressible parameter as keyword-only, including '
        '`id`, `description` and `defer_loading`, plus a `**unsupported: Any` catch-all. See '
        "`SpendLimits.from_spec` for the reference shape, and this module's docstring for the four "
        'ways a signature ends up publishing nothing.\n'
        f'If the capability should not be configurable from a spec at all, override '
        f'`get_serialization_name` to return `None` instead (as `SubAgents` does).'
    )


@pytest.mark.parametrize('capability', _cases(_NO_SCHEMA_ENTRY | _NO_BASE_FIELDS))
def test_capability_spec_accepts_base_fields(capability: type[AbstractCapability[Any]]) -> None:
    name = _name(capability)
    params = _defs(capability).get(f'spec_params_{name}')
    assert params is not None, (
        f'{name} publishes only a one-parameter short form, so a spec cannot set '
        f'{sorted(_BASE_FIELDS)} on it. Either the `from_spec` signature names a single parameter, '
        "or a field type with no JSON schema caused the full form to be dropped (see this module's "
        'docstring, cause 3).'
    )
    missing = sorted(_BASE_FIELDS - set(params['properties']))
    assert not missing, (
        f'{name} does not publish {missing} to the AgentSpec schema, so a spec cannot set those '
        f'fields on it. `AbstractCapability` defines them for every capability; a `from_spec` '
        'override has to name them and pass them through to the constructor, or they reach '
        '`cls()` only by accident through `**kwargs` and never reach the schema.'
    )
