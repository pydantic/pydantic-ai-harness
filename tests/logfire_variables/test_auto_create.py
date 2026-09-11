"""Tests for auto-create-on-first-use in the `ManagedVariableCapability` base.

When a managed variable is used but doesn't exist in Logfire yet, the base creates it in the
background with the code default as its value, so the Logfire UI becomes the editing surface without
a manual create-in-UI step. These tests exercise that path through `ManagedPrompt` (the smallest
full capability built on the base) against a `LocalVariableProvider`, which -- like the remote
provider -- reports an unknown variable as `'unrecognized_variable'` and supports `create_variable`.

Because real creation happens on a fire-and-forget daemon thread, the tests replace the module-level
`_spawn_create` seam with an inline version (via the `spawned`/`spawned_inline` fixtures) so the
attempt is deterministic and any warning surfaces on the calling thread.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import logfire
import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import (
    ResolvedVariable,
    Rollout,
    Variable,
    VariableAlreadyExistsError,
    VariableConfig,
    VariablesConfig,
)
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import ManagedPrompt
from pydantic_ai_harness.logfire import _managed_variable

from ._helpers import variables_provider

pytestmark = pytest.mark.anyio

DEFAULT = 'You are a helpful assistant.'


def instructions_seen(result_messages: list[ModelMessage]) -> list[str]:
    """Collect the rendered instructions from each `ModelRequest` in a run."""
    return [m.instructions for m in result_messages if isinstance(m, ModelRequest) and m.instructions is not None]


def logged_messages(capfire: CaptureLogfire) -> list[str]:
    """The message of each exported log record, skipping the variable-resolution spans."""
    return [
        span['attributes']['logfire.msg']
        for span in capfire.exporter.exported_spans_as_dict()
        if span['attributes'].get('logfire.span_type') == 'log'
    ]


def _existing_config(name: str) -> VariablesConfig:
    """A config that already knows `name` (so resolution reports `'resolved'`, not unknown)."""
    return VariablesConfig(
        variables={name: VariableConfig(name=name, labels={}, rollout=Rollout(labels={}), overrides=[])}
    )


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    # The once-per-process guard is module-level state; clear it so each test starts fresh.
    _managed_variable._reset_auto_create_guard()
    _managed_variable._durable_write_warnings.clear()


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record spawned variable names without creating anything.

    Leaving the provider untouched means a repeated run still resolves as unknown, so the
    once-per-process guard -- not a now-existing variable -- is what governs repeat attempts.
    """
    names: list[str] = []

    def record(variable: Variable[Any], config: VariableConfig) -> None:
        names.append(variable.name)

    monkeypatch.setattr(_managed_variable, '_spawn_create', record)
    return names


@pytest.fixture
def spawned_inline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record spawned names and run creation inline on the calling thread for determinism."""
    names: list[str] = []

    def record_and_create(variable: Variable[Any], config: VariableConfig) -> None:
        names.append(variable.name)
        _managed_variable._create_variable(variable, config)

    monkeypatch.setattr(_managed_variable, '_spawn_create', record_and_create)
    return names


async def test_unknown_variable_is_auto_created(capfire: CaptureLogfire, spawned_inline: list[str]) -> None:
    config = VariablesConfig(variables={})
    with variables_provider(capfire, config):
        agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_new', default=DEFAULT)])
        result = await agent.run('hello')

    # The triggering run still uses the code default and succeeds.
    assert instructions_seen(result.all_messages()) == [DEFAULT]

    # The variable was created carrying its name, JSON schema, and the code default as the example.
    assert spawned_inline == ['prompt__auto_new']
    created = config.variables['prompt__auto_new']
    assert created.name == 'prompt__auto_new'
    assert created.json_schema == {'type': 'string'}
    assert created.example == '"You are a helpful assistant."'

    # Creating a variable puts a persistent, teammate-visible object in the user's project, so it is
    # reported into that same project rather than passing silently.
    assert logged_messages(capfire) == [
        'Created Logfire managed variable prompt__auto_new from the code default; '
        'set a value in Logfire to manage this agent from there'
    ]


async def test_durable_context_skips_auto_create_and_warns_once(capfire: CaptureLogfire, spawned: list[str]) -> None:
    class DurableContext(AbstractCapability[object]):
        @property
        def in_durable_context(self) -> bool:
            return True

    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(
            TestModel(), capabilities=[ManagedPrompt('durable_auto_create', default=DEFAULT), DurableContext()]
        )
        with pytest.warns(UserWarning, match='Create the variable in the Logfire UI') as caught:
            await agent.run('hello')
            await agent.run('again')
    assert len(caught) == 1
    assert spawned == []


async def test_code_default_fallback_reason_still_triggers_create(
    capfire: CaptureLogfire, spawned_inline: list[str]
) -> None:
    # Regression for logfire >= 4.37: an unrecognized variable's fallback to the code default is
    # reported as the generic `'code_default'` reason (older SDKs surfaced `'unrecognized_variable'`
    # as the final reason). Auto-create must key off the provider not recognizing the name -- not the
    # exact reason string -- so it still fires here. Assert both that the run saw a code-default
    # fallback reason and that the variable was created.
    capability = ManagedPrompt('auto_reason', default=DEFAULT)
    seen_reasons: list[str | None] = []

    def record_reason() -> str:
        resolved = capability.resolved
        seen_reasons.append(_managed_variable.resolution_reason(resolved) if resolved is not None else None)
        return 'ok'

    config = VariablesConfig(variables={})
    with variables_provider(capfire, config):
        agent = Agent(TestModel(), tools=[record_reason], capabilities=[capability])
        await agent.run('hello')

    # The value fell back to the code default (whichever reason this logfire surfaces for it)...
    assert seen_reasons == ['code_default'] or seen_reasons == ['unrecognized_variable']
    # ...and that still triggered creation of the previously-unknown variable.
    assert spawned_inline == ['prompt__auto_reason']
    assert 'prompt__auto_reason' in config.variables


async def test_real_background_thread_creates_variable(capfire: CaptureLogfire) -> None:
    # Exercise the genuine fire-and-forget path (no `_spawn_create` monkeypatch): a daemon thread
    # creates the variable in the local provider off the run's thread. Poll inside the provider
    # block so the creation lands on the local provider before it is reconfigured away on exit.
    config = VariablesConfig(variables={})
    with variables_provider(capfire, config):
        agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_thread', default=DEFAULT)])
        result = await agent.run('hello')

        deadline = time.monotonic() + 5.0
        while 'prompt__auto_thread' not in config.variables and time.monotonic() < deadline:
            time.sleep(0.01)  # pragma: no cover - timing race; skipped when the thread wins before the first check

    assert instructions_seen(result.all_messages()) == [DEFAULT]
    assert config.variables['prompt__auto_thread'].example == '"You are a helpful assistant."'


async def test_known_variable_is_not_created(capfire: CaptureLogfire, spawned: list[str]) -> None:
    with variables_provider(capfire, _existing_config('prompt__auto_known')):
        agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_known', default=DEFAULT)])
        await agent.run('hello')

    assert spawned == []


async def test_auto_create_false_skips_creation(capfire: CaptureLogfire, spawned: list[str]) -> None:
    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_off', default=DEFAULT, auto_create=False)])
        await agent.run('hello')

    assert spawned == []


async def test_code_default_with_no_provider_does_not_create(
    monkeypatch: pytest.MonkeyPatch, spawned: list[str]
) -> None:
    def code_default(_resolved: ResolvedVariable[Any]) -> str:
        return 'code_default'

    monkeypatch.setattr(_managed_variable, 'resolution_reason', code_default)
    await Agent(TestModel(), capabilities=[ManagedPrompt('no_provider', default=DEFAULT)]).run('hello')
    assert spawned == []


async def test_creation_attempted_once_per_process(capfire: CaptureLogfire, spawned: list[str]) -> None:
    # `spawned` deliberately does not create the variable, so both runs still resolve it as unknown;
    # only the module-level guard keeps the second run from spawning again.
    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_once', default=DEFAULT)])
        await agent.run('hello')
        await agent.run('hello again')

    assert spawned == ['prompt__auto_once']


async def test_creation_is_attempted_once_per_logfire_instance(spawned_inline: list[str]) -> None:
    # A process can serve several Logfire projects. The guard is keyed by the destination instance as
    # well as the name, so creating the variable in the first project doesn't stand in for the second.
    configs = [VariablesConfig(variables={}), VariablesConfig(variables={})]
    instances = [
        logfire.configure(
            local=True,
            send_to_logfire=False,
            console=False,
            variables=logfire.LocalVariablesOptions(config=config),
        )
        for config in configs
    ]

    for instance in instances:
        capability = ManagedPrompt('auto_two_projects', default=DEFAULT, logfire_instance=instance)
        await Agent(TestModel(), capabilities=[capability]).run('hello')
        # A second run against the same instance is still guarded.
        await Agent(TestModel(), capabilities=[capability]).run('hello')

    assert spawned_inline == ['prompt__auto_two_projects', 'prompt__auto_two_projects']
    assert all('prompt__auto_two_projects' in config.variables for config in configs)


async def test_already_exists_is_swallowed(capfire: CaptureLogfire, spawned_inline: list[str]) -> None:
    with variables_provider(capfire, VariablesConfig(variables={})):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()

        def raise_already_exists(config: VariableConfig) -> VariableConfig:
            raise VariableAlreadyExistsError(f"Variable '{config.name}' already exists")

        provider.create_variable = raise_already_exists  # type: ignore[method-assign]

        agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_race', default=DEFAULT)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = await agent.run('hello')

    # A race with another writer is fine: no warning, run unaffected.
    assert [w for w in caught if 'auto-create' in str(w.message)] == []
    assert instructions_seen(result.all_messages()) == [DEFAULT]


async def test_generic_failure_warns_once(capfire: CaptureLogfire, spawned_inline: list[str]) -> None:
    with variables_provider(capfire, VariablesConfig(variables={})):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()

        def raise_forbidden(config: VariableConfig) -> VariableConfig:
            raise RuntimeError('403 Forbidden')

        provider.create_variable = raise_forbidden  # type: ignore[method-assign]

        agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_forbidden', default=DEFAULT)])
        with pytest.warns(UserWarning, match="Failed to auto-create Logfire managed variable 'prompt__auto_forbidden'"):
            result = await agent.run('hello')

        # The warning is raised on the creation thread, where it is easy to miss, so the failure is
        # logged to Logfire as well.
        assert logged_messages(capfire) == ['Failed to auto-create Logfire managed variable prompt__auto_forbidden']

    # The failure is surfaced but the run is unaffected.
    assert instructions_seen(result.all_messages()) == [DEFAULT]


async def test_no_provider_makes_no_attempt(spawned: list[str]) -> None:
    # No `variables_provider`: the default `NoOpVariableProvider` reports `'no_provider'`, which must
    # not trigger auto-create (there is nothing to create into).
    agent = Agent(TestModel(), capabilities=[ManagedPrompt('auto_no_provider', default=DEFAULT)])
    result = await agent.run('hello')

    assert spawned == []
    assert instructions_seen(result.all_messages()) == [DEFAULT]


async def test_unreachable_provider_probe_does_not_fail_the_run(
    capfire: CaptureLogfire, spawned: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto-create is best-effort, and the provider probe is its one remote call on the run thread.

    A remote provider that is unreachable -- or that raises for any other reason -- must leave the run
    with the code default rather than take it down, which is what the capability documents.
    """
    with variables_provider(capfire, VariablesConfig(variables={})):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()

        def unreachable(name: str) -> Any:
            raise ConnectionError('provider unreachable')

        monkeypatch.setattr(provider, 'get_variable_config', unreachable)
        result = await Agent(TestModel(), capabilities=[ManagedPrompt('auto_unreachable', default=DEFAULT)]).run(
            'hello'
        )

    assert result.output.startswith('success')
    assert spawned == []
