"""Regression tests for one-off capability and toolset IDs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


def _no_arguments() -> dict[str, Any]:
    return {}


def _path_argument(name: str) -> Callable[[], dict[str, Any]]:
    return lambda: {name: Path('.')}


def _browser_arguments() -> dict[str, Any]:
    return {'browser_agent': lambda task: None}


def _exa_search_arguments() -> dict[str, Any]:
    return {'client': object()}


def _exa_agent_arguments() -> dict[str, Any]:
    return {'runs': object()}


def _sub_agents_arguments() -> dict[str, Any]:
    subagents = pytest.importorskip('pydantic_ai_harness.subagents')
    return {
        'agents': [subagents.SubAgent(Agent(TestModel(), name='worker'))],
        'agent_folders': [],
    }


def _dynamic_workflow_arguments() -> dict[str, Any]:
    return {'agents': [Agent(TestModel(), name='worker')]}


def _spend_arguments() -> dict[str, Any]:
    return {'expose_tools': True}


def _you_arguments() -> dict[str, Any]:
    return {'client': object()}


CapabilityFactory = Callable[[], dict[str, Any]]


def _leaf_ids(toolset: Any) -> list[str | None]:
    ids: list[str | None] = []
    toolset.apply(lambda leaf: ids.append(leaf.id))
    return ids


CAPABILITY_ID_CASES: tuple[tuple[str, str, str | None, str, CapabilityFactory], ...] = (
    ('pydantic_ai_harness.shell', 'Shell', 'shell', 'shell', _no_arguments),
    ('pydantic_ai_harness.filesystem', 'FileSystem', 'file_system', 'file_system', _no_arguments),
    ('pydantic_ai_harness.subagents', 'SubAgents', 'sub_agents', 'sub_agents', _sub_agents_arguments),
    ('pydantic_ai_harness.modal_sandbox', 'ModalSandbox', 'modal_sandbox', 'modal_sandbox', _no_arguments),
    ('pydantic_ai_harness.exa', 'ExaSearch', 'exa_search', 'exa_search', _exa_search_arguments),
    ('pydantic_ai_harness.exa', 'ExaAgent', 'exa_agent', 'exa_agent', _exa_agent_arguments),
    (
        'pydantic_ai_harness.capability_creation',
        'CapabilityCreation',
        'capability_creation',
        'capability_creation',
        _path_argument('directory'),
    ),
    ('pydantic_ai_harness.browser_use', 'BrowserUse', 'browser_use', 'browser_use', _browser_arguments),
    (
        'pydantic_ai_harness.repo_context',
        'RepoContext',
        'repo_context',
        'repo_context',
        _path_argument('workspace_dir'),
    ),
    ('pydantic_ai_harness.pydantic_ai_docs', 'PydanticAIDocs', 'pydantic_ai_docs', 'pydantic_ai_docs', _no_arguments),
    ('pydantic_ai_harness.localstack', 'LocalStack', 'local_stack', 'local_stack', _no_arguments),
    ('pydantic_ai_harness.macroscope', 'Macroscope', 'macroscope', 'macroscope', _no_arguments),
    ('pydantic_ai_harness.playwright', 'PlaywrightBrowser', 'playwright', 'playwright', _no_arguments),
    (
        'pydantic_ai_harness.tool_output_limits',
        'ToolOutputLimits',
        'tool_output_limits',
        'tool_output_limits',
        _no_arguments,
    ),
    (
        'pydantic_ai_harness.dynamic_workflow',
        'DynamicWorkflow',
        'dynamic_workflow',
        'dynamic_workflow',
        _dynamic_workflow_arguments,
    ),
    # Multiple memories are a supported composition, so the capability keeps its
    # auto-suffixed id while its leaf toolset retains the established durable id.
    ('pydantic_ai_harness.memory', 'Memory', None, 'memory', _no_arguments),
    ('pydantic_ai_harness.planning', 'Planning', 'planning', 'planning', _no_arguments),
    ('pydantic_ai_harness.spend', 'SpendLimits', 'spend', 'spend', _spend_arguments),
    ('pydantic_ai_harness.youdotcom', 'YouSearch', 'you_search', 'you_search', _you_arguments),
    ('pydantic_ai_harness.youdotcom', 'YouResearch', 'you_research', 'you_research', _you_arguments),
)


@pytest.mark.parametrize(
    ('module_name', 'class_name', 'default_capability_id', 'default_toolset_id', 'arguments'), CAPABILITY_ID_CASES
)
def test_default_and_custom_ids_reach_toolsets(
    module_name: str,
    class_name: str,
    default_capability_id: str | None,
    default_toolset_id: str,
    arguments: CapabilityFactory,
) -> None:
    module = pytest.importorskip(module_name)
    capability_class: Any = getattr(module, class_name)
    kwargs = arguments()

    default_capability = capability_class(**kwargs)
    assert default_capability.id == default_capability_id
    assert _leaf_ids(default_capability.get_toolset()) == [default_toolset_id]

    custom_id = f'custom_{default_toolset_id}'
    custom_capability = capability_class(**kwargs, id=custom_id)
    assert custom_capability.id == custom_id
    assert _leaf_ids(custom_capability.get_toolset()) == [custom_id]


@pytest.mark.parametrize(
    ('module_name', 'class_name'),
    (
        ('pydantic_ai_harness.shell', 'Shell'),
        ('pydantic_ai_harness.modal_sandbox', 'ModalSandbox'),
        ('pydantic_ai_harness.localstack', 'LocalStack'),
    ),
)
def test_run_scoped_toolsets_keep_custom_id(module_name: str, class_name: str) -> None:
    module = pytest.importorskip(module_name)
    capability_class: Any = getattr(module, class_name)
    toolset = capability_class(id='custom').get_toolset()

    run_toolset = asyncio.run(toolset.for_run(None))

    assert run_toolset.id == 'custom'


def test_filtered_filesystem_keeps_the_leaf_id() -> None:
    filesystem = pytest.importorskip('pydantic_ai_harness.filesystem')

    toolset = filesystem.FileSystem(id='custom', read_only=True).get_toolset()

    assert _leaf_ids(toolset) == ['custom']


@pytest.mark.parametrize(
    ('module_name', 'class_name', 'toolset_id', 'arguments'),
    (
        ('pydantic_ai_harness.memory', 'Memory', 'memory', _no_arguments),
        ('pydantic_ai_harness.planning', 'Planning', 'planning', _no_arguments),
        ('pydantic_ai_harness.spend', 'SpendLimits', 'spend', _spend_arguments),
        ('pydantic_ai_harness.playwright', 'PlaywrightBrowser', 'playwright', _no_arguments),
    ),
)
def test_established_toolset_ids_survive_explicit_none(
    module_name: str,
    class_name: str,
    toolset_id: str,
    arguments: CapabilityFactory,
) -> None:
    module = pytest.importorskip(module_name)
    capability_class: Any = getattr(module, class_name)
    capability = capability_class(**arguments(), id=None)

    assert capability.id is None
    assert _leaf_ids(capability.get_toolset()) == [toolset_id]
