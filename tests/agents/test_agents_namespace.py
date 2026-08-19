"""The `pydantic_ai_harness.agents` namespace exports the complete agents, and the pre-move paths still import."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic_ai import Agent

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.agents import Coder, Researcher
from pydantic_ai_harness.agents.coder import DEFAULT_ALLOWED_COMMANDS
from pydantic_ai_harness.agents.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS


def test_namespace_exports_the_capabilities() -> None:
    import pydantic_ai_harness.agents

    assert pydantic_ai_harness.agents.DEFAULT_ALLOWED_COMMANDS is DEFAULT_ALLOWED_COMMANDS
    assert pydantic_ai_harness.agents.DEFAULT_RESEARCHER_INSTRUCTIONS is DEFAULT_RESEARCHER_INSTRUCTIONS


@pytest.mark.parametrize('name', ['coder_agent', 'researcher_agent'])
def test_namespace_exports_the_agents(name: str) -> None:
    import pydantic_ai_harness.agents

    assert isinstance(getattr(pydantic_ai_harness.agents, name), Agent)


def test_namespace_agent_exports_are_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import pydantic_ai_harness.agents; '
            "assert 'pydantic_ai_harness.agents.coder._agent' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_namespace_unknown_export() -> None:
    import pydantic_ai_harness.agents

    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.agents.__getattr__('missing')


@pytest.mark.parametrize(
    ('old_path', 'capability', 'agent_name'),
    [
        ('pydantic_ai_harness.coder', Coder, 'coder_agent'),
        ('pydantic_ai_harness.researcher', Researcher, 'researcher_agent'),
    ],
)
def test_fresh_import_warns_and_aliases_old_path(old_path: str, capability: type[object], agent_name: str) -> None:
    from importlib import import_module

    sys.modules.pop(old_path, None)
    with pytest.warns(HarnessDeprecationWarning, match=r'has moved to `pydantic_ai_harness\.agents\.'):
        module = import_module(old_path)

    assert getattr(module, capability.__name__) is capability
    assert isinstance(getattr(module, agent_name), Agent)
    with pytest.raises(AttributeError, match='has no attribute'):
        module.__getattr__('missing')
