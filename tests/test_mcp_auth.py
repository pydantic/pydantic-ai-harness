"""Test the shared hosted MCP helpers: credential lookup and read-only selection."""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_harness._mcp import credential, is_read_only


def test_explicit_credential_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', 'deployment-token')
    assert credential('user-token', env='WHOAMI_TOKEN', service='whoami') == 'user-token'


def test_environment_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', 'deployment-token')
    assert credential(None, env='WHOAMI_TOKEN', service='whoami') == 'deployment-token'


@pytest.mark.parametrize('env', ['WHOAMI_TOKEN', None])
@pytest.mark.parametrize('auth', [None, ''])
def test_missing_credential_raises(auth: str | None, env: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('WHOAMI_TOKEN', '')
    with pytest.raises(UserError, match='to connect to whoami'):
        credential(auth, env=env, service='whoami')


def test_oauth_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(UserError, match='Browser OAuth is not supported'):
        credential('oauth', env=None, service='whoami')
    monkeypatch.setenv('WHOAMI_TOKEN', 'oauth')
    with pytest.raises(UserError, match='Browser OAuth is not supported'):
        credential(None, env='WHOAMI_TOKEN', service='whoami')


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({'annotations': {'readOnlyHint': True}}, True),
        ({'annotations': {'readOnlyHint': False}}, False),
        ({'annotations': {}}, False),
        (None, False),
    ],
)
def test_is_read_only_requires_an_explicit_hint(metadata: dict[str, object] | None, expected: bool) -> None:
    assert is_read_only(ToolDefinition(name='tool', metadata=metadata)) is expected
