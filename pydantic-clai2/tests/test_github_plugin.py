"""The built-in `github` plugin: declared disabled, authenticated from `GITHUB_TOKEN`, refusing to load without it."""

import io
from pathlib import Path

import pytest
from pydantic import JsonValue
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.github import GitHub
from rich.console import Console

from pydantic_clai2 import DEFAULT_PLUGINS, api_keys
from pydantic_clai2.capability_catalog import HARNESS_PLUGINS
from pydantic_clai2.commands import Commands
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'github')


def loader_for(tmp_path: Path, settings: dict[str, JsonValue] | None = None) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'config.db')
    declaration = BUILTIN if settings is None else BUILTIN.model_copy(update={'settings': settings})
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(declaration,),
    )


def test_declared_as_disabled_clai_plugin_not_raw_catalog_entry() -> None:
    assert BUILTIN.factory == 'pydantic_clai2.github'
    assert not BUILTIN.enabled
    assert BUILTIN.settings == {}
    assert all(plugin.factory != 'pydantic_ai_harness.github:GitHub' for plugin in HARNESS_PLUGINS)


async def test_enable_reads_environment_token_and_defaults_to_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('GITHUB_TOKEN', ' env-token ')
    api_keys.save_key(name='GITHUB_TOKEN', value='saved-token')
    loader = loader_for(tmp_path)
    await loader.enable('github')
    assert loader.capabilities() == [GitHub[None](auth='env-token', read_only=True)]


async def test_saved_key_is_used_without_environment_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GITHUB_TOKEN', '  ')
    api_keys.save_key(name='GITHUB_TOKEN', value='saved-token')
    loader = loader_for(tmp_path, {'read_only': False})
    await loader.enable('github')
    assert loader.capabilities() == [GitHub[None](auth='saved-token', read_only=False)]


async def test_missing_token_fails_the_load_and_registers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    loader = loader_for(tmp_path)
    with pytest.raises(PluginError, match=r'set `GITHUB_TOKEN`, or save an API key named GITHUB_TOKEN'):
        await loader.enable('github')
    assert not loader.capabilities()
    assert 'GITHUB_TOKEN' in (loader.entries()[0].error or '')


async def test_unknown_setting_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GITHUB_TOKEN', 'env-token')
    with pytest.raises(PluginError, match='toolsets'):
        await loader_for(tmp_path, {'toolsets': ['repos']}).enable('github')
