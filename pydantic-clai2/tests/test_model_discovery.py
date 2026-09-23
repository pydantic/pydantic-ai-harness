"""Recorded provider catalogs and offline discovery behavior."""

import os
from pathlib import Path

import anyio
import httpx2
import pytest
from menu_script import Script, make_context, pick
from pydantic import ValidationError
from pydantic_ai.providers.openai_codex import (
    CredentialsPersistenceError,
    OpenAICodexCredentials,
    OpenAICodexProvider,
)
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import TypedDict

from pydantic_clai2.auth import CodexCredentials
from pydantic_clai2.model_catalog import genai_prices_models, provider_catalog
from pydantic_clai2.model_discovery import CodexModels, discover_models
from pydantic_clai2.model_menu import ModelMenu, open_add_model_menu


class RecordedResponse(TypedDict):
    body: dict[str, str | bytes]
    headers: dict[str, list[str]]


def sanitize_response(response: RecordedResponse) -> RecordedResponse:
    decoded = httpx2.Response(
        200,
        headers=[(key, value) for key, values in response['headers'].items() for value in values],
        content=response['body']['string'],
    ).content
    try:
        decoded = CodexModels.model_validate_json(decoded).model_dump_json().encode()
    except ValidationError:
        pass
    response['body']['string'] = decoded
    response['headers'] = {'content-type': ['application/json']}
    return response


@pytest.fixture
def vcr_config() -> dict[str, object]:
    return {
        'filter_headers': ['authorization', 'x-api-key', 'chatgpt-account-id', 'openai-organization', 'openai-project'],
        'before_record_response': sanitize_response,
    }


@pytest.mark.vcr
@pytest.mark.parametrize('provider', ['openai', 'anthropic', 'deepseek'])
async def test_live_provider_catalog(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    key = f'{provider.upper()}_API_KEY'
    monkeypatch.setenv(key, os.getenv(key, 'recorded-key'))
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
    models, notice = await provider_catalog(provider=provider, current=f'{provider}:my-current-model')
    assert notice is None
    assert len(models) > 1
    assert all(model.provider == provider for model in models)
    names = [model.name for model in models]
    assert names == sorted(set(names))
    assert f'{provider}:my-current-model' in names


@pytest.mark.vcr
async def test_openai_compatible_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.deepseek.com')
    monkeypatch.setenv('OPENAI_API_KEY', os.getenv('DEEPSEEK_API_KEY', 'recorded-key'))
    names, notice = await discover_models(provider='openai-chat')
    assert notice is None and names
    assert all(name.startswith('openai-chat:deepseek-') for name in names)


@pytest.mark.vcr
async def test_codex_live_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = OpenAICodexCredentials(
        access_token='recorded-token', refresh_token='not-used', account_id='recorded-account'
    )
    if os.getenv('CLAI_RECORD_CODEX'):
        async with OpenAICodexProvider() as provider:
            credentials.access_token = provider.credentials.access_token
            credentials.account_id = provider.credentials.account_id

    async def load(self: CodexCredentials) -> OpenAICodexCredentials:
        return credentials

    monkeypatch.setattr(CodexCredentials, 'load', load)
    names, notice = await discover_models(provider='openai-codex')
    assert notice is None
    assert names and 'openai-codex:gpt-6-astra' in names
    assert 'openai-codex:codex-auto-review' not in names
    assert names == sorted(set(names))


@pytest.mark.parametrize(
    'provider', ['openai', 'openai-chat', 'openai-responses', 'anthropic', 'deepseek', 'openai-codex']
)
async def test_missing_credentials_keep_catalog(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPSEEK_API_KEY', 'OPENAI_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    models, notice = await provider_catalog(provider=provider, current=f'{provider}:saved-model')
    assert models
    assert notice and 'Using the built-in catalog' in notice
    if provider == 'openai-codex':
        assert '/login openai-codex' in notice


@pytest.mark.parametrize('alias', ['openai-chat', 'openai-responses'])
async def test_openai_aliases_retain_metadata(alias: str, monkeypatch: pytest.MonkeyPatch) -> None:
    original = next(model for model in genai_prices_models() if model.provider == 'openai' and model.context_window)
    name = f'{alias}:{original.name.partition(":")[2]}'

    async def discover(*, provider: str) -> tuple[list[str], None]:
        assert provider == alias
        return [name, f'{alias}:unknown-model'], None

    monkeypatch.setattr('pydantic_clai2.model_discovery.discover_models', discover)
    models, notice = await provider_catalog(provider=alias)
    assert notice is None
    found = next(model for model in models if model.name == name)
    assert found.provider == alias
    assert found.label == original.label
    assert found.context_window == original.context_window
    assert found.prices == original.prices
    unknown = next(model for model in models if model.name == f'{alias}:unknown-model')
    assert unknown.context_window is None and unknown.prices is None


async def test_empty_discovery_keeps_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    async def empty(*, provider: str) -> tuple[list[str], None]:
        return [], None

    monkeypatch.setattr('pydantic_clai2.model_discovery.discover_models', empty)
    models, notice = await provider_catalog(provider='anthropic')
    assert models and notice and 'No live models returned' in notice


@pytest.mark.vcr
async def test_rejected_api_key_keeps_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'deliberately-invalid-key')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    models, notice = await provider_catalog(provider='openai')
    assert models and notice and 'Check credentials' in notice
    assert 'deliberately-invalid-key' not in notice


async def test_unsupported_provider_stays_offline() -> None:
    assert await discover_models(provider='bedrock') == (None, None)


@pytest.mark.parametrize('error', [httpx2.ConnectError('secret'), ValueError('secret'), TimeoutError('secret')])
async def test_failed_discovery_closes_client(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[httpx2.AsyncClient] = []

    async def fail(self: httpx2.AsyncClient, url: str, *, params: dict[str, str]) -> httpx2.Response:
        clients.append(self)
        assert url == 'https://chatgpt.com/backend-api/codex/models'
        assert params == {'client_version': '0.156.1'}
        raise error

    monkeypatch.setattr(httpx2.AsyncClient, 'get', fail)
    names, notice = await discover_models(provider='openai-codex')
    assert names is None and notice and 'secret' not in notice
    assert len(clients) == 1 and clients[0].is_closed


async def test_credential_save_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(self: httpx2.AsyncClient, url: str, *, params: dict[str, str]) -> httpx2.Response:
        raise CredentialsPersistenceError('Could not save refreshed credentials')

    monkeypatch.setattr(httpx2.AsyncClient, 'get', fail)
    with pytest.raises(CredentialsPersistenceError):
        await discover_models(provider='openai-codex')


async def test_cancellation_closes_discovery_client(monkeypatch: pytest.MonkeyPatch) -> None:
    started = anyio.Event()
    clients: list[httpx2.AsyncClient] = []
    finished: list[bool] = []

    async def wait(self: httpx2.AsyncClient, url: str, *, params: dict[str, str]) -> httpx2.Response:
        clients.append(self)
        started.set()
        await anyio.sleep_forever()
        raise AssertionError('Cancellation must propagate')  # pragma: no cover -- sleep_forever cannot return.

    async def discover() -> None:
        await discover_models(provider='openai-codex')
        finished.append(True)

    monkeypatch.setattr(httpx2.AsyncClient, 'get', wait)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(discover)
            await started.wait()
            tasks.cancel_scope.cancel()
    assert not finished
    assert len(clients) == 1 and clients[0].is_closed


def test_codex_catalog_schema() -> None:
    parsed = CodexModels.model_validate(
        {'models': [{'slug': 'codex-model', 'visibility': 'list'}, {'slug': 'hidden-model', 'visibility': 'hide'}]}
    )
    assert [model.slug for model in parsed.models if model.visibility == 'list'] == ['codex-model']
    with pytest.raises(ValidationError):
        CodexModels.model_validate({'models': [{'slug': '', 'visibility': 'list'}]})


async def test_live_catalog_selection_and_back_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queried: list[str] = []

    async def discover(*, provider: str) -> tuple[list[str], None]:
        queried.append(provider)
        return [f'{provider}:new-model'], None

    monkeypatch.setattr('pydantic_clai2.model_discovery.discover_models', discover)
    context, applied = make_context(tmp_path)
    models, notice = await provider_catalog(provider='anthropic', current=context.settings.model)
    assert [model.name for model in models] == ['anthropic:new-model']
    assert notice is None
    script = Script(
        lists=[pick('anthropic'), MenuResult(cancelled=True), pick('openai'), pick('openai:new-model')],
        choices=[],
        texts=[],
    )
    assert await open_add_model_menu(context, runners=script.runners) == 'Saved model. Applied.'
    assert context.settings.model == 'openai:new-model'
    assert context.store.models() == ['openai-codex:gpt-6-astra', 'openai:new-model']
    assert applied == ['model']
    assert queried == ['anthropic', 'anthropic', 'openai']


def test_fallback_notice_is_visible_in_menu(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    menu = ModelMenu(context, notice='Discovery failed. Using the built-in catalog.')
    assert context.settings.model is not None
    assert 'Discovery failed' in menu.details(MenuItem('current', value=context.settings.model))
