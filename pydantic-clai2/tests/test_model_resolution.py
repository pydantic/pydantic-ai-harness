"""Both interfaces retain CLAI-owned provider credentials and endpoints."""

import io

from pydantic import SecretStr
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.vllm import VLLMProvider
from rich.console import Console

from pydantic_clai2 import openrouter, vllm
from pydantic_clai2.auth import CodexAuth
from pydantic_clai2.model_resolution import resolve_model


async def test_clai_provider_resolution() -> None:
    auth = CodexAuth(Console(file=io.StringIO()))
    assert await resolve_model('test', auth=auth) == 'test'
    codex = await resolve_model('openai-codex:gpt-6-astra', auth=auth)
    assert isinstance(codex, OpenAICodexModel)
    assert codex.provider is auth.provider
    second = await resolve_model('openai-codex:another-model', auth=auth)
    assert isinstance(second, OpenAICodexModel)
    assert second.provider is codex.provider
    async with codex:
        assert codex.model_name == 'gpt-6-astra'

    openrouter.save_connection(openrouter.Connection(token=SecretStr('saved-openrouter-token')))
    router = await resolve_model('openrouter:vendor/model', auth=auth)
    assert isinstance(router, OpenRouterModel)
    async with router:
        assert router.model_name == 'vendor/model'
        assert isinstance(router.provider, OpenRouterProvider)
        assert router.provider.client.api_key == 'saved-openrouter-token'

    vllm.save_connection(vllm.Connection(url='http://127.0.0.1:8999/custom/v1', token=SecretStr('saved-vllm-token')))
    local = await resolve_model('vllm:local-model', auth=auth)
    assert isinstance(local, OpenAIChatModel)
    async with local:
        assert local.model_name == 'local-model'
        assert isinstance(local.provider, VLLMProvider)
        assert str(local.provider.client.base_url) == 'http://127.0.0.1:8999/custom/v1/'
        assert local.provider.client.api_key == 'saved-vllm-token'
