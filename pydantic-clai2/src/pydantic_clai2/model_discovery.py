"""Live model names using core's provider authentication and credential lifecycle."""

from typing import Annotated

import anyio
import httpx2
from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic
from openai import OpenAIError
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openai_codex import CredentialsPersistenceError, OpenAICodexProvider

from .auth import CodexCredentials


class CodexModel(BaseModel):
    """The picker fields in Codex's non-OpenAI-compatible model catalog."""

    slug: str = Field(min_length=1)
    visibility: str


class CodexModels(BaseModel):
    """Codex wraps model entries in `models`, not `data`."""

    models: list[CodexModel]


async def discover_models(*, provider: str) -> tuple[list[str] | None, str | None]:
    """Return qualified names and a safe fallback notice; unsupported providers stay offline."""
    if provider not in ('openai', 'openai-chat', 'openai-responses', 'openai-codex', 'anthropic', 'deepseek'):
        return None, None
    try:
        with anyio.fail_after(10):
            async with httpx2.AsyncClient(timeout=10, follow_redirects=False) as client:
                if provider == 'openai-codex':
                    codex = OpenAICodexProvider(credential_source=CodexCredentials(), http_client=client)
                    response = await client.get(
                        f'{codex.base_url}/models',
                        # Codex versions its catalog by client compatibility, not CLAI's package version.
                        params={'client_version': '0.156.1'},
                    )
                    response.raise_for_status()
                    names = [
                        model.slug
                        for model in CodexModels.model_validate_json(response.content).models
                        if model.visibility == 'list'
                    ]
                elif provider == 'anthropic':
                    anthropic = AnthropicProvider(http_client=client)
                    assert isinstance(anthropic.client, AsyncAnthropic)
                    models_api = anthropic.client.with_options(max_retries=0).models
                    assert models_api is not None
                    names = [model.id async for model in models_api.list()]
                else:
                    openai = (DeepSeekProvider if provider == 'deepseek' else OpenAIProvider)(http_client=client)
                    names = [model.id async for model in openai.client.with_options(max_retries=0).models.list()]
                names = TypeAdapter(list[Annotated[str, Field(min_length=1)]]).validate_python(names)
    except CredentialsPersistenceError:
        raise
    except (httpx2.HTTPError, OpenAIError, AnthropicAPIError, ModelAPIError, UserError, ValueError, TimeoutError):
        hint = (
            'Run /login openai-codex to reconnect.'
            if provider == 'openai-codex'
            else 'Check credentials and connectivity.'
        )
        return None, f'Live model discovery unavailable.\nUsing the built-in catalog.\n{hint}'
    return sorted({f'{provider}:{name}' for name in names}), None
