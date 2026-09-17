"""Connect to a trusted vLLM server using core's OpenAI-compatible provider."""

import asyncio
import json

import httpx
import keyring
from prompt_toolkit import PromptSession
from pydantic import BaseModel, Field, HttpUrl, SecretStr, TypeAdapter
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]

from .command_context import CommandContext
from .menu_worker import menu_key, run_worker

_SERVICE = 'pydantic-clai2.vllm'


class Connection(BaseModel):
    """Endpoint and optional credential, stored together in keyring."""

    url: str
    token: SecretStr = Field(default_factory=lambda: SecretStr(''))


class ServedModel(BaseModel):
    """One OpenAI-compatible discovery result."""

    id: str = Field(min_length=1)


class ModelList(BaseModel):
    """Validated discovery response."""

    data: list[ServedModel]


def api_url(value: str) -> str:
    """Accept a server root or API root, but not credentials or query parameters."""
    url = TypeAdapter(HttpUrl).validate_python(value.strip())
    if url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP(S) server URL without credentials, query, or fragment.')
    root = str(url).rstrip('/')
    return root if root.endswith('/v1') else root + '/v1'


async def discover(connection: Connection, *, transport: httpx.AsyncBaseTransport | None = None) -> list[str]:
    """Query only the requested endpoint; do not forward credentials across redirects."""
    token = connection.token.get_secret_value()
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    async with httpx.AsyncClient(transport=transport, timeout=20, follow_redirects=False, trust_env=False) as client:
        try:
            response = await client.get(f'{api_url(connection.url)}/models', headers=headers)
            response.raise_for_status()
        except httpx.HTTPError:
            raise UserError('Model discovery failed. Check the server URL, token, and connectivity.') from None
    names = sorted({model.id for model in ModelList.model_validate_json(response.content).data})
    if not names:
        raise UserError('The server returned no models.')
    return names


def save_connection(connection: Connection) -> None:
    """Keep credentials out of command history and SQLite."""
    value = connection.model_dump()
    value['token'] = connection.token.get_secret_value()
    keyring.set_password(_SERVICE, 'connection', json.dumps(value))


def model(name: str) -> OpenAIChatModel:
    """Resolve a saved vLLM selection through core, without global API-key fallbacks."""
    raw = keyring.get_password(_SERVICE, 'connection')
    if raw is None:
        raise UserError('Connect first with /vllm.')
    connection = Connection.model_validate_json(raw)
    provider = OpenAIProvider(
        base_url=api_url(connection.url), api_key=connection.token.get_secret_value() or 'not-required'
    )
    return OpenAIChatModel(name.removeprefix('vllm:'), provider=provider)


def choose(names: list[str]) -> str | None:  # pragma: no cover -- terminal ownership.
    """Pick one discovered model in Termflow."""
    result = (
        MenuBuilder('vLLM models')
        .items([MenuItem(name, value=name) for name in names])
        .searchable()
        .key_source(menu_key)
        .build()
        .run()
    )
    return result.item.value if not result.cancelled and result.item and isinstance(result.item.value, str) else None


async def connect(context: CommandContext, args: list[str]) -> str:
    """Prompt privately, discover models, then persist only after selection."""
    if args:
        raise ValueError('Usage: /vllm (URL and optional token are prompted separately)')
    prompt: PromptSession[str] = PromptSession()
    try:
        url = api_url(await prompt.prompt_async('vLLM server URL: '))
        token = await prompt.prompt_async('Token (optional, Enter for none): ', is_password=True)
    except (EOFError, KeyboardInterrupt):
        return 'Connection cancelled.'
    connection = Connection(url=url, token=SecretStr(token))
    names = await discover(connection)
    selected = await run_worker(lambda: choose(names))
    if selected is None:
        return 'Connection cancelled.'
    await asyncio.to_thread(save_connection, connection)
    return context.set_setting(['model', f'vllm:{selected}'])
