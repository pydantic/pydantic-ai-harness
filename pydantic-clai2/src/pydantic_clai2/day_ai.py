"""The built-in `day_ai` plugin: harness `DayAI`, with a token from `/keys` or a browser sign-in.

Settings hold at most the name of a `/keys` entry, never a token. Without one, the plugin uses the conventional
`DAY_AI_ACCESS_TOKEN` entry when it exists, and otherwise signs in the way `/mcp` does for an OAuth server:
FastMCP's browser flow, with tokens kept in the keyring under `mcp-day_ai`. `/mcp` server names cannot contain
underscores, so that credential never belongs to one of your servers. The environment is not read.
"""

import asyncio
from typing import Literal

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.day_ai import DayAI

from . import theme
from .api_keys import KeyReference, SavedKey, load_keys, prompt_api_key, save_key
from .commands import Command
from .mcp import TokenStore, browser_sign_in, http_client
from .plugins import DepsT, PluginHost, SessionStart

DAY_AI_MCP_URL = 'https://day.ai/api/mcp'
"""The hosted MCP endpoint harness `DayAI` connects to when it is given a token."""

KEY_NAME = 'DAY_AI_ACCESS_TOKEN'
"""The conventional `/keys` label, the variable harness `DayAI` documents. Only a label; not read from the environment."""

TOKEN_ACCOUNT = 'day_ai'
"""The `/mcp` token store name, so the keyring credential is `mcp-day_ai`."""

SETUP = f'Run /day_ai connect to choose or enter a token, or add {KEY_NAME} in /keys.'
_USAGE = 'Usage: /day_ai [connect]'
_RELOAD = 'Run /plugins reload day_ai to connect with it.'
_LABEL = f'Day AI access token, saved in /keys as {KEY_NAME} (leave empty to sign in through the browser): '


class DayAISettings(BaseModel):
    """The JSON a `day_ai` declaration may carry. It names a token in `/keys`; it can never hold one."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    token: KeyReference | None = Field(
        default=None,
        description=f'The saved API key in /keys to connect with. Unset uses {KEY_NAME} when saved, else the browser.',
    )


def activate(host: PluginHost[DepsT]) -> None:
    """Add `DayAI` with a `/keys` token resolved on every run, or sign in through the browser before loading."""
    settings = host.settings(DayAISettings)
    saved = load_keys()
    token = settings.token or (KeyReference(name=KEY_NAME) if KEY_NAME in saved else None)
    _register_command(host, settings, token)
    if token is not None:
        host.add(DayAI[DepsT](auth=SavedKey(name=token.name, setup=SETUP)))
        if token.name not in saved:
            # Loading anyway keeps `/day_ai connect` available; each run fails closed until the key is saved.
            host.console.print(
                f'Day AI has no token: {token.name} is not in /keys. {SETUP}',
                style=theme.color(theme.WARNING),
                markup=False,
            )
        return
    host.add(DayAI[DepsT](client=_transport()))

    @host.on('session_start')
    async def sign_in(_: SessionStart) -> None:  # pyright: ignore[reportUnusedFunction]
        if await to_thread.run_sync(TokenStore(TOKEN_ACCOUNT).signed_in):
            return
        if not host.console.is_terminal:
            raise UserError(f'Save {KEY_NAME} in /keys, or sign in to Day AI from an interactive CLAI session first.')
        host.console.print('Opening your browser to sign in to Day AI.', style=theme.color(theme.MUTED))
        # A throwaway connection runs the sign-in now, so a failure fails the load rather than the next prompt.
        async with Client(_transport()):
            pass


def _register_command(host: PluginHost[DepsT], settings: DayAISettings, token: KeyReference | None) -> None:
    async def day_ai(args: list[str]) -> str:
        if args == ['connect']:
            choice = await _choose()
            if choice is None:
                return 'Day AI connection unchanged.'
            if choice == 'browser':
                host.save_settings(settings.model_copy(update={'token': None}))
                return f'Day AI will sign in through the browser. {_RELOAD}'
            host.save_settings(settings.model_copy(update={'token': choice}))
            return f'Day AI will use the saved key {choice.name}; manage it in /keys. {_RELOAD}'
        if args:
            raise ValueError(_USAGE)
        if token is None:
            return 'Day AI signs in through the browser. /day_ai connect chooses a token from /keys instead.'
        saved = token.name in await asyncio.to_thread(load_keys)
        return f'Day AI token: {token.name} in /keys ({"saved" if saved else "missing"}).'

    host.commands.register(
        Command(
            name='day_ai',
            description='Show or choose how Day AI signs in: a token in /keys or the browser',
            handler=day_ai,
            complete=lambda _: ('connect',),
        )
    )


async def _choose() -> KeyReference | Literal['browser'] | None:
    """Pick a saved key, save a masked new value as `DAY_AI_ACCESS_TOKEN`, or choose the browser; `None` cancels."""
    prompt: PromptSession[str] = PromptSession()
    choice = await prompt_api_key(prompt=prompt, label=_LABEL, optional=True)
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        return 'browser'
    if KEY_NAME in await asyncio.to_thread(load_keys):
        try:
            answer = await prompt.prompt_async(f'Replace {KEY_NAME} for every plugin that uses it? [y/N]: ')
        except (EOFError, KeyboardInterrupt):
            return None
        if answer.strip().lower() != 'y':
            return None
    await asyncio.to_thread(save_key, name=KEY_NAME, value=value)
    return KeyReference(name=KEY_NAME)


def _transport() -> StreamableHttpTransport:
    return StreamableHttpTransport(
        DAY_AI_MCP_URL, auth=browser_sign_in(TOKEN_ACCOUNT), httpx_client_factory=http_client
    )
