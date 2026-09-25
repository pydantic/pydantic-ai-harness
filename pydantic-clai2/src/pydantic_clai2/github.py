"""The built-in `github` plugin: GitHub's hosted MCP tools, through harness `GitHub`.

The token lives in `/keys`; the plugin's settings hold only its name, so other plugins can share it.
"""

import asyncio

from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai_harness.github import GitHub

from . import theme
from .api_keys import KeyReference, SavedKey, load_keys, prompt_api_key, save_key
from .commands import Command
from .plugins import DepsT, PluginHost

KEY_NAME = 'GITHUB_TOKEN'
"""The conventional `/keys` label, shared by every plugin that uses a GitHub token. Not read from the environment."""
SETUP = f'Run /github connect to choose or enter a token, or add {KEY_NAME} in /keys.'
_USAGE = 'Usage: /github [connect]'


class GitHubSettings(BaseModel):
    """The JSON a `github` declaration may carry. It names the token; it can never hold one."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    token: KeyReference = Field(
        default_factory=lambda: KeyReference(name=KEY_NAME), description='The saved API key in /keys to connect with.'
    )
    read_only: bool = Field(
        default=True, description="Offer only GitHub's read tools, so the agent cannot change repositories."
    )


def activate(host: PluginHost[DepsT]) -> None:
    """Add `GitHub` with a token resolved from `/keys` on every run, plus `/github` to choose it."""
    settings = host.settings(GitHubSettings)
    token = SavedKey(name=settings.token.name, setup=SETUP)
    host.add(GitHub[DepsT](auth=token, read_only=settings.read_only))

    async def github(args: list[str]) -> str:
        if args == ['connect']:
            reference = await _choose(token.name)
            if reference is None:
                return 'GitHub token unchanged.'
            token.name = reference.name
            host.save_settings(settings.model_copy(update={'token': reference}))
            return f'GitHub uses the saved key {reference.name} from the next turn. Manage it in /keys.'
        if args:
            raise ValueError(_USAGE)
        saved = token.name in await asyncio.to_thread(load_keys)
        tools = 'read-only' if settings.read_only else 'read and write'
        return f'GitHub token: {token.name} in /keys ({"saved" if saved else "missing"}); {tools} tools.'

    host.commands.register(
        Command(
            name='github',
            description='Show or choose the GitHub token saved in /keys',
            handler=github,
            complete=lambda _: ('connect',),
        )
    )
    if token.name not in load_keys():
        # Loading anyway keeps `/github connect` available; each run fails closed until a token is saved.
        host.console.print(
            f'GitHub has no token: {token.name} is not in /keys. {SETUP}',
            style=theme.color(theme.WARNING),
            markup=False,
        )


async def _choose(name: str) -> KeyReference | None:
    """Pick a saved key, or save a masked new value under `name`; `None` means cancelled."""
    prompt: PromptSession[str] = PromptSession()
    choice = await prompt_api_key(prompt=prompt, label=f'GitHub token (saved in /keys as {name}): ')
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        raise ValueError('A GitHub token is required.')
    if name in await asyncio.to_thread(load_keys):
        try:
            answer = await prompt.prompt_async(f'Replace {name} for every plugin and connection that uses it? [y/N]: ')
        except (EOFError, KeyboardInterrupt):
            return None
        if answer.strip().lower() != 'y':
            return None
    await asyncio.to_thread(save_key, name=name, value=value)
    return KeyReference(name=name)
