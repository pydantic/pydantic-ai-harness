"""The optional `configure` hook: a plugin's own settings menu, opened by `/plugins configure`, add, and enable.

A plugin module that defines `async def configure(config: PluginConfig) -> str` gets a settings menu. It runs
on the event loop between prompts and shows its widgets through `run_worker`, usually with
`field_menu.run_flow_async`. Each edit calls `config.save`, which writes the plugin's settings immediately; the
loader reloads an enabled plugin afterwards so the new settings apply. Settings are stored in plaintext, so a
hook keeps secrets in `/keys` and saves only a key's name.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import JsonValue

from .field_menu import TERMINAL, Runners


@dataclass(frozen=True, kw_only=True)
class PluginConfig:
    """What a `configure` hook reads and writes."""

    name: str
    """The plugin id, as `/plugins` shows it."""
    settings: Callable[[], dict[str, JsonValue]]
    """The saved settings right now, including edits made earlier in this menu."""
    save: Callable[[dict[str, JsonValue]], None]
    """Replace the saved settings. Validate first: they are loaded as-is on the next activation."""
    runners: Runners = TERMINAL
    """How widgets are shown; tests pass scripted runners."""
