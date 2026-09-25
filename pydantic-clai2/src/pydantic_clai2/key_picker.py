"""Choose a plugin's secret from `/keys` inside its settings menu; only a `KeyReference` comes back.

A plugin's settings menu runs on a menu worker thread, but `prompt_api_key` is a coroutine that opens its
own widgets through `run_worker`. `pick_key` hands it back to the event loop and waits, watching the menu's
stop signal so cancelling the menu cancels the picker too.
"""

import asyncio
import concurrent.futures

from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder  # pyright: ignore[reportMissingTypeStubs]

from ._rendering import markdown_style
from .api_keys import KeyReference, load_keys, prompt_api_key, save_key
from .field_menu import Runners
from .menu_worker import menu_key, run_worker, worker_stopping


class MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked termflow input, matching the settings menu around it."""

    def __init__(self, runners: Runners) -> None:
        """Show widgets through `runners`, which tests replace with scripted ones."""
        self._runners = runners

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        """Ask for the secret; Esc raises `EOFError`, which `prompt_api_key` reads as cancellation."""
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Value: ')
            .placeholder('Paste the secret; it is saved in /keys, not in plugin settings')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: self._runners.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError
        return result.value


async def choose_key(*, name: str, label: str, runners: Runners) -> KeyReference | None:
    """Pick a saved key, or save a masked new value under `name`; `None` means nothing changed."""
    choice = await prompt_api_key(prompt=MaskedPrompt(runners), label=label)
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        return None
    if name in await asyncio.to_thread(load_keys) and not await run_worker(lambda: _confirm_replace(name, runners)):
        return None
    await asyncio.to_thread(save_key, name=name, value=value)
    return KeyReference(name=name)


def pick_key(loop: asyncio.AbstractEventLoop, *, name: str, label: str, runners: Runners) -> KeyReference | None:
    """Run `choose_key` on `loop` from a menu worker thread; a stopping worker cancels it and returns `None`."""
    picking = asyncio.run_coroutine_threadsafe(choose_key(name=name, label=label, runners=runners), loop)
    while True:
        try:
            return picking.result(timeout=0.05)
        except concurrent.futures.TimeoutError:
            if worker_stopping():  # pragma: no cover -- timing-dependent; the picker's own keys see the stop.
                picking.cancel()
                return None


def _confirm_replace(name: str, runners: Runners) -> bool:
    menu = (
        MenuBuilder(f'{name} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved key', value=False),
                MenuItem(f'Replace {name} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = runners.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True
