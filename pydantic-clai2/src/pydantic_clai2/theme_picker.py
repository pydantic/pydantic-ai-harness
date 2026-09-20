"""Select a bundled Termflow palette through the existing settings path."""

from termflow.ansi.color import bg_color, fg_color  # pyright: ignore[reportMissingTypeStubs]
from termflow.themes import PALETTES  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]

from ._rendering import markdown_style
from .command_context import CommandContext
from .field_menu import TERMINAL, Runners
from .menu_worker import menu_key, run_worker


def build_theme_picker(context: CommandContext) -> Menu:
    """List only Termflow's palettes, without changing colours while browsing."""
    names = list(PALETTES)

    def preview(item: MenuItem) -> str:
        palette = PALETTES[str(item.value)]
        return (
            f'{palette.name}\n\n{bg_color(palette.bg)}{fg_color(palette.fg)} Sample text \x1b[0m\n\n'
            + ''.join(f'{bg_color(color)}  ' for color in palette.ansi)
            + '\x1b[0m\n\n'
            'Enter saves and applies. Esc keeps the current theme.'
        )

    return (
        MenuBuilder('Select theme')
        .style(markdown_style())
        .items(
            [MenuItem(f'{name}{" (current)" if name == context.settings.theme else ""}', value=name) for name in names]
        )
        .searchable()
        .initial_index(names.index(context.settings.theme))
        .preview(preview)
        .footer_hint('type to filter - Enter apply - Esc close')
        .key_source(menu_key)
        .build()
    )


async def theme_command(context: CommandContext, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Persist a palette by name or picker; cancellation leaves the preference unchanged."""
    if len(args) > 1:
        raise ValueError('Usage: /theme [NAME]. Use /theme to browse Termflow palettes.')
    if args:
        name = args[0]
    else:
        result = await run_worker(lambda: runners.run_list(build_theme_picker(context)))
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return 'No changes.'
        name = result.item.value
    return context.set_setting(['display.theme', name])
