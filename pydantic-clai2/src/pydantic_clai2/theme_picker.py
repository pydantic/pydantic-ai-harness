"""Select a terminal appearance through the same settings path as `/set`."""

from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from ._rendering import markdown_style
from .command_context import CommandContext
from .field_menu import TERMINAL, Runners
from .menu_worker import menu_key, run_worker


def build_theme_picker(context: CommandContext) -> Menu:
    """Build a searchable picker with a role sample, without changing the active theme."""
    names = list(theme.THEMES)

    def preview(item: MenuItem) -> str:
        selected = next(value for name, value in theme.THEMES.items() if name == item.value)
        samples = '\n'.join(
            f'{theme.sgr(color)}{label}\x1b[0m'
            for label, color in (
                ('Heading / tool name', selected.primary),
                ('Status / thinking', selected.thinking),
                ('Hint / tool preview', selected.muted),
                ('Warning', selected.warning),
                ('Error', selected.error),
            )
        )
        return f'{selected.description}\n\n{samples}\n\nEnter saves and applies. Esc keeps the current theme.'

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
    """Persist a named theme or pick one; cancelling does not alter the preference."""
    if len(args) > 1:
        raise ValueError('Usage: /theme [NAME]. Choose pydantic, light, or system.')
    if args:
        name = args[0]
    else:
        result = await run_worker(lambda: runners.run_list(build_theme_picker(context)))
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return 'No changes.'
        name = result.item.value
    if name not in theme.THEMES:
        raise ValueError(f'Unknown theme: {name}. Choose pydantic, light, or system.')
    return context.set_setting(['display.theme', name])
