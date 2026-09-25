"""The `/plugins` menu, driven headless."""

import asyncio
import io
from collections.abc import Coroutine, Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.commands import Commands
from pydantic_clai2.plugin_loader import PluginError, PluginLoader
from pydantic_clai2.plugin_menu import PluginMenu, open_plugins_menu
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.settings_store import SettingsStore

PLUGIN = 'from pydantic_clai2.plugins import PluginHost\ndef activate(host: PluginHost) -> None:\n    pass\n'
TUNED = PLUGIN + (
    'async def configure(config):\n'
    "    config.save({'runs': int(config.settings().get('runs', 0)) + 1})\n"
    "    return f'configured {config.name}'\n"
)
SYNC = PLUGIN + "def configure(config):\n    return 'sync'\n"
BAD = PLUGIN + 'def configure(config):\n    return 1\n'
RAISES = PLUGIN + "async def configure(config):\n    raise RuntimeError('menu broke')\n"


class FakeMenu:
    def __init__(self) -> None:
        self.redraws: list[Sequence[MenuItem]] = []

    def replace_items(self, items: Sequence[MenuItem]) -> None:
        self.redraws.append(items)


def make_loader(tmp_path: Path, *names: str, sources: dict[str, str] | None = None) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    for name in names:
        (store.plugins_dir / f'{name}.py').write_text(PLUGIN)
    for name, source in (sources or {}).items():
        (store.plugins_dir / f'{name}.py').write_text(source)
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
    )


def run_now(action: Coroutine[object, object, object]) -> None:
    asyncio.run(action)


def test_rows_details_and_keys(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'alpha', 'beta')
    menu = PluginMenu(loader, apply=run_now)
    labels = [item.label for item in menu.items()]
    assert labels[0].startswith('[ ] alpha') and labels[1].startswith('[ ] beta')
    fake = FakeMenu()
    alpha = menu.items()[0]
    menu.toggle(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('[x] alpha')
    assert 'state   enabled, loaded' in menu.details(alpha)
    assert 'adds    0 commands' in menu.details(alpha)
    menu.reload(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('[x] alpha')
    menu.toggle(fake, alpha)
    assert fake.redraws[-1][0].label.startswith('[ ] alpha')
    assert 'state   disabled' in menu.details(alpha)
    menu.remove(fake, alpha)
    assert 'state   disabled' in menu.details(alpha)
    assert menu.details(MenuItem('stray', value=None)) == ''
    assert menu.details(MenuItem('typed', value=0)) == ''
    assert len(fake.redraws) == 4
    assert menu.close(fake, alpha).item is alpha
    assert menu.build() is not None


def test_errors_become_a_notice(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'broken', 'fine')
    (loader.plugins_dir / 'broken.py').write_text('raise RuntimeError("nope")')
    menu = PluginMenu(loader, apply=run_now)
    fake = FakeMenu()
    broken, fine = menu.items()
    menu.toggle(fake, broken)
    assert menu.notice is not None and 'RuntimeError: nope' in menu.notice
    assert 'notice  ' in menu.details(broken) and 'error   RuntimeError: nope' in menu.details(broken)
    assert menu.details(MenuItem('stray', value=None)) == menu.notice
    menu.toggle(fake, fine)
    assert menu.notice is None
    for handler in (menu.toggle, menu.reload, menu.remove):
        handler(fake, MenuItem('stray', value=None))
    assert len(fake.redraws) == 5


def test_empty_state(tmp_path: Path) -> None:
    loader = make_loader(tmp_path)
    items = PluginMenu(loader, apply=run_now).items()
    assert len(items) == 1 and items[0].disabled and str(loader.plugins_dir) in items[0].label


async def test_open_menu_applies_actions_from_the_menu_thread(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'gamma')

    def run(menu: PluginMenu[None]) -> None:
        menu.toggle(FakeMenu(), menu.items()[0])

    assert await open_plugins_menu(loader, run=run) == ''
    assert loader.entries()[0].host is not None
    listing = await loader.command(['list'])
    assert 'gamma:' in listing and '(enabled, loaded)' in listing


@pytest.mark.parametrize('names', [(), ('gamma',)])
async def test_open_menu_closes_quietly_without_changes(tmp_path: Path, names: tuple[str, ...]) -> None:
    loader = make_loader(tmp_path, *names)

    def run(menu: PluginMenu[None]) -> None:
        assert menu.items()

    assert await open_plugins_menu(loader, run=run) == ''
    assert all(entry.host is None for entry in loader.entries())


def settings(loader: PluginLoader[None], name: str) -> object:
    return next(entry for entry in loader.entries() if entry.name == name).declaration.settings


async def test_configure_hook_through_commands(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'plain', sources={'tuned': TUNED, 'sync': SYNC, 'bad': BAD, 'raises': RAISES})
    with pytest.raises(PluginError, match="Plugin 'raises': RuntimeError: menu broke"):
        await loader.command(['enable', 'raises'])
    assert next(entry for entry in loader.entries() if entry.name == 'raises').host is None
    with pytest.raises(ValueError, match='Plugin plain has no settings menu'):
        await loader.command(['configure', 'plain'])
    assert await loader.command(['enable', 'plain']) == 'Enabled plain.'
    assert await loader.command(['enable', 'tuned']) == 'configured tuned\nEnabled tuned.'
    assert settings(loader, 'tuned') == {'runs': 1}
    [first] = [entry.host for entry in loader.entries() if entry.name == 'tuned']
    assert first is not None

    assert await loader.command(['configure', 'tuned']) == 'configured tuned'
    [tuned] = [entry for entry in loader.entries() if entry.name == 'tuned']
    assert tuned.declaration.settings == {'runs': 2}
    assert tuned.host is not None and tuned.host is not first, 'reactivated with the new settings'

    assert await loader.command(['configure', 'sync']) == 'sync'
    assert next(entry for entry in loader.entries() if entry.name == 'sync').host is None, 'still disabled'
    with pytest.raises(PluginError, match='configure must return the message to show'):
        await loader.command(['configure', 'bad'])


async def test_menu_opens_settings_before_enabling_and_on_c(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, 'plain', sources={'tuned': TUNED, 'bad': BAD})
    shown: list[str | None] = []

    def by_name(menu: PluginMenu[None], name: str) -> MenuItem:
        return next(item for item in menu.items() if item.value == name)

    def run(menu: PluginMenu[None]) -> None:
        fake = FakeMenu()
        step = len(shown)
        shown.append(menu.notice)
        if step == 0:
            assert menu.toggle(fake, by_name(menu, 'tuned')) is not None, 'closes to open the settings'
        elif step == 1:
            assert menu.configure(fake, by_name(menu, 'plain')) is None
            assert menu.notice == 'Plugin plain has no settings menu.' and fake.redraws
            assert menu.configure(fake, MenuItem('stray', value=None)) is None
            assert menu.configure(fake, by_name(menu, 'tuned')) is not None
        elif step == 2:
            assert menu.configure(fake, by_name(menu, 'bad')) is not None

    message = await open_plugins_menu(loader, run=run)
    assert len(shown) == 4, 'the list opens again after each settings menu'
    assert message.splitlines() == [
        'configured tuned',
        'configured tuned',
        "Plugin 'bad': TypeError: configure must return the message to show",
    ]
    assert settings(loader, 'tuned') == {'runs': 2}
    assert next(entry for entry in loader.entries() if entry.name == 'tuned').host is not None


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
