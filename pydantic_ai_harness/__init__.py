"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .home_automation import HomeAutomation

__all__ = ['CodeMode', 'HomeAutomation']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name == 'HomeAutomation':
        from .home_automation import HomeAutomation

        return HomeAutomation
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
