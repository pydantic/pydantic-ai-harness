"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .skills import Skill, Skills, load_skills_from_directory

__all__ = ['CodeMode', 'Skill', 'Skills', 'load_skills_from_directory']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in ('Skill', 'Skills', 'load_skills_from_directory'):
        from .skills import Skill, Skills, load_skills_from_directory

        _exports = {'Skill': Skill, 'Skills': Skills, 'load_skills_from_directory': load_skills_from_directory}
        return _exports[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
