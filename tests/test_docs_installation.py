"""Keep user-facing Python installation examples available for pip and uv."""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_PAGES = sorted((_ROOT / 'docs').rglob('*.md'))
_READMES = [_ROOT / 'README.md', *sorted((_ROOT / 'pydantic_ai_harness').rglob('README.md'))]
_INSTALL = re.compile(r'^\s*(pip install|uv add) (.+)$', re.MULTILINE)


@pytest.mark.parametrize('path', [*_PAGES, *_READMES], ids=lambda path: str(path.relative_to(_ROOT)))
def test_installation_commands_are_paired(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    commands = _INSTALL.findall(text)
    pip_args = [args for command, args in commands if command == 'pip install']
    uv_args = [args for command, args in commands if command == 'uv add']
    assert pip_args == uv_args, f'{path}: pip and uv must install the same packages in the same order'
    for manager, command in commands:
        label = 'pip' if manager == 'pip install' else 'uv'
        if path in _PAGES:
            expected = f'=== "{label}"\n\n    ```bash\n    {manager} {command}\n'
        else:
            expected = f'{label}:\n\n```bash\n{manager} {command}\n'
        assert expected in text, f'{path}: installation command needs its {label} label or tab'


@pytest.mark.parametrize('path', [*_PAGES, *_READMES], ids=lambda path: str(path.relative_to(_ROOT)))
def test_installation_commands_are_not_inline(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    assert not re.search(r'`(?:pip install|uv add) [^`]+`', text), (
        f'{path}: move inline installation commands into paired pip / uv blocks'
    )
