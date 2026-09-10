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
    indent = '    ' if path in _PAGES else ''
    pip_label = '=== "pip"' if path in _PAGES else 'pip:'
    uv_label = '=== "uv"' if path in _PAGES else 'uv:'
    pair = re.compile(
        rf'^{re.escape(pip_label)}\n\n{indent}```bash\n'
        rf'{indent}pip install (?P<args>[^\n]+)\n'
        rf'(?P<pip_followup>(?:(?!{indent}```)[^\n]*\n)*){indent}```\n\n'
        rf'{re.escape(uv_label)}\n\n{indent}```bash\n'
        rf'{indent}uv add (?P=args)\n'
        rf'(?P<uv_followup>(?:(?!{indent}```)[^\n]*\n)*){indent}```',
        re.MULTILINE,
    )
    for match in pair.finditer(text):
        assert not _INSTALL.search(match['pip_followup'] + match['uv_followup']), (
            f'{path}: additional installation commands need their own paired blocks'
        )
    assert not _INSTALL.search(pair.sub('', text)), (
        f'{path}: each installation needs adjacent labeled pip / uv blocks with identical package arguments'
    )


@pytest.mark.parametrize('path', [*_PAGES, *_READMES], ids=lambda path: str(path.relative_to(_ROOT)))
def test_installation_commands_are_not_inline(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    assert not re.search(r'`(?:pip install|uv add) [^`]+`', text), (
        f'{path}: move inline installation commands into paired pip / uv blocks'
    )
