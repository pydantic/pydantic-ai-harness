"""One in-workspace POSIX search for command-capable workspaces without ripgrep."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import TypeVar

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import Workspace

from ._ripgrep import Record

_T = TypeVar('_T')
_MAX_OUTPUT_BYTES = 8 << 20
_MAX_RECORD_BYTES = 1 << 20


async def run_posix_search(
    workspace: Workspace,
    *,
    cwd: str,
    target: str = '.',
    pattern: str | None = None,
    literal: bool = False,
    ignore_case: bool = False,
    context: int = 0,
    limit: int,
    accept: Callable[[Record], _T | None],
) -> tuple[list[_T], bool]:
    """Enumerate sorted files and optionally grep them without transferring their contents to the host."""
    # Git handles nested .gitignore files; find is used only outside a repository.
    # Keep the entire enumeration and scan in one sandbox command, even for large trees.
    enumeration = (
        'if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then '
        'git ls-files -co --exclude-standard -z -- '
        + shlex.quote(target)
        + '; else find '
        + shlex.quote(target)
        + " ! -path . -name '.*' -prune -o -type f -print0; fi | LC_ALL=C sort -z"
    )
    if pattern is None:
        processing = 'xargs -0 sh -c \'for file do printf "%s\\0" "$file"; done\' sh'
    else:
        flags = '-nIhE' if not literal else '-nIhF'
        if ignore_case:
            flags += 'i'
        if context:
            flags += f' -C {context}'
        quoted_pattern = shlex.quote(pattern)
        processing = (
            "xargs -0 sh -c 'for file do "
            f'grep {flags} -e "$1" -- "$file" | '
            'while IFS= read -r line; do printf "%s\\0%s\\n" "$file" "$line"; done; '
            f"done' sh {quoted_pattern}"
        )
    script = f'{{ {enumeration} | {processing}; }} | head -c {_MAX_OUTPUT_BYTES}'
    result = await workspace.run(script, shell=True, cwd=cwd, timeout=120)
    if result.exit_code != 0:
        raise ModelRetry(f'POSIX search failed: {result.stderr.strip() or result.exit_code}')
    output = result.stdout
    cut = len(output.encode('utf-8', errors='surrogateescape')) >= _MAX_OUTPUT_BYTES
    results: list[_T] = []
    start = 0
    while (end := output.find('\0', start)) >= 0:
        path = output[start:end]
        if pattern is None:
            text = ''
            start = end + 1
        else:
            line_end = output.find('\n', end + 1)
            if line_end < 0:
                cut = True
                break
            text = output[end + 1 : line_end]
            start = line_end + 1
        if start - end > _MAX_RECORD_BYTES:
            cut = True
            break
        kept = accept(Record(path=path, text=text))
        if kept is not None:
            if len(results) >= limit:
                cut = True
                break
            results.append(kept)
    return results, cut
