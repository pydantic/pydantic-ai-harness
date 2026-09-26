"""One in-workspace POSIX search for command-capable workspaces without ripgrep."""

from __future__ import annotations

import re
import shlex
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import Workspace

from ._ripgrep import Record

_T = TypeVar('_T')
_MAX_OUTPUT_BYTES = 8 << 20
_MAX_RECORD_BYTES = 1 << 20
_STATUS = '__harness_posix_status='


def validate_posix_pattern(pattern: str) -> None:
    """Reject rg syntax that POSIX ERE would interpret as different text."""
    # POSIX ERE lacks rg's \d/\w/\s classes, lazy quantifiers, named groups,
    # inline flags and Unicode properties; lookaround/backreferences are unsupported by rg too.
    if re.search(r'\\[dDsSwWpP]|\(\?|[+*?]\?|\{[0-9,]+\}\?', pattern):
        raise ValueError(
            'This regex requires ripgrep; POSIX grep does not support that syntax. Use literal=True or install rg.'
        )


async def run_posix_search(
    workspace: Workspace,
    *,
    cwd: str,
    target: str = '.',
    pattern: str | None = None,
    literal: bool = False,
    ignore_case: bool = False,
    context: int = 0,
    include_hidden: bool = False,
    limit: int,
    accept: Callable[[Record], Awaitable[_T | None]],
    prepare: Callable[[list[Record]], Awaitable[None]] | None = None,
) -> tuple[list[_T], bool]:
    """Enumerate sorted files and optionally grep them without transferring their contents to the host."""
    if pattern is not None and not literal:
        validate_posix_pattern(pattern)
    # Git applies nested .gitignore files, but --exclude-from=.ignore only reads the
    # search-root .ignore; nested .ignore rules need rg. Without git, find ignores neither.
    enumeration = (
        'if [ -f .ignore ]; then extra=--exclude-from=.ignore; else extra=; fi; '
        'if command -v git >/dev/null 2>&1; then '
        'if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then '
        'git ls-files -co --exclude-standard $extra -z -- '
        + shlex.quote(target)
        + '; else tmp=$(mktemp -d) || exit 2; '
        'git init --bare -q "$tmp" || exit 2; '
        'GIT_DIR="$tmp" GIT_WORK_TREE="$PWD" git ls-files -o --exclude-standard '
        '-z $extra -- ' + shlex.quote(target) + '; status=$?; rm -rf -- "$tmp"; [ "$status" -eq 0 ]; fi; '
        'else find '
        + shlex.quote(target)
        + (" ! -path . -name '.*' -prune -o " if not include_hidden else ' ')
        + '-type f -print0; fi'
    )
    if pattern is None:
        processing = 'xargs -0 sh -c \'for file do printf "%s\\0" "$file"; done\' sh'
    else:
        flags = '-nIhF' if literal else '-nIhE'
        if ignore_case:
            flags += 'i'
        if context:
            flags += f' -C {context}'
        # Grep's 1 means no matches, while 2 (including an invalid ERE or a read error)
        # must not be turned into a plausible empty result by xargs or the output pipe.
        processing = (
            "xargs -0 sh -c 'tmp=$(mktemp) || exit 2; pattern=$1; shift; for file do "
            f'grep {flags} -e "$pattern" -- "$file" > "$tmp"; code=$?; '
            'if [ "$code" -gt 1 ]; then rm -f -- "$tmp"; exit "$code"; fi; '
            'while IFS= read -r line; do printf "%s\\0%s\\n" "$file" "$line"; done < "$tmp"; '
            'done; rm -f -- "$tmp"\' sh ' + shlex.quote(pattern)
        )
    # Capture enumeration's status before sorting: a failed git/find must not masquerade
    # as an empty successful search through the pipeline's final xargs status.
    script = (
        '{ list=$(mktemp) || exit 2; '
        f'{{ {enumeration}; }} > "$list"; code=$?; '
        'if [ "$code" -eq 0 ]; then LC_ALL=C sort -z "$list" | '
        f'{processing}; code=$?; fi; rm -f -- "$list"; '
        f'echo "{_STATUS}$code" >&2; }} | head -c {_MAX_OUTPUT_BYTES}'
    )
    result = await workspace.run(script, shell=True, cwd=cwd, timeout=120)
    stderr, _, status = result.stderr.rpartition(_STATUS)
    if result.exit_code != 0 or not status or status.strip() != '0':
        raise ModelRetry(f'POSIX search failed: {stderr.strip() or result.stderr.strip() or result.exit_code}')
    output = result.stdout
    cut = len(output.encode('utf-8', errors='surrogateescape')) >= _MAX_OUTPUT_BYTES
    results: list[_T] = []
    records: list[Record] = []
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
        records.append(Record(path=path, text=text))
    if start < len(output):
        cut = True
    if prepare is not None:
        await prepare(records)
    for record in records:
        kept = await accept(record)
        if kept is not None:
            if len(results) >= limit:
                cut = True
                break
            results.append(kept)
    return results, cut
