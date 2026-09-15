"""Ripgrep-backed content search.

Ripgrep is optional. `_toolset.py` falls back to scanning in-process when the binary is missing, so the capability
never requires an external tool; when the binary is present it does the walking, because it does not pull a file into
Python to decide whether the file matches.

The walk is ripgrep's; the policy is not. Every path it reports is re-checked against the toolset's containment,
permission, and binary rules before a match reaches the model, so a faster walk never widens what the agent reads.
"""

from __future__ import annotations

import base64
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio
from pydantic import BaseModel, ValidationError

TIMEOUT_SECONDS = 30.0
"""A search that cannot finish in this long is a tool error, not a run that hangs."""

MAX_FILE_BYTES = 5 * 1024 * 1024
"""Skip files larger than this. `read_file` can still fetch a larger file the model names explicitly."""

_MAX_FILESIZE_ARGUMENT = '5M'
"""`MAX_FILE_BYTES` expressed for `rg --max-filesize`."""


class RipgrepError(Exception):
    """A ripgrep invocation that could not produce usable output."""


@dataclass(frozen=True)
class Match:
    """One hit as ripgrep reported it, before the toolset applies its policy."""

    path: str
    line_number: int
    line: str


class _Path(BaseModel):
    """A reported path: text when it is valid UTF-8, base64 when it is not."""

    text: str | None = None
    bytes: str | None = None


class _Lines(BaseModel):
    """A matched line: text when it is valid UTF-8, base64 when it is not."""

    text: str | None = None
    bytes: str | None = None


class _MatchData(BaseModel):
    """The `data` payload of a `match` event."""

    path: _Path = _Path()
    lines: _Lines = _Lines()
    line_number: int | None = None


class _Event(BaseModel):
    """One record from ripgrep's NDJSON stream. Unknown keys are ignored."""

    type: str = ''
    data: _MatchData | None = None


def find_ripgrep() -> Path | None:
    """Return the `rg` binary, or `None` to mean 'scan in-process'.

    `PATH` is checked first, then the directory holding the running interpreter, where `uv` puts the binaries of an
    environment it manages. The probe runs on every call rather than caching: caching would freeze the answer for the
    life of the process, so a ripgrep installed later in the session would go unnoticed.
    """
    on_path = shutil.which('rg')
    if on_path is not None:
        return Path(on_path)

    interpreter_dir = Path(sys.executable).parent
    for name in ('rg', 'rg.exe'):
        candidate = interpreter_dir / name
        if candidate.exists():
            return candidate
    return None


async def search(
    executable: Path,
    *,
    root: Path,
    pattern: str,
    include_glob: str | None,
    max_matches: int,
) -> list[Match]:
    """Content-search `root` and return hits in path order.

    `--json` because a path can contain a colon, which makes `file:line:text` ambiguous to parse from outside.
    `--sort path` because ripgrep otherwise reports files as they finish, and tool output has to be reproducible.
    `--max-count` is a per-file cap, so the caller still owns the total budget; asking for one hit past that budget is
    what lets the caller report truncation instead of guessing at it. `--no-ignore` because the toolset's documented
    policy is the only thing that decides what is skipped: ripgrep's hidden-file default already covers dotfiles, and
    `.gitignore` does not get a vote.

    The root is passed as `.` with `cwd=root`, never as an absolute path. Ripgrep matches ignore patterns against the
    paths it walks, so an absolute root lets a pattern naming one of the root's own ancestors veto the whole search: a
    root under a directory named by such a pattern returns no matches and no error at all.

    `RipgrepError` also covers a pattern ripgrep cannot compile. Its dialect has no backreferences or lookaround, which
    Python's `re` accepts, so the caller can retry those in-process rather than reporting them as invalid.
    """
    arguments = [
        str(executable),
        '--json',
        '--sort',
        'path',
        '--max-count',
        str(max_matches + 1),
        '--max-filesize',
        _MAX_FILESIZE_ARGUMENT,
        '--no-ignore',
        '--no-messages',
        '-e',
        pattern,
    ]
    if include_glob is not None:
        arguments.extend(['-g', include_glob])
    arguments.append('.')

    returncode, stdout, stderr = await _run(arguments, cwd=root)
    if returncode not in (0, 1):
        raise RipgrepError(stderr.strip() or f'ripgrep exited with code {returncode}.')
    return _parse(stdout)


async def _run(arguments: Sequence[str], *, cwd: Path) -> tuple[int, str, str]:
    """Run one bounded ripgrep command, returning `(returncode, stdout, stderr)`.

    `anyio.run_process` owns the subprocess: it waits on it, closes both pipes, and kills it as its context manager
    unwinds, so the enclosing `fail_after` leaves nothing running. Decoding replaces invalid UTF-8 rather than raising,
    so one oddly encoded file cannot fail a whole search.
    """
    try:
        with anyio.fail_after(TIMEOUT_SECONDS):
            completed = await anyio.run_process(list(arguments), cwd=cwd, check=False)
    except TimeoutError as error:
        raise RipgrepError(f'Search timed out after {TIMEOUT_SECONDS:.0f}s.') from error

    return (
        completed.returncode,
        completed.stdout.decode('utf-8', errors='replace'),
        completed.stderr.decode('utf-8', errors='replace'),
    )


def _parse(stdout: str) -> list[Match]:
    """Turn ripgrep's NDJSON stream into matches.

    Only `match` events carry a line; `begin`, `end`, and `summary` are bookkeeping, and a record that does not
    validate is skipped. Text ripgrep could not decode as UTF-8 arrives base64-encoded, and is decoded leniently here
    rather than dropped, matching what the in-process scan reports for the same file.
    """
    matches: list[Match] = []
    for raw_line in stdout.splitlines():
        if not raw_line:
            continue
        try:
            event = _Event.model_validate_json(raw_line)
        except ValidationError:  # pragma: no cover
            continue
        data = event.data
        if event.type != 'match' or data is None or data.line_number is None:
            continue
        path = _path_text(data.path)
        if path is None:
            continue
        line = _line_text(data.lines)
        if line is None:
            continue
        matches.append(Match(path=path, line_number=data.line_number, line=line))

    return matches


def _path_text(path: _Path) -> str | None:
    """Read a reported path normalized, decoding the base64 form used for a name that is not valid UTF-8.

    Normalized because ripgrep reports each path relative to the search root with a `.` separator prefix, and the
    caller orders these hits beside its own relative paths: an unnormalized `./a.md` sorts before every name starting
    after `.` in the byte order, which would put a merged hit in the wrong place.
    """
    if path.text is not None:
        return os.path.normpath(path.text)
    if path.bytes is None:
        return None
    return os.path.normpath(os.fsdecode(base64.b64decode(path.bytes)))


def _line_text(lines: _Lines) -> str | None:
    """Read a matched line, without its terminator, decoding the base64 form used for invalid UTF-8."""
    if lines.text is not None:
        return lines.text.rstrip('\r\n')
    if lines.bytes is None:
        return None
    return os.fsdecode(base64.b64decode(lines.bytes)).rstrip('\r\n')
