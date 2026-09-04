"""Filesystem toolset providing sandboxed file operations."""

from __future__ import annotations

import codecs
import errno
import fnmatch
import functools
import hashlib
import os
import posixpath
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Concatenate, ParamSpec

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.sandboxes import SandboxError, SandboxUnavailableError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness._sandbox import sandbox_path

_P = ParamSpec('_P')

READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'}
)
"""Names of filesystem tools that do not modify the workspace."""

# Errors that mean "the model asked for something the tool couldn't do" -- a
# missing file, a denied path, a stale edit. pyai only feeds `ModelRetry` back
# to the model; any other exception aborts the whole run. `_recoverable`
# converts these so the agent can correct itself and continue.
_RECOVERABLE_ERRORS = (
    PermissionError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    FileExistsError,
    ValueError,
)

# The same idea one level down, for failures the backend raises as a bare
# `OSError` with no dedicated subclass for `_RECOVERABLE_ERRORS` to name.
# Entries are explicit so other errors keep aborting the run; for example,
# retrying cannot fix `ENOSPC` or `EROFS`.
#
# Keyed by `OSError.errno`, which the stdlib types as `int | None`.
_RECOVERABLE_ERRNOS: dict[int | None, str] = {
    errno.ENAMETOOLONG: 'The path name is too long.',
    errno.ELOOP: 'The path resolves through a symlink loop.',
    errno.EILSEQ: 'The path name contains a byte sequence the filesystem cannot represent.',
}
_OUTSIDE_WORKSPACE = '<outside-workspace>'
"""Shown instead of an absolute path that is not inside the workspace root."""

_NOT_A_PATH = '<not-a-path>'
"""Shown when an error's `filename` is not a path value at all."""


def _model_safe_filename(filename: str | bytes, root: str) -> str:
    """Return the path relative to the workspace root.

    Paths not inside the root become `_OUTSIDE_WORKSPACE`; values that are
    not paths at all become `_NOT_A_PATH`.
    """
    try:
        raw = os.fsdecode(filename)
    except TypeError:
        return _NOT_A_PATH
    if not posixpath.isabs(raw):
        return raw
    path = posixpath.normpath(raw)
    if path == root or path.startswith(root + '/'):
        return posixpath.relpath(path, root)
    return _OUTSIDE_WORKSPACE


def _sanitize_recoverable_error(error: BaseException, root: str) -> str:
    """Render a recoverable error without exposing absolute paths.

    Errors without an OS-supplied `filename` keep their original message.
    OS errors keep `errno` and `strerror`, with the path rewritten relative
    to `root` (see `_model_safe_filename` for the fallback placeholders).
    """
    if not isinstance(error, OSError) or error.filename is None:
        return str(error)

    filename = _model_safe_filename(error.filename, root)
    return f'[Errno {error.errno}] {error.strerror}: {filename!r}'


def _recoverable(
    fn: Callable[Concatenate[FileSystemToolset, RunContext[Any], _P], Awaitable[str]],
) -> Callable[Concatenate[FileSystemToolset, RunContext[Any], _P], Awaitable[str]]:
    """Surface model-correctable tool errors as `ModelRetry`."""

    @functools.wraps(fn)
    async def wrapper(self: FileSystemToolset, ctx: RunContext[Any], *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, ctx, *args, **kwargs)
        except _RECOVERABLE_ERRORS as e:
            raise ModelRetry(_sanitize_recoverable_error(e, await self._root_for(ctx))) from e  # pyright: ignore[reportPrivateUsage]
        # A dead sandbox and a misconfigured one (`UserError`, e.g. no sandbox attached) are the
        # user's to fix; deliberate backend failures are recoverable, but programming errors
        # still propagate.
        except (SandboxUnavailableError, UserError):
            raise
        except SandboxError as e:
            if isinstance(e, TimeoutError):
                raise ModelRetry(f'{fn.__name__} timed out.') from e
            raise ModelRetry(str(e)) from e
        except OSError as e:
            reason = _RECOVERABLE_ERRNOS.get(e.errno)
            if reason is None:
                raise
            # The full error may embed the absolute path; the reason is path-free.
            raise ModelRetry(reason) from e

    return wrapper


def _format_lines(lines: Sequence[str], *, first_line_number: int, has_more: bool) -> str:
    """Number a window of lines, with a hint for continuing past its end."""
    if not lines:
        return '(empty file)\n'

    result = ''.join(f'{number:>6}\t{line}\n' for number, line in enumerate(lines, start=first_line_number))
    if has_more:
        result += f'... (more lines. Use offset={first_line_number - 1 + len(lines)} to continue reading.)\n'
    return result


def _is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Detect binary content by checking for null bytes or invalid UTF-8 in the sample."""
    sample = data[:sample_size]
    if b'\x00' in sample:
        return True
    try:
        codecs.getincrementaldecoder('utf-8')().decode(sample, final=False)
    except UnicodeDecodeError:
        return True
    return False


def _content_hash(content: str) -> str:
    """Compute a short content hash for conflict detection."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]


class FileSystemToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing filesystem operations inside the run's sandbox, scoped to a root directory.

    Security model:
    - All paths resolved relative to the root inside the sandbox, with textual
      containment checks; symlinks are not resolved for pattern matching, and
      the sandbox itself is the isolation boundary
    - Glob-based allow/deny filtering
    - Protected path patterns (e.g. `.git/`, `.env`)
    - Binary file detection blocks text operations
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        allowed_patterns: Sequence[str],
        denied_patterns: Sequence[str],
        protected_patterns: Sequence[str],
        max_read_lines: int,
        max_list_results: int,
        max_search_results: int,
        max_find_results: int,
    ) -> None:
        super().__init__()
        # A sandbox path: absolute, or relative to the sandbox working directory.
        self._root = sandbox_path(root_dir)
        self._allowed_patterns = list(allowed_patterns)
        self._denied_patterns = list(denied_patterns)
        self._protected_patterns = list(protected_patterns)
        self._max_read_lines = max_read_lines
        self._max_list_results = max_list_results
        self._max_search_results = max_search_results
        self._max_find_results = max_find_results

        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.edit_file, name='edit_file')
        self.add_function(self.list_directory, name='list_directory')
        self.add_function(self.search_files, name='search_files')
        self.add_function(self.find_files, name='find_files')
        self.add_function(self.create_directory, name='create_directory')
        self.add_function(self.file_info, name='file_info')

    def _matches(self, path: str, pattern: str) -> bool:
        """Glob-match a relative path, treating a leading `**/` as 'any directory, including the root'.

        `fnmatch` has no recursive `**`, so a bare `**/secrets*` would miss a
        root-level `secrets.yaml` -- there's no leading directory to match.
        Retrying with the `**/` prefix stripped covers the zero-directory case.
        """
        if fnmatch.fnmatch(path, pattern):
            return True
        if pattern.startswith('**/'):
            return fnmatch.fnmatch(path, pattern[3:])
        return False

    def _first_matching_pattern(self, path: str, patterns: list[str]) -> str | None:
        """Return the first pattern that matches path, or None."""
        return next((p for p in patterns if self._matches(path, p)), None)

    def _check_access(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        """Validate path against allow/deny/protected patterns.

        `check_allowed=False` skips the `allowed_patterns` gate. Walkers
        (`list_directory`, `search_files`, `find_files`) pass it so their root
        directory isn't required to match `allowed_patterns` itself -- `.` or
        `src` would never match a file pattern like `src/*.py`. The walk's
        entries are still filtered against `allowed_patterns` per-entry via
        `_is_accessible`. Denied patterns continue to gate the root.
        """
        if write and self._protected_patterns:
            matched = self._first_matching_pattern(path, self._protected_patterns)
            if matched:
                raise PermissionError(f'Path {path!r} is protected (matches {matched!r}).')

        if self._denied_patterns:
            matched = self._first_matching_pattern(path, self._denied_patterns)
            if matched:
                raise PermissionError(f'Path {path!r} is denied by pattern {matched!r}.')

        if check_allowed and self._allowed_patterns:
            if not any(self._matches(path, p) for p in self._allowed_patterns):
                raise PermissionError(f'Path {path!r} does not match any allowed pattern.')

    def _is_accessible(self, path: str) -> bool:
        """Predicate form of the read-level `_check_access` checks.

        Protected patterns are not consulted: they gate writes, and the walkers
        only read.
        """
        if self._denied_patterns:
            if self._first_matching_pattern(path, self._denied_patterns) is not None:
                return False
        if self._allowed_patterns and not any(self._matches(path, p) for p in self._allowed_patterns):
            return False
        return True

    async def _root_for(self, ctx: RunContext[AgentDepsT]) -> str:
        """The absolute sandbox path of the configured root."""
        return posixpath.normpath(await ctx.sandbox.resolve(self._root))

    async def _resolve(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        *,
        write: bool = False,
        check_allowed: bool = True,
    ) -> tuple[str, str]:
        """Resolve and access-check a path in one step, returning `(root, resolved)`.

        Resolution happens first so the access check matches patterns against
        the canonical path relative to the root, collapsing `.`/`..`/`//`
        segments that would otherwise slip past a literal pattern (e.g.
        `config/./secret.txt` evading a `config/secret.txt` deny rule).
        """
        root = await self._root_for(ctx)
        resolved = await ctx.sandbox.resolve(path, base=root)
        # `resolve` is spelling, not confinement (its own contract): the containment check is
        # ours and shapes policy only. Symlinks are left to the sandbox isolation boundary.
        if resolved != root and not resolved.startswith(root + '/'):
            raise PermissionError(f'Path {path!r} resolves outside the root directory.')
        self._check_access(posixpath.relpath(resolved, root), write=write, check_allowed=check_allowed)
        return root, resolved

    @staticmethod
    def _relative(root: str, path: str) -> str:
        return posixpath.relpath(posixpath.normpath(path), root)

    @staticmethod
    def _is_hidden(path: str) -> bool:
        return any(part.startswith('.') and part != '.' for part in path.split('/'))

    @_recoverable
    async def read_file(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read a text file with line numbers.

        Args:
            ctx: The current agent run context.
            path: Relative paths always resolve from the configured root.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        if limit is None:
            limit = self._max_read_lines
        _, resolved = await self._resolve(ctx, path)
        try:
            window = await ctx.sandbox.read_file(resolved, offset=offset + 1, limit=limit)
        except IsADirectoryError as e:
            raise FileNotFoundError(f"'{path}' is a directory, not a file.") from e
        except FileNotFoundError as e:
            raise FileNotFoundError(f'File not found: {path}') from e

        if offset > 0 and not window.lines:
            # Reading the file just to count its lines would defeat the bounded read.
            raise ValueError(f'Offset {offset} exceeds file length.')

        # The facade decodes with replacement, so binary detection uses the returned text window.
        if '\ufffd' in window.text or '\x00' in window.text:
            entry = await ctx.sandbox.stat(resolved)
            size = entry.size or 0
            return f'[Binary file: {size} bytes. Use a binary-aware tool to inspect.]'

        lines = window.lines
        if offset == 0 and not window.has_more:
            # The whole file is in the window, so report the hash write_file and edit_file verify
            # against. It comes from the file itself: a window drops the trailing newline and any
            # `\r`, so hashing the window text would report a hash they never accept. A partial
            # window has no whole-file hash to report, so it omits one.
            content = (await ctx.sandbox.read_bytes(resolved)).decode('utf-8', errors='replace')
            header = f'[{path} | {len(lines)} lines | hash:{_content_hash(content)}]\n'
        else:
            header = f'[{path} | lines {offset + 1}-{offset + len(lines)}]\n'
        return header + _format_lines(lines, first_line_number=offset + 1, has_more=window.has_more)

    @_recoverable
    async def write_file(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        """Create or overwrite a file with conflict detection.

        Args:
            ctx: The current agent run context.
            path: Relative paths always resolve from the configured root.
            content: The text content to write.
            expected_hash: If provided, the write is rejected when the file exists
                and its current hash doesn't match (optimistic concurrency).

        Returns:
            Confirmation message with new hash.
        """
        root, resolved = await self._resolve(ctx, path, write=True)
        try:
            entry = await ctx.sandbox.stat(resolved)
        except (FileNotFoundError, NotADirectoryError):
            entry = None
        if entry is not None and entry.is_dir:
            raise ModelRetry(f'Path {path!r} exists and is not a regular file.')

        parent = posixpath.dirname(resolved)
        try:
            parent_entry = await ctx.sandbox.stat(parent)
        except FileNotFoundError as e:
            parent_rel = self._relative(root, parent)
            raise FileNotFoundError(
                f"Parent directory '{parent_rel}' does not exist. Use create_directory first."
            ) from e
        if not parent_entry.is_dir:
            parent_rel = self._relative(root, parent)
            raise FileNotFoundError(f"Parent directory '{parent_rel}' does not exist. Use create_directory first.")

        if expected_hash is not None and entry is not None:
            current = (await ctx.sandbox.read_bytes(resolved)).decode('utf-8', errors='replace')
            current_hash = _content_hash(current)
            if current_hash != expected_hash:
                raise ValueError(
                    f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                    f'got hash:{current_hash}). Re-read the file and retry.'
                )

        await ctx.sandbox.write_bytes(resolved, content.encode('utf-8'))
        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        return f'Wrote {len(content)} chars ({lines} lines) to {path}. [hash:{new_hash}]'

    @_recoverable
    async def edit_file(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        """Edit a file by exact string replacement with conflict detection.

        The old_text must appear exactly once in the file. Include surrounding
        context lines to ensure uniqueness.

        Args:
            ctx: The current agent run context.
            path: Relative paths always resolve from the configured root.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash for subsequent operations.
        """
        _, resolved = await self._resolve(ctx, path, write=True)
        try:
            raw = await ctx.sandbox.read_bytes(resolved)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'File not found: {path}') from e
        text = raw.decode('utf-8', errors='replace')
        current_hash = _content_hash(text)
        if expected_hash is not None and current_hash != expected_hash:
            raise ValueError(
                f'Conflict: file {path!r} has changed (expected hash:{expected_hash}, '
                f'got hash:{current_hash}). Re-read the file and retry.'
            )

        count = text.count(old_text)
        if count == 0:
            raise ValueError(f'old_text not found in {path}.')
        if count > 1:
            raise ValueError(
                f'old_text found {count} times in {path}. Include more surrounding context to make the match unique.'
            )

        new_content = text.replace(old_text, new_text, 1)
        await ctx.sandbox.write_bytes(resolved, new_content.encode('utf-8'))
        return f'Edited {path}. [hash:{_content_hash(new_content)}]'

    @_recoverable
    async def list_directory(self, ctx: RunContext[AgentDepsT], path: str = '.') -> str:
        """List the contents of a directory.

        Args:
            ctx: The current agent run context.
            path: Relative paths always resolve from the configured root.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        root, resolved = await self._resolve(ctx, path, check_allowed=False)
        try:
            root_entry = await ctx.sandbox.stat(resolved)
        except FileNotFoundError as e:
            raise NotADirectoryError(f'Not a directory: {path}') from e
        if not root_entry.is_dir:
            raise NotADirectoryError(f'Not a directory: {path}')

        entries: list[str] = []
        for entry in sorted(await ctx.sandbox.list_dir(resolved), key=lambda item: item.path):
            rel = self._relative(root, entry.path)
            if self._is_hidden(rel) or not self._is_accessible(rel):
                continue
            line = f'{rel}/' if entry.is_dir else f'{rel}  ({entry.size or 0} bytes)'
            if len(entries) >= self._max_list_results:
                entries.append(f'[... truncated at {self._max_list_results} entries]')
                break
            entries.append(line)
        return '\n'.join(entries) if entries else '(empty directory)'

    @_recoverable
    async def search_files(
        self,
        ctx: RunContext[AgentDepsT],
        pattern: str,
        *,
        path: str = '.',
        include_glob: str | None = None,
    ) -> str:
        """Search file contents using a regular expression.

        Args:
            ctx: The current agent run context.
            pattern: Regex pattern to search for.
            path: Relative paths always resolve from the configured root.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').

        Returns:
            str: Matching lines formatted as file:line_number:text.
        """
        root, resolved = await self._resolve(ctx, path, check_allowed=False)
        per_file_cap = max(1, self._max_search_results + 1)
        result = await ctx.sandbox.run(
            ['grep', '-rn', '-I', '-H', '-m', str(per_file_cap), '--', pattern, resolved],
            timeout=30,
        )
        if result.exit_code >= 2:
            raise ModelRetry(result.stderr.strip() or f'grep exited with code {result.exit_code}.')
        if result.exit_code != 0 and not result.stdout:
            return 'No matches found.'

        matches: list[str] = []
        for line in result.stdout.splitlines():
            parts = line.split(':', 2)
            if len(parts) != 3:
                continue
            absolute, line_number, text = parts
            rel = self._relative(root, absolute)
            if self._is_hidden(rel) or not self._is_accessible(rel):
                continue
            if include_glob is not None and not self._matches(rel, include_glob):
                continue
            if len(matches) >= self._max_search_results:
                matches.append(f'[... truncated at {self._max_search_results} matches]')
                break
            matches.append(f'{rel}:{line_number}:{text}')
        return '\n'.join(matches) if matches else 'No matches found.'

    @_recoverable
    async def find_files(self, ctx: RunContext[AgentDepsT], pattern: str, *, path: str = '.') -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            ctx: The current agent run context.
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Relative paths always resolve from the configured root.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        if posixpath.isabs(pattern):
            raise ValueError(f'Pattern {pattern!r} must be relative to the search path, not absolute.')
        root, resolved = await self._resolve(ctx, path, check_allowed=False)
        try:
            root_entry = await ctx.sandbox.stat(resolved)
        except FileNotFoundError as e:
            raise NotADirectoryError(f'Not a directory: {path}') from e
        if not root_entry.is_dir:
            raise NotADirectoryError(f'Not a directory: {path}')

        # Patterns follow glob semantics: `*` stays in one directory and `**/` recurses.
        # `find -name` always recurses, so translate the two documented pattern shapes;
        # other shapes anchor on `-path` (whose `*` may cross separators).
        if pattern.startswith('**/') and '/' not in pattern[3:]:
            argv = ['find', resolved, '-name', pattern[3:]]
        elif '/' not in pattern:
            argv = ['find', resolved, '-mindepth', '1', '-maxdepth', '1', '-name', pattern]
        else:
            argv = ['find', resolved, '-path', posixpath.join(resolved, pattern)]
        result = await ctx.sandbox.run(argv, timeout=30)
        if result.exit_code != 0:
            raise ModelRetry(result.stderr.strip() or f'find exited with code {result.exit_code}.')

        matches: list[str] = []
        for absolute in sorted(result.stdout.splitlines()):
            rel = self._relative(root, absolute)
            if self._is_hidden(rel) or not self._is_accessible(rel):
                continue
            if len(matches) >= self._max_find_results:
                matches.append(f'[... truncated at {self._max_find_results} matches]')
                break
            try:
                entry = await ctx.sandbox.stat(absolute)
            except FileNotFoundError:  # deleted mid-walk
                continue
            matches.append(f'{rel}{"/" if entry.is_dir else ""}')
        return '\n'.join(matches) if matches else 'No matches found.'

    @_recoverable
    async def create_directory(self, ctx: RunContext[AgentDepsT], path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            ctx: The current agent run context.
            path: Relative paths always resolve from the configured root.

        Returns:
            Confirmation message.
        """
        _, resolved = await self._resolve(ctx, path, write=True)
        await ctx.sandbox.make_dir(resolved)
        return f'Created directory: {path}'

    @_recoverable
    async def file_info(self, ctx: RunContext[AgentDepsT], path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            ctx: The current agent run context.
            path: Relative paths always resolve from the configured root.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        _, resolved = await self._resolve(ctx, path)
        try:
            entry = await ctx.sandbox.stat(resolved)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Path not found: {path}') from e

        parts = [
            f'path: {path}',
            f'type: {"directory" if entry.is_dir else "file"}',
            f'size: {entry.size or 0} bytes',
        ]
        if not entry.is_dir:
            raw = await ctx.sandbox.read_bytes(resolved)
            is_bin = _is_binary(raw)
            parts.append(f'binary: {is_bin}')
            if not is_bin:
                text = raw.decode('utf-8', errors='replace')
                parts.append(f'lines: {len(text.splitlines())}')
                parts.append(f'hash: {_content_hash(text)}')
        return '\n'.join(parts)
