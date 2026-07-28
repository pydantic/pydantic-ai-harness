"""Conservative shell-command analysis: split, gate, unwrap, classify.

The design follows the two harnesses mined in
https://github.com/pydantic/pydantic-ai-notes/blob/main/features/harness-comparison/2026-07-08%20oss%20implementation%20mining%20-%20oh-my-pi%2C%20codex%2C%20opencode.md
(section 2.1):

- **Codex's conservative-parse gate** (`shell-command/src/bash.rs`): a command is only
  eligible for auto-approval if it reduces to a list of *plain* commands joined by the
  safe control operators `&&`, `||`, `;`, `|`. The moment we see a subshell, command
  substitution (`$(...)`, backticks), redirection, variable expansion, background `&`,
  brace/glob metacharacters, or a shell assignment prefix, we stop trusting our parse and
  degrade the whole call to `ask`. We never guess-allow.
- **opencode's per-segment requirement** (`tool/shell.ts`): every segment must
  independently pass; one bad segment poisons the sequence.
- **Wrapper stripping** (`timeout`/`nice`/`env`/`xargs`/`sudo`): peel to the inner command
  before matching, and degrade to `ask` on any wrapper shape that could smuggle execution
  (`env FOO=...`, `sudo -u`, ...).

We do this with the standard library only -- a quote-aware scanner plus `shlex` -- so the
capability adds **no new dependency**. `bashlex`/`tree-sitter-bash` would give a real AST,
but the conservative gate means anything our scanner cannot prove plain already degrades to
`ask`; a heavier parser would only *narrow* the set of commands that degrade, never change a
wrong-allow into a right-allow. See the PR body for this trade-off.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal

from ._safelist import (
    BASE64_WRITE_FLAGS,
    FIND_EXEC_FLAGS,
    GIT_BRANCH_READONLY_OPTIONS,
    GIT_READONLY_SUBCOMMANDS,
    GIT_UNSAFE_GLOBAL_OPTIONS,
    INTERPRETERS,
    RG_EXEC_FLAGS,
    UNCONDITIONAL_SAFE,
    WRAPPERS,
)

Verdict = Literal['allow', 'ask', 'deny']

# Control operators at which we split into independent command segments. `&&` and `||`
# are checked before the single-character operators so we never mistake them for `&`/`|`.
_TWO_CHAR_OPERATORS = ('&&', '||')
_ONE_CHAR_OPERATORS = (';', '|')

# Unquoted characters that make a command "not plainly parseable" -- their presence trips
# the conservative gate. `$` covers `$(...)`, `${...}`, and `$VAR`; `` ` `` covers legacy
# substitution; `<`/`>` are redirection; `(`/`)`/`{`/`}` are subshells/groups; `*`/`?`/`[`
# are globs; `~` is home expansion; `!` is history/negation; `\n`/`\r` are extra statements.
_GATE_CHARS = frozenset('$`()<>{}*?[]~!\n\r')

_ASSIGNMENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')


@dataclass(frozen=True)
class PreparedCommand:
    """A shell command reduced to plain, wrapper-stripped argv segments.

    `confident` is `True` only when the whole command passed the conservative gate *and*
    every wrapper peeled cleanly. When `False`, `segments` is empty and callers must treat
    the command as `ask` (never `allow`); `reason` explains why.
    """

    confident: bool
    segments: tuple[tuple[str, ...], ...]
    reason: str = ''


def _consume_quote(command: str, i: int, quote: str, current: list[str]) -> int | None:
    """Append a full quoted span (from the opening quote at `i`) to `current`.

    Returns the index just past the closing quote, or `None` if the quote never closes.
    Only double quotes honor backslash escaping (matching POSIX shell).
    """
    n = len(command)
    current.append(command[i])
    i += 1
    while i < n:
        ch = command[i]
        current.append(ch)
        if quote == '"' and ch == '\\' and i + 1 < n:
            current.append(command[i + 1])
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    return None  # unbalanced quotes


def _split_segments(command: str) -> list[str] | None:
    """Quote-aware split into raw segment strings, or `None` if the gate trips.

    Walks the string, consuming quoted spans whole and splitting at unquoted control
    operators. Returns `None` (gate tripped) on any unquoted gate character, a lone `&`
    (background), an unbalanced quote, or a trailing backslash.
    """
    segments: list[str] = []
    current: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch in ('"', "'"):
            closed = _consume_quote(command, i, ch, current)
            if closed is None:
                return None
            i = closed
            continue
        if ch == '\\':
            if i + 1 >= n:
                return None  # trailing backslash / line continuation
            current.append(ch)
            current.append(command[i + 1])
            i += 2
            continue
        operator_len = _operator_length(command, i, bool(current))
        if operator_len == 0:
            return None  # gate char (incl. lone `&` / start-of-word `#`)
        if operator_len is None:
            current.append(ch)
            i += 1
            continue
        segments.append(''.join(current))
        current = []
        i += operator_len
    segments.append(''.join(current))
    return segments


def _operator_length(command: str, i: int, has_current: bool) -> int | None:
    """Classify `command[i]`: split-operator length, `0` to gate, or `None` for a plain char."""
    ch = command[i]
    if ch == '&':
        return 2 if command[i : i + 2] == '&&' else 0  # lone `&` (background) gates
    if command[i : i + 2] in _TWO_CHAR_OPERATORS:
        return 2
    if ch in _ONE_CHAR_OPERATORS:
        return 1
    if ch == '#' and (not has_current or command[i - 1].isspace()):
        return 0  # start-of-word comment
    if ch in _GATE_CHARS:
        return 0
    return None


def _strip_wrappers(argv: list[str]) -> list[str] | None:
    """Peel known wrappers to the inner command, or `None` to degrade to `ask`.

    Only the simplest wrapper shapes peel. Anything that could smuggle execution or change
    identity/environment (`env FOO=...`, `env -i`, `sudo -u`, `timeout --signal=... `,
    `nice -n` without its value) returns `None` so the caller degrades to `ask`.
    """
    argv = list(argv)
    while argv and argv[0] in WRAPPERS:
        head = argv[0]
        rest = argv[1:]
        if not rest:
            return None
        if head == 'timeout':
            # timeout DURATION cmd... -- options (e.g. -s, --signal) degrade.
            if rest[0].startswith('-'):
                return None
            argv = rest[1:]
        elif head == 'nice':
            if rest[0] == '-n':
                if len(rest) < 2:
                    return None
                argv = rest[2:]
            elif rest[0].startswith('-'):
                return None
            else:
                argv = rest
        elif head in ('env', 'nohup', 'stdbuf', 'ionice', 'doas'):
            # No options, no `NAME=value` assignments (LD_PRELOAD risk).
            if rest[0].startswith('-') or '=' in rest[0]:
                return None
            argv = rest
        elif head == 'sudo':
            if rest[0].startswith('-'):
                return None
            argv = rest
        else:  # xargs
            if rest[0].startswith('-'):
                return None
            argv = rest
        if not argv:
            return None
    return argv


def prepare_command(command: str) -> PreparedCommand:
    """Reduce a shell command to plain, wrapper-stripped argv segments (or degrade)."""
    raw_segments = _split_segments(command)
    if raw_segments is None:
        return PreparedCommand(False, (), 'contains a shell construct that cannot be safely analyzed')
    segments: list[tuple[str, ...]] = []
    for raw in raw_segments:
        raw = raw.strip()
        if not raw:
            return PreparedCommand(False, (), 'contains an empty command segment')
        try:
            argv = shlex.split(raw)
        except ValueError:
            return PreparedCommand(False, (), 'could not be tokenized')
        if not argv:  # pragma: no cover - the gate already rejects comment/empty-only segments
            return PreparedCommand(False, (), 'contains an empty command segment')
        stripped = _strip_wrappers(argv)
        if stripped is None or not stripped:
            return PreparedCommand(False, (), 'uses a wrapper in a shape that cannot be safely analyzed')
        if _ASSIGNMENT_RE.match(stripped[0]):
            return PreparedCommand(False, (), 'sets an environment variable inline')
        segments.append(tuple(stripped))
    return PreparedCommand(True, tuple(segments))


@dataclass(frozen=True)
class CommandVerdict:
    """Built-in safety verdict for a shell command (channel B)."""

    verdict: Verdict | None
    reason: str = ''
    retryable: bool = True


def _classify_git(argv: tuple[str, ...]) -> CommandVerdict:
    # Scan global options up to the first non-option token (the subcommand).
    idx = 1
    while idx < len(argv) and argv[idx].startswith('-'):
        if argv[idx] in GIT_UNSAFE_GLOBAL_OPTIONS or argv[idx].split('=')[0] in GIT_UNSAFE_GLOBAL_OPTIONS:
            return CommandVerdict('ask', f'`git {argv[idx]}` broadens scope and is not auto-approved')
        idx += 1
    if idx >= len(argv):
        return CommandVerdict(None)  # bare `git` -- no opinion
    subcommand = argv[idx]
    if subcommand not in GIT_READONLY_SUBCOMMANDS:
        return CommandVerdict(None)  # add/commit/push/... -- defer to rules/default
    if subcommand == 'branch':
        options = argv[idx + 1 :]
        readonly = any(opt in GIT_BRANCH_READONLY_OPTIONS or opt.startswith('--format=') for opt in options)
        if any(not opt.startswith('-') for opt in options) or not readonly:
            # A positional (branch name) or no read-only flag means it may create/delete.
            return CommandVerdict('ask', '`git branch` without a read-only option may modify branches')
    return CommandVerdict('allow')


def _classify_segment(argv: tuple[str, ...]) -> CommandVerdict:
    cmd = argv[0]
    flags = frozenset(a for a in argv[1:] if a.startswith('-'))
    if cmd in INTERPRETERS:
        return CommandVerdict('ask', f'`{cmd}` runs an arbitrary interpreter/shell and requires approval')
    if cmd == 'rm':
        if flags & {'-r', '-R', '--recursive', '-f', '--force', '-rf', '-fr', '-rF'}:
            return CommandVerdict('deny', '`rm` with a recursive/force flag is never auto-approved', retryable=False)
        return CommandVerdict('ask', '`rm` deletes files and is not auto-approved')
    if cmd == 'git':
        return _classify_git(argv)
    if cmd == 'find':
        if flags & FIND_EXEC_FLAGS:
            return CommandVerdict(
                'deny', '`find` with an exec/delete/write action is never auto-approved', retryable=False
            )
        return CommandVerdict('allow')
    if cmd == 'rg':
        if flags & RG_EXEC_FLAGS:
            return CommandVerdict(
                'deny', '`rg` with a preprocessor/decompression flag can run arbitrary binaries', retryable=False
            )
        return CommandVerdict('allow')
    if cmd == 'base64':
        if flags & BASE64_WRITE_FLAGS:
            return CommandVerdict('ask', '`base64 -o` writes a file and is not auto-approved')
        return CommandVerdict('allow')
    if cmd == 'sed':
        # Only the read-only `sed -n {N|M,N}p` form is safe.
        if _is_safe_sed(argv):
            return CommandVerdict('allow')
        return CommandVerdict('ask', 'only `sed -n {N}p` is auto-approved; this `sed` may edit in place')
    if cmd in UNCONDITIONAL_SAFE:
        return CommandVerdict('allow')
    return CommandVerdict(None)  # unknown command -- defer to rules/default


_SED_N_ARG_RE = re.compile(r'^\d+(,\d+)?p$')


def _is_safe_sed(argv: tuple[str, ...]) -> bool:
    # `sed -n {N|M,N}p [file]` -- at most one script argument, `-n` required.
    if len(argv) < 3 or argv[1] != '-n':
        return False
    return bool(_SED_N_ARG_RE.match(argv[2]))


def analyze_command(prepared: PreparedCommand) -> CommandVerdict:
    """Built-in safety verdict (channel B) for an already-prepared command.

    - not confident -> `ask` (conservative gate; never `allow`)
    - any segment dangerous / exec-flagged -> `deny` (one bad segment poisons the sequence)
    - any segment needs approval -> `ask`
    - every segment on the read-only safelist -> `allow`
    - otherwise (some unknown, none dangerous) -> no opinion (`None`)
    """
    if not prepared.confident:
        return CommandVerdict('ask', prepared.reason, retryable=True)
    per_segment = [_classify_segment(argv) for argv in prepared.segments]
    denies = [s for s in per_segment if s.verdict == 'deny']
    if denies:
        return denies[0]
    asks = [s for s in per_segment if s.verdict == 'ask']
    if asks:
        return asks[0]
    if all(s.verdict == 'allow' for s in per_segment):
        return CommandVerdict('allow')
    return CommandVerdict(None)


def command_matches_prefix(prefix: str, prepared: PreparedCommand) -> bool:
    """Whether every segment of `prepared` starts with the whole words of `prefix`.

    Word-boundary semantics: `git status` matches `git status -sb` but never `git status-evil`.
    Only matches a confidently-parsed command; a command we could not parse never matches a
    prefix rule (so a broad `allow` rule cannot green-light a command we could not analyze).
    """
    if not prepared.confident:
        return False
    try:
        words = tuple(shlex.split(prefix))
    except ValueError:  # pragma: no cover - defensive
        return False
    if not words:
        return True
    for argv in prepared.segments:
        if argv[: len(words)] != words:
            return False
    return True
