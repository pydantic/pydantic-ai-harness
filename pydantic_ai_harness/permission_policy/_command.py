"""Conservative shell-command analysis: split, gate, unwrap, classify.

The overall shape follows two harnesses surveyed for this capability's design (see the
README): Codex CLI's **conservative-parse gate** -- a command is only eligible for
auto-approval if it reduces to a list of *plain* commands joined by the safe control
operators `&&`, `||`, `;`, `|`; the moment we see a subshell, command substitution,
redirection, variable expansion, background `&`, or a glob/brace metacharacter, we stop
trusting our parse and degrade the whole call to `ask` -- and opencode's **per-segment
requirement**: every segment of a compound command must independently pass, or one bad
segment poisons the sequence. We use the standard library only (a quote-aware scanner plus
`shlex`); a full AST parser would only ever *narrow* what degrades to `ask`, never turn a
wrong-allow into a right-allow, given the gate already refuses to guess.

## Why "checks a fixed argv position" is the wrong shape for a validator here

An earlier design (see git history / the linked upstream PR discussion) wrote each
command-specific check as a small set of assertions about *where* a flag or script argument
would appear: "argv[1] must be `-n`", "argv[2] is the script", "flags are exactly the tokens
that start with `-`". Real shell syntax does not hold still for that: a flag can be repeated,
shifted after an option that takes no value, spelled as `--flag=value` instead of
`--flag value`, or bundled with other short options (`-rf` is `-r` plus `-f`). A validator
that only inspects specific positions is trivially defeated by anything that shifts them --
appending a second `-e` script to `sed`, spelling `--pre=cmd` instead of `--pre cmd`, or
running `git` through an absolute path. Any of those degrade a "based on this argv shape it's
safe" predicate into "safe assuming nothing else is present", which is exactly the assumption
an adversarial (or just unusually-spelled) command call breaks.

The fix applied throughout this module is to stop checking positions and instead **consume
the whole argv**: canonicalize every option token the same way (`_flag_key` strips a
`--flag=value` value before comparison) before any denylist/allowlist membership test runs,
and validators that need to prove a command is *exactly* one safe shape (`sed`) scan every
remaining token rather than assuming nothing follows the tokens they already recognized.
Where a command's flag surface is too large or under-specified for us to safely canonicalize
(ripgrep and `find`'s single-character option bundling), we say so in `_safelist.py` rather
than pretend the gap is closed.
"""

from __future__ import annotations

import os
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
    GIT_WRITE_CAPABLE_OPTIONS,
    INTERPRETERS,
    RG_EXEC_FLAGS,
    RM_DESTRUCTIVE_FLAGS,
    UNCONDITIONAL_SAFE,
    WRAPPERS,
)

Verdict = Literal['allow', 'ask', 'deny']

# Control operators at which we split into independent command segments. `&&` and `||` are
# checked before the single-character operators so we never mistake them for `&`/`|`.
_TWO_CHAR_OPERATORS = ('&&', '||')
_ONE_CHAR_OPERATORS = (';', '|')

# Unquoted characters that make a command "not plainly parseable" -- their presence trips the
# conservative gate. `$` covers `$(...)`, `${...}`, and `$VAR`; `` ` `` covers legacy
# substitution; `<`/`>` are redirection; `(`/`)`/`{`/`}` are subshells/groups; `*`/`?`/`[` are
# globs; `~` is home expansion; `!` is history/negation; `\n`/`\r` are extra statements.
_GATE_CHARS = frozenset('$`()<>{}*?[]~!\n\r')

# Inside a *double*-quoted span, ordinary text is inert, but `$` and `` ` `` still trigger
# command/variable substitution (`"$(...)"`, `` "`...`" ``, `"$VAR"`) -- a naive scanner that
# treats a whole quoted span as opaque literal text misses that a command hides inside it.
# A single-quoted span has no such exception: POSIX shells never expand anything inside `'...'`.
_DOUBLE_QUOTE_GATE_CHARS = frozenset('$`')

_ASSIGNMENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')


@dataclass(frozen=True)
class PreparedCommand:
    """A shell command reduced to plain, wrapper-stripped argv segments.

    `confident` is `True` only when the whole command passed the conservative gate *and*
    every wrapper peeled cleanly. When `False`, `segments` is empty and callers must treat the
    command as `ask` (never `allow`); `reason` explains why.
    """

    confident: bool
    segments: tuple[tuple[str, ...], ...]
    reason: str = ''


def _consume_quote(command: str, i: int, quote: str, current: list[str]) -> int | None:
    """Append a full quoted span (from the opening quote at `i`) to `current`.

    Returns the index just past the closing quote, or `None` if the quote never closes or (for
    a double-quoted span) an unescaped `$`/backtick shows a substitution hiding inside it.
    """
    n = len(command)
    current.append(command[i])
    i += 1
    while i < n:
        ch = command[i]
        if quote == '"' and ch == '\\' and i + 1 < n:
            current.append(ch)
            current.append(command[i + 1])
            i += 2
            continue
        if quote == '"' and ch in _DOUBLE_QUOTE_GATE_CHARS:
            return None  # `"$(...)"` / `` "`...`" `` / `"$VAR"` still expand
        current.append(ch)
        if ch == quote:
            return i + 1
        i += 1
    return None  # unbalanced quotes


def _split_segments(command: str) -> list[str] | None:
    """Quote-aware split into raw segment strings, or `None` if the gate trips.

    Walks the string, consuming quoted spans whole and splitting at unquoted control
    operators. Returns `None` (gate tripped) on any unquoted gate character, a substitution
    hiding inside a double-quoted span, a lone `&` (background), an unbalanced quote, or a
    trailing backslash.
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
    identity/environment returns `None` so the caller degrades to `ask`. `sudo` and `doas`
    change the identity the *rest* of the command runs as, and `xargs` builds its real argv
    from stdin at run time rather than from the tokens on the command line -- neither is a
    "peel to the inner command and reuse its verdict" wrapper, so both always return `None`
    here rather than exposing an inner command for the caller to (wrongly) approve.
    """
    argv = list(argv)
    while argv and argv[0] in WRAPPERS:
        head = argv[0]
        rest = argv[1:]
        if not rest:
            return None
        if head in ('sudo', 'doas', 'xargs'):
            # Never peel: `sudo`/`doas` change who the inner command runs as, and `xargs`'s
            # real argv depends on stdin, which we cannot see from the command string alone.
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
        else:  # env, nohup, stdbuf, ionice
            # No options, no `NAME=value` assignments (LD_PRELOAD risk).
            if rest[0].startswith('-') or '=' in rest[0]:
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


def _flag_key(token: str) -> str:
    """Canonical form of an option token for denylist/allowlist membership tests.

    Strips a GNU `--flag=value` value so `--pre=/bin/sh` compares equal to `--pre`, and a
    bare `--output=path` compares equal to `--output`. Every membership check in this module
    against `_safelist.py`'s flag sets goes through this function rather than comparing raw
    tokens directly -- see the module docstring for why a raw-token comparison is the
    positional-argument class of bug this analyzer must not repeat.
    """
    if '=' in token and token.startswith('-'):
        return token.split('=', 1)[0]
    return token


def _rm_flags(argv: tuple[str, ...]) -> frozenset[str]:
    """Canonical flag set for an `rm` invocation, with bundled short options decomposed.

    `rm`'s short options are all boolean (`-r`, `-f`, `-i`, `-v`, ... -- none takes an attached
    value), so a bundle like `-rf` or `-vfr` can be split one character at a time without
    ambiguity: `-rf` becomes `{-r, -f}`. This is what lets the destructive-flag check below
    catch `rm -vrf file` the same way it catches `rm -r -f file`. We do *not* do this for
    `find`/`rg` (see `_safelist.py`): their short-option alphabets include value-taking flags
    where blind character-splitting would silently misclassify the command.
    """
    flags: set[str] = set()
    for token in argv[1:]:
        if token == '--':
            break
        if token.startswith('--'):
            flags.add(_flag_key(token))
        elif token.startswith('-') and len(token) > 1:
            flags.update(f'-{ch}' for ch in token[1:])
    return frozenset(flags)


def _classify_git(argv: tuple[str, ...]) -> CommandVerdict:
    # Scan global options up to the first non-option token (the subcommand).
    idx = 1
    while idx < len(argv) and argv[idx].startswith('-'):
        if _flag_key(argv[idx]) in GIT_UNSAFE_GLOBAL_OPTIONS:
            return CommandVerdict('ask', f'`git {argv[idx]}` broadens scope and is not auto-approved')
        idx += 1
    if idx >= len(argv):
        return CommandVerdict(None)  # bare `git` -- no opinion
    subcommand = argv[idx]
    if subcommand not in GIT_READONLY_SUBCOMMANDS:
        return CommandVerdict(None)  # add/commit/push/... -- defer to rules/default
    options = argv[idx + 1 :]
    if any(_flag_key(opt) in GIT_WRITE_CAPABLE_OPTIONS for opt in options if opt.startswith('-')):
        return CommandVerdict('ask', f'`git {subcommand} --output`/`-o` writes a caller-chosen file')
    if subcommand == 'branch':
        readonly = any(opt in GIT_BRANCH_READONLY_OPTIONS or opt.startswith('--format=') for opt in options)
        if any(not opt.startswith('-') for opt in options) or not readonly:
            # A positional (branch name) or no read-only flag means it may create/delete.
            return CommandVerdict('ask', '`git branch` without a read-only option may modify branches')
    return CommandVerdict('allow')


def _classify_segment(argv: tuple[str, ...]) -> CommandVerdict:
    # `_classify_segment`'s dangerous-command checks are normalized to the executable's
    # basename (`/usr/bin/rm` is still `rm`) so a path qualifier alone cannot dodge them.
    # `UNCONDITIONAL_SAFE` below deliberately stays keyed on the *literal* argv[0] -- widening
    # what counts as "safe" via path normalization would trade one bypass for another (a
    # maliciously-placed binary at an unexpected path shadowing a trusted name); narrowing what
    # counts as "dangerous" has no such downside, so only the dangerous-command checks
    # normalize. See the module docstring.
    cmd = argv[0]
    basename = os.path.basename(cmd)
    flags = frozenset(_flag_key(a) for a in argv[1:] if a.startswith('-'))
    if basename in INTERPRETERS:
        return CommandVerdict('ask', f'`{basename}` runs an arbitrary interpreter/shell and requires approval')
    if basename == 'rm':
        if _rm_flags(argv) & RM_DESTRUCTIVE_FLAGS:
            return CommandVerdict('deny', '`rm` with a recursive/force flag is never auto-approved', retryable=False)
        return CommandVerdict('ask', '`rm` deletes files and is not auto-approved')
    if basename == 'git':
        return _classify_git(argv)
    if basename == 'find':
        if flags & FIND_EXEC_FLAGS:
            return CommandVerdict(
                'deny', '`find` with an exec/delete/write action is never auto-approved', retryable=False
            )
        return CommandVerdict('allow')
    if basename == 'rg':
        if flags & RG_EXEC_FLAGS:
            return CommandVerdict(
                'deny', '`rg` with a preprocessor/decompression flag can run arbitrary binaries', retryable=False
            )
        return CommandVerdict('allow')
    if basename == 'base64':
        if flags & BASE64_WRITE_FLAGS:
            return CommandVerdict('ask', '`base64 -o` writes a file and is not auto-approved')
        return CommandVerdict('allow')
    if basename == 'sed':
        if _is_safe_sed(argv):
            return CommandVerdict('allow')
        return CommandVerdict('ask', 'only `sed -n {N}p` is auto-approved; this `sed` may edit in place')
    if cmd in UNCONDITIONAL_SAFE:
        return CommandVerdict('allow')
    return CommandVerdict(None)  # unknown command -- defer to rules/default


_SED_N_ARG_RE = re.compile(r'^\d+(,\d+)?p$')


def _is_safe_sed(argv: tuple[str, ...]) -> bool:
    """Whether `argv` is *exactly* `sed -n {N|M,N}p [file...]` and nothing else.

    Only two tokens are ever recognized as the safe shape (`-n` and one print-range script);
    everything else in `argv` must be a plain positional filename. This is a whole-argv
    consumption check, not a fixed-position one: an earlier version of this predicate checked
    only `argv[1]`/`argv[2]` and ignored anything appended after them, so `sed -n 2p -e "3w
    /tmp/out"` -- a second, destructive `-e` script tacked on after the recognized safe
    prefix -- passed unnoticed and was auto-approved. Checking that *nothing remains* beyond
    the recognized shape closes that: any additional flag or script anywhere in `argv`
    (`-e`, `-i`, `-f`, a repeated `-n`, ...) now fails this check.
    """
    if len(argv) < 3 or argv[1] != '-n':
        return False
    if not _SED_N_ARG_RE.match(argv[2]):
        return False
    return all(not token.startswith('-') for token in argv[3:])


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
