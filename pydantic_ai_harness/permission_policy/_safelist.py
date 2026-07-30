"""Read-only command safelist and flag denylists for shell-command analysis.

The command lists below (which binaries are read-only, which flags turn a normally-safe
command dangerous) are adopted, with attribution, from OpenAI Codex CLI's `codex-rs`
command-safety module (`command_safety/is_safe_command.rs` and `is_dangerous_command.rs`,
Apache-2.0) -- see the survey referenced in this capability's README. Each entry encodes a
hard-won lesson: `git -p` can shell out through a pager, `find -fprintf` writes files,
`rg --pre` runs an arbitrary preprocessor, `sed` is only safe restricted to `-n {N}p`.

We use the *lists* as data. The matching algorithm in `_command.py` -- including how flags
are canonicalized before they're compared against these sets -- is our own; see that module's
docstring for the design rationale and the security history behind it.
"""

from __future__ import annotations

# Commands that only ever read or inspect state, safe to auto-allow regardless of arguments.
UNCONDITIONAL_SAFE: frozenset[str] = frozenset(
    {
        'cat',
        'cd',
        'cut',
        'echo',
        'expr',
        'false',
        'grep',
        'head',
        'id',
        'ls',
        'nl',
        'numfmt',
        'paste',
        'pwd',
        'rev',
        'seq',
        'stat',
        'tac',
        'tail',
        'tr',
        'true',
        'uname',
        'uniq',
        'wc',
        'which',
        'whoami',
    }
)

# `rm`'s short options are all boolean (none take an attached value), so `_command.py` can
# safely decompose a bundled token like `-rf` into `-r` + `-f` before checking membership
# here -- unlike `find`/`rg` below, where the short-option alphabet includes value-taking
# flags and blind bundling would misparse. See `_command.py:_rm_flags`.
RM_DESTRUCTIVE_FLAGS: frozenset[str] = frozenset({'-r', '-R', '--recursive', '-f', '--force'})

# `base64` is safe unless asked to *write* a file.
BASE64_WRITE_FLAGS: frozenset[str] = frozenset({'-o', '--output'})

# `find` is safe unless asked to execute something or write/delete files. Presence of any of
# these is treated as arbitrary code execution or mutation. Long-form only -- `find` has no
# single-dash multi-letter bundling to worry about, but see the module docstring for the
# `--flag=value` canonicalization that still applies to these.
FIND_EXEC_FLAGS: frozenset[str] = frozenset(
    {
        '-exec',
        '-execdir',
        '-ok',
        '-okdir',
        '-delete',
        '-fls',
        '-fprint',
        '-fprint0',
        '-fprintf',
    }
)

# `rg` (ripgrep) is safe unless told to run an external binary or shell out to a decompressor.
# Residual risk: `-z` is a single-character flag and ripgrep supports bundling short options
# (e.g. `-oz`); we do not decompose ripgrep's short-option bundles (see the module docstring),
# so a bundled `-z` currently evades this set. Prefer `--search-zip` detection or a stricter
# `default_verdict` if this matters for your threat model.
RG_EXEC_FLAGS: frozenset[str] = frozenset({'--pre', '--hostname-bin', '-z', '--search-zip'})

# `git` read-only subcommands. Anything else (add, commit, push, checkout, ...) is not
# auto-allowed by the built-in analysis.
GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset({'status', 'log', 'diff', 'show', 'branch'})

# Flags that let an otherwise-read-only `git` subcommand write an arbitrary caller-chosen
# file (`git diff --output=path`, `git log -o path`, `git show --output path` all write).
GIT_WRITE_CAPABLE_OPTIONS: frozenset[str] = frozenset({'-o', '--output'})

# `git` *global* options (before the subcommand) that broaden scope enough that the command
# should no longer be auto-allowed: `-C` changes directory, `-c`/`--config-env` inject config,
# `-p`/`--paginate` can invoke a pager that shells out, etc.
GIT_UNSAFE_GLOBAL_OPTIONS: frozenset[str] = frozenset(
    {
        '-C',
        '-c',
        '--config-env',
        '--exec-path',
        '--git-dir',
        '--namespace',
        '--super-prefix',
        '--work-tree',
        '-p',
        '--paginate',
    }
)

# `git branch` is read-only only with these options; a bare `git branch <name>` *creates* a
# branch, so an explicit read-only option is required.
GIT_BRANCH_READONLY_OPTIONS: frozenset[str] = frozenset(
    {
        '--list',
        '-l',
        '--show-current',
        '-a',
        '--all',
        '-r',
        '-v',
        '-vv',
        '--verbose',
    }
)

# Shell wrappers whose *inner* command determines what actually runs, so the analyzer peels
# them before matching. `sudo`/`doas` and `xargs` are members of this set for detection
# purposes only -- `_command.py` never actually unwraps them (see its module docstring for
# why both are treated as always-`ask`, never peeled).
WRAPPERS: frozenset[str] = frozenset({'timeout', 'nice', 'env', 'xargs', 'sudo', 'nohup', 'stdbuf', 'ionice', 'doas'})

# Arbitrary-code interpreters/shells. Even under a broad `allow` rule these must never be
# auto-allowed: `bash -c '...'`, `python -c '...'`, etc. can run anything, so the built-in
# analysis routes them to `ask` (most-restrictive-wins overrides the allow). This is the
# execution-channel twin of `BANNED_PREFIX_SUGGESTIONS` below (the rule-persistence channel).
INTERPRETERS: frozenset[str] = frozenset(
    {
        'bash',
        'sh',
        'zsh',
        'fish',
        'dash',
        'ksh',
        'csh',
        'tcsh',
        'python',
        'python2',
        'python3',
        'node',
        'deno',
        'bun',
        'perl',
        'ruby',
        'php',
        'lua',
        'Rscript',
        'osascript',
        'pwsh',
        'powershell',
        'source',
        'eval',
        'exec',
    }
)

# Model-proposed "always allow this prefix" suggestions that must never be honored: each would
# allow arbitrary code execution behind an innocuous-looking prefix. Adopted from Codex CLI's
# `BANNED_PREFIX_SUGGESTIONS` (`core/src/exec_policy.rs`), Apache-2.0. Used to vet any rule an
# escalation flow might propose to persist.
BANNED_PREFIX_SUGGESTIONS: frozenset[str] = frozenset(
    {
        'python',
        'python3',
        'python -c',
        'python3 -c',
        'bash',
        'bash -c',
        'bash -lc',
        'sh',
        'sh -c',
        'zsh',
        'zsh -c',
        'env',
        'sudo',
        'node',
        'node -e',
        'deno',
        'perl',
        'perl -e',
        'ruby',
        'ruby -e',
        'php',
        'php -r',
        'lua',
        'lua -e',
        'osascript',
        'pwsh',
        'pwsh -Command',
        'powershell',
        'eval',
        'exec',
    }
)
