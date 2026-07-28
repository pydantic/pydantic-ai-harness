"""Read-only command safelist and flag denylists for shell-command analysis.

The data below is adopted, with attribution, from OpenAI Codex CLI's
`codex-rs` command-safety module (`command_safety/is_safe_command.rs` and
`is_dangerous_command.rs`), Apache-2.0. See the mining doc:
https://github.com/pydantic/pydantic-ai-notes/blob/main/features/harness-comparison/2026-07-08%20oss%20implementation%20mining%20-%20oh-my-pi%2C%20codex%2C%20opencode.md
(section 2.1, "The read-only safelist"). Each entry encodes a hard-won lesson --
e.g. `git -p` (pager can shell out), `find -fprintf` (writes files),
`rg --pre` (runs an arbitrary preprocessor), `sed` restricted to `-n {N}p`.

We extract the *lists* as data; the surrounding algorithm is our own. No Codex
code is copied verbatim.
"""

from __future__ import annotations

# Commands that only ever read/inspect -- safe to auto-allow with any arguments.
# (Codex `is_safe_command.rs` unconditional set, plus Linux-only `tac`/`numfmt`.)
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

# `base64` is safe unless it is asked to *write* a file.
BASE64_WRITE_FLAGS: frozenset[str] = frozenset({'-o', '--output'})

# `find` is safe unless it is asked to execute something or write/delete files.
# Presence of any of these is treated as arbitrary code execution / mutation.
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

# `rg` (ripgrep) is safe unless it is told to run an external binary or shell out
# to a decompressor.
RG_EXEC_FLAGS: frozenset[str] = frozenset({'--pre', '--hostname-bin', '-z', '--search-zip'})

# `git` read-only subcommands. Anything else (add, commit, push, checkout, ...) is
# not auto-allowed.
GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset({'status', 'log', 'diff', 'show', 'branch'})

# `git` *global* options (before the subcommand) that broaden scope enough that the
# command should no longer be auto-allowed: `-C` changes directory, `-c`/`--config-env`
# inject config, `-p`/`--paginate` can invoke a pager that shells out, etc.
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

# `git branch` is read-only only with these options; a bare `git branch <name>`
# *creates* a branch, so we require an explicit read-only option.
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

# Shell wrappers whose *inner* command is what actually runs; we peel them before
# matching (adopted from the CC/Codex "wrapper stripping" behavior, mining doc 2.1).
# `env`/`sudo`/`nice` peel only in their simplest, argument-free-ish shapes -- any
# option or assignment degrades to `ask` (an `env FOO=... cmd` can inject
# `LD_PRELOAD`, a `sudo -u` changes identity), which is handled in `_command.py`.
WRAPPERS: frozenset[str] = frozenset({'timeout', 'nice', 'env', 'xargs', 'sudo', 'nohup', 'stdbuf', 'ionice', 'doas'})

# Arbitrary-code interpreters/shells. Even under a broad `allow` rule these must never be
# auto-allowed -- `bash -c '...'`, `python -c '...'`, etc. can run anything, so the built-in
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

# Model-proposed "always allow this prefix" suggestions that must never be honored:
# each would allow arbitrary code execution behind an innocuous-looking prefix.
# Adopted from Codex `BANNED_PREFIX_SUGGESTIONS` (`core/src/exec_policy.rs`), Apache-2.0.
# Used to vet any persistence proposal an escalation flow surfaces.
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
