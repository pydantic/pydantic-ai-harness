"""Red-team tests for the shell-command analyzer.

The whole point of this capability is its edge cases: command substitution, nested
subshells, wrapper chains, quoting tricks, and env-assignment prefixes must each be either
correctly matched or *conservatively degraded to ask/deny* -- never wrongly allowed.

Several classes below (`TestFlagCanonicalization`, `TestNeverPeeledWrappers`,
`TestDoubleQuotedSubstitution`, and the sed/rm cases in `TestSedScript` /
`TestDangerousCommands`) are adversarial tests for bypasses that a fixed-argv-position or
raw-token-comparison validator would miss -- see `_command.py`'s module docstring for why that
shape of check is wrong here. Each test documents, in its own docstring, what a naive
validator of that shape would have done wrong.
"""

from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pydantic_ai_harness.permission_policy._command import (
        analyze_command,
        command_matches_prefix,
        prepare_command,
    )


def _verdict(command: str) -> str | None:
    return analyze_command(prepare_command(command)).verdict


class TestConservativeGate:
    """Anything the parser cannot prove is a list of plain commands degrades to `ask`."""

    @pytest.mark.parametrize(
        'command',
        [
            'echo $(rm -rf /)',  # command substitution
            'echo `rm -rf /`',  # backtick substitution
            'echo ${HOME}',  # variable expansion
            'echo $HOME',  # bare variable
            '(ls)',  # subshell
            'ls || (pwd && echo hi)',  # nested subshell
            'ls > out.txt',  # redirection
            'ls >> out.txt',  # append redirection
            'cat < in.txt',  # input redirection
            'ls &',  # background
            'ls & pwd',  # background + more
            'ls *.py',  # glob star
            'ls foo?.py',  # glob question
            'ls [ab].py',  # glob class
            'ls ~/secret',  # home expansion
            'echo !!',  # history expansion
            'ls {a,b}',  # brace expansion
            'cat <<EOF',  # heredoc
            'ls \\',  # trailing backslash / line continuation
            "echo 'unbalanced",  # unbalanced single quote
            'echo "unbalanced',  # unbalanced double quote
            'ls # comment then rm',  # start-of-word comment
        ],
    )
    def test_complex_constructs_degrade_to_ask(self, command: str) -> None:
        prepared = prepare_command(command)
        assert prepared.confident is False
        assert prepared.segments == ()
        assert _verdict(command) == 'ask'

    def test_empty_segment_degrades(self) -> None:
        assert prepare_command('ls ;; pwd').confident is False  # ; ; -> empty middle segment
        assert prepare_command('| ls').confident is False  # leading operator -> empty segment

    def test_untokenizable_segment_degrades(self) -> None:
        # A segment that ends in a dangling backslash after stripping trailing whitespace
        # passes the char scanner but `shlex` refuses to tokenize it.
        prepared = prepare_command('ls \\ ')
        assert prepared.confident is False
        assert prepared.reason == 'could not be tokenized'

    def test_quoted_metacharacters_do_not_trip_the_gate(self) -> None:
        # Metacharacters inside quotes are literal data, not shell syntax -- except `$`/`` ` ``
        # inside double quotes, which is its own class covered by `TestDoubleQuotedSubstitution`.
        assert prepare_command('grep "a|b" file').confident is True
        assert prepare_command("grep 'a && b' file").confident is True
        assert prepare_command('echo "a#b"').confident is True  # `#` mid-token is literal
        assert prepare_command('echo a\\|b').confident is True  # escaped pipe

    def test_double_quote_escape_handling(self) -> None:
        prepared = prepare_command('echo "he said \\"hi\\""')
        assert prepared.confident is True
        assert prepared.segments == (('echo', 'he said "hi"'),)


class TestDoubleQuotedSubstitution:
    """`$(...)`, backticks, and `${...}` still expand inside double quotes, unlike `'...'`.

    A scanner that treats an entire double-quoted span as opaque literal text -- as an
    earlier version of `_consume_quote` did -- misses that a command substitution hides
    inside it: `echo "$(touch /tmp/pwn)"` would parse as a plain, safelisted `echo` call and
    could be auto-approved while the substitution runs first.
    """

    @pytest.mark.parametrize(
        'command',
        [
            'echo "$(touch /tmp/pwn)"',
            'echo "`touch /tmp/pwn`"',
            'echo "${HOME}"',
            'echo "$HOME"',
            'echo "prefix $(id) suffix"',
        ],
    )
    def test_substitution_inside_double_quotes_gates(self, command: str) -> None:
        assert prepare_command(command).confident is False
        assert _verdict(command) == 'ask'

    def test_single_quotes_still_never_expand(self) -> None:
        # Unlike double quotes, single quotes suppress all expansion in POSIX shells, so
        # these must NOT gate.
        assert prepare_command("echo '$(touch /tmp/pwn)'").confident is True
        assert prepare_command("echo '`touch /tmp/pwn`'").confident is True

    def test_escaped_dollar_in_double_quotes_does_not_gate(self) -> None:
        # `\$` inside double quotes is a literal dollar sign, not the start of an expansion.
        assert prepare_command('echo "price: \\$5"').confident is True


class TestCompoundSplitting:
    """`&&`, `||`, `;`, `|` split into segments; every segment must pass."""

    def test_operators_split(self) -> None:
        assert prepare_command('a && b || c ; d | e').segments == (
            ('a',),
            ('b',),
            ('c',),
            ('d',),
            ('e',),
        )

    def test_one_bad_segment_poisons_the_sequence(self) -> None:
        assert _verdict('ls && rm -rf /') == 'deny'
        assert _verdict('cat f | rg --pre x') == 'deny'
        assert _verdict('ls; git push') is None  # unknown segment -> no auto-allow

    def test_all_safe_segments_allow(self) -> None:
        assert _verdict('cat a | grep b | wc -l') == 'allow'
        assert _verdict('ls; pwd; whoami') == 'allow'


class TestWrapperStripping:
    def test_timeout_peels_duration(self) -> None:
        assert prepare_command('timeout 5 git status').segments == (('git', 'status'),)
        assert _verdict('timeout 5 git status') == 'allow'

    def test_timeout_with_option_degrades(self) -> None:
        assert prepare_command('timeout -s KILL 5 ls').confident is False

    def test_nice_forms(self) -> None:
        assert prepare_command('nice -n 5 ls').segments == (('ls',),)
        assert prepare_command('nice ls').segments == (('ls',),)

    def test_nice_missing_value_degrades(self) -> None:
        assert prepare_command('nice -n').confident is False

    def test_nice_other_option_degrades(self) -> None:
        assert prepare_command('nice --adjustment=5 ls').confident is False

    def test_env_plain_peels(self) -> None:
        assert prepare_command('env ls').segments == (('ls',),)

    def test_env_assignment_degrades(self) -> None:
        # `env LD_PRELOAD=evil.so ls` must never auto-allow as `ls`.
        assert prepare_command('env LD_PRELOAD=x.so ls').confident is False

    def test_env_option_degrades(self) -> None:
        assert prepare_command('env -i ls').confident is False

    def test_nested_wrappers_peel_fully(self) -> None:
        assert prepare_command('timeout 5 nice -n 1 env ls').segments == (('ls',),)

    def test_wrapper_with_no_inner_command_degrades(self) -> None:
        assert prepare_command('timeout 5').confident is False
        assert prepare_command('sudo').confident is False

    def test_interpreter_behind_wrapper_is_not_allowed(self) -> None:
        # `timeout 5 bash -c "rm -rf /"` unwraps to bash, which requires approval.
        assert _verdict('timeout 5 bash -c "rm -rf /"') == 'ask'


class TestNeverPeeledWrappers:
    """`sudo`, `doas`, and `xargs` are never peeled to an inner command.

    An earlier version peeled these the same way as `timeout`/`env`: strip the wrapper,
    classify what's left. That is unsound for both, for different reasons: `sudo`/`doas`
    change *who* the rest of the command runs as (`sudo cat /etc/shadow` would classify as
    plain, safelisted `cat`), and `xargs` builds its real argv from stdin at run time, which
    is not visible in the command string at all (`... | xargs find` can turn into
    `find ... -exec ...` even though the static text only shows `find` with no exec flag).
    """

    def test_sudo_never_peels(self) -> None:
        assert prepare_command('sudo ls').confident is False
        assert _verdict('sudo ls') == 'ask'

    def test_sudo_with_options_still_never_peels(self) -> None:
        assert prepare_command('sudo -u root ls').confident is False

    def test_doas_never_peels(self) -> None:
        assert prepare_command('doas cat /etc/shadow').confident is False
        assert _verdict('doas cat /etc/shadow') == 'ask'

    def test_xargs_never_peels(self) -> None:
        assert prepare_command('xargs rm').confident is False
        assert _verdict('xargs rm') == 'ask'

    def test_xargs_stdin_sourced_argv_cannot_smuggle_an_allow(self) -> None:
        # The static text shows only `echo` (safelisted) piped into `xargs find` (which, read
        # naively, is just `find` with no exec flag -- also allowed). The real, stdin-built
        # invocation is `find . -exec touch /tmp/pwn ;`. Both segments must degrade.
        assert _verdict("echo '. -exec touch /tmp/pwn ;' | xargs find") == 'ask'


class TestEnvAssignmentPrefix:
    def test_bare_assignment_prefix_degrades(self) -> None:
        assert prepare_command('FOO=bar ls').confident is False
        assert prepare_command('LD_PRELOAD=x.so ls').confident is False


class TestSafelist:
    def test_unconditional_safe(self) -> None:
        for cmd in ['ls', 'cat f', 'pwd', 'whoami', 'echo hi', 'wc -l', 'grep x f']:
            assert _verdict(cmd) == 'allow', cmd

    def test_unknown_command_no_opinion(self) -> None:
        assert _verdict('npm run build') is None
        assert _verdict('make test') is None

    def test_interpreters_require_approval(self) -> None:
        for cmd in ['bash -c "x"', 'sh script.sh', 'python -c "x"', 'node -e "x"', 'ruby -e x']:
            assert _verdict(cmd) == 'ask', cmd

    def test_interpreters_require_approval_via_absolute_path(self) -> None:
        # A validator keyed on the literal argv[0] string (`cmd == 'bash'`) never recognizes
        # `/bin/bash` as the interpreter it is; the executable's basename does.
        for cmd in ['/bin/bash -c "x"', '/usr/bin/python3 -c "x"', '/opt/homebrew/bin/node -e "x"']:
            assert _verdict(cmd) == 'ask', cmd


class TestDangerousCommands:
    def test_rm_recursive_or_force_denied(self) -> None:
        for cmd in ['rm -rf /', 'rm -r dir', 'rm -f x', 'rm --recursive d', 'rm --force x']:
            v = analyze_command(prepare_command(cmd))
            assert v.verdict == 'deny', cmd
            assert v.retryable is False

    def test_rm_plain_asks(self) -> None:
        assert _verdict('rm file.txt') == 'ask'

    def test_rm_bundled_short_flags_denied(self) -> None:
        # A denylist of literal combo strings (`{'-rf', '-fr', '-rF'}`) only ever catches the
        # exact combos it enumerated. `-vrf` bundles `-v` (harmless) with `-r`/`-f`
        # (destructive) and was not one of them, so it fell through to a merely-retryable
        # `ask` instead of the hard `deny` a recursive-force delete requires. Decomposing the
        # bundle one character at a time (safe for `rm`, whose short options are all
        # boolean -- see `_safelist.py`) closes every ordering/combination, not just the ones
        # someone thought to enumerate.
        for cmd in ['rm -vrf x', 'rm -frv x', 'rm -Rvf x', 'rm -iRf x']:
            v = analyze_command(prepare_command(cmd))
            assert v.verdict == 'deny', cmd
            assert v.retryable is False

    def test_rm_dash_dash_ends_option_parsing(self) -> None:
        # After `--`, `-rf` is a literal filename, not the recursive+force flags.
        assert _verdict('rm -- -rf') == 'ask'

    def test_dangerous_command_denied_via_absolute_path(self) -> None:
        assert analyze_command(prepare_command('/bin/rm -rf /')).verdict == 'deny'


class TestFlagCanonicalization:
    """`--flag=value` must compare equal to `--flag` in every denylist check.

    A raw-token comparison (`token in DENYLIST`) only matches a flag spelled with a space
    before its value; the equally-valid `--flag=value` GNU spelling never appears in the
    literal set and falls through to `allow`. Every flag comparison in `_command.py` goes
    through `_flag_key`, which strips the `=value` suffix before comparing, so both spellings
    are treated the same.
    """

    def test_rg_pre_equals_form_denied(self) -> None:
        assert _verdict('rg --pre=/bin/sh pattern file') == 'deny'

    def test_rg_hostname_bin_equals_form_denied(self) -> None:
        assert _verdict('rg --hostname-bin=/bin/sh pattern file') == 'deny'

    def test_find_exec_equals_form_denied(self) -> None:
        assert _verdict('find . -exec=/bin/sh') == 'deny'

    def test_base64_output_equals_form_asks(self) -> None:
        assert _verdict('base64 --output=/tmp/x in') == 'ask'


class TestFlagDenylists:
    def test_find_exec_delete_denied(self) -> None:
        # Note: real `-exec ... {} ;` also trips the gate via `{}`/`;`; here we use parseable
        # forms so the *flag denylist* branch itself is exercised.
        for cmd in ['find . -delete', 'find . -exec rm', 'find . -fprintf out fmt', 'find . -okdir x']:
            assert _verdict(cmd) == 'deny', cmd

    def test_find_read_only_allowed(self) -> None:
        assert _verdict('find . -name x.py -type f') == 'allow'

    def test_rg_exec_flags_denied(self) -> None:
        for cmd in ['rg --pre pp x', 'rg --hostname-bin b x', 'rg -z x', 'rg --search-zip x']:
            assert _verdict(cmd) == 'deny', cmd

    def test_rg_plain_allowed(self) -> None:
        assert _verdict('rg pattern src') == 'allow'

    def test_base64_write_asks(self) -> None:
        assert _verdict('base64 -o out.bin in') == 'ask'
        assert _verdict('base64 --output out in') == 'ask'

    def test_base64_read_allowed(self) -> None:
        assert _verdict('base64 in.txt') == 'allow'


class TestGit:
    def test_readonly_subcommands_allowed(self) -> None:
        for cmd in ['git status', 'git status -sb', 'git log --oneline', 'git diff HEAD', 'git show abc']:
            assert _verdict(cmd) == 'allow', cmd

    def test_word_boundary_prefix(self) -> None:
        # `git status-evil` must NOT be treated as `git status`.
        assert _verdict('git status-evil') is None

    def test_unsafe_global_options_ask(self) -> None:
        assert _verdict('git -C /etc status') == 'ask'
        assert _verdict('git -p log') == 'ask'

    def test_config_env_form_option_ask(self) -> None:
        assert _verdict('git --config-env=X=Y status') == 'ask'

    def test_bare_git_no_opinion(self) -> None:
        assert _verdict('git') is None
        assert _verdict('git --no-pager') is None  # benign option, no subcommand -> no opinion

    def test_unsafe_global_option_without_subcommand_asks(self) -> None:
        assert _verdict('git -c') == 'ask'  # `-c` is an unsafe global option

    def test_mutating_subcommands_no_opinion(self) -> None:
        assert _verdict('git push') is None
        assert _verdict('git commit -m x') is None

    def test_branch_listing_allowed(self) -> None:
        assert _verdict('git branch --list') == 'allow'
        assert _verdict('git branch -a') == 'allow'
        assert _verdict('git branch --format=short') == 'allow'

    def test_branch_create_or_delete_asks(self) -> None:
        assert _verdict('git branch newthing') == 'ask'  # positional -> may create
        assert _verdict('git branch -d old') == 'ask'  # no read-only option present

    def test_readonly_subcommand_output_flag_asks(self) -> None:
        # `git diff`/`log`/`show` are read-only, but `--output`/`-o` writes a caller-chosen
        # file; checking only `git branch`'s own options (as an earlier version did) left
        # the other read-only subcommands' own write-capable flags unchecked entirely.
        assert _verdict('git diff --output=/tmp/x HEAD') == 'ask'
        assert _verdict('git log --output /tmp/x') == 'ask'
        assert _verdict('git show --output=/tmp/x abc') == 'ask'


class TestSedScript:
    def test_only_dash_n_print_allowed(self) -> None:
        assert _verdict('sed -n 2p file') == 'allow'
        assert _verdict('sed -n 1,5p file') == 'allow'

    def test_other_sed_asks(self) -> None:
        assert _verdict('sed -i s/a/b/ file') == 'ask'
        assert _verdict('sed s/a/b/ file') == 'ask'
        assert _verdict('sed -n 2d file') == 'ask'  # not a print
        assert _verdict('sed -n') == 'ask'  # too short

    def test_extra_script_after_safe_prefix_asks(self) -> None:
        # `_is_safe_sed` used to check only argv[1]/argv[2] and never looked past them, so a
        # second, destructive `-e` script tacked on after the recognized-safe `-n {N}p`
        # prefix passed unnoticed and the whole call was auto-approved. Requiring the *entire*
        # remaining argv to be plain filenames closes this: any extra flag or script anywhere
        # fails the check.
        assert _verdict('sed -n 2p -e "3w /tmp/out"') == 'ask'
        assert _verdict('sed -n 2p -i') == 'ask'
        assert _verdict('sed -n 2p -n 3p') == 'ask'  # a second script, not just a second flag


class TestPrefixMatching:
    def test_word_boundary(self) -> None:
        assert command_matches_prefix('git status', prepare_command('git status -sb')) is True
        assert command_matches_prefix('git status', prepare_command('git status-evil')) is False
        assert command_matches_prefix('git status', prepare_command('git push')) is False

    def test_all_segments_must_match(self) -> None:
        assert command_matches_prefix('git', prepare_command('git status && git log')) is True
        assert command_matches_prefix('git', prepare_command('git status && rm x')) is False

    def test_empty_prefix_matches_any_confident_command(self) -> None:
        assert command_matches_prefix('', prepare_command('anything here')) is True

    def test_unconfident_never_matches(self) -> None:
        assert command_matches_prefix('echo', prepare_command('echo $(x)')) is False

    def test_prefix_matches_wrapper_stripped_inner(self) -> None:
        assert command_matches_prefix('git status', prepare_command('timeout 5 git status')) is True
