"""Red-team tests for the shell-command analyzer.

The whole point of this capability is its edge cases: command substitution, nested
subshells, wrapper chains, quoting tricks, and env-assignment prefixes must each be either
correctly matched or *conservatively degraded to ask/deny* -- never wrongly allowed.
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
        # Metacharacters inside quotes are literal data, not shell syntax.
        assert prepare_command('grep "a|b" file').confident is True
        assert prepare_command("grep 'a && b' file").confident is True
        assert prepare_command('echo "a#b"').confident is True  # `#` mid-token is literal
        assert prepare_command('echo a\\|b').confident is True  # escaped pipe

    def test_double_quote_escape_handling(self) -> None:
        prepared = prepare_command('echo "he said \\"hi\\""')
        assert prepared.confident is True
        assert prepared.segments == (('echo', 'he said "hi"'),)


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

    def test_sudo_peels_but_options_degrade(self) -> None:
        assert prepare_command('sudo ls').segments == (('ls',),)
        assert prepare_command('sudo -u root ls').confident is False

    def test_xargs_peels_inner_command(self) -> None:
        assert prepare_command('xargs rm').segments == (('rm',),)
        assert _verdict('xargs rm') == 'ask'  # rm still needs approval after unwrapping

    def test_xargs_option_degrades(self) -> None:
        assert prepare_command('xargs -0 rm').confident is False

    def test_nested_wrappers_peel_fully(self) -> None:
        assert prepare_command('timeout 5 nice -n 1 env ls').segments == (('ls',),)

    def test_wrapper_with_no_inner_command_degrades(self) -> None:
        assert prepare_command('timeout 5').confident is False
        assert prepare_command('sudo').confident is False

    def test_interpreter_behind_wrapper_is_not_allowed(self) -> None:
        # `timeout 5 bash -c "rm -rf /"` unwraps to bash, which requires approval.
        assert _verdict('timeout 5 bash -c "rm -rf /"') == 'ask'


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


class TestDangerousCommands:
    def test_rm_recursive_or_force_denied(self) -> None:
        for cmd in ['rm -rf /', 'rm -r dir', 'rm -f x', 'rm --recursive d', 'rm --force x']:
            v = analyze_command(prepare_command(cmd))
            assert v.verdict == 'deny', cmd
            assert v.retryable is False

    def test_rm_plain_asks(self) -> None:
        assert _verdict('rm file.txt') == 'ask'


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


class TestSedScript:
    def test_only_dash_n_print_allowed(self) -> None:
        assert _verdict('sed -n 2p file') == 'allow'
        assert _verdict('sed -n 1,5p file') == 'allow'

    def test_other_sed_asks(self) -> None:
        assert _verdict('sed -i s/a/b/ file') == 'ask'
        assert _verdict('sed s/a/b/ file') == 'ask'
        assert _verdict('sed -n 2d file') == 'ask'  # not a print
        assert _verdict('sed -n') == 'ask'  # too short


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
