"""Tests for the rule model: ordering, matching, and channel combination."""

from __future__ import annotations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pydantic_ai_harness.permission_policy._command import prepare_command
    from pydantic_ai_harness.permission_policy._rules import (
        Rule,
        bare_name_verdict,
        last_matching_rule,
        resolve,
    )


def _resolve(rules: list[Rule], tool: str, command: str | None = None, *, default: str = 'ask', analyze: bool = True):
    prepared = prepare_command(command) if command is not None else None
    args = {'command': command} if command is not None else {}
    return resolve(rules, tool, args, prepared, default=default, analyze_shell=analyze)  # type: ignore[arg-type]


class TestLastMatchWins:
    def test_later_rule_overrides_earlier(self) -> None:
        # opencode's `{orchestrator-*: deny, orchestrator-fast: allow}` => allow.
        rules = [Rule('deny', tool='orchestrator-*'), Rule('allow', tool='orchestrator-fast')]
        assert last_matching_rule(rules, 'orchestrator-fast', {}, None).verdict == 'allow'  # type: ignore[union-attr]
        assert last_matching_rule(rules, 'orchestrator-slow', {}, None).verdict == 'deny'  # type: ignore[union-attr]

    def test_no_match_returns_none(self) -> None:
        assert last_matching_rule([Rule('allow', tool='foo')], 'bar', {}, None) is None

    def test_command_prefix_last_match(self) -> None:
        rules = [Rule('deny', tool='*', command='git'), Rule('allow', tool='*', command='git status')]
        assert _resolve(rules, 'run_command', 'git status').verdict == 'allow'
        assert _resolve(rules, 'run_command', 'git push').verdict == 'deny'


class TestArgMatcher:
    def test_predicate_matches_on_args(self) -> None:
        rules = [Rule('deny', tool='write_file', args=lambda a: a.get('path', '').startswith('/etc'))]
        prepared = None
        assert (
            resolve(rules, 'write_file', {'path': '/etc/passwd'}, prepared, default='allow', analyze_shell=True).verdict
            == 'deny'
        )
        assert (
            resolve(rules, 'write_file', {'path': '/tmp/x'}, prepared, default='allow', analyze_shell=True).verdict
            == 'allow'
        )

    def test_command_rule_needs_shell_command(self) -> None:
        # A `command` rule never matches a call with no prepared command.
        rules = [Rule('allow', tool='*', command='git')]
        assert last_matching_rule(rules, 'some_tool', {}, None) is None


class TestChannelCombination:
    def test_command_safety_overrides_user_allow(self) -> None:
        # User allows all git; the flag denylist still forces `git -C` to ask (most-restrictive).
        rules = [Rule('allow', tool='run_command', command='git')]
        assert _resolve(rules, 'run_command', 'git -C /etc status').verdict == 'ask'

    def test_dangerous_command_overrides_user_allow(self) -> None:
        rules = [Rule('allow', tool='run_command', command='rm')]
        decision = _resolve(rules, 'run_command', 'rm -rf /')
        assert decision.verdict == 'deny'
        assert decision.retryable is False  # never-allowed

    def test_user_deny_overrides_safelist_allow(self) -> None:
        rules = [Rule('deny', tool='run_command', command='ls')]
        decision = _resolve(rules, 'run_command', 'ls -la')
        assert decision.verdict == 'deny'
        assert decision.retryable is True  # user deny -> may re-request

    def test_safelist_allows_without_any_rule(self) -> None:
        assert _resolve([], 'run_command', 'ls -la').verdict == 'allow'

    def test_user_rule_extends_safelist_for_unknown_command(self) -> None:
        rules = [Rule('allow', tool='run_command', command='npm')]
        assert _resolve(rules, 'run_command', 'npm run build').verdict == 'allow'

    def test_conservative_gate_caps_broad_allow(self) -> None:
        # Even `allow *` cannot green-light a command substitution.
        rules = [Rule('allow', tool='run_command')]
        assert _resolve(rules, 'run_command', 'echo $(rm -rf /)').verdict == 'ask'

    def test_analyze_shell_disabled_skips_command_safety(self) -> None:
        rules = [Rule('allow', tool='run_command')]
        # With analysis off, the dangerous command is governed by rules alone.
        assert _resolve(rules, 'run_command', 'rm -rf /', analyze=False).verdict == 'allow'


class TestDefault:
    def test_default_when_no_channel_has_an_opinion(self) -> None:
        assert _resolve([], 'run_command', 'npm run build').verdict == 'ask'
        assert _resolve([], 'run_command', 'npm run build', default='deny').verdict == 'deny'

    def test_default_applies_to_non_shell_tool(self) -> None:
        assert resolve([], 'weird_tool', {}, None, default='deny', analyze_shell=True).verdict == 'deny'
        assert resolve([], 'weird_tool', {}, None, default='allow', analyze_shell=True).verdict == 'allow'


class TestBareNameVerdict:
    def test_only_bare_rules_considered(self) -> None:
        rules = [
            Rule('deny', tool='run_command'),  # bare
            Rule('allow', tool='run_command', command='ls'),  # not bare -> ignored here
        ]
        assert bare_name_verdict(rules, 'run_command') == 'deny'

    def test_last_bare_rule_wins(self) -> None:
        rules = [Rule('deny', tool='*'), Rule('allow', tool='run_command')]
        assert bare_name_verdict(rules, 'run_command') == 'allow'

    def test_none_when_no_bare_rule(self) -> None:
        assert bare_name_verdict([Rule('deny', tool='x', command='y')], 'x') is None
