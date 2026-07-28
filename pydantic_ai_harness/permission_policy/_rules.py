"""The allow/ask/deny rule model and its evaluation.

Two matcher philosophies are combined, following the mining doc's recommendation
(section 2.1, "For our implementation"):

- **opencode's last-match-wins ordering** for user rules: later rules override earlier ones,
  so specificity comes from position (`[deny 'git *', allow 'git status']` => `git status`
  is allowed). No action-precedence lattice.
- **Codex's most-restrictive-wins** to merge the user-rule verdict with the built-in
  command-safety verdict: `deny > ask > allow`. This is what makes the flag denylist
  "override allows" -- a user `allow git *` cannot green-light `git -C /etc ...`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from ._command import PreparedCommand, Verdict, analyze_command, command_matches_prefix

_RANK: dict[Verdict, int] = {'allow': 0, 'ask': 1, 'deny': 2}


@dataclass(frozen=True)
class Rule:
    """One allow/ask/deny rule over tool calls.

    A rule matches a tool call when the tool name matches `tool` (an `fnmatch` glob) **and**
    the argument matcher matches. The argument matcher is, in order of precedence:

    - `command`: a whitespace-separated *prefix* matched against a shell tool's command with
      word-boundary semantics. Every segment of a compound command must start with the prefix
      (`git status` matches `git status -sb` but not `git status-evil` or `git push`).
      Only meaningful for shell-class tools; a non-shell call never matches a `command` rule.
    - `args`: an arbitrary predicate over the tool's argument dict. Not serializable; use for
      non-shell tools that need argument inspection.
    - neither: the rule matches the tool regardless of arguments (a "bare-name" rule).

    Rules are evaluated **last-match-wins**: the last rule in the list that matches decides.
    """

    verdict: Verdict
    tool: str = '*'
    command: str | None = None
    args: Callable[[dict[str, Any]], bool] | None = None

    def is_bare(self) -> bool:
        """Whether this rule ignores arguments (matches on tool name alone)."""
        return self.command is None and self.args is None


@dataclass(frozen=True)
class Decision:
    """The resolved verdict for a tool call, with an explanation and the deciding source."""

    verdict: Verdict
    reason: str
    retryable: bool
    source: str  # 'rule', 'command-safety', or 'default'


def _rule_matches(rule: Rule, tool_name: str, args: dict[str, Any], prepared: PreparedCommand | None) -> bool:
    if not fnmatchcase(tool_name, rule.tool):
        return False
    if rule.command is not None:
        if prepared is None:
            return False
        return command_matches_prefix(rule.command, prepared)
    if rule.args is not None:
        return rule.args(args)
    return True


def last_matching_rule(
    rules: list[Rule], tool_name: str, args: dict[str, Any], prepared: PreparedCommand | None
) -> Rule | None:
    """Return the last rule that matches this call, or `None` (opencode last-match-wins)."""
    match: Rule | None = None
    for rule in rules:
        if _rule_matches(rule, tool_name, args, prepared):
            match = rule
    return match


def bare_name_verdict(rules: list[Rule], tool_name: str) -> Verdict | None:
    """Last-match-wins verdict considering only bare-name rules (no argument matcher).

    Used to decide whether `deny_removes_tool` should drop a tool entirely (the Claude Code
    bare-name-deny pattern): a tool is only removed when it is denied *regardless* of
    arguments.
    """
    verdict: Verdict | None = None
    for rule in rules:
        if rule.is_bare() and fnmatchcase(tool_name, rule.tool):
            verdict = rule.verdict
    return verdict


def resolve(
    rules: list[Rule],
    tool_name: str,
    args: dict[str, Any],
    prepared: PreparedCommand | None,
    *,
    default: Verdict,
    analyze_shell: bool,
) -> Decision:
    """Combine the user-rule channel and the command-safety channel into one decision."""
    candidates: list[Decision] = []

    rule = last_matching_rule(rules, tool_name, args, prepared)
    if rule is not None:
        # A user deny is presented as retryable-with-justification; only the built-in hard
        # denies (dangerous commands) are flagged never-allowed (`retryable=False`).
        candidates.append(
            Decision(
                rule.verdict,
                reason=f'matched policy rule (tool={rule.tool!r}, verdict={rule.verdict!r})',
                retryable=True,
                source='rule',
            )
        )

    if analyze_shell and prepared is not None:
        cmd = analyze_command(prepared)
        if cmd.verdict is not None:
            candidates.append(
                Decision(
                    cmd.verdict,
                    reason=cmd.reason or 'built-in command-safety analysis',
                    retryable=cmd.retryable,
                    source='command-safety',
                )
            )

    if not candidates:
        return Decision(default, reason='no rule matched; using the default verdict', retryable=True, source='default')

    winner = candidates[0]
    for candidate in candidates[1:]:
        if _RANK[candidate.verdict] > _RANK[winner.verdict]:
            winner = candidate
    return winner


__all__ = ['Decision', 'Rule', 'bare_name_verdict', 'last_matching_rule', 'resolve']
