"""Permission policy: an allow/ask/deny rule engine over tool calls.

See `README.md` in this package for the full model, the command-analysis mechanics, and the
red-team edge cases the test suite locks down.
"""

from ._capability import DEFAULT_SHELL_TOOLS, ContextPolicy, PermissionPolicy, PermissionRequest
from ._command import PreparedCommand, Verdict, analyze_command, prepare_command
from ._rules import Decision, Rule, resolve
from ._safelist import BANNED_PREFIX_SUGGESTIONS

__all__ = [
    'BANNED_PREFIX_SUGGESTIONS',
    'DEFAULT_SHELL_TOOLS',
    'ContextPolicy',
    'Decision',
    'PermissionPolicy',
    'PermissionRequest',
    'PreparedCommand',
    'Rule',
    'Verdict',
    'analyze_command',
    'prepare_command',
    'resolve',
]
