"""The `PermissionPolicy` capability: an allow/ask/deny engine over tool calls."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolApprovalResult,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)

from ._command import PreparedCommand, prepare_command
from ._rules import Decision, Rule, bare_name_verdict, resolve

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart

# Marker key under which we stash our own metadata on an `ApprovalRequired`, so
# `handle_deferred_tool_calls` only resolves the asks *this* capability raised and leaves
# other capabilities' deferred calls untouched (good composition citizen).
_MARKER = 'pydantic_ai_harness.permission_policy'

# Provenance marker stamped onto the `metadata` of a `deny`'s `ToolReturnPart`, so the app
# (and any provenance-aware consumer) can tell a *harness* denial apart from a genuine tool
# result. Mirrors the pattern of core PR #6319's `SYNTHESIZED_TOOL_RETURN_METADATA_KEY`.
#
# NOTE: `wrap_tool_execute` cannot set `ToolReturnPart.outcome` — the only value return that
# reaches `outcome='denied'` is a `ToolDenied` on the *deferred* approval path (core
# `_tool_execution._call_tool`); a synchronous `deny` verdict has no such channel, and
# `outcome` is not serialized to the wire regardless. So the denial reaches the model as a
# structurally-successful `ToolReturnPart` distinguished only by its prose. We keep an
# explicit, unambiguous prose marker (below) load-bearing for the model, and the `metadata`
# marker load-bearing for the app. When core grows a provenance channel that renders to the
# wire (pydantic-ai#6404), route this through it.
_DENY_METADATA_KEY = 'pydantic_ai_harness_permission_denied'

# Explicit prose prefix so the *model* can attribute the denial to the harness policy layer,
# not to the tool itself failing.
_DENY_PROSE_MARKER = '[permission-policy]'

# Default set of tool names treated as shell-class (their command argument is analyzed).
DEFAULT_SHELL_TOOLS: frozenset[str] = frozenset(
    {'run_command', 'start_command', 'bash', 'shell', 'run_shell_command', 'execute_command', 'exec'}
)

_ESCALATION_NOTE = (
    ' Note: calls to this tool are checked against a permission policy. If a call is denied, '
    'you may restate it once with a brief justification when the action is genuinely necessary; '
    'do not retry a command that was reported as never allowed -- use a safer alternative instead.'
)


@dataclass
class PermissionRequest:
    """The context handed to an `on_ask` handler for a call that needs approval."""

    tool_name: str
    """Name of the tool being called."""
    args: dict[str, Any]
    """The (validated) arguments of the call."""
    command: str | None
    """The shell command string, if this is a shell-class tool call."""
    reason: str
    """Why the policy routed this call to `ask`."""


# An `on_ask` handler returns True to approve, False to deny with the default message, or a
# string to deny with that message. May be sync or async.
OnAsk = Callable[[RunContext[AgentDepsT], PermissionRequest], 'bool | str | Awaitable[bool | str]']


@dataclass
class PermissionPolicy(AbstractCapability[AgentDepsT]):
    """Allow / ask / deny rule engine over tool calls, evaluated as each tool runs.

    Rules are an **ordered** list; the **last** matching rule wins (opencode semantics). For
    shell-class tools the argument matcher operates on the *command*, with compound-command
    splitting, a conservative-parse gate, wrapper stripping, a read-only safelist, and a flag
    denylist -- so a broad `allow` can never green-light a command the engine cannot prove
    safe. See this capability's README for the full model.

    Verdicts:

    - **allow** -- the tool runs.
    - **deny** -- the call is blocked and an explanatory message is returned to the model
      (it is told whether re-requesting with justification can help, or the action is never
      allowed). With `deny_removes_tool=True`, a tool denied *regardless of arguments* is
      also removed from the model's toolset entirely (the Claude Code bare-name-deny pattern).
    - **ask** -- the call is routed through Pydantic AI's deferred-approval machinery by
      raising [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired]. If an `on_ask`
      handler is set, this capability resolves the approval inline; otherwise the call
      surfaces as [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] for the
      caller to resolve (e.g. with a real human, or a
      [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] capability).

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.permission_policy import PermissionPolicy, Rule

    policy = PermissionPolicy[None](
        rules=[
            Rule('deny', tool='run_command', command='git push'),
            Rule('allow', tool='run_command', command='git status'),
        ],
    )
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[policy])
    ```
    """

    rules: list[Rule] = field(default_factory=list[Rule])
    """Ordered allow/ask/deny rules. The **last** matching rule wins."""

    default_verdict: str = 'ask'
    """Verdict when no rule matches and command-safety analysis has no opinion. `'ask'` by
    default (fail-safe); set `'allow'` for an allowlist-by-exception posture or `'deny'` to
    block everything not explicitly allowed."""

    shell_tools: frozenset[str] = DEFAULT_SHELL_TOOLS
    """Tool names whose command argument is analyzed as a shell command."""

    command_arg: str = 'command'
    """Name of the argument holding the shell command for shell-class tools."""

    analyze_shell_commands: bool = True
    """Whether to run the built-in command-safety analysis (conservative gate, read-only
    safelist, flag denylist, dangerous-command detection) for shell-class tools. When
    `False`, shell tools are governed by `rules` and `default_verdict` alone."""

    deny_removes_tool: bool = False
    """When `True`, a tool that is denied *regardless of arguments* (by a bare-name rule) is
    removed from the model's toolset entirely, rather than denied per-call."""

    add_escalation_note: bool = True
    """When `True`, append a short note to guarded tools' descriptions telling the model it
    may re-request a denied call with justification (Codex-style escalation protocol)."""

    on_ask: OnAsk[AgentDepsT] | None = None
    """Optional handler that resolves `ask` verdicts inline. Receives the `RunContext` and a
    [`PermissionRequest`][pydantic_ai_harness.permission_policy.PermissionRequest];
    return `True` to approve, `False` to deny with the default message, or a string to deny
    with that message. When `None`, `ask` calls surface as `DeferredToolRequests` instead."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not serializable: rules may carry Python callables (`args`, `on_ask`)."""
        return None

    # --- helpers -----------------------------------------------------------------

    def _prepare(self, tool_name: str, args: dict[str, Any]) -> tuple[PreparedCommand | None, str | None]:
        """Return the prepared shell command for a shell-class call (or `None`)."""
        if tool_name not in self.shell_tools:
            return None, None
        command = args.get(self.command_arg)
        if not isinstance(command, str):
            return None, None
        return prepare_command(command), command

    def _is_guarded(self, tool_name: str) -> bool:
        from fnmatch import fnmatchcase

        if tool_name in self.shell_tools:
            return True
        return any(fnmatchcase(tool_name, rule.tool) for rule in self.rules)

    def _decide(self, tool_name: str, args: dict[str, Any]) -> tuple[Decision, str | None]:
        prepared, command = self._prepare(tool_name, args)
        verdict = self.default_verdict if self.default_verdict in ('allow', 'ask', 'deny') else 'ask'
        decision = resolve(
            self.rules,
            tool_name,
            args,
            prepared,
            default=verdict,  # pyright: ignore[reportArgumentType]  -- narrowed to the Verdict literals above
            analyze_shell=self.analyze_shell_commands,
        )
        return decision, command

    def _deny_message(self, tool_name: str, decision: Decision) -> str:
        head = f'{_DENY_PROSE_MARKER} Permission denied for `{tool_name}`: {decision.reason}.'
        if decision.retryable:
            tail = (
                ' If this action is genuinely required, you may restate the request once with a '
                'brief justification. Otherwise, choose a different approach.'
            )
        else:
            tail = ' This will not be allowed even with justification; use a safer alternative.'
        return head + tail

    def _deny_return(self, tool_name: str, decision: Decision) -> ToolReturn:
        """Build the `deny` tool return.

        Explicit prose marker for the model + `metadata` provenance marker for the app. See
        `_DENY_METADATA_KEY` for why `outcome` can't be set from here.
        """
        return ToolReturn(
            return_value=self._deny_message(tool_name, decision),
            metadata={_DENY_METADATA_KEY: True},
        )

    # --- hooks -------------------------------------------------------------------

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Optionally drop bare-name-denied tools and annotate guarded tool descriptions."""
        result: list[ToolDefinition] = []
        for tool_def in tool_defs:
            if self.deny_removes_tool and bare_name_verdict(self.rules, tool_def.name) == 'deny':
                continue
            if self.add_escalation_note and self._is_guarded(tool_def.name):
                description = (tool_def.description or '').rstrip() + _ESCALATION_NOTE
                tool_def = replace(tool_def, description=description)
            result.append(tool_def)
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Evaluate the policy and allow / deny / ask before the tool runs."""
        decision, command = self._decide(tool_def.name, args)
        if decision.verdict == 'allow':
            return await handler(args)
        if decision.verdict == 'deny':
            return self._deny_return(tool_def.name, decision)
        # ask
        if ctx.tool_call_approved:
            return await handler(args)
        raise ApprovalRequired(metadata={_MARKER: {'reason': decision.reason, 'command': command, 'args': dict(args)}})

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        """Resolve the `ask` approvals this capability raised, using `on_ask`."""
        if self.on_ask is None:
            return None
        approvals: dict[str, bool | DeferredToolApprovalResult] = {}
        for approval in requests.approvals:
            meta = requests.metadata.get(approval.tool_call_id, {}).get(_MARKER)
            if meta is None:
                continue
            request = PermissionRequest(
                tool_name=approval.tool_name,
                args=meta.get('args', {}),
                command=meta.get('command'),
                reason=meta.get('reason', ''),
            )
            outcome = self.on_ask(ctx, request)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if outcome is True:
                approvals[approval.tool_call_id] = ToolApproved()
            elif outcome is False:
                approvals[approval.tool_call_id] = ToolDenied(
                    message=f'Permission denied for `{approval.tool_name}`: not approved.'
                )
            else:
                approvals[approval.tool_call_id] = ToolDenied(message=outcome)
        if not approvals:
            return None
        return DeferredToolResults(approvals=approvals)
