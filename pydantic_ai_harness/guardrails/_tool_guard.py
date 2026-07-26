"""Tool guardrail capability.

`ToolGuard` is the third guardrail edge. `InputGuard` screens the user prompt
and `OutputGuard` screens the agent output, but neither sees a tool call:
`InputGuard` evaluates only the first model request, so everything a tool sends
out and everything a tool brings back passes unchecked. That matters because
tool results are where untrusted content enters an agent loop -- fetched pages,
file contents, MCP server responses.

`ToolGuard` inspects both sides of a tool call with the same
[`GuardResult`][pydantic_ai_harness.GuardResult] vocabulary the other two
guards use:

- `guard` sees the validated arguments before the tool runs.
- `result_guard` sees the tool's return value before it reaches the model.

Verdicts map onto Pydantic AI control flow rather than a parallel mechanism:
`block` uses [`SkipToolExecution`][pydantic_ai.exceptions.SkipToolExecution] so
the refusal becomes the tool result and the agent can recover, `retry` raises
[`ModelRetry`][pydantic_ai.exceptions.ModelRetry], and `approve` raises
[`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] so the call joins
the run's deferred approval flow.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import ApprovalRequired, ModelRetry, SkipToolExecution, UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import assert_never

from pydantic_ai_harness.guardrails._shared import (
    GuardOutcome,
    evaluate,
    trace_approval,
    trace_block,
    trace_redaction,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition


_DEFAULT_ARGS_BLOCK_MESSAGE = 'Tool call blocked by tool guardrail.'
_DEFAULT_RESULT_BLOCK_MESSAGE = 'Tool result blocked by tool guardrail.'
_DEFAULT_RETRY_MESSAGE = 'Tool call rejected by tool guardrail.'

_ARGS_DIRECTION = 'tool args'
_RESULT_DIRECTION = 'tool result'


def _is_arguments(value: object) -> TypeGuard[Mapping[str, Any]]:
    """Whether a `replace` verdict carries tool arguments.

    Only the mapping shape is checked. A mapping with non-string keys fails
    later against the tool's own signature, in an error that names the tool.
    """
    return isinstance(value, Mapping)


@dataclass(frozen=True, kw_only=True)
class ToolCallInfo:
    """The tool call an argument guard inspects."""

    name: str
    """Name of the tool the model called."""

    args: Mapping[str, Any]
    """Validated arguments, after the tool's own schema validation."""

    tool_call_id: str
    """Identifier of this call, for correlating with the run's messages."""


@dataclass(frozen=True, kw_only=True)
class ToolResultInfo(ToolCallInfo):
    """The tool call and its return value, as a result guard sees them."""

    result: object
    """What the tool returned, before any other capability transforms it."""


ToolGuardFunc = (
    Callable[[ToolCallInfo], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], ToolCallInfo], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `ToolGuard(guard=...)`.

The callable receives a [`ToolCallInfo`][pydantic_ai_harness.ToolCallInfo] and
returns `True` / `GuardResult`. It may optionally take a
[`RunContext`][pydantic_ai.tools.RunContext] as a first argument, and may be
sync or async. Raising an exception is treated as a hard failure and propagates
up to the caller.
"""

ToolResultGuardFunc = (
    Callable[[ToolResultInfo], GuardOutcome | Awaitable[GuardOutcome]]
    | Callable[[RunContext[AgentDepsT], ToolResultInfo], GuardOutcome | Awaitable[GuardOutcome]]
)
"""Signature of the callable passed to `ToolGuard(result_guard=...)`.

The callable receives a [`ToolResultInfo`][pydantic_ai_harness.ToolResultInfo],
whose `result` is the tool's return value unchanged -- no stringification, so a
structured result arrives as the object the tool produced.
"""


@dataclass
class ToolGuard(AbstractCapability[AgentDepsT]):
    """Validate tool arguments before execution and tool results after it.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import GuardResult, ToolCallInfo, ToolGuard


    def no_writes_outside_workspace(call: ToolCallInfo) -> GuardResult:
        path = call.args.get('path', '')
        if call.name == 'write_file' and not str(path).startswith('/workspace/'):
            return GuardResult.block(f'{path} is outside the workspace.')
        return GuardResult.allow()


    agent = Agent('openai:gpt-5.4', capabilities=[ToolGuard(guard=no_writes_outside_workspace)])
    ```

    Verdicts, by stage:

    | Outcome | `guard` (arguments) | `result_guard` (result) |
    |---|---|---|
    | `allow` | run the tool | return the result unchanged |
    | `block` | skip execution; the refusal message becomes the tool result | the refusal message replaces the result |
    | `replace` | run the tool with substituted arguments (a mapping) | substitute a sanitized result |
    | `retry` | ask the model to redo the call | ask the model to redo the call |
    | `approve` | defer the call for human approval | not valid; raises `UserError` |

    `block` is the graceful path on both stages: the agent sees the refusal text
    where it expected a tool result and can choose another approach. To fail the
    run instead, raise [`ToolBlocked`][pydantic_ai_harness.ToolBlocked] (or any
    exception) from the guard.

    Either guard may take a [`RunContext`][pydantic_ai.tools.RunContext] as a
    first parameter when it needs run state, such as `deps` for role-aware
    policy. The parameter is detected from the signature.

    Scope: output tools do not fire tool-execution hooks in Pydantic AI, so a
    guard never sees the call that produces the agent's structured output --
    use `OutputGuard` for that.

    Ordering: declares `position='innermost'`. Argument hooks run outermost
    first, so the guard sees the arguments every other capability has already
    finished modifying; result hooks run innermost first, so the guard sees the
    raw tool result before a capability such as `OverflowingToolOutput`
    truncates or offloads it.
    """

    guard: ToolGuardFunc[AgentDepsT] | None = None
    """Callable that decides what to do with a tool call before it executes."""

    result_guard: ToolResultGuardFunc[AgentDepsT] | None = None
    """Callable that decides what to do with a tool result before the model sees it."""

    tools: Sequence[str] | None = None
    """Restrict both guards to these tool names. `None` guards every tool."""

    hidden: Sequence[str] = field(default_factory=tuple)
    """Tool names to withhold from the model entirely.

    Hiding differs from blocking: a hidden tool is dropped from the tool
    definitions sent to the model, so it costs no tokens and the model never
    tries to call it. A blocked tool stays visible and the model learns it was
    refused. Hiding is a static name list; for policy that depends on `deps` or
    the arguments, use `guard`.
    """

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost: see the final arguments, and the tool result before other capabilities rewrite it."""
        return CapabilityOrdering(position='innermost')

    def _guards(self, tool_name: str) -> bool:
        """Whether this guard applies to `tool_name`."""
        return self.tools is None or tool_name in self.tools

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Drop the `hidden` tools from what the model is offered."""
        if not self.hidden:
            return tool_defs
        hidden = set(self.hidden)
        return [tool_def for tool_def in tool_defs if tool_def.name not in hidden]

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate `guard` against the validated arguments and act on its verdict."""
        if self.guard is None or not self._guards(call.tool_name):
            return args

        info = ToolCallInfo(name=call.tool_name, args=args, tool_call_id=call.tool_call_id)
        verdict = await evaluate(self.guard, ctx, info)
        match verdict.action:
            case 'allow':
                return args
            case 'block':
                message = verdict.message or _DEFAULT_ARGS_BLOCK_MESSAGE
                trace_block(ctx, direction=_ARGS_DIRECTION, message=message, tool_name=call.tool_name)
                raise SkipToolExecution(message)
            case 'retry':
                raise ModelRetry(verdict.message or _DEFAULT_RETRY_MESSAGE)
            case 'approve':
                # A resumed run re-evaluates the guard, and a policy that asked for approval
                # once asks again. Honoring `tool_call_approved` -- the same flag
                # `ApprovalRequiredToolset` reads -- is what ends the round trip instead of
                # deferring the call forever.
                if ctx.tool_call_approved:
                    return args
                trace_approval(ctx, direction=_ARGS_DIRECTION, tool_name=call.tool_name)
                raise ApprovalRequired
            case 'replace':
                if not _is_arguments(verdict.replacement):
                    raise UserError(
                        'GuardResult.replace() for a tool argument guard must provide replacement arguments '
                        '(a mapping).'
                    )
                replacement = dict(verdict.replacement)
                trace_redaction(
                    ctx,
                    direction=_ARGS_DIRECTION,
                    original=args,
                    replacement=replacement,
                    tool_name=call.tool_name,
                )
                return replacement
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Evaluate `result_guard` against the tool result and act on its verdict."""
        if self.result_guard is None or not self._guards(call.tool_name):
            return result

        info = ToolResultInfo(name=call.tool_name, args=args, tool_call_id=call.tool_call_id, result=result)
        verdict = await evaluate(self.result_guard, ctx, info)
        match verdict.action:
            case 'allow':
                return result
            case 'block':
                message = verdict.message or _DEFAULT_RESULT_BLOCK_MESSAGE
                trace_block(ctx, direction=_RESULT_DIRECTION, message=message, tool_name=call.tool_name)
                return message
            case 'retry':
                raise ModelRetry(verdict.message or _DEFAULT_RETRY_MESSAGE)
            case 'approve':
                raise UserError(
                    'A tool result guard cannot return GuardResult.approve() -- the tool has already run. '
                    'Return approve() from the argument guard instead.'
                )
            case 'replace':
                trace_redaction(
                    ctx,
                    direction=_RESULT_DIRECTION,
                    original=result,
                    replacement=verdict.replacement,
                    tool_name=call.tool_name,
                )
                return verdict.replacement
            case _:  # pragma: no cover - assert_never exhaustiveness guard
                assert_never(verdict.action)
