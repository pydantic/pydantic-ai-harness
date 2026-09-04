"""Background tools capability that runs selected tools concurrently."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ToolFailedError,
    ToolRetryError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnPart, UserContent
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolRequests,
    RunContext,
    ToolDefinition,
    ToolSelector,
    matches_tool_selector,
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities import WrapRunHandler, WrapToolExecuteHandler
    from pydantic_ai.run import AgentRunResult


_INSTRUCTIONS = """\
Some tools run in the background: when you call them you'll get an immediate \
acknowledgment. If the run remains active, the result will be delivered \
automatically as a follow-up message when the task completes. Continue working on other \
things in the meantime; do not block waiting for the result.\
"""

logger = logging.getLogger(__name__)


def _format_background_error(error: Exception) -> str:
    """Format a background-tool error without exposing unexpected exception details."""
    if isinstance(error, (ApprovalRequired, CallDeferred)):
        return f'{type(error).__name__} was raised; background tools cannot defer a running task.'
    if isinstance(error, ToolRetryError):
        content = error.tool_retry.content
        return content if isinstance(content, str) and content else type(error).__name__
    if isinstance(error, ToolFailedError):
        content = error.tool_failed.content
        return content if isinstance(content, str) and content else type(error).__name__
    return type(error).__name__


def _format_background_result(tool_name: str, task_id: str, result: Any) -> tuple[UserContent, ...]:
    """Format a tool result as model-visible user content without application metadata."""
    if isinstance(result, ToolReturn):
        return_value: object = result.return_value
        extra_content = result.content
    else:
        return_value = result
        extra_content = None

    return_part = ToolReturnPart(tool_name=tool_name, tool_call_id=task_id, content=return_value)
    return_text = return_part.model_response_str()
    content: list[UserContent] = [return_text, *return_part.files]
    if isinstance(extra_content, str):
        content.append(extra_content)
    elif extra_content is not None:
        content.extend(extra_content)

    prefix = f"Background tool '{tool_name}' (task {task_id}) completed.\nResult:"
    content[0] = f'{prefix} {content[0]}'

    if all(isinstance(item, str) for item in content):
        return ('\n'.join(item for item in content if isinstance(item, str)),)
    return tuple(content)


@dataclass
class BackgroundTools(AbstractCapability[AgentDepsT]):
    """Run selected tools concurrently with the current agent run.

    When the model calls a tool that matches the selector, the capability spawns the
    tool's handler in a run-owned task and immediately returns an acknowledgment
    string to the agent. When the task completes, its result (or error) is formatted as
    user content and enqueued via
    [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] as an `'asap'` message.
    Pydantic AI's pending message queue delivers it on the next model request, or
    redirects the agent to a fresh request if it would otherwise end, so the model
    receives the result and can act on it while the run remains active.

    ```python
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai_harness import BackgroundTools

    # Default: any tool with `metadata={'background': True}` runs in the background.
    agent = Agent('openai:gpt-5.6-sol', capabilities=[BackgroundTools()])

    @agent.tool_plain(metadata={'background': True})
    async def slow_research(query: str) -> str:
        await asyncio.sleep(60)  # stand-in for a long-running job
        return f'Research findings for {query!r}'
    ```

    Combine with [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] to mark
    several tools at once, or with `FunctionToolset.with_metadata(...)` to mark a whole
    toolset. Or pass a name list / predicate via `tools=...` to ignore metadata entirely.

    Warning:
        Run cleanup cancels and drains background tasks before it completes. Async tools
        must propagate cancellation; suppressing cancellation can keep cleanup open.
        Python cannot stop a synchronous tool's worker thread, so it may continue after
        the cancelled run returns.

        A synchronous background tool runs concurrently with the agent. Make mutable
        dependencies and other shared state it uses thread-safe.

    `ToolReturn.return_value` and `ToolReturn.content` remain model-visible, including
    multimodal content. Application-only `ToolReturn.metadata` and deferred tool names from
    `ToolReturn.tools` are not carried into the follow-up message. Raised exceptions become
    failure results. Cancelling one background tool does not cancel its siblings; call
    `ctx.cancel()` when a background tool needs to stop the run and all live background tasks.

    `run_stream()` waits for live background tasks before it returns, but does not take
    the extra model turn that delivers their results. Use `agent.run()` or a driven
    `agent.iter()` loop when result delivery is required.

    Realtime sessions already execute tools concurrently. `BackgroundTools` leaves
    them on the realtime session's native tool-result path instead of replacing their
    result with an acknowledgment and follow-up message.

    `BackgroundTools` composes with Temporal durable execution. With DBOS, ordinary
    function tools are not automatically durable steps, so a background tool must
    delegate durable work to an explicit DBOS step. A tool handler running inside a
    durable activity or task must not call `ctx.enqueue()` because replay restores only
    its return value.
    """

    tools: ToolSelector[AgentDepsT] = field(default_factory=lambda: {'background': True})
    """Which tools should run in the background.

    - `dict[str, Any]` (default `{'background': True}`): tools whose metadata deeply
      includes the given key-value pairs.
    - `'all'`: every tool in the agent's toolset (rarely what you want).
    - `Sequence[str]`: tools with matching names.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: custom predicate.
    """

    id: str | None = 'background_tools'

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Combine selectors so each matching tool is scheduled exactly once."""
        merged = super().combine(capabilities)
        # Core only groups instances of the same capability class under one id.
        assert isinstance(merged, cls)

        async def matches_any(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
            for capability in capabilities:
                assert isinstance(capability, cls)
                if await matches_tool_selector(capability.tools, ctx, tool_def):
                    return True
            return False

        return replace(merged, tools=matches_any)

    _tasks: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]], init=False, repr=False)
    _realtime: bool = field(default=False, init=False, repr=False)
    _completed: list[tuple[UserContent, ...]] = field(
        default_factory=list[tuple[UserContent, ...]], init=False, repr=False
    )
    _task_errors: list[BaseException] = field(default_factory=list[BaseException], init=False, repr=False)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return None if self._realtime else _INSTRUCTIONS

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        run_capability = replace(self)
        run_capability._realtime = ctx.realtime
        return run_capability

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        run_sequential = ctx.tool_manager is not None and ctx.tool_manager.get_parallel_execution_mode() == 'sequential'
        if (
            self._realtime
            or run_sequential
            or tool_def.sequential
            or not await matches_tool_selector(self.tools, ctx, tool_def)
        ):
            return await handler(args)

        task_id = call.tool_call_id
        tool_name = call.tool_name

        async def _run() -> None:
            try:
                result = await handler(args)
                message = _format_background_result(tool_name, task_id, result)
            except UnexpectedModelBehavior as e:
                self._task_errors.append(e)
                raise
            except Exception as e:
                if not isinstance(e, (ApprovalRequired, CallDeferred, ToolRetryError, ToolFailedError)):
                    logger.exception('Background tool %s failed', tool_name)
                error = _format_background_error(e)
                message = (f"Background tool '{tool_name}' (task {task_id}) failed: {error}",)
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                self._task_errors.append(e)
                raise
            self._completed.append(message)

        def task_done(task: asyncio.Task[None]) -> None:
            self._tasks.discard(task)
            ctx.usage.tool_calls -= 1
            if not task.cancelled():
                task.exception()

        ctx.usage.tool_calls += 1
        task = asyncio.create_task(_run(), name=f'background tool {tool_name} ({task_id})')
        self._tasks.add(task)
        task.add_done_callback(task_done)
        return (
            f"Tool '{tool_name}' is running in background (task {task_id}). "
            f'If this run remains active, you will receive the result automatically when it completes. '
            f'Continue with other work in the meantime.'
        )

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        from pydantic_graph import End

        if isinstance(result, End) and isinstance(result.data.output, DeferredToolRequests):
            # Background results are dropped when the run pauses.
            return result

        # Let the outer drain deliver anything already queued before waiting for
        # another completion.
        while (
            isinstance(result, End)
            and self._tasks
            and not self._completed
            and not self._task_errors
            and not ctx.pending_messages
        ):
            done, _ = await asyncio.wait(tuple(self._tasks), return_when=asyncio.FIRST_COMPLETED)
            self._tasks.difference_update(done)

        if self._task_errors:
            raise self._task_errors.pop(0)
        for message in self._completed:
            ctx.enqueue(*message)
        self._completed.clear()
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        if self._realtime:
            return await handler()

        try:
            result = await handler()
        finally:
            tasks = tuple(self._tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        if self._task_errors:
            raise self._task_errors.pop(0)
        return result
