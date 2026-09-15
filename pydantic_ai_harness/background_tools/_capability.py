"""Background tools capability that runs selected tools concurrently."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import anyio
import anyio.abc
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult, RawToolArgs
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailedError,
    ToolRetryError,
    UnexpectedModelBehavior,
    UserError,
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
Some tools return right away and deliver their result later as a follow-up message. \
Pass `run_in_background=true` to a tool that accepts it when you want to keep working \
and get its result later.\
"""

_RUN_IN_BACKGROUND = 'run_in_background'


def _instructions(ctx: RunContext[Any]) -> str | None:
    # Realtime sessions run tools concurrently already; see `_background_mode`.
    return None if ctx.realtime else _INSTRUCTIONS


logger = logging.getLogger(__name__)


def _with_run_in_background(tool_def: ToolDefinition) -> ToolDefinition:
    """Add the optional `run_in_background` argument to a tool's schema."""
    properties = tool_def.parameters_json_schema.get('properties', {})
    if _RUN_IN_BACKGROUND in properties:
        raise UserError(
            f"Tool '{tool_def.name}' already has a '{_RUN_IN_BACKGROUND}' parameter, "
            'so it cannot be an optional background tool.'
        )
    flag = {
        'type': 'boolean',
        'description': 'Set to true to keep working and get the result later as a follow-up message.',
    }
    schema = {**tool_def.parameters_json_schema, 'properties': {**properties, _RUN_IN_BACKGROUND: flag}}
    return replace(tool_def, parameters_json_schema=schema)


_Outcome = tuple[UserContent, ...] | BaseException
"""What a finished background task hands to the run: the follow-up to deliver, or the error that ends the run."""


def _deliver(ctx: RunContext[Any], outcome: _Outcome) -> None:
    if isinstance(outcome, BaseException):
        raise outcome
    ctx.enqueue(*outcome)


def _format_background_error(error: ApprovalRequired | CallDeferred | ToolRetryError | ToolFailedError) -> str:
    """Describe a tool-signalled failure to the model."""
    if isinstance(error, (ApprovalRequired, CallDeferred)):
        return f'{type(error).__name__} was raised; background tools cannot defer a running task.'
    content = error.tool_retry.content if isinstance(error, ToolRetryError) else error.tool_failed.content
    return content if isinstance(content, str) and content else type(error).__name__


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
    Set the key to `'optional'` instead of `True` to let the model choose per call.

    Warning:
        Run cleanup cancels live background tasks and waits for them, so async tools must
        propagate cancellation. A synchronous tool's worker thread cannot be interrupted, so
        cleanup waits until it returns. It runs concurrently with the agent: keep the state
        it touches thread-safe.

    Exceptions raised by the tool become failure messages; running out of retries or raising
    `CancelledError` ends the run, as it would for a sequential tool. See the docs page for
    streaming, realtime and durable-execution limits.
    """

    tools: ToolSelector[AgentDepsT] = field(default_factory=lambda: {'background': True})
    """Which tools should run in the background.

    - `dict[str, Any]` (default `{'background': True}`): tools whose metadata deeply
      includes the given key-value pairs.
    - `'all'`: every tool in the agent's toolset (rarely what you want).
    - `Sequence[str]`: tools with matching names.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: custom predicate.

    A tool with `metadata={'background': 'optional'}` that this selector does not match gains an
    optional boolean `run_in_background` argument, which the tool function never receives; a call
    runs in the background only when the model passes `true`. Tools that cannot run in the
    background in this run (sequential tools, sequential runs, realtime sessions) are left unchanged.
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

    _task_group: anyio.abc.TaskGroup = field(init=False, repr=False, compare=False)
    """Owns the run's background tasks. `wrap_run` opens it around the run, so no task outlives the run."""
    _live: int = field(default=0, init=False, repr=False)
    """Background tasks that have not handed over their outcome yet."""
    _send: MemoryObjectSendStream[_Outcome] = field(init=False, repr=False, compare=False)
    _outcomes: MemoryObjectReceiveStream[_Outcome] = field(init=False, repr=False, compare=False)
    """Outcomes in completion order; `after_node_run` takes them as they arrive."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _instructions

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BackgroundTools[AgentDepsT]:
        return replace(self)

    async def _background_mode(
        self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition
    ) -> Literal['always', 'optional'] | None:
        """Whether `tool_def` always runs in the background, may on request, or cannot in this run."""
        run_sequential = ctx.tool_manager is not None and ctx.tool_manager.get_parallel_execution_mode() == 'sequential'
        if ctx.realtime or run_sequential or tool_def.sequential:
            return None
        if await matches_tool_selector(self.tools, ctx, tool_def):
            return 'always'
        if (tool_def.metadata or {}).get('background') == 'optional':
            return 'optional'
        return None

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return [
            _with_run_in_background(tool_def) if await self._background_mode(ctx, tool_def) == 'optional' else tool_def
            for tool_def in tool_defs
        ]

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        if await self._background_mode(ctx, tool_def) != 'optional':
            return args
        parsed: Any = args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except ValueError:
                return args  # Core turns malformed JSON into a retry.
        if not isinstance(parsed, dict):
            return args
        # The tool's validator rejects unknown arguments, so the flag is removed and checked here.
        stripped: dict[str, Any] = {**parsed}
        if not isinstance(stripped.pop(_RUN_IN_BACKGROUND, False), bool):
            raise ModelRetry(f'`{_RUN_IN_BACKGROUND}` must be true or false.')
        return stripped

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        mode = await self._background_mode(ctx, tool_def)
        # The flag was removed before validation, so it is read from the call as the model sent it.
        if mode is None or (mode == 'optional' and call.args_as_dict().get(_RUN_IN_BACKGROUND) is not True):
            return await handler(args)

        task_id = call.tool_call_id
        tool_name = call.tool_name

        async def _run() -> None:
            outcome: _Outcome
            try:
                # A task the run cancelled before it got to start must not run the tool.
                await anyio.lowlevel.checkpoint_if_cancelled()
                try:
                    result = await handler(args)
                except (ApprovalRequired, CallDeferred, ToolRetryError, ToolFailedError) as e:
                    outcome = (f"Background tool '{tool_name}' (task {task_id}) failed: {_format_background_error(e)}",)
                except asyncio.CancelledError as e:
                    if self._task_group.cancel_scope.cancel_called:
                        raise
                    # The tool raised this itself: it ends the run, as it would for a sequential tool.
                    outcome = e
                except UnexpectedModelBehavior as e:
                    # The retry budget ran out: it ends the run, as it would for a sequential tool.
                    outcome = e
                except Exception as e:
                    # Unexpected errors are logged in full; the model only learns the type.
                    logger.exception('Background tool %s failed', tool_name)
                    outcome = (f"Background tool '{tool_name}' (task {task_id}) failed: {type(e).__name__}",)
                except BaseException as e:
                    outcome = e
                else:
                    outcome = _format_background_result(tool_name, task_id, result)
                self._send.send_nowait(outcome)
            finally:
                self._live -= 1
                # Core counts a tool call when its handler returns, so the slot was held while the task ran.
                ctx.usage.tool_calls -= 1

        ctx.usage.tool_calls += 1
        self._live += 1
        self._task_group.start_soon(_run, name=f'background tool {tool_name} ({task_id})')
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
            # Finished tasks are left unswept so that a deferred-tool pause is not turned into
            # another model request by the end-of-run drain.
            return result

        for outcome in self._arrived():
            _deliver(ctx, outcome)
        if isinstance(result, End) and self._live and not ctx.pending_messages:
            # The model is ending the run while tasks are still live: wait for the next outcome, so
            # that the end-of-run drain turns its follow-up into another model request.
            _deliver(ctx, await self._outcomes.receive())
        return result

    def _arrived(self) -> Iterator[_Outcome]:
        """Outcomes that have arrived so far, in completion order."""
        with suppress(anyio.WouldBlock):
            while True:
                yield self._outcomes.receive_nowait()

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        self._send, self._outcomes = anyio.create_memory_object_stream[_Outcome](math.inf)
        result: AgentRunResult[Any] | None = None
        with self._send, self._outcomes:
            async with anyio.create_task_group() as self._task_group:
                result = await handler()
                # Tasks still live after a deferred-tool pause or `run_stream()` are dropped.
                self._task_group.cancel_scope.cancel()
            # An error from a task that finished after the last node boundary still ends the run.
            for outcome in self._arrived():
                if isinstance(outcome, BaseException):
                    raise outcome
        # `result` is bound: the group re-raises the run's own exception, and no task cancels the group.
        assert result is not None
        return result
