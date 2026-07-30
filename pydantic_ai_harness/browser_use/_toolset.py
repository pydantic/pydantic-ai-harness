"""The `browse_web` toolset and the factory contract for building browser-use agents"""
# ruff: noqa: D415

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, TypeAlias

import anyio
from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.browser_use._model import resolve_chat_model
from pydantic_ai_harness.browser_use._settings import BrowserAgentSettings

try:
    from browser_use import Agent as _BrowserUseAgent
    from browser_use.agent.views import AgentOutput
    from browser_use.browser import BrowserProfile, BrowserSession
    from browser_use.browser.views import BrowserStateSummary
    from browser_use.llm.base import BaseChatModel
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'browser-use is required for BrowserUse. Install it with: pip install "pydantic-ai-harness[browser-use]"'
    ) from _import_error

logger = logging.getLogger(__name__)

_TOOL_NAME = 'browse_web'

StepCallback: TypeAlias = Callable[[BrowserStateSummary, AgentOutput, int], None]
"""Sync form of browser-use's `register_new_step_callback`."""


def _narrator(sink: Callable[[str], None]) -> StepCallback:
    """Report each step's goal to `sink`."""

    def narrate(browser_state_summary: BrowserStateSummary, output: AgentOutput, step_number: int) -> None:
        goal = (output.next_goal or output.evaluation_previous_goal or '').strip()
        if goal:
            sink(f'  - {goal}')

    return narrate


# teardown is shielded from cancellation, so bound it -- a wedged browser must not hang exit
_TEARDOWN_TIMEOUT = 30


async def _kill(session: BrowserSession) -> bool:
    """Close a browser session, even while the caller is being cancelled"""
    succeeded = True
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(_TEARDOWN_TIMEOUT) as timeout_scope:
            try:
                await session.kill()
            except Exception:
                succeeded = False
                logger.warning('browser-use session teardown failed; retaining the session for retry', exc_info=True)
        if timeout_scope.cancel_called:
            succeeded = False
            logger.warning(
                'browser-use session teardown timed out after %s seconds; retaining the session for retry',
                _TEARDOWN_TIMEOUT,
            )
    return succeeded


class BrowserAgentHistory(Protocol):
    """The subset of browser-use's `AgentHistoryList` that the `browse_web` tool reads"""

    def final_result(self) -> None | str:
        """The text of the final result, or `None` when the agent never finished."""
        ...  # pragma: no cover

    def errors(self) -> list[str | None]:
        """One entry per step: the step's error message, or `None` for clean steps."""
        ...  # pragma: no cover

    def is_successful(self) -> bool | None:
        """The agent's own success verdict for a finished task; `None` while not done."""
        ...  # pragma: no cover

    @property
    def structured_output(self) -> BaseModel | None:
        """The final result parsed against the configured output schema, if any."""
        ...  # pragma: no cover


class BrowserAgent(Protocol):
    """A ready-to-run browser agent for one task, as built by a `BrowserAgentFactory`."""

    def run(self, max_steps: int = 500) -> Awaitable[BrowserAgentHistory]:
        """Run the loop until done or `max_steps`.

        `Awaitable` rather than `async def` so `browser_use.Agent.run`'s
        decorated `Coroutine` return type satisfies the protocol.
        """
        ...  # pragma: no cover


@dataclass
class BrowserTask:
    """What `browse_web` passes to a `BrowserAgentFactory` for one call.

    A dataclass so new fields don't break existing factories.
    """

    task: str
    """The natural-language goal for the browser agent."""

    llm: BaseChatModel | None
    """The resolved chat model; `None` means browser-use's own default."""

    browser_session: BrowserSession
    """The session to browse in; the tool owns its lifecycle."""

    use_vision: bool | Literal['auto']
    """Send page screenshots to the model; `'auto'` follows the model's capabilities."""

    output_schema: type[BaseModel] | None
    """Forwarded as browser-use's `output_model_schema`."""

    sensitive_data: dict[str, str | dict[str, str]] | None = field(repr=False)
    """Secret placeholders browser-use substitutes in the browser. Out of `repr()` to stay out of logs."""

    extend_system_message: str | None
    """Extra instructions appended to the browser agent's system prompt."""

    settings: BrowserAgentSettings
    """The remaining browser-use `Agent` options; `*_llm` fields arrive resolved."""

    on_step: StepCallback | None = None
    """Narration callback (from `progress`); forward as `register_new_step_callback`."""


class BrowserAgentFactory(Protocol):
    """Builds the browser agent `browse_web` runs for one task.

    Don't start or stop the session (the tool owns it), and keep
    `enable_signal_handler=False`.
    """

    def __call__(self, request: BrowserTask) -> BrowserAgent:
        """Build a runnable browser agent for one call."""
        ...  # pragma: no cover


def default_browser_agent(request: BrowserTask) -> BrowserAgent:
    """Build a real `browser_use.Agent` (the default factory)."""
    settings = request.settings
    # signal handler off: no SIGINT hijacking inside a host app
    return _BrowserUseAgent[None, BaseModel](
        task=request.task,
        llm=request.llm,
        browser_session=request.browser_session,
        use_vision=request.use_vision,
        output_model_schema=request.output_schema,
        sensitive_data=request.sensitive_data,
        extend_system_message=request.extend_system_message,
        enable_signal_handler=False,
        register_new_step_callback=request.on_step,
        tools=settings.tools,
        override_system_message=settings.override_system_message,
        max_failures=settings.max_failures,
        max_actions_per_step=settings.max_actions_per_step,
        use_thinking=settings.use_thinking,
        flash_mode=settings.flash_mode,
        max_history_items=settings.max_history_items,
        page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
        fallback_llm=resolve_chat_model(settings.fallback_llm),
        use_judge=settings.use_judge,
        judge_llm=resolve_chat_model(settings.judge_llm),
        ground_truth=settings.ground_truth,
        calculate_cost=settings.calculate_cost,
        vision_detail_level=settings.vision_detail_level,
        llm_screenshot_size=settings.llm_screenshot_size,
        llm_timeout=settings.llm_timeout,
        step_timeout=settings.step_timeout,
        directly_open_url=settings.directly_open_url,
        include_recent_events=settings.include_recent_events,
        final_response_after_failure=settings.final_response_after_failure,
        enable_planning=settings.enable_planning,
        planning_replan_on_stall=settings.planning_replan_on_stall,
        planning_exploration_limit=settings.planning_exploration_limit,
        loop_detection_enabled=settings.loop_detection_enabled,
        loop_detection_window=settings.loop_detection_window,
        message_compaction=settings.message_compaction,
        max_clickable_elements_length=settings.max_clickable_elements_length,
        include_tool_call_examples=settings.include_tool_call_examples,
        initial_actions=settings.initial_actions,
        available_file_paths=settings.available_file_paths,
        file_system_path=settings.file_system_path,
        display_files_in_done_text=settings.display_files_in_done_text,
        save_conversation_path=settings.save_conversation_path,
        save_conversation_path_encoding=settings.save_conversation_path_encoding,
        include_attributes=settings.include_attributes,
        extraction_schema=settings.extraction_schema,
        sample_images=settings.sample_images,
        skills=settings.skills,
        skill_ids=settings.skill_ids,
        pricing_url=settings.pricing_url,
        generate_gif=settings.generate_gif,
        demo_mode=settings.demo_mode,
    )


class BrowserUseToolset(FunctionToolset[AgentDepsT]):
    """Provides the `browse_web` tool: run an autonomous browser-use agent per task."""

    def __init__(
        self,
        *,
        browser_agent: BrowserAgentFactory,
        llm: BaseChatModel | None,
        browser_profile: BrowserProfile | None,
        allowed_domains: list[str] | None,
        headless: bool | None,
        max_steps: int,
        use_vision: bool | Literal['auto'],
        output_schema: type[BaseModel] | None,
        sensitive_data: dict[str, str | dict[str, str]] | None,
        extend_system_message: str | None,
        settings: BrowserAgentSettings,
        session_scope: Literal['call', 'agent'],
        cdp_url: str | None,
        use_cloud: bool | None,
        progress: Callable[[str], None] | None,
    ) -> None:
        super().__init__()
        self._browser_agent = browser_agent
        self._llm = llm
        self._browser_profile = browser_profile
        self._allowed_domains = allowed_domains
        self._headless = headless
        self._max_steps = max_steps
        self._use_vision: bool | Literal['auto'] = use_vision
        self._output_schema = output_schema
        self._sensitive_data = sensitive_data
        self._extend_system_message = extend_system_message
        # resolve *_llm once so every factory gets ready-to-use models
        self._settings = replace(
            settings,
            page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
            fallback_llm=resolve_chat_model(settings.fallback_llm),
            judge_llm=resolve_chat_model(settings.judge_llm),
        )
        self._session_scope: Literal['call', 'agent'] = session_scope
        self._cdp_url = cdp_url
        self._use_cloud = use_cloud
        self._progress = progress
        self._on_step = _narrator(progress) if progress is not None else None
        self._shared_session: BrowserSession | None = None
        self._pending_cleanup: list[BrowserSession] = []
        self._session_closed = False
        self._active_call_sessions = 0
        self._call_cleanup_in_progress = False
        self._call_condition = asyncio.Condition()
        self._cleanup_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self.add_function(self.browse_web, name=_TOOL_NAME)

    def _build_session(self) -> BrowserSession:
        """A fresh session: the capability's fields override the profile."""
        # headless default only applies to a plain local launch
        headless = self._headless
        if headless is None and self._browser_profile is None and not self._use_cloud:
            headless = True
        profile = self._browser_profile
        if self._use_cloud is not None:
            # folded into the profile: the session's typed overloads split cloud and local kwargs
            if profile is None:
                profile = BrowserProfile(use_cloud=self._use_cloud)
            else:
                profile = profile.model_copy(update={'use_cloud': self._use_cloud})
        # keep_alive in 'agent' scope, or browser_use.Agent kills the session after each run
        return BrowserSession(
            cdp_url=self._cdp_url,
            browser_profile=profile,
            headless=headless,
            allowed_domains=self._allowed_domains,
            keep_alive=True if self._session_scope == 'agent' else None,
        )

    async def _run_agent(self, task: str, session: BrowserSession) -> BrowserAgentHistory:
        """Build the sub-agent for `task` against `session` and run its loop."""
        agent = self._browser_agent(
            BrowserTask(
                task=task,
                llm=self._llm,
                browser_session=session,
                use_vision=self._use_vision,
                output_schema=self._output_schema,
                sensitive_data=self._sensitive_data,
                extend_system_message=self._extend_system_message,
                settings=self._settings,
                on_step=self._on_step,
            )
        )
        return await agent.run(max_steps=self._max_steps)

    def _render_result(self, history: BrowserAgentHistory) -> str:
        """The tool result for a finished run: text, schema JSON, or a failure report."""
        result = history.final_result()
        if result is None:
            step_errors = [error for error in history.errors() if error]
            detail = '; '.join(step_errors) if step_errors else 'no further details'
            return f'The browser agent stopped without producing a result ({detail}).'
        answer = self._render_answer(result, history)
        # The verdict is applied to whatever the answer turned out to be, schema JSON included:
        # `structured_output` parses the final result whether or not the sub-agent called `done`
        # with `success=False`, so reading it alone would present a run it gave up on as a clean
        # answer.
        if history.is_successful() is False:
            return f'The browser agent could not fully complete the task. Its final result: {answer}'
        return answer

    def _render_answer(self, result: str, history: BrowserAgentHistory) -> str:
        """The answer itself: schema JSON when one is configured, otherwise the agent's own text."""
        if self._output_schema is None:
            return result
        try:
            structured = history.structured_output
        except ValidationError as error:
            raise ModelRetry(
                f'The browser agent finished, but its result did not match the configured output schema: {error}'
            ) from error
        # A `None` here is unreachable with browser-use's own history, which parses whenever there
        # is a final result and a schema -- both already true. Only a custom factory's history can
        # land here, and its prose is a better answer than an invented failure.
        return structured.model_dump_json() if structured is not None else result

    async def browse_web(self, task: str) -> str:
        """Have an autonomous browser agent carry out a web task and return its result.

        Args:
            task: One self-contained web goal in natural language, e.g.
                "find the price of the Pro plan on example.com and return it".

        Returns:
            The browser agent's final text result, or JSON conforming to the
            configured output schema when one is set.
        """
        if self._progress is not None:
            self._progress(f'* {task}')
        await self._retry_pending_cleanup()
        if self._session_scope == 'call':
            history = await self._run_in_fresh_session(task)
        else:
            history = await self._run_in_shared_session(task)
        return self._render_result(history)

    async def _close_session(self, session: BrowserSession) -> None:
        """Close `session`, retaining its identity when cleanup needs another attempt."""
        with anyio.CancelScope(shield=True):
            if not await _kill(session):
                async with self._cleanup_lock:
                    self._pending_cleanup.append(session)

    async def _retry_pending_cleanup(self) -> None:
        """Retry sessions whose previous teardown failed or timed out."""
        async with self._cleanup_lock:
            pending, self._pending_cleanup = self._pending_cleanup, []
            for session in pending:
                if not await _kill(session):
                    self._pending_cleanup.append(session)

    async def _run_in_fresh_session(self, task: str) -> BrowserAgentHistory:
        """One disposable session for one call, killed when the call ends, on success or failure."""
        session = self._build_session()
        async with self._call_condition:
            await self._call_condition.wait_for(lambda: not self._call_cleanup_in_progress)
            self._active_call_sessions += 1
        try:
            return await self._run_agent(task, session)
        finally:
            with anyio.CancelScope(shield=True):
                await self._close_session(session)
                async with self._call_condition:
                    self._active_call_sessions -= 1
                    self._call_condition.notify_all()

    async def _run_in_shared_session(self, task: str) -> BrowserAgentHistory:
        """The `'agent'`-scoped shared session; the lock serializes calls -- one browser, one driver at a time."""
        async with self._session_lock:
            if self._session_closed:
                # A call that was queued behind `aclose()` reaches the lock after the browser
                # is gone. Without this it would lazily start a fresh `keep_alive` session that
                # nothing is left to close, so the process would exit with a live Chromium.
                raise RuntimeError(
                    'The shared browser session is closed: `aclose()` was called, so `browse_web` '
                    'cannot open another one. Build a new capability to browse again.'
                )
            if self._shared_session is None:
                self._shared_session = self._build_session()
            try:
                return await self._run_agent(task, self._shared_session)
            except BaseException:
                # A failed or cancelled run can leave the shared browser in an
                # unknown state; kill it so the next call starts fresh.
                session, self._shared_session = self._shared_session, None
                await self._close_session(session)
                raise

    async def aclose(self) -> None:
        """Kill the shared browser session and refuse to open another"""
        if self._session_scope == 'call':
            async with self._call_condition:
                await self._call_condition.wait_for(lambda: not self._call_cleanup_in_progress)
                self._call_cleanup_in_progress = True
                try:
                    await self._call_condition.wait_for(lambda: self._active_call_sessions == 0)
                    await self._retry_pending_cleanup()
                finally:
                    self._call_cleanup_in_progress = False
                    self._call_condition.notify_all()
            return

        async with self._session_lock:
            self._session_closed = True
            if self._shared_session is not None:
                session, self._shared_session = self._shared_session, None
                await self._close_session(session)
        await self._retry_pending_cleanup()
