"""`LoopDetection`: detect when an agent is stuck in a repeated-action loop and intervene.

Five major coding harnesses (Gemini CLI, OpenHands, Roo, Crush, goose) converged on some
form of loop detection, because an autonomous agent that gets stuck repeats the same action
until it exhausts its step or token budget. This capability watches the run's tool calls and
model responses for the signatures of a stuck loop and, on detection, either nudges the model
with a harness-marked message, raises an error, or hands a structured `LoopDetected` to a
user callback.

Detection tiers:

- **Tier 1, exact repetition** -- the same `(tool_name, canonical_args)` fingerprint occurs
  at least `repeat_threshold` times within a sliding `window` of recent tool calls. Unlike a
  consecutive-run check this also catches a loop that interleaves an occasional different call.
- **Tier 2a, error cycle** -- the same tool call returns a byte-identical result on
  `error_cycle_threshold` consecutive executions. A tool that keeps failing the same way is
  the common case: coding-agent tools (shell, grep, file reads) usually report failure as an
  ordinary error-shaped result rather than by raising, so an identical repeated result is the
  observable signature of a call that is not making progress.
- **Tier 2b, alternation** -- two distinct call fingerprints alternate A-B-A-B for
  `alternation_cycles` full cycles (a common two-step thrash, e.g. edit then re-read).
- **Tier 2c, monologue** -- `monologue_threshold` consecutive model responses carry no tool
  call and near-identical text (normalized prefix match), i.e. the model is narrating instead
  of acting.

Per-run state lives on a fresh copy returned from `for_run`, so concurrent runs never share
counters.
"""

from __future__ import annotations

import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from opentelemetry import trace
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import ToolDefinition

__all__ = ['LoopDetected', 'LoopDetectedError', 'LoopDetection', 'LoopTier', 'OnLoop']

LoopTier = Literal['exact_repetition', 'error_cycle', 'alternation', 'monologue']
"""Which detection tier fired. See the module docstring for what each one means."""


@dataclass(frozen=True)
class LoopDetected:
    """A structured description of a detected loop, passed to an `on_loop` callback.

    Also carried by `LoopDetectedError` (as `.detected`) when `on_loop='error'`.
    """

    tier: LoopTier
    """Which detection tier fired."""

    tool_name: str | None
    """The tool driving the loop, or `None` for a `monologue` (which has no tool call)."""

    count: int
    """How many times the looping signal was observed (repetitions, cycles, or responses)."""

    window: int
    """The number of recent items the tier considered when it fired.

    For `exact_repetition` this is the current sliding-window occupancy; for the other tiers
    it is the number of consecutive items that formed the pattern.
    """

    fingerprints: tuple[str, ...]
    """The canonical call fingerprint(s) involved, each `tool_name(canonical_json_args)`.

    One entry for `exact_repetition` / `error_cycle`, the two alternating fingerprints for
    `alternation`, and empty for `monologue`.
    """

    message: str
    """A human-readable, harness-marked explanation. Used verbatim as the nudge text."""


class LoopDetectedError(Exception):
    """Raised by `LoopDetection` when `on_loop='error'` and a loop is detected.

    The structured `LoopDetected` is available as `.detected`.
    """

    def __init__(self, detected: LoopDetected) -> None:
        self.detected = detected
        super().__init__(detected.message)


OnLoop = Literal['nudge', 'error'] | Callable[[LoopDetected], None] | Callable[[LoopDetected], Awaitable[None]]
"""What `LoopDetection` does when it detects a loop.

- `'nudge'` (default): enqueue a harness-marked message that the model sees on its next
  request, asking it to change approach.
- `'error'`: raise `LoopDetectedError` to abort the run.
- a callable: called with the `LoopDetected`; may be sync or async. Raise from it to abort,
  or use it to log / record / enqueue custom steering.
"""

# Structured, unambiguous harness marker prefixing every nudge. The nudge is delivered via
# `ctx.enqueue` (see `_act`), which lands it as a *durable* `UserPromptPart` -- wire-
# indistinguishable from a real user turn without this prefix. Naming the harness and the
# capability up front lets the model (and any downstream model that reads the history)
# attribute the steering to the framework, not the user. Matches the bracketed
# `harness:<capability>` convention used across the harness's explicit-marker paths.
#
# TODO(pydantic-ai#6404): once core grows a provenance channel that renders a model-neutral
# `source` marker on request parts per `ModelProfile` at `prepare_messages` time (developer
# role / attribution tag), migrate the nudge onto it instead of baking this presentation
# prefix into the durable transcript. Until then the prefix is the only signal available.
_MARKER = '[harness:loop-detection]'
_CHANGE_APPROACH = 'Change approach, or state plainly what is blocking you.'


def _canonical_args(args: str | dict[str, Any] | None) -> str:
    """Return a stable, order-independent string for a tool call's arguments."""
    if args is None:
        return ''
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return args
    return json.dumps(args, sort_keys=True, default=str)


def _fingerprint(part: ToolCallPart) -> str:
    """A canonical `tool_name(canonical_args)` fingerprint for a tool call."""
    return f'{part.tool_name}({_canonical_args(part.args)})'


def _fingerprint_tool_name(fingerprint: str) -> str:
    """Recover the tool name from a fingerprint produced by `_fingerprint`."""
    return fingerprint.split('(', 1)[0]


def _normalize_text(response: ModelResponse, prefix_chars: int) -> str:
    """Lowercased, whitespace-collapsed prefix of a response's text, for monologue matching.

    Returns `''` when the response carries no text (which never counts as a monologue).
    """
    text = ' '.join(part.content for part in response.parts if isinstance(part, TextPart))
    normalized = ' '.join(text.split()).lower()
    return normalized[:prefix_chars]


@dataclass
class _LoopState:
    """Mutable per-run detection state. Rebuilt for every run by `LoopDetection.for_run`."""

    calls: deque[str] = field(default_factory=lambda: deque[str](maxlen=1))
    """Sliding window of recent call fingerprints (tier 1)."""

    alternation: deque[str] = field(default_factory=lambda: deque[str](maxlen=2))
    """Tail of recent call fingerprints sized to the alternation check (tier 2b)."""

    last_outcome: tuple[str, str] | None = None
    """The most recent `(fingerprint, result_repr)` seen, for the error-cycle tier (2a)."""

    outcome_repeats: int = 0
    """Consecutive repetitions of `last_outcome` (tier 2a)."""

    last_monologue: str | None = None
    """The most recent normalized monologue prefix (tier 2c)."""

    monologue_repeats: int = 0
    """Consecutive near-identical monologue responses (tier 2c)."""


@dataclass
class LoopDetection(AbstractCapability[AgentDepsT]):
    """Detect when an agent is stuck in a repeated-action loop and intervene.

    Attach it to any agent that runs autonomously for many steps. It adds no tools,
    instructions, or model settings, so it composes with any toolset or other capability.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.loop_detection import LoopDetection

    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[LoopDetection()])
    await agent.run('...')  # a stuck loop nudges the model to change approach
    ```

    See the module docstring for the detection tiers. On detection it runs `on_loop`: a nudge
    (default), a raised `LoopDetectedError`, or a user callback. After a tier fires, its
    counters reset so the same loop has to rebuild before it fires again, rather than
    triggering on every subsequent step.
    """

    repeat_threshold: int = 5
    """Tier 1: warn once a call fingerprint occurs this many times within `window`."""

    window: int = 10
    """Tier 1: how many recent tool calls the sliding window keeps."""

    error_cycle_threshold: int = 3
    """Tier 2a: warn once a tool call returns an identical result this many times in a row."""

    alternation_cycles: int = 3
    """Tier 2b: warn once two calls alternate A-B for this many full cycles."""

    monologue_threshold: int = 3
    """Tier 2c: warn once this many consecutive responses are near-identical text with no tool call."""

    monologue_prefix_chars: int = 200
    """Tier 2c: how many leading normalized characters two responses must share to count as near-identical."""

    on_loop: OnLoop = 'nudge'
    """What to do on detection. See `OnLoop`."""

    _state: _LoopState = field(default_factory=_LoopState, compare=False, repr=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Return a fresh copy with empty per-run counters so concurrent runs don't share state."""
        state = _LoopState(
            calls=deque(maxlen=max(self.window, 1)),
            alternation=deque(maxlen=max(self.alternation_cycles * 2, 2)),
        )
        return replace(self, _state=state)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Track this response's tool calls (tiers 1 and 2b) or its monologue text (tier 2c)."""
        state = self._state
        tool_calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        if tool_calls:
            state.last_monologue = None
            state.monologue_repeats = 0
            for call in tool_calls:
                detected = self._observe_call(state, call)
                if detected is not None:
                    await self._act(ctx, detected)
                    return response
        else:
            detected = self._observe_monologue(state, response)
            if detected is not None:
                await self._act(ctx, detected)
        return response

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Track the tool result for the error-cycle tier (2a)."""
        state = self._state
        outcome = (_fingerprint(call), repr(result))
        if outcome == state.last_outcome:
            state.outcome_repeats += 1
        else:
            state.last_outcome = outcome
            state.outcome_repeats = 1
        if state.outcome_repeats >= self.error_cycle_threshold:
            count = state.outcome_repeats
            state.last_outcome = None
            state.outcome_repeats = 0
            message = (
                f'{_MARKER} The tool `{call.tool_name}` returned the same result {count} times in a row; '
                f'repeating it is not making progress. {_CHANGE_APPROACH}'
            )
            await self._act(
                ctx,
                LoopDetected(
                    tier='error_cycle',
                    tool_name=call.tool_name,
                    count=count,
                    window=count,
                    fingerprints=(outcome[0],),
                    message=message,
                ),
            )
        return result

    def _observe_call(self, state: _LoopState, call: ToolCallPart) -> LoopDetected | None:
        """Record a tool call and return a `LoopDetected` if tier 1 or 2b fires."""
        fingerprint = _fingerprint(call)
        state.calls.append(fingerprint)
        state.alternation.append(fingerprint)

        count = state.calls.count(fingerprint)
        if count >= self.repeat_threshold:
            window = len(state.calls)
            state.calls.clear()
            state.alternation.clear()
            message = (
                f'{_MARKER} You appear to be repeating the same action (`{call.tool_name}` called {count} '
                f'times with identical arguments). {_CHANGE_APPROACH}'
            )
            return LoopDetected(
                tier='exact_repetition',
                tool_name=call.tool_name,
                count=count,
                window=window,
                fingerprints=(fingerprint,),
                message=message,
            )

        alternation = _detect_alternation(state.alternation, self.alternation_cycles)
        if alternation is not None:
            a, b = alternation
            window = len(state.alternation)
            state.alternation.clear()
            a_name, b_name = _fingerprint_tool_name(a), _fingerprint_tool_name(b)
            message = (
                f'{_MARKER} You appear to be alternating between `{a_name}` and `{b_name}` without making '
                f'progress ({self.alternation_cycles} cycles). {_CHANGE_APPROACH}'
            )
            return LoopDetected(
                tier='alternation',
                tool_name=a_name,
                count=self.alternation_cycles,
                window=window,
                fingerprints=(a, b),
                message=message,
            )
        return None

    def _observe_monologue(self, state: _LoopState, response: ModelResponse) -> LoopDetected | None:
        """Record a no-tool-call response and return a `LoopDetected` if tier 2c fires."""
        text = _normalize_text(response, self.monologue_prefix_chars)
        if not text:
            return None
        if state.last_monologue is not None and text == state.last_monologue:
            state.monologue_repeats += 1
        else:
            state.monologue_repeats = 1
        state.last_monologue = text
        if state.monologue_repeats >= self.monologue_threshold:
            count = state.monologue_repeats
            state.last_monologue = None
            state.monologue_repeats = 0
            message = (
                f'{_MARKER} You have produced {count} near-identical responses without taking any action. '
                f'Take a concrete next step, or state plainly what is blocking you.'
            )
            return LoopDetected(
                tier='monologue',
                tool_name=None,
                count=count,
                window=count,
                fingerprints=(),
                message=message,
            )
        return None

    async def _act(self, ctx: RunContext[AgentDepsT], detected: LoopDetected) -> None:
        """Run the configured `on_loop` action, after recording a span event."""
        _emit_event(detected)
        on_loop = self.on_loop
        if callable(on_loop):
            result = on_loop(detected)
            if inspect.isawaitable(result):
                await result
            return
        if on_loop == 'error':
            raise LoopDetectedError(detected)
        # `enqueue` delivers the nudge on the next request but persists it as a durable
        # `UserPromptPart`; the `_MARKER` prefix is what keeps it attributable as harness
        # steering rather than a real user turn. See the `_MARKER` TODO for the intended
        # migration to core's provenance channel (pydantic-ai#6404) once it exists.
        try:
            ctx.enqueue(detected.message, priority='asap')
        except UserError:  # pragma: no cover - no queue only in synthetic contexts outside a run
            pass


def _detect_alternation(fingerprints: deque[str], cycles: int) -> tuple[str, str] | None:
    """Return the two alternating fingerprints if the tail is a full A-B-A-B pattern, else `None`.

    One cycle is A-B, so `cycles` full cycles need `cycles * 2` entries.
    """
    needed = cycles * 2
    if len(fingerprints) < needed:
        return None
    tail = list(fingerprints)[-needed:]
    a, b = tail[0], tail[1]
    if a == b:
        return None
    for index, fingerprint in enumerate(tail):
        if fingerprint != (a if index % 2 == 0 else b):
            return None
    return a, b


def _emit_event(detected: LoopDetected) -> None:
    """Add a span event to the active OTel span (a no-op when no span is recording)."""
    trace.get_current_span().add_event(
        'loop_detection.detected',
        {
            'loop.tier': detected.tier,
            'loop.tool_name': detected.tool_name or '',
            'loop.count': detected.count,
            'loop.window': detected.window,
        },
    )
