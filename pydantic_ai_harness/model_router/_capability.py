"""Model routing through Pydantic AI's model-selection hook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from json import dumps
from math import isfinite
from typing import Literal, TypeGuard

from opentelemetry.trace import NoOpTracer, Status, StatusCode, Tracer
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ModelSelection, ModelSelector
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserContent, UserPromptPart
from pydantic_ai.models import ModelSelectionContext
from pydantic_ai.output import OutputSpec
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness._usage import reserved_usage_limits

_SPAN_NAME = 'model_router.select'


@dataclass(frozen=True)
class ModelChoice:
    """A selectable model and the guidance for choosing it."""

    model: ModelSelection
    """Model name or `Model` instance selected for this choice."""

    description: str
    """When the router should select this choice."""


@dataclass
class ModelRouter(AbstractCapability[AgentDepsT]):
    """Pick the model for a run from a named menu using another model.

    The router is itself a Pydantic AI agent whose output type is a `Literal`
    of the configured choice keys. This works with language models and with
    models that only produce typed output.

    In `once` mode, the capability makes one router request and keeps that
    choice for the run. In `per_step` mode it routes before every logical model
    request. If the router request raises, the declared `default` choice is
    returned and the main run continues.
    """

    choices: Mapping[str, ModelChoice]
    """Named models available to the router."""

    router_model: ModelSelection
    """Model name or `Model` instance that chooses from `choices`."""

    default: str
    """Choice used when routing fails or reported confidence is too low."""

    mode: Literal['once', 'per_step'] = 'once'
    """Whether to route once for the run or before every request step."""

    confidence_threshold: float | None = None
    """Minimum reported router confidence, from 0 to 1, before accepting its pick.

    A router that reports no confidence keeps its pick.
    """

    _router_agent: Agent[None, str] = field(init=False, repr=False, compare=False)
    _run_ready: bool = field(default=False, init=False, repr=False)
    _run_prompt: str | Sequence[UserContent] | None = field(default=None, init=False, repr=False)
    _cached_choice: str | None = field(default=None, init=False, repr=False)
    _tracer: Tracer = field(default_factory=NoOpTracer, init=False, repr=False)
    _usage_limits: UsageLimits | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the routing menu and build the internal router agent."""
        self.choices = dict(self.choices)
        if not self.choices:
            raise UserError('ModelRouter.choices must not be empty')
        if any(not name for name in self.choices):
            raise UserError('ModelRouter choice names must not be empty')
        if any(not choice.description for choice in self.choices.values()):
            raise UserError('ModelRouter choice descriptions must not be empty')
        if self.default not in self.choices:
            raise UserError(f'ModelRouter.default must name a configured choice, got {self.default!r}')
        if self.mode not in {'once', 'per_step'}:
            raise UserError("ModelRouter.mode must be 'once' or 'per_step'")
        if self.confidence_threshold is not None and not 0 <= self.confidence_threshold <= 1:
            raise UserError('ModelRouter.confidence_threshold must be between 0 and 1')
        # Built here rather than per selection: `per_step` would otherwise re-infer the router
        # model on every step, and an unresolvable `router_model` would be swallowed by the
        # fallback and silently route every request to `default` for the life of the agent.
        output_type: OutputSpec[str] = Literal[tuple(self.choices)]  # type: ignore[valid-type]
        self._router_agent = Agent[None, str](
            self.router_model,
            name='model_router',
            deps_type=type(None),
            output_type=output_type,
            instructions=self._instructions(),
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ModelRouter[AgentDepsT]:
        """Return a run-scoped router with the new prompt and isolated selection cache."""
        router = replace(self)
        router._run_ready = True
        router._run_prompt = ctx.prompt
        router._tracer = ctx.tracer
        router._usage_limits = ctx.usage_limits
        return router

    def get_model(self) -> ModelSelector[AgentDepsT]:
        """Return the selector used by Pydantic AI before each request step."""
        return self._select_model

    async def _select_model(self, ctx: ModelSelectionContext[AgentDepsT]) -> ModelSelection:
        if not self._run_ready:
            return self.choices[self.default].model
        if self.mode == 'once' and self._cached_choice is not None:
            return self.choices[self._cached_choice].model

        with self._tracer.start_as_current_span(_SPAN_NAME) as span:
            picked = self.default
            confidence: float | None = None
            fallback_reason = 'none'
            try:
                result = await self._router_agent.run(
                    self._routing_input(ctx),
                    usage=ctx.usage,
                    usage_limits=reserved_usage_limits(self._usage_limits),
                )
                candidate = result.output
                picked = candidate
                confidence = _confidence(result.response.provider_details)
                if (
                    confidence is not None
                    and self.confidence_threshold is not None
                    and confidence < self.confidence_threshold
                ):
                    picked = self.default
                    fallback_reason = 'low_confidence'
            except Exception as error:
                picked = self.default
                fallback_reason = 'error'
                if span.is_recording():
                    span.set_attribute('model_router.error.type', type(error).__name__)
                    span.set_status(Status(StatusCode.ERROR, 'Router request failed; used the default choice.'))

            if span.is_recording():
                attributes: dict[str, str | int | float] = {
                    'model_router.choice': picked,
                    'model_router.fallback_reason': fallback_reason,
                    'model_router.mode': self.mode,
                    'model_router.run_step': ctx.run_step,
                }
                if confidence is not None:
                    attributes['model_router.confidence'] = confidence
                span.set_attributes(attributes)

        if self.mode == 'once':
            self._cached_choice = picked
        return self.choices[picked].model

    def _instructions(self) -> str:
        menu = '\n'.join(f'- {dumps(name)}: {choice.description}' for name, choice in self.choices.items())
        return (
            'Choose which configured model should handle the next request. Return exactly one choice key. '
            'Use the descriptions as routing policy.\n\nAvailable choices:\n' + menu
        )

    def _routing_input(self, ctx: ModelSelectionContext[AgentDepsT]) -> str:
        messages: list[ModelMessage] = list(ctx.messages)
        if ctx.run_step == 1 and self._run_prompt is not None:
            messages.append(ModelRequest(parts=[UserPromptPart(self._run_prompt)]))
        return ModelMessagesTypeAdapter.dump_json(messages).decode()

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable because choices can hold live `Model` instances."""
        return None


def _confidence(provider_details: Mapping[str, object] | None) -> float | None:
    if provider_details is None:
        return None
    raw = provider_details.get('confidence')
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return _checked_confidence(raw)
    if _is_object_mapping(raw):
        response = raw.get('response')
        if not isinstance(response, bool) and isinstance(response, int | float):
            return _checked_confidence(response)
        values = [
            _checked_confidence(value)
            for value in raw.values()
            if not isinstance(value, bool) and isinstance(value, int | float)
        ]
        if values:
            return min(values)
    return None


def _checked_confidence(value: int | float) -> float:
    confidence = float(value)
    if not isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f'Router reported invalid confidence: {value!r}')
    return confidence


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)
