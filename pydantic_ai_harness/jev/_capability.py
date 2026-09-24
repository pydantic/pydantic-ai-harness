"""Jev capability composer: Jev composes a sub-agent for each prompt and hands it the turn."""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, CapabilityEvent, Choices, UseEnumMemberDocstrings
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    WebFetch,
    WebSearch,
    WrapModelRequestHandler,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserContent, UserPromptPart
from pydantic_ai.models import KnownModelName, Model, ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.guardrails import InputGuardrail
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.pydantic_ai_docs import PydanticAIDocs
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.skills import Skills
from pydantic_ai_harness.subagents import ModelOption
from pydantic_ai_harness.subagents._models import as_option, model_label

if TYPE_CHECKING:
    from opentelemetry.trace import Span

_NAME = 'jev_capability_composer'


class Thinking(UseEnumMemberDocstrings, str, Enum):
    """How much reasoning effort a request needs."""

    # Jev reads these docstrings as each option's meaning.

    low = 'low'
    """Routine or single-step work"""
    medium = 'medium'
    """Moderate multi-step work"""
    high = 'high'
    """Hard problems: debugging, design, or large changes"""


ComposeAction: TypeAlias = Literal['compose', 'escalate', 'fallthrough']
"""What the composer did with Jev's pick.

- `compose`: a sub-agent on the model and capabilities Jev picked handled the turn.
- `escalate`: Jev was unsure of the model, so the sub-agent ran its capabilities on `unsure_model`.
- `fallthrough`: Jev picked no capabilities, so the agent's own model handled the turn.
"""


@dataclass(frozen=True, kw_only=True)
class ComposableCapability:
    """One catalog entry: a capability class, what to build it with, and when a run needs it."""

    description: str
    """What the capability is for. Jev reads it to decide whether a request needs it."""

    capability: type[AbstractCapability[object]]
    """The capability class, built fresh for each run that picks it."""

    arguments: Mapping[str, object] = field(default_factory=dict[str, object])
    """Arguments the capability is built with, passed to its `from_spec` as in an `AgentSpec` entry."""

    @classmethod
    def of(
        cls,
        capability: type[AbstractCapability[object]],
        *,
        description: str | None = None,
        arguments: Mapping[str, object] | None = None,
    ) -> ComposableCapability:
        """A catalog entry described by the first line of `capability`'s docstring unless `description` is given.

        A docstring says what a capability is; Jev decides best from what a request needs it for, so a
        description written for that tends to route better.
        """
        doc = inspect.getdoc(capability) or ''
        if doc.startswith(f'{capability.__name__}('):
            doc = ''  # the signature `dataclass` writes for a class without a docstring
        summary = description or next(iter(doc.splitlines()), '')
        if not summary:
            raise UserError(f'{capability.__name__} has no docstring to describe it; pass `description=`.')
        return cls(description=summary, capability=capability, arguments=dict(arguments or {}))


SKILLS_DIRECTORY = Path('.agents/skills')
"""Where `default_catalog` looks for an Agent Skills library, relative to the working directory."""


def default_catalog() -> dict[str, ComposableCapability]:
    """The allowlist `JevCapabilityComposer` picks from when no `catalog` is given.

    Every entry needs no third-party API key and gets its configuration from a default: the working
    directory for `RepoContext`, and `SKILLS_DIRECTORY` for `Skills`. An entry whose requirement is not
    met here is left out instead of failing when a run picks it: `skills` when the directory does not
    exist or the `skills` extra is missing, `code_mode` without the `code-mode` extra, and `web_fetch` without the `web-fetch` extra,
    whose local fetcher covers models with no native URL fetching. `web_search` falls back to DuckDuckGo
    when the `duckduckgo` extra is installed.
    """
    catalog: dict[str, ComposableCapability] = {
        'filesystem': ComposableCapability(
            description='Read, search, and edit files. Any request that looks at or changes code needs this',
            capability=FileSystem,
        ),
        'shell': ComposableCapability(
            # Jev does not infer that changing code means running its tests, so the description says so: this
            # took shell recall on code changes from 25% to 89% on a labelled set.
            description=(
                'Run commands such as tests, linters, builds, and git. '
                'Any request that changes code needs this to check the change works'
            ),
            capability=Shell,
        ),
        'planning': ComposableCapability(description='Track a multi-step plan across a long task', capability=Planning),
        'repo_context': ComposableCapability(
            description="Follow this repository's own agent instructions and conventions (AGENTS.md, CLAUDE.md)",
            capability=RepoContext,
            # A `Path`, not a string: `RepoContext` does not coerce `workspace_dir` when loaded from a spec.
            arguments={'workspace_dir': Path('.')},
        ),
        'pydantic_ai_docs': ComposableCapability(
            description='Look up Pydantic AI documentation', capability=PydanticAIDocs
        ),
        'web_search': ComposableCapability(
            description='Search the public web for current information',
            capability=WebSearch,
            # DuckDuckGo stands in on a model without native web search.
            arguments={'local': 'duckduckgo'} if importlib.util.find_spec('ddgs') is not None else {},
        ),
    }
    if importlib.util.find_spec('markdownify') is not None:
        # Only with the local fetcher: native URL fetching is missing on common models (OpenAI's among them).
        catalog['web_fetch'] = ComposableCapability(
            description='Fetch and read a specific web page or URL', capability=WebFetch, arguments={'local': True}
        )
    if SKILLS_DIRECTORY.is_dir() and importlib.util.find_spec('yaml') is not None:
        catalog['skills'] = ComposableCapability(
            # Scoped to named skills: a general description drew it onto most coding requests.
            description='Follow a skill from the project skill library, only when the request names one',
            capability=Skills,
            arguments={'directories': [SKILLS_DIRECTORY]},
        )
    if importlib.util.find_spec('pydantic_monty') is not None:
        # Imported here: `code_mode` refuses to import without `pydantic-monty`.
        from pydantic_ai_harness.code_mode import CodeMode

        catalog['code_mode'] = ComposableCapability(
            description='Chain many tool calls in one Python script, for bulk or repetitive work',
            capability=CodeMode,
        )
    return catalog


@dataclass(frozen=True, kw_only=True)
class Composition:
    """What Jev picked for one prompt."""

    model: str
    """The key of the chosen `models` entry."""

    thinking: Thinking
    capabilities: tuple[str, ...]
    """Keys of the chosen catalog entries, in catalog order."""

    confidence: Mapping[str, float]
    """Jev's confidence per field (`model`, `thinking`, `capabilities`), from 0 to 1."""


@dataclass(kw_only=True)
class CapabilitiesComposedEvent(CapabilityEvent, namespace='jev', name='capabilities_composed'):
    """Jev composed a sub-agent, which is about to handle the turn."""

    model: str
    """The `models` key the sub-agent runs on: Jev's pick, or `unsure_model` when `escalated`."""
    thinking: Thinking
    capabilities: tuple[str, ...]
    """The catalog keys the sub-agent is built with. A pick of a `shared_capabilities` class is not listed."""
    escalated: bool
    """Whether Jev was unsure of the model, so the sub-agent runs on `unsure_model` instead of its pick."""


class _Picks(BaseModel):
    """The fields Jev fills. `_picks_type` narrows `model` and `capabilities` to one composer's options."""

    model: str
    thinking: Thinking
    capabilities: list[str]


def _picks_type(models: Mapping[str, str], catalog: Mapping[str, str]) -> type[_Picks]:
    """The output type for one model menu and catalog, each option described by its `Choices` entry."""
    model_key = Choices(models, name='ModelPick')
    capability_key = Choices(catalog, name='Capability')

    class Composition(_Picks):
        """Compose an agent to handle this request: its model, reasoning effort, and capabilities."""

        model: Annotated[str, model_key] = Field(description='Which model should handle this request?')
        thinking: Thinking = Field(description='How much reasoning effort does this request need?')
        capabilities: list[Annotated[str, capability_key]] = Field(
            description='Does handling this request need this capability?'
        )

    return Composition


_CONFIDENCE = TypeAdapter(dict[str, float])


@dataclass
class JevCapabilityComposer(AbstractCapability[AgentDepsT]):
    """Let [Jev](https://typesafe.ai) compose a sub-agent for each prompt and hand it the turn.

    On a run's first model request, one Jev request picks a model from `models`, a thinking effort, and the
    capabilities from `catalog` the prompt needs. The composer builds that sub-agent, runs it on the
    conversation so far, and returns its answer as the model response, so the agent's own model is not
    called. When Jev is unsure of the model, the sub-agent runs on `unsure_model`, the strongest entry by
    default. When it picks no capabilities, the agent's own model handles the prompt as usual.

    The sub-agent is an independent run. Of the agent's configuration it gets the conversation, `deps`,
    usage and usage limits, and `shared_capabilities`; the agent's history records only its answer.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.jev import JevCapabilityComposer

    agent = Agent(
        'openai-codex:gpt-6-sol',
        capabilities=[
            JevCapabilityComposer(
                models={
                    'fast': 'openai-codex:gpt-6-luna',
                    'medium': 'openai-codex:gpt-6-sol',
                    'max': 'openai-codex:gpt-6-astra',
                },
            )
        ],
    )
    ```

    Jev runs through Pydantic AI's `TypeSafeModel`, which reads `TYPESAFE_API_KEY`.
    """

    models: Mapping[str, Model | KnownModelName | str | ModelOption]
    """The model menu Jev picks from. A `ModelOption.description` tells Jev what the entry is for, and
    `ModelOption.settings` apply to the sub-agent, overriding the thinking effort Jev picked."""

    catalog: Mapping[str, ComposableCapability] = field(default_factory=default_catalog)
    """The capabilities Jev picks from. Defaults to `default_catalog()`, an allowlist that needs no API keys."""

    instructions: str | None = None
    """Instructions for the sub-agent. The agent's own instructions are not passed on."""

    shared_capabilities: Sequence[AbstractCapability[AgentDepsT]] = ()
    """Capabilities every sub-agent gets besides its picks, such as guardrails, approval policies, or limits.

    A pick of the same class as one of these is left out, so these keep their configuration."""

    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None
    """Receives the sub-agent's events, such as its tool calls, which the agent's own event stream does not carry."""

    confidence_threshold: float = 0.4
    """Minimum Jev confidence in the model pick to use it; below it the sub-agent runs on `unsure_model`.

    A picker that reports no confidence is trusted."""

    unsure_model: str | None = None
    """The `models` key to run on when Jev is unsure of the model pick. Defaults to the last entry, so order
    the menu from cheapest to strongest: an unsure pick then costs a stronger model rather than a wrong one."""

    jev_model: Model | str = 'typesafe:jev-latest'
    """The model that composes. Pin a version (`typesafe:jev-1.13.0`) once the threshold is tuned."""

    _options: dict[str, ModelOption] = field(init=False, repr=False, compare=False)
    _unsure: str = field(init=False, repr=False, compare=False)
    _picker: Agent[None, _Picks] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.models:
            raise UserError('JevCapabilityComposer needs at least one entry in `models`.')
        if not self.catalog:
            raise UserError('JevCapabilityComposer needs at least one entry in `catalog`.')
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise UserError(f'confidence_threshold must be between 0 and 1, got {self.confidence_threshold}.')
        if self.unsure_model is not None and self.unsure_model not in self.models:
            raise UserError(f'unsure_model {self.unsure_model!r} is not a key of `models`.')
        self._options = {key: as_option(value) for key, value in self.models.items()}
        self._unsure = self.unsure_model or list(self.models)[-1]
        self._picker = Agent(
            self.jev_model,
            name=_NAME,
            output_type=_picks_type(
                {key: option.description or model_label(option.model) for key, option in self._options.items()},
                {key: entry.description for key, entry in self.catalog.items()},
            ),
            defer_model_check=True,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: the catalog holds classes and the model menu may hold `Model` instances."""
        return None

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost and inside `InputGuardrail`, so Jev reads the prompt as the capabilities before it leave it."""
        return CapabilityOrdering(position='innermost', wrapped_by=[InputGuardrail])

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """On a run's first request, hand the turn to a sub-agent Jev composed, or fall through to `handler`."""
        parameters = request_context.model_request_parameters
        # The sub-agent answers in text, so a run that needs structured output is left to the agent's model.
        takes_text = parameters.allow_text_output and parameters.output_mode in ('text', 'tool')
        prompt = _pending_text(request_context.messages) if ctx.run_step == 1 and takes_text else None
        if prompt is None:
            return await handler(request_context)

        with ctx.tracer.start_as_current_span(_NAME + ' compose') as span:
            composition = await self.compose(prompt, usage_ctx=ctx)
            action = self._action(composition)
            _record(span, ctx, prompt, composition, action, self._unsure)
        if action == 'fallthrough':
            return await handler(request_context)

        shared = {type(capability) for capability in self.shared_capabilities}
        composition = replace(
            composition,
            model=self._unsure if action == 'escalate' else composition.model,
            capabilities=tuple(key for key in composition.capabilities if self.catalog[key].capability not in shared),
        )
        await ctx.emit(
            CapabilitiesComposedEvent(
                model=composition.model,
                thinking=composition.thinking,
                capabilities=composition.capabilities,
                escalated=action == 'escalate',
            )
        )
        result = await self.build_agent(composition, deps_type=type(ctx.deps)).run(
            # A copy: the sub-agent's run writes to the messages it continues, which belong to this run.
            message_history=deepcopy(request_context.messages),
            deps=ctx.deps,
            usage=ctx.usage,
            usage_limits=ctx.usage_limits,
            event_stream_handler=self.event_stream_handler,
        )
        return ModelResponse(parts=[TextPart(content=result.output)], model_name=result.response.model_name)

    async def compose(self, prompt: str, *, usage_ctx: RunContext[AgentDepsT] | None = None) -> Composition:
        """Ask Jev what `prompt` needs. Usage counts toward `usage_ctx`'s run and its limits when given."""
        result = await self._picker.run(
            prompt,
            usage=usage_ctx.usage if usage_ctx is not None else None,
            usage_limits=usage_ctx.usage_limits if usage_ctx is not None else None,
        )
        picks = result.output
        details = result.response.provider_details or {}
        return Composition(
            model=picks.model,
            thinking=picks.thinking,
            capabilities=tuple(key for key in self.catalog if key in picks.capabilities),
            confidence=_CONFIDENCE.validate_python(details.get('confidence', {})),
        )

    def build_agent(self, composition: Composition, *, deps_type: type[AgentDepsT]) -> Agent[AgentDepsT, str]:
        """The sub-agent `composition` describes, with `shared_capabilities` after its catalog entries."""
        option = self._options[composition.model]
        entries = [
            self.catalog[key].capability.from_spec(**self.catalog[key].arguments) for key in composition.capabilities
        ]
        return Agent(
            option.model,
            deps_type=deps_type,
            name=_NAME + '_sub_agent',
            instructions=self.instructions,
            model_settings={'thinking': composition.thinking.value, **(option.settings or {})},
            capabilities=[*entries, *self.shared_capabilities],
        )

    def _action(self, composition: Composition) -> ComposeAction:
        if not composition.capabilities:
            return 'fallthrough'
        if composition.confidence.get('model', 1.0) < self.confidence_threshold:
            return 'escalate'
        return 'compose'


def _pending_text(messages: Sequence[ModelMessage]) -> str | None:
    """The text of the user prompt in the request about to be sent; `None` when it has none."""
    prompts = [part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)]
    return _text_of(prompts[-1]) if prompts else None


def _text_of(prompt: str | Sequence[UserContent]) -> str | None:
    """The prompt's text parts, which are all Jev reads; `None` when there are none."""
    if isinstance(prompt, str):
        return prompt
    texts = [item for item in prompt if isinstance(item, str)]
    return '\n'.join(texts) if texts else None


def _record(
    span: Span,
    ctx: RunContext[AgentDepsT],
    prompt: str,
    composition: Composition,
    action: ComposeAction,
    unsure: str,
) -> None:
    if not span.is_recording():
        return
    span.set_attributes(
        {
            'jev_composer.action': action,
            'jev_composer.model': composition.model,
            'jev_composer.thinking': composition.thinking.value,
            'jev_composer.capabilities': list(composition.capabilities),
            **{'jev_composer.confidence.' + key: value for key, value in composition.confidence.items()},
        }
    )
    if action != 'fallthrough':
        span.set_attribute('jev_composer.run_model', unsure if action == 'escalate' else composition.model)
    if ctx.trace_include_content:
        span.set_attribute('jev_composer.prompt', prompt)
