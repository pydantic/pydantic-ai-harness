"""Jev capability composer: Jev picks each run's model, thinking effort, and capabilities."""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, CapabilityEvent, Choices, UseEnumMemberDocstrings
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WebFetch, WebSearch, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, UserContent, UserPromptPart
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, AgentToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.guardrails import InputGuardrail
from pydantic_ai_harness.guardrails._shared import as_guards, evaluate_all
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


ComposeAction: TypeAlias = Literal['compose', 'escalate', 'fallthrough', 'blocked']
"""What the composer did with Jev's pick.

- `compose`: the run uses the model and capabilities Jev picked.
- `escalate`: Jev was unsure of the model, so the run uses its capabilities on `unsure_model`.
- `fallthrough`: Jev picked no capabilities, so the run is left as the agent configured it.
- `blocked`: one of the agent's input guardrails blocked the prompt, so Jev was not asked.
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
    """Jev's picks were applied to this run, before its first model request."""

    model: str
    """The `models` key the run uses: Jev's pick, or `unsure_model` when `escalated`."""
    thinking: Thinking
    capabilities: tuple[str, ...]
    """The catalog keys added to the run. A pick the agent already has is not added twice, so it is not listed."""
    escalated: bool
    """Whether Jev was unsure of the model, so the run uses `unsure_model` instead of its pick."""


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
    """Let [Jev](https://typesafe.ai) pick each run's model, thinking effort, and capabilities.

    Before a run starts, one Jev request picks a model from `models`, a thinking effort, and the
    capabilities from `catalog` the prompt needs, and the run goes ahead with them. Everything else on the
    agent -- its instructions, output type, guardrails, persistence, and limits -- applies as usual, because
    the picks join the run rather than replacing it. When Jev is unsure of the model, the run uses
    `unsure_model`, the strongest entry by default. When it picks no capabilities, the run is left as it was.

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
    `ModelOption.settings` apply to the run, overriding the thinking effort Jev picked. A `model` passed to
    `Agent.run` takes precedence over the pick."""

    catalog: Mapping[str, ComposableCapability] = field(default_factory=default_catalog)
    """The capabilities Jev picks from. Defaults to `default_catalog()`, an allowlist that needs no API keys."""

    confidence_threshold: float = 0.4
    """Minimum Jev confidence in the model pick to use it; below it the run uses `unsure_model`.

    A picker that reports no confidence is trusted."""

    unsure_model: str | None = None
    """The `models` key to use when Jev is unsure of the model pick. Defaults to the last entry, so order
    the menu from cheapest to strongest: an unsure pick then costs a stronger model rather than a wrong one."""

    jev_model: Model | str = 'typesafe:jev-latest'
    """The model that composes. Pin a version (`typesafe:jev-1.13.0`) once the threshold is tuned."""

    _options: dict[str, ModelOption] = field(init=False, repr=False, compare=False)
    _unsure: str = field(init=False, repr=False, compare=False)
    _picker: Agent[None, _Picks] = field(init=False, repr=False, compare=False)
    _run_model: Model | KnownModelName | str | None = field(default=None, init=False, repr=False, compare=False)
    _run_settings: ModelSettings | None = field(default=None, init=False, repr=False, compare=False)
    _run_event: CapabilitiesComposedEvent | None = field(default=None, init=False, repr=False, compare=False)

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

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Ask Jev about the run's prompt, and return its picks as this run's capabilities."""
        prompt = _latest_prompt(ctx)
        text = _text_of(prompt) if prompt is not None else None
        # A copy from `capability_for` already holds a run's picks.
        if text is None or self._run_event is not None:
            return self

        agent_capabilities = _leaves(ctx.agent.root_capability if ctx.agent is not None else None)
        with ctx.tracer.start_as_current_span(_NAME + ' compose') as span:
            screened = await _screen(ctx, text, agent_capabilities)
            if screened is None:
                _record(span, ctx, None, None, 'blocked', self._unsure)
                return self
            composition = await self.compose(screened, usage_ctx=ctx)
            action = self._action(composition)
            _record(span, ctx, screened, composition, action, self._unsure)
        if action == 'fallthrough':
            return self
        present = {type(capability) for capability in agent_capabilities}
        return self.capability_for(
            replace(
                composition,
                model=self._unsure if action == 'escalate' else composition.model,
                capabilities=tuple(
                    key for key in composition.capabilities if self.catalog[key].capability not in present
                ),
            ),
            escalated=action == 'escalate',
        )

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

    def capability_for(self, composition: Composition, *, escalated: bool = False) -> AbstractCapability[AgentDepsT]:
        """The capability that applies `composition` to a run: its catalog entries, model, and thinking effort."""
        option = self._options[composition.model]
        entries: list[AbstractCapability[AgentDepsT]] = [
            _Pick(wrapped=self.catalog[key].capability.from_spec(**self.catalog[key].arguments))
            for key in composition.capabilities
        ]
        # A copy of the composer carries the picks, as `for_run` intends for per-run state.
        picked = copy(self)
        picked._run_model = option.model
        picked._run_settings = {'thinking': composition.thinking.value, **(option.settings or {})}
        picked._run_event = CapabilitiesComposedEvent(
            model=composition.model,
            thinking=composition.thinking,
            capabilities=composition.capabilities,
            escalated=escalated,
        )
        return CombinedCapability([*entries, picked])

    def get_model(self) -> Model | KnownModelName | str | None:
        return self._run_model

    def get_model_settings(self) -> ModelSettings | None:
        return self._run_settings

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        if self._run_event is not None:
            await ctx.emit(self._run_event)

    def _action(self, composition: Composition) -> ComposeAction:
        if not composition.capabilities:
            return 'fallthrough'
        if composition.confidence.get('model', 1.0) < self.confidence_threshold:
            return 'escalate'
        return 'compose'


@dataclass
class _Pick(WrapperCapability[AgentDepsT]):
    """A catalog entry Jev picked for one run.

    Its tools give way to another capability of its class that the run also has: one passed to
    `Agent.run`, which `for_run` cannot see, or the same entry picked by an earlier composer. Both would
    register the same tool names, and the run would fail on the conflict.
    """

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolset = self.wrapped.get_toolset()
        if toolset is None:
            return None
        if not isinstance(toolset, AbstractToolset):
            return DynamicToolset[AgentDepsT](toolset).filtered(self._unless_shadowed)
        return toolset.filtered(self._unless_shadowed)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    def _unless_shadowed(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
        rivals = [leaf for leaf in _leaves(ctx.root_capability) if isinstance(_unpicked(leaf), type(self.wrapped))]
        first = next((_unpicked(leaf) for leaf in rivals), self.wrapped)
        return first is self.wrapped and all(isinstance(leaf, _Pick) for leaf in rivals)


def _unpicked(capability: AbstractCapability[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
    return capability.wrapped if isinstance(capability, _Pick) else capability


def _leaves(capability: AbstractCapability[AgentDepsT] | None) -> list[AbstractCapability[AgentDepsT]]:
    """Every capability in the tree, wrappers and the capabilities they wrap included."""
    leaves: list[AbstractCapability[AgentDepsT]] = []
    if capability is not None:
        capability.apply(leaves.append)
    return leaves


async def _screen(
    ctx: RunContext[AgentDepsT], prompt: str, capabilities: Sequence[AbstractCapability[AgentDepsT]]
) -> str | None:
    """The prompt as the agent's input guardrails leave it for Jev.

    Redacted where one redacts; `None` where one blocks it or returns anything but prompt text. The
    guardrails run again, as usual, on the run's first model request.
    """
    for guardrail in capabilities:
        if not isinstance(guardrail, InputGuardrail):
            continue
        verdict, _ = await evaluate_all(as_guards(guardrail.guard, capability='InputGuardrail'), ctx, prompt)
        if verdict.action == 'replace' and isinstance(verdict.replacement, str):
            prompt = verdict.replacement
        elif verdict.action != 'allow':
            return None
    return prompt


def _latest_prompt(ctx: RunContext[AgentDepsT]) -> str | Sequence[UserContent] | None:
    """The run's prompt, or the latest user prompt in the history it continues when it was given none."""
    if ctx.prompt is not None:
        return ctx.prompt
    pending = ctx.messages[-1] if ctx.messages else None
    if isinstance(pending, ModelRequest):
        prompts = [part.content for part in pending.parts if isinstance(part, UserPromptPart)]
        if prompts:
            return prompts[-1]
    return None


def _text_of(prompt: str | Sequence[UserContent]) -> str | None:
    """The prompt's text parts, which are all Jev reads; `None` when there are none."""
    if isinstance(prompt, str):
        return prompt
    texts = [item for item in prompt if isinstance(item, str)]
    return '\n'.join(texts) if texts else None


def _record(
    span: Span,
    ctx: RunContext[AgentDepsT],
    prompt: str | None,
    composition: Composition | None,
    action: ComposeAction,
    unsure: str,
) -> None:
    if not span.is_recording():
        return
    span.set_attribute('jev_composer.action', action)
    if composition is None:
        # Blocked: the prompt is not recorded, since it may hold what the guardrail blocked it for.
        return
    span.set_attributes(
        {
            'jev_composer.model': composition.model,
            'jev_composer.thinking': composition.thinking.value,
            'jev_composer.capabilities': list(composition.capabilities),
            **{'jev_composer.confidence.' + key: value for key, value in composition.confidence.items()},
        }
    )
    if action != 'fallthrough':
        span.set_attribute('jev_composer.run_model', unsure if action == 'escalate' else composition.model)
    if prompt is not None and ctx.trace_include_content:
        span.set_attribute('jev_composer.prompt', prompt)
