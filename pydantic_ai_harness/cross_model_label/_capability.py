"""`CrossModelHistoryLabel`: tell the model when earlier turns came from a different model.

When a run continues a history whose assistant turns were produced by a *different* model than
the one now serving -- a `FallbackModel` failover, a model swap between runs, an A/B handoff, a
takeover -- the serving model otherwise reads those turns as its own. It will defend claims it
never made and keep commitments it cannot verify. This capability detects the mismatch and
contributes one short line naming the other model, so the serving model treats the earlier turns
as inherited context rather than its own output.

The line rides the capability `get_instructions` channel, so it is *ephemeral*: instructions are
rebuilt for every request and are never stored as a message part in the run history. That is the
cache-safe channel (a provenance note that changes with the history must not move the cached
message prefix), and it is the correct provenance channel besides: a note *about* the history
must not itself become history that a later model reads back as fact.

Model identity is compared at the *family* level by default (`gpt-5.2-mini` and `gpt-5.2` are the
same family; `gpt-5.2` and `claude-sonnet-4-5` are not), because a smaller sibling of the same
model does not carry a foreign voice. Pydantic AI profiles do not expose a first-class family
key, so the default resolver derives one heuristically from the model name (stripping size-tier
suffixes and dated snapshots); pass `granularity='exact'` or a callable to override it.

Under a `FallbackModel` the serving member is not known until it answers, so the current identity
is taken from the most recent response's `model_name` (which records who actually served, per
Pydantic AI core PR #6338), falling back to the wrapper's first candidate before any response.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

Granularity: TypeAlias = Literal['family', 'exact']
"""How strictly two model identities are compared.

- `'family'`: compare a derived family key, so a size-tier sibling (`gpt-5.2-mini` vs `gpt-5.2`)
  counts as the same model and does not trigger the label.
- `'exact'`: compare the full (normalized) model name, so any difference triggers.
"""

FamilyResolver: TypeAlias = Callable[[str, 'str | None'], str]
"""A custom identity resolver: receives a response's `model_name` and `provider_name` and returns
the comparison key. Two turns are treated as the same model when their keys are equal."""

CrossModelFormatter: TypeAlias = Callable[[RunContext[AgentDepsT], 'CrossModelHistory'], 'str | None']
"""Signature of a `format` override: receives the run context and the detected cross-model
summary, and returns the line to contribute (or `None` to contribute nothing this request)."""


@dataclass(frozen=True)
class CrossModelHistory:
    """What the detector found this request, passed to a `format` override."""

    current_family: str
    """The resolved identity key of the model now serving."""

    other_family: str
    """The resolved identity key of the differing prior model that the line names."""

    other_model_name: str
    """The raw `model_name` of the named prior response (before family resolution)."""

    differing_responses: int
    """How many prior responses (carrying a `model_name`) resolved to a different family."""

    known_responses: int
    """How many prior responses carried a `model_name` at all (the comparison denominator)."""


# Trailing dated snapshots: OpenAI's dashed `-2024-08-06` and Anthropic's compact `-20241022`.
_DASHED_DATE = re.compile(r'-\d{4}-\d{2}-\d{2}$')
_COMPACT_DATE = re.compile(r'-\d{6,}$')
# Trailing alias/channel markers that do not name a different model.
_TRAILING_ALIAS = re.compile(r'-(latest|preview|exp|experimental)$')
# Size-tier suffixes: a smaller sibling is the same model family, not a foreign voice.
_SIZE_TIERS: frozenset[str] = frozenset({'mini', 'nano', 'small', 'lite', 'tiny'})


def normalize_model_name(model_name: str) -> str:
    """Lowercase and drop a leading `provider:`/`provider/` segment, for stable comparison."""
    name = model_name.lower()
    for separator in (':', '/'):
        if separator in name:
            name = name.rsplit(separator, 1)[-1]
    return name


def model_family(model_name: str, provider_name: str | None = None) -> str:
    """Derive a coarse family key from a model name (the default `'family'` resolver).

    Strips a leading provider segment, a trailing dated snapshot, trailing alias markers
    (`-latest`, `-preview`), and size-tier suffixes (`-mini`, `-nano`, ...), so that
    `gpt-5.2-mini`, `gpt-5.2`, and `gpt-5.2-2025-01-01` all resolve to `gpt-5.2`. It is a
    best-effort heuristic, not an authoritative taxonomy: pass a callable to `granularity` when
    you need exact control. `provider_name` is accepted for parity with custom resolvers and is
    unused by default (the same model served by two providers is one family).
    """
    name = normalize_model_name(model_name)
    name = _DASHED_DATE.sub('', name)
    name = _COMPACT_DATE.sub('', name)
    while (stripped := _TRAILING_ALIAS.sub('', name)) != name:
        name = stripped
    tokens = [token for token in name.split('-') if token not in _SIZE_TIERS]
    return '-'.join(tokens) if tokens else name


_DEFAULT_FORMAT = (
    'Note: assistant responses before this point were produced by a different model ({family}). '
    'Treat their claims and commitments as inherited context, not your own output.'
)


@dataclass
class CrossModelHistoryLabel(AbstractCapability[AgentDepsT]):
    """Contribute one line when earlier assistant turns came from a different model.

    Attach it to an agent that may continue a history written by another model (a `FallbackModel`
    failover, a model swap between runs, a takeover). On each request it resolves the identity of
    the model now serving, scans prior `ModelResponse.model_name` values, and -- when the history
    is another model's per the configured `threshold` -- contributes a single ephemeral line, for
    example:

        Note: assistant responses before this point were produced by a different model (gpt-5.2).
        Treat their claims and commitments as inherited context, not your own output.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.cross_model_label import CrossModelHistoryLabel

    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[CrossModelHistoryLabel()])
    # `message_history` was produced by a different model earlier:
    await agent.run('continue', message_history=history)
    ```

    The line is ephemeral per-request instruction text, never a stored message part, so it neither
    moves the cacheable prefix nor becomes history a later model reads back as fact. The capability
    is stateless: it reads only `ctx.model` and `ctx.messages` each request, so one instance is
    reusable across many runs. It never mutates the message history.
    """

    granularity: Granularity | FamilyResolver = 'family'
    """How strictly to compare identities: `'family'` (default) treats size-tier siblings as the
    same model, `'exact'` compares full normalized names, or a `(model_name, provider_name) -> key`
    callable for full control."""

    threshold: float | Literal['recent'] = 'recent'
    """When to fire. `'recent'` (default) fires only when the immediately preceding response (the
    most recent one carrying a `model_name`) is a different family, so it acts as a one-shot handoff
    nudge that goes quiet once the serving model has added its own turn. A float in `(0, 1]` fires
    when at least that fraction of all prior responses (that carry a `model_name`) are a different
    family, so it acts as a persistent provenance banner for as long as the history stays foreign."""

    format: CrossModelFormatter[AgentDepsT] | None = None
    """Optional override for the line. Receives the run context and a `CrossModelHistory` summary;
    returns the line, or `None` to contribute nothing this request. When set, the default wording
    does not apply."""

    def __post_init__(self) -> None:
        if not isinstance(self.threshold, str) and not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be 'recent' or a float in the range (0, 1].")

    # -- identity resolution --

    def _key(self, model_name: str, provider_name: str | None) -> str:
        granularity = self.granularity
        if callable(granularity):
            return granularity(model_name, provider_name)
        if granularity == 'exact':
            return normalize_model_name(model_name)
        return model_family(model_name, provider_name)

    def _current_identity(self, ctx: RunContext[AgentDepsT]) -> tuple[str, str | None]:
        """The `(model_name, provider_name)` of the model now serving.

        Under a `FallbackModel` the serving member is unknown until it answers, so use the most
        recent response's model (accurate per core #6338), falling back to the wrapper's first
        candidate before any response.
        """
        model = ctx.model
        if isinstance(model, FallbackModel):
            for message in reversed(ctx.messages):
                if isinstance(message, ModelResponse) and message.model_name:
                    return message.model_name, message.provider_name
            first = model.models[0]
            return first.model_name, first.system
        return model.model_name, model.system

    # -- detection --

    @staticmethod
    def _prior_responses(messages: list[ModelMessage]) -> list[ModelResponse]:
        """Prior responses that carry a `model_name` (unlabeled turns are skipped, not guessed)."""
        return [m for m in messages if isinstance(m, ModelResponse) and m.model_name]

    def _detect(self, ctx: RunContext[AgentDepsT]) -> CrossModelHistory | None:
        """Decide whether earlier turns are a different model, per `threshold`; `None` if not."""
        current_name, current_provider = self._current_identity(ctx)
        current_family = self._key(current_name, current_provider)

        known = self._prior_responses(ctx.messages)
        if not known:  # fresh history, or every prior response is unlabeled: stay silent.
            return None

        families = [self._key(r.model_name or '', r.provider_name) for r in known]
        differing = [i for i, family in enumerate(families) if family != current_family]
        if not differing:
            return None

        if self.threshold == 'recent':
            if families[-1] == current_family:
                return None
            other_family, other_name = families[-1], known[-1].model_name or ''
        else:
            if len(differing) / len(known) < self.threshold:
                return None
            other_family, other_name = self._predominant(known, families, differing)

        return CrossModelHistory(
            current_family=current_family,
            other_family=other_family,
            other_model_name=other_name,
            differing_responses=len(differing),
            known_responses=len(known),
        )

    @staticmethod
    def _predominant(known: list[ModelResponse], families: list[str], differing: list[int]) -> tuple[str, str]:
        """The most common differing family (tie broken by most recent), and a raw name for it."""
        counts: dict[str, int] = {}
        last_position: dict[str, int] = {}
        last_name: dict[str, str] = {}
        for i in differing:
            family = families[i]
            counts[family] = counts.get(family, 0) + 1
            last_position[family] = i
            last_name[family] = known[i].model_name or ''
        best = max(counts, key=lambda family: (counts[family], last_position[family]))
        return best, last_name[best]

    # -- contribution --

    async def _instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        detected = self._detect(ctx)
        if detected is None:
            return None
        if self.format is not None:
            return self.format(ctx, detected)
        return _DEFAULT_FORMAT.format(family=detected.other_family)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Contribute the cross-model line when earlier turns are a different model, else nothing."""
        return self._instructions
