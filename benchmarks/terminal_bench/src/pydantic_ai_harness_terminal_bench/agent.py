"""The reference agent: a Pydantic AI `Agent` wired to harness capabilities.

Deliberately small. Every line should be explainable as a general capability
with Terminal-Bench as the evidence, not as a benchmark-shaped heuristic. The
weight that makes it competitive lives in the harness capabilities it composes,
where it is reusable, not in this file.

Composed today (all on harness `main`):
- bash tool over the environment (see `tools.py`).
- `TieredCompaction` from the compaction menu: a cheap zero-LLM pass first
  (clear old tool results), summarizing only when the history still exceeds the
  target. This is the direct A/B against Terminus 2's 3-step subagent
  summarization.
- A short, byte-stable system prompt (see `prompts.py`).
- Sensible `UsageLimits` (see `config.py`), applied by the caller at run time.

Capabilities that slot in once their PRs merge are marked below with commented
config, so the wiring is obvious the day they land.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
)

from pydantic_ai_harness_terminal_bench.config import DEFAULT_COMPACTION_TARGET_TOKENS
from pydantic_ai_harness_terminal_bench.prompts import SYSTEM_PROMPT
from pydantic_ai_harness_terminal_bench.tools import TerminalBenchDeps, build_bash_toolset


def build_compaction(
    *,
    summarizer_model: str | Model,
    target_tokens: int = DEFAULT_COMPACTION_TARGET_TOKENS,
) -> TieredCompaction[TerminalBenchDeps]:
    """The tiered compaction strategy, cheap-to-expensive.

    Order matters: the zero-LLM tier runs first and is often enough on a
    tool-output-heavy terminal trajectory, so the one paid summary call only
    happens when clearing old tool results cannot get under `target_tokens`.

    `DeduplicateFileReads` is intentionally not a tier here: it keys on a
    dedicated read-file tool, and this agent reads files through `bash cat`,
    which it cannot identify. A bash-only agent's two useful tiers are clearing
    old tool output, then summarizing.
    """
    # Each tier still needs a trigger to construct, even though `TieredCompaction`
    # drives it directly and bypasses that trigger (matches the compaction menu's
    # own example). The minimal `max_tokens=1` / `max_messages=1` is a placeholder.
    return TieredCompaction[TerminalBenchDeps](
        tiers=[
            ClearToolResults(max_tokens=1, keep_pairs=3),
            SummarizingCompaction(model=summarizer_model, max_messages=1, keep_messages=20),
        ],
        target_tokens=target_tokens,
    )


def build_agent(
    *,
    model: str | Model,
    summarizer_model: str | Model | None = None,
    instructions: str = SYSTEM_PROMPT,
    compaction_target_tokens: int = DEFAULT_COMPACTION_TARGET_TOKENS,
    model_settings: ModelSettings | None = None,
    extra_capabilities: Sequence[AbstractCapability[TerminalBenchDeps]] = (),
) -> Agent[TerminalBenchDeps, str]:
    """Build the reference agent.

    Args:
        model: The Pydantic AI model (a `provider:model` id or a `Model`
            instance -- the tests pass a scripted `FunctionModel`).
        summarizer_model: Model for the compaction summary tier. Defaults to
            `model`; point it at a cheaper model to cut compaction cost.
        instructions: System prompt. Keep it short and stable; the default is
            the whole point.
        compaction_target_tokens: History size that compaction keeps under.
        model_settings: Optional model settings (e.g. prompt-cache flags,
            thinking budget) forwarded to the agent.
        extra_capabilities: Additional harness capabilities to compose in, for
            ablations. See the commented block below for the ones landing soon.
    """
    capabilities: list[AbstractCapability[TerminalBenchDeps]] = [
        build_compaction(
            summarizer_model=summarizer_model if summarizer_model is not None else model,
            target_tokens=compaction_target_tokens,
        ),
        # Slots in once these harness PRs merge (see README, "New capabilities"):
        #   LoopDetection(...),      # harness #336 -- break repeated identical bash calls
        #   BudgetDisclosure(...),   # harness #334 -- tell the model its remaining budget
        #   StalenessTracker(...),   # harness #333 -- flag context gone stale after edits
        *extra_capabilities,
    ]
    return Agent(
        model,
        deps_type=TerminalBenchDeps,
        toolsets=[build_bash_toolset()],
        instructions=instructions,
        capabilities=capabilities,
        model_settings=model_settings,
    )
