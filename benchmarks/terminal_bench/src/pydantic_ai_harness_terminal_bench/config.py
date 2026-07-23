"""Run configuration: model-name mapping, usage limits, slice, and cost table.

Everything here is data plus small pure functions, so it is unit-testable
without Harbor, a model, or Docker.
"""

from __future__ import annotations

from pydantic_ai.usage import UsageLimits

DEFAULT_MODEL = 'anthropic:claude-opus-4-6'
"""Default when Harbor is invoked without `-m`. Opus 4.6 is the survey's headline
same-model comparison target (Terminus 2 scored 62.9 with it)."""

# Terminal-Bench trajectories are long; these bound a runaway agent without
# cutting off legitimate multi-step work. Tune against real runs before a
# leaderboard submission -- they are a safety envelope, not a tuned optimum.
DEFAULT_REQUEST_LIMIT = 120
"""Max model requests (turns) per task."""

DEFAULT_TOOL_CALLS_LIMIT = 200
"""Max `bash` calls per task."""

DEFAULT_TOTAL_TOKENS_LIMIT = 4_000_000
"""Max total tokens per task, across all turns."""

DEFAULT_TOOL_TIMEOUT_SEC = 120
"""Per-command timeout handed to `environment.exec`."""

# Compaction fires when the running history is estimated to exceed this. Set
# below a frontier model's real context window so the summary lands before the
# provider would reject the request.
DEFAULT_COMPACTION_TARGET_TOKENS = 150_000


def build_usage_limits(
    *,
    request_limit: int = DEFAULT_REQUEST_LIMIT,
    tool_calls_limit: int = DEFAULT_TOOL_CALLS_LIMIT,
    total_tokens_limit: int = DEFAULT_TOTAL_TOKENS_LIMIT,
) -> UsageLimits:
    """The per-task usage envelope.

    Defaults are the safety-envelope values above. A cost-sensitive run (the
    live CI slice) tightens them through the agent's `--ak` kwargs so a single
    task cannot run away with the budget.
    """
    return UsageLimits(
        request_limit=request_limit,
        tool_calls_limit=tool_calls_limit,
        total_tokens_limit=total_tokens_limit,
    )


GATEWAY_ENV_VAR = 'PYDANTIC_AI_GATEWAY_API_KEY'
"""When set, the live agent routes the model through the Pydantic AI Gateway
(so traces land in Logfire) instead of calling the provider directly."""


def resolve_model_name(name: str | None, *, use_gateway: bool) -> str:
    """Map a Harbor model id to a Pydantic AI model string.

    When `use_gateway` is set, the provider is prefixed with `gateway/` so the
    request is routed through the Pydantic AI Gateway (see
    `pydantic_ai.providers.gateway`). `anthropic:claude-sonnet-4-6` becomes
    `gateway/anthropic:claude-sonnet-4-6`; the direct path is unchanged.

    Examples:
        >>> resolve_model_name('anthropic/claude-sonnet-4-6', use_gateway=False)
        'anthropic:claude-sonnet-4-6'
        >>> resolve_model_name('anthropic/claude-sonnet-4-6', use_gateway=True)
        'gateway/anthropic:claude-sonnet-4-6'
    """
    base = DEFAULT_MODEL if name is None else convert_model_name(name)
    return f'gateway/{base}' if use_gateway else base


def parse_trial_ids(session_id: str) -> tuple[str, str]:
    """Derive `(task, trial)` from a Harbor `session_id`.

    Harbor names a session `{trial_name}__{role}`, where `trial_name` is
    `{task}__{suffix}` -- e.g. `fix-git__bZZeEkw__env`. The task is the first
    segment; the trial is everything up to the role. Used to tag observability
    spans (VStorm's `tb.task` / `tb.trial` convention). An empty or role-less id
    degrades to `(session_id, session_id)` rather than raising.
    """
    if not session_id:
        return session_id, session_id
    parts = session_id.split('__')
    if len(parts) < 2:
        return session_id, session_id
    task = parts[0]
    trial = '__'.join(parts[:-1])
    return task, trial


def convert_model_name(name: str) -> str:
    """Map a Harbor `provider/model` id to a Pydantic AI `provider:model` id.

    Adapted from VStorm's pydantic-deep Harbor adapter (MIT), which solves the
    same Harbor-to-Pydantic-AI naming gap. Harbor uses `/`; Pydantic AI uses
    `:`. A name that already carries a `:` provider prefix, or a bare model name
    with no `/`, is returned unchanged. Google is the one provider that needs a
    branch: Harbor's `google`/`gemini`/`vertex` prefixes map onto Pydantic AI's
    `google:` (Gemini Developer API) and `google-cloud:` (Vertex AI).

    Examples:
        >>> convert_model_name('anthropic/claude-opus-4-6')
        'anthropic:claude-opus-4-6'
        >>> convert_model_name('vertex/gemini-3.1-pro')
        'google-cloud:gemini-3.1-pro'
        >>> convert_model_name('openai:gpt-5.2')
        'openai:gpt-5.2'
    """
    if ':' in name or '/' not in name:
        return name
    provider, model = name.split('/', 1)
    provider_lc = provider.lower()
    if provider_lc in ('vertex', 'google-vertex', 'google-cloud'):
        return f'google-cloud:{model}'
    if provider_lc in ('google', 'gemini'):
        return f'google:{model}'
    return f'{provider}:{model}'


# --- Nightly-slice defaults (survey 2026-07-06, section 3.2) -----------------
# A curated subset for the harness-quality CI tripwire: short-running tasks,
# mixed categories, 3 trials each, reported mean +/- spread. Harbor selects the
# subset with `-i/--include-task-name` (glob), `-x/--exclude-task-name`, and
# `-l/--n-tasks`; see scripts/run_slice.sh. Task-name globs must be validated
# against the live `terminal-bench/terminal-bench-2` dataset -- Terminal-Bench
# task names are not enumerable offline, so the runnable default falls back to
# "first N tasks" (`-l`), which is dataset-version independent.
SLICE_MAX_TASKS = 15
SLICE_TRIALS = 3


# --- Cost estimates (survey 2026-07-06, sections 3.1 and 3.2) ----------------
# Copied verbatim from the survey doc, which flags them as reasoned, not sourced
# from a price sheet. Publishing cost-per-task next to score is on-brand; treat
# these as order-of-magnitude until calibrated on a real run.
COST_ESTIMATES: tuple[tuple[str, str, str], ...] = (
    ('Full leaderboard run', '89 tasks x 5 trials, frontier model', '$450 - $2,200'),
    ('Single-trial dev run', '89 tasks x 1 trial', '$90 - $450'),
    ('Nightly slice', '~15 tasks x 3 trials, mid-tier model', '$15 - $60'),
)
"""Rows of `(label, shape, estimated USD)`. Source: survey sections 3.1/3.2."""
