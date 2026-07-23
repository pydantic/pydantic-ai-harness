# Terminal-Bench reference agent (Pydantic AI on Harbor)

A minimal [Pydantic AI](https://ai.pydantic.dev/) agent, wrapped in a
[Harbor](https://www.harborframework.com/) `BaseAgent` adapter, so it can be
evaluated on [Terminal-Bench 2.x](https://www.tbench.ai/). The agent is
deliberately small: its weight lives in
[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness)
capabilities, not in the prompt.

This is a standalone project inside the harness repo. It has its own
`pyproject.toml` and pulls in Harbor (a heavy, web/DB-stack dependency), so it is
kept out of the harness's default install and CI. See
[Why it is standalone](#why-it-is-standalone).

## What this is

The pitch is **the neutral instrument, instrumented**. Frontier labs pick
scaffolds for reliability, attributability, and third-party reproducibility, not
for score-maxxing (a lab would never launch on a score-maxxing harness -- the
number has to be attributable to the model). This agent aims to be a
mini-swe-agent-simple bash agent on a substrate that adds the things Terminus 2
and mini-SWE-agent get wrong:

- **One tool: bash.** No tmux keystroke loop. Anthropic publicly indicted
  Terminus 2's tmux waiting for 2.7x more timeouts at xhigh effort; a plain
  bash tool avoids it. A file editor could be a second tool later.
- **Compaction from the harness menu** (`TieredCompaction`): cheap zero-LLM
  passes first (clear old tool results, deduplicate file reads), a summary call
  only when they cannot get under target. This is the direct A/B against
  Terminus 2's 3-step subagent summarization.
- **A short, byte-stable system prompt.** A stable prefix is what lets provider
  prompt caching land -- the cost lever across 5 trials x 89 tasks.
- **Sensible `UsageLimits`** as a safety envelope.
- **Every trajectory is a Logfire trace.** Wire up
  [`logfire`](https://logfire.pydantic.dev/) and each run is a clickable trace;
  the leaderboard entry doubles as a Logfire demo.

Every line of the agent should be explainable as a general capability with the
benchmark as evidence, not a benchmark-shaped heuristic. Keeping it minimal is
the enforcement mechanism: capability weight must live in the framework, where
it is reusable, or not exist.

## How it works

The agent runs **host-side**. Its `bash` tool calls back into the task container
through Harbor's `environment.exec`, so nothing is installed in the container.

```
Harbor task container
        ^  environment.exec(command)      +-------------------------+
        |  <-- stdout/stderr/exit code    |  harbor_agent.py         |
        +---------------------------------|  PydanticAITerminalBench |
                                          |  Agent (BaseAgent)       |
   instruction --> prompt                 |    |                      |
                                          |    v                      |
                                          |  agent.py build_agent()   |
                                          |    Pydantic AI Agent      |
                                          |    + bash tool            |
                                          |    + TieredCompaction     |
                                          |    + UsageLimits          |
                                          +-------------------------+
```

| File | Role |
|---|---|
| `src/.../tools.py` | The `bash` tool over a `CommandExecutor` protocol. Substrate-agnostic. |
| `src/.../agent.py` | `build_agent()` -- the ~120-line Pydantic AI agent + capabilities. |
| `src/.../harbor_agent.py` | The Harbor `BaseAgent` adapter: `environment.exec` in, usage out. |
| `src/.../prompts.py` | The short, byte-stable system prompt. |
| `src/.../config.py` | Model-name mapping, `UsageLimits`, slice defaults, cost table. |
| `src/.../smoke.py` | A keyless scripted agent for the CI Docker smoke test. |
| `tasks/smoke-hello/` | A minimal local task for that smoke test. |

## Install

```bash
cd benchmarks/terminal_bench
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install harbor            # the Terminal-Bench harness
docker info                      # Docker must be running for real runs
```

## Run

A full run and a task subset both go through `harbor run`:

```bash
# Full Terminal-Bench 2 run, 5 trials/task, Opus 4.6 (leaderboard shape).
harbor run \
  -d terminal-bench/terminal-bench-2 \
  -a pydantic_ai_harness_terminal_bench.harbor_agent:PydanticAITerminalBenchAgent \
  -m anthropic/claude-opus-4-6 \
  -k 5 --timeout-multiplier 1.0

# Nightly slice: first 15 tasks, 3 trials, a mid-tier model (the CI tripwire).
scripts/run_slice.sh
```

`scripts/run_slice.sh` and `scripts/run_full.sh` wrap those commands. Harbor
selects a subset with `-i/--include-task-name` (glob), `-x/--exclude-task-name`,
`-l/--n-tasks`, or `-t/--task` (a single task). Curated slice task-name globs
live in `config.py`, but the runnable default falls back to "first N tasks"
(`-l`) because Terminal-Bench task names are not enumerable offline.

Agent options are passed with `--ak key=value`, e.g.
`--ak tool_timeout_sec=180 --ak summarizer_model=anthropic/claude-haiku-4`.

### Cost estimates

Copied from the harness-comparison survey (2026-07-06, sections 3.1/3.2), which
flags them as reasoned, not quoted from a price sheet:

| Run | Shape | Estimated USD |
|---|---|---|
| Full leaderboard run | 89 tasks x 5 trials, frontier model | $450 - $2,200 |
| Single-trial dev run | 89 tasks x 1 trial | $90 - $450 |
| Nightly slice | ~15 tasks x 3 trials, mid-tier model | $15 - $60 |

Cache discipline (the byte-stable prompt) should push toward the low end.
Publishing cost-per-task next to score is on-brand.

## The Terminus-2 comparison plan

The headline is **same-model, beat Terminus 2**. On the TB 2.0 leaderboard,
Opus 4.6 scored 62.9 under Terminus 2 and 58.0 under Claude Code, an 18.4-point
spread from the harness alone. The target: beat 62.9 with an Opus 4.6-class
model; stretch, beat Claude Code's 58.0 with an Anthropic model. Each harness
capability is an ablation arm (compaction on/off, and the new capabilities
below), so the per-capability deltas write the launch blog for free.

## New capabilities landing soon

These harness capabilities are on open PRs, not yet on `main`. When they merge,
they slot into `build_agent` via `extra_capabilities` (see the commented block
in `agent.py`):

- **LoopDetection** (harness #336) -- break repeated identical bash calls.
- **BudgetDisclosure** (harness #334) -- tell the model its remaining budget.
- **StalenessTracker** (harness #333) -- flag context gone stale after edits.

## Testing

The seam is proven without Docker. Unit tests use a fake environment that
records the commands the agent issues and returns scripted results:

```bash
uv run pytest        # adapter, agent, tools, config, smoke-model tests
```

The Docker path is validated in CI (`.github/workflows/benchmark-smoke.yml`):
one `tasks/smoke-hello` container is run with the **scripted, keyless** smoke
agent (a `FunctionModel` issues `echo`/`cat`, no API keys), asserting the trial
completes, `environment.exec` round-trips, and a write from a host-side tool call
lands in the container. Full real-model runs happen wherever Docker and API keys
live.

## Leaderboard: build now, submit later

Terminal-Bench submissions were temporarily closed in early July 2026 pending a
new integrity process. Build and publish the agent and self-run numbers now;
submit (PR to the
[HF leaderboard dataset](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard),
>= 5 trials/task, `timeout_multiplier == 1.0`) the moment it reopens.

## Why it is standalone

Harbor pulls a large web/DB stack (uvicorn, supabase, tiktoken, tokenizers) and
is not typed to the harness's pyright-strict bar. Adding it to the harness's
default install would bloat the lock and break the strict typecheck and
lowest-version resolution jobs. So this project has its own `pyproject.toml` and
is excluded from the root package build and CI. It is a runnable
example-with-heavy-deps, in the spirit of a leaderboard entry named "Pydantic AI
Harness". Whether it should eventually move to its own repo (for a cleaner
leaderboard entry name and release cadence) is a
[question for the maintainers](#).

## License

MIT. The Harbor model-name mapping in `config.convert_model_name` is adapted
(not copied) from VStorm's
[pydantic-deep](https://github.com/vstorm-co/pydantic-deep) Harbor adapter (MIT),
which solves the same Harbor-to-Pydantic-AI naming gap. Their adapter is a
`BaseInstalledAgent` (it installs their CLI in each container); this one is an
in-process `BaseAgent`, which is smaller and keeps the agent unit-testable
without Docker.
