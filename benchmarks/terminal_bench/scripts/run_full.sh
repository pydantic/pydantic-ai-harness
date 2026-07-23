#!/usr/bin/env bash
# Full Terminal-Bench 2 run in the leaderboard shape: 5 trials/task,
# timeout_multiplier == 1.0. Set the model with MODEL (Harbor provider/model id).
#
# Usage: MODEL=anthropic/claude-opus-4-6 scripts/run_full.sh [extra harbor args]
set -euo pipefail

MODEL="${MODEL:-anthropic/claude-opus-4-6}"
AGENT='pydantic_ai_harness_terminal_bench.harbor_agent:PydanticAITerminalBenchAgent'

harbor run \
    -d terminal-bench/terminal-bench-2 \
    -a "${AGENT}" \
    -m "${MODEL}" \
    -k 5 \
    --timeout-multiplier 1.0 \
    "$@"
