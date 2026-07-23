#!/usr/bin/env bash
# Nightly slice: a small task subset as the harness-quality tripwire (survey
# section 3.2). Defaults to the first 15 tasks x 3 trials on a mid-tier model.
#
# Task names are not enumerable offline, so the default uses -l (first N tasks),
# which is dataset-version independent. To pin a curated subset instead, pass
# include globs, e.g.:
#   scripts/run_slice.sh -i 'hello-*' -i 'sparql-*'
#
# Usage: MODEL=anthropic/claude-haiku-4 scripts/run_slice.sh [extra harbor args]
set -euo pipefail

MODEL="${MODEL:-anthropic/claude-haiku-4}"
N_TASKS="${N_TASKS:-15}"
N_TRIALS="${N_TRIALS:-3}"
AGENT='pydantic_ai_harness_terminal_bench.harbor_agent:PydanticAITerminalBenchAgent'

harbor run \
    -d terminal-bench/terminal-bench-2 \
    -a "${AGENT}" \
    -m "${MODEL}" \
    -l "${N_TASKS}" \
    -k "${N_TRIALS}" \
    --timeout-multiplier 1.0 \
    "$@"
