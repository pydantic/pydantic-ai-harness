"""The live-model entrypoint for the real Terminal-Bench slice (CI `bench-live`).

Same reference agent as `harbor_agent.PydanticAITerminalBenchAgent`, with two
run-environment behaviours the leaderboard/nightly paths do not need:

- Model routing: when `PYDANTIC_AI_GATEWAY_API_KEY` is set, the model is routed
  through the Pydantic AI Gateway so traces land in Logfire; otherwise it calls
  the provider directly (`ANTHROPIC_API_KEY`). Selection is by environment, so
  the same Harbor `-m` value works either way.
- Observability: when `LOGFIRE_TOKEN` is set, each task runs inside a Logfire
  span tagged with the Harbor task/trial ids. Absent the token, nothing changes.

Run it in CI (see `.github/workflows/benchmark-live.yml`):

    harbor run -d terminal-bench/terminal-bench-2 \\
        -a pydantic_ai_harness_terminal_bench.live:LiveBenchAgent \\
        -m anthropic/claude-sonnet-4-6 -i fix-git -k 1 -n 1
"""

from __future__ import annotations

import os

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic_ai.models import Model

from pydantic_ai_harness_terminal_bench.config import GATEWAY_ENV_VAR, parse_trial_ids, resolve_model_name
from pydantic_ai_harness_terminal_bench.harbor_agent import PydanticAITerminalBenchAgent
from pydantic_ai_harness_terminal_bench.observability import trial_span


class LiveBenchAgent(PydanticAITerminalBenchAgent):
    """Reference agent wired for a real-model CI run: Gateway routing + Logfire spans."""

    def resolve_model(self) -> str | Model:
        """Route through the Gateway when its key is set, else call the provider directly."""
        use_gateway = bool(os.environ.get(GATEWAY_ENV_VAR))
        return resolve_model_name(self.model_name, use_gateway=use_gateway)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        session_id = getattr(environment, 'session_id', '') or ''
        task_id, trial_id = parse_trial_ids(session_id)
        with trial_span(task_id, trial_id):
            await super().run(instruction, environment, context)
