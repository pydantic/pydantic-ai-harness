"""Nimble Web Search Agent capability -- Agent API V2 lifecycle tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.nimble._toolset import (
    AgentEffort,
    NimbleClient,
    NimbleSource,
    _json_dump,  # pyright: ignore[reportPrivateUsage]
    _OwnedClientLifecycle,  # pyright: ignore[reportPrivateUsage]
    _recoverable,  # pyright: ignore[reportPrivateUsage]
    _source_list,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

AgentUseCase = Literal['research', 'enrichment', 'dataset_building']

_INSTRUCTIONS = (
    'You can run Nimble Web Search Agents (Agent API V2). Prefer Mode 1: call '
    '`agent_run_start` with `agent_name` (create-or-reuse) plus `input`, optional '
    '`use_case` (locked at create: research | enrichment | dataset_building), '
    '`skill` (domain expertise), `effort`, and run overrides (`sources`, '
    '`output_schema`, `input_data` for enrichment). Pass `agent_id` (`wsa_…`) only '
    'when reusing a known agent (Mode 2). Preserve returned `web_search_agent_id` '
    'and run id (`task_run_…`) for `agent_run_status` / `agent_run_result` across '
    'turns. Runs often take 3–15 minutes — never block-poll inside one tool call. '
    'Optional discovery: `agents_list` / `agent_templates_list`.'
)


def _trust_sources(payload: Mapping[str, Any]) -> list[NimbleSource]:
    """Extract citation URLs from a completed result payload into ToolReturn sources."""
    output = payload.get('output')
    if not isinstance(output, Mapping):
        return []
    trust = cast(Mapping[str, Any], output).get('trust')
    if not isinstance(trust, Mapping):
        return []
    sources = cast(Mapping[str, Any], trust).get('sources')
    if not isinstance(sources, Sequence):
        return []
    urls: dict[str, str | None] = {}
    for raw in cast(Sequence[Any], sources):
        if not isinstance(raw, Mapping):
            continue
        item = cast(Mapping[str, Any], raw)
        url = item.get('url')
        if isinstance(url, str) and url:
            title = item.get('title')
            urls[url] = title if isinstance(title, str) else None
    return _source_list(urls)


class NimbleAgentToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent Nimble Web Search Agent (Agent API V2) lifecycle tools.

    Tools return immediately without polling — use start, then status/result
    across turns (resumable start / status / result).

    **Mode choice (stateless host default = Mode 1):**
    - Mode 1: `agent_name` create-or-reuse via `POST /v2/agents/runs`
    - Mode 2: explicit `agent_id` (`wsa_…`) via `POST /v2/agents/{id}/runs`
    - Mode 3: omit both for an anonymous one-shot (still returns `wsa_…`)
    """

    def __init__(self, *, client: NimbleClient) -> None:
        super().__init__()
        self._client = client
        self.add_function(self.agents_list, name='agents_list')
        self.add_function(self.agent_templates_list, name='agent_templates_list')
        self.add_function(self.agent_run_start, name='agent_run_start')
        self.add_function(self.agent_run_status, name='agent_run_status')
        self.add_function(self.agent_run_result, name='agent_run_result')

    @_recoverable
    async def agents_list(self, limit: int | None = None, offset: int | None = None) -> ToolReturn[str]:
        """List available Nimble Web Search Agents.

        Args:
            limit: Maximum number of agents to return.
            offset: Pagination offset.

        Returns:
            Agent records as JSON.
        """
        list_kwargs: dict[str, Any] = {}
        if limit is not None:
            list_kwargs['limit'] = limit
        if offset is not None:
            list_kwargs['offset'] = offset
        response = await self._client.agents.list(**list_kwargs)
        items = [item.model_dump(mode='json') for item in response.items]
        return ToolReturn(_json_dump(items), metadata={'sources': []})

    @_recoverable
    async def agent_templates_list(self, limit: int | None = None, offset: int | None = None) -> ToolReturn[str]:
        """List available Nimble Web Search Agent templates.

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            Template records as JSON.
        """
        list_kwargs: dict[str, Any] = {}
        if limit is not None:
            list_kwargs['limit'] = limit
        if offset is not None:
            list_kwargs['offset'] = offset
        response = await self._client.agents.templates.list(**list_kwargs)
        items = [item.model_dump(mode='json') for item in response.items]
        return ToolReturn(_json_dump(items), metadata={'sources': []})

    @_recoverable
    async def agent_run_start(
        self,
        input: str,
        agent_id: str | None = None,
        agent_name: str | None = None,
        use_case: AgentUseCase | None = None,
        skill: str | None = None,
        effort: AgentEffort | None = None,
        sources: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        input_data: list[dict[str, Any]] | dict[str, Any] | None = None,
        enable_events: bool | None = None,
    ) -> ToolReturn[str]:
        """Start a Nimble Web Search Agent run (does not wait for completion).

        Prefer Mode 1: pass `agent_name` (and optional `use_case` / `skill` on first
        create). Pass `agent_id` for Mode 2. Omit both for Mode 3 anonymous create.

        Args:
            input: The prompt / task instructions for the run.
            agent_id: Existing agent id (`wsa_…`) for Mode 2. Takes precedence over
                `agent_name` when both are set.
            agent_name: Mode 1 create-or-reuse name (stateless hosts).
            use_case: `research` | `enrichment` | `dataset_building`. Locked at agent
                create — against an existing agent must match or be omitted (else 422).
            skill: Plain-text operating instructions / domain expertise. One-time
                override except Mode 1 first create (persists).
            effort: Effort tier (`low`…`max`). Expect 3–15 minutes for deeper tiers.
            sources: One-time source guidance override (`allow`/`block`/`avoid`/`prioritize`).
            output_schema: JSON Schema override for structured output.
            input_data: Enrichment payload (rows/object); distinct from `output_schema`.
            enable_events: When true, enables SSE via the runs events endpoint later.

        Returns:
            Run metadata including `task_run_…` id and `web_search_agent_id` (`wsa_…`).
        """
        run_body: dict[str, Any] = {'input': input}
        if effort is not None:
            run_body['effort'] = effort
        if sources is not None:
            run_body['sources'] = sources
        if output_schema is not None:
            run_body['output_schema'] = output_schema
        if input_data is not None:
            run_body['input_data'] = input_data
        if enable_events is not None:
            run_body['enable_events'] = enable_events

        # Typed SDK params for agent_name / use_case / skill land in nimble_python>=1.2;
        # until the extra floor is bumped, pass them via extra_body.
        extra_body: dict[str, Any] = {}
        if use_case is not None:
            extra_body['use_case'] = use_case
        if skill is not None:
            extra_body['skill'] = skill

        if agent_id:
            response = await self._client.agents.runs.create(
                agent_id,
                **run_body,
                extra_body=extra_body or None,
            )
        else:
            if agent_name is not None:
                extra_body['agent_name'] = agent_name
            response = await self._client.agents.run(
                **run_body,
                extra_body=extra_body or None,
            )

        payload = response.model_dump(mode='json')
        run_id = payload.get('id')
        wsa_id = payload.get('web_search_agent_id') or agent_id
        mode = 'mode2' if agent_id else ('mode1' if agent_name else 'mode3')
        return ToolReturn(
            f'Started agent run {run_id!r} (agent {wsa_id!r}, {mode}). '
            f'Use agent_run_status / agent_run_result across turns; '
            f'deeper effort tiers often take 3–15 minutes.\n\n{_json_dump(payload)}',
            metadata={
                'agent_id': wsa_id,
                'run_id': run_id,
                'web_search_agent_id': wsa_id,
                'mode': mode,
                'sources': [],
            },
        )

    @_recoverable
    async def agent_run_status(self, agent_id: str, run_id: str) -> ToolReturn[str]:
        """Get the status of a Nimble Web Search Agent run.

        Args:
            agent_id: The `wsa_…` id from `agent_run_start` (`web_search_agent_id`).
            run_id: The `task_run_…` id from `agent_run_start`.

        Returns:
            Run status metadata as JSON (`queued` | `running` | `completed` | …).
        """
        response = await self._client.agents.runs.get(run_id, agent_id=agent_id)
        payload = response.model_dump(mode='json')
        return ToolReturn(
            _json_dump(payload),
            metadata={
                'agent_id': agent_id,
                'run_id': run_id,
                'web_search_agent_id': agent_id,
                'sources': [],
            },
        )

    @_recoverable
    async def agent_run_result(self, agent_id: str, run_id: str) -> ToolReturn[str]:
        """Get the result of a Nimble Web Search Agent run.

        Args:
            agent_id: The `wsa_…` id from `agent_run_start`.
            run_id: The `task_run_…` id from `agent_run_start`.

        Returns:
            Completed output (text/json + trust) or structured failure as JSON.
            Active runs may return a resumable error (e.g. HTTP 409) — call status again.
        """
        response = await self._client.agents.runs.result(run_id, agent_id=agent_id)
        if hasattr(response, 'model_dump'):
            raw_payload = response.model_dump(mode='json')
        else:  # pragma: no cover
            raw_payload = {'result': response}
        payload = cast(dict[str, Any], raw_payload)
        return ToolReturn(
            _json_dump(payload),
            metadata={
                'agent_id': agent_id,
                'run_id': run_id,
                'web_search_agent_id': agent_id,
                'sources': _trust_sources(payload),
            },
        )


@dataclass
class NimbleAgent(AbstractCapability[AgentDepsT]):
    """Nimble [Web Search Agents](https://www.nimbleway.com/) (Agent API V2).

    Resumable lifecycle tools for research / enrichment / dataset building.
    **Default bootstrap is Mode 1** (`agent_name` create-or-reuse) because a
    typical Pydantic AI agent is a stateless host for `wsa_…` persistence —
    the model keeps ids in message history across turns. Pass `agent_id` for
    Mode 2 when the application already stores a `wsa_…`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.nimble import NimbleAgent, NimbleSearch

    agent = Agent(
        'openai:gpt-5.2',
        capabilities=[NimbleSearch(), NimbleAgent()],
    )
    ```

    Authentication comes from the `NIMBLE_API_KEY` environment variable by
    default; pass `client` to configure it explicitly. Factory-built clients
    send `X-Client-Source: pydantic-ai` and are closed after each agent run.
    """

    guidance: str | None = None
    """Custom guidance for the system prompt.

    Leave as `None` for the default Web Search Agent guidance, or set `''` to
    contribute no instructions at all.
    """

    client: NimbleClient | None = None
    """Nimble client to use; when `None`, an `AsyncNimble` is built from `NIMBLE_API_KEY`.

    Factory-built clients send `X-Client-Source: pydantic-ai` and are closed when
    the last concurrent run ends.
    """

    _client_lifecycle: _OwnedClientLifecycle = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client_lifecycle = _OwnedClientLifecycle(explicit_client=self.client)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static guidance for discovering and running Web Search Agents."""
        if self.guidance is not None:
            return self.guidance or None
        return _INSTRUCTIONS

    def get_toolset(self) -> NimbleAgentToolset[AgentDepsT]:
        """Build the toolset providing Web Search Agent lifecycle tools."""
        return NimbleAgentToolset[AgentDepsT](client=self._client_lifecycle.resolve())

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Retain a factory-built client for this run (safe under concurrency)."""
        await self._client_lifecycle.retain_for_run()

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Release a factory-built client; close it when no runs remain."""
        await self._client_lifecycle.release_after_run()
        return result

    @classmethod
    def from_spec(cls, *, guidance: str | None = None) -> NimbleAgent[AgentDepsT]:
        """Construct the capability from serializable spec options.

        The `client` field is not spec-serializable, so spec-loaded instances
        always build the default `AsyncNimble` from `NIMBLE_API_KEY`.
        """
        return cls(guidance=guidance)
