# Coder

`Coder` gives a Pydantic AI agent a complete, opinionated stack for working in a local codebase.
It is a regular [combined capability](https://pydantic.dev/docs/ai/capabilities/custom/#composition-and-middleware-semantics) made from the [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) below, so you can use it as-is or take it apart.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.workspaces import LocalWorkspace
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder('.')])

result = agent.run_sync(
    'Find out why tests/test_parser.py fails and fix the bug it caught.',
    workspace=LocalWorkspace(root=Path.cwd()),
)
print(result.output)
#> Found it: `parse()` returned None on empty input instead of raising. Fixed in src/parser.py; tests pass now.
```

Programmatic callers pass a workspace explicitly. Coder's Shell and FileSystem tools run against the run's workspace, so passing `workspace=` (or a `WorkspaceRef`) directs them at that environment, and the bundled `coder_agent` supplies the current checkout as a local workspace when none is given. Interfaces that start runs for you accept `workspace=` too, so pass it to [`agent.to_cli_sync(workspace=...)`](https://pydantic.dev/docs/ai/cli/) or [`agent.to_web(workspace=...)`](https://pydantic.dev/docs/ai/web/) the same way you pass it to `run()`.

Or skip the file entirely and run the exported `coder_agent` with [`clai`](https://pydantic.dev/docs/ai/cli/#custom-agents) (the Pydantic AI CLI), via [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
```

The bundled `coder_agent` supplies the current checkout as a local workspace when no explicit reference is supplied.

It is literally these capabilities combined, in this order:

- [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/): read, write, edit, and search tools rooted at the workspace, path-traversal and symlink safe
- [`Shell`](https://pydantic.dev/docs/ai/harness/shell/): allowlisted commands rooted at the workspace (a guardrail, not a security boundary), with common LLM provider API-key variables filtered from inherited command environments
- [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/): repository instructions and structure
- [`Planning`](https://pydantic.dev/docs/ai/harness/planning/): a plan the agent creates and keeps current during multi-step work
- [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/): delegation, with a read-only `explorer` sub-agent by default
- [`ClearToolResults`](https://pydantic.dev/docs/ai/harness/compaction/): clears stale tool results at 70% of the model context window
- [`WarnNearLimits`](https://pydantic.dev/docs/ai/harness/compaction/): warns the agent at 90% of the model context window
- [`ToolOutputLimits`](https://pydantic.dev/docs/ai/harness/tool-output-limits/): bounds how much context any single tool result can consume

Pass `subagents=[]` to disable delegation, or supply your own `SubAgent` entries.

`Coder` ships with **no default instructions**: modern models don't need procedural coaching, and each composed capability already contributes its own tool guidance. Pass `instructions='...'` to add your own: identity, tone, or house rules. The exported `coder_agent` separately carries the identity instruction `You are a coding agent built on Pydantic AI.`

Other capabilities pair well with `Coder`; add them alongside it in `capabilities`: core [Web Search](https://pydantic.dev/docs/ai/capabilities/web-search/) and [Web Fetch](https://pydantic.dev/docs/ai/capabilities/web-fetch/) to look up docs and error messages, [Skills](https://pydantic.dev/docs/ai/harness/skills/), [Memory](https://pydantic.dev/docs/ai/harness/memory/), [Conversation Search](https://pydantic.dev/docs/ai/harness/conversation-search/), [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/), and [Dynamic Workflow](https://pydantic.dev/docs/ai/harness/dynamic-workflow/).

The command allowlist is a guardrail against accidents, not a security boundary. Validation checks only the first token, and allowlisted commands such as `python`, `git`, `uv`, and `make` can spawn arbitrary processes, so a model that wants to work around the allowlist can. For untrusted work, run the agent process inside a container or another OS-level sandbox.

## Blown-out equivalent

This expands the capabilities used by `coder_agent`. Pass a local workspace when running this agent, as in the example above:

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import (
    ClearToolResults,
    FileSystem,
    LLM_API_KEY_ENV_PATTERNS,
    Planning,
    RepoContext,
    Shell,
    SubAgent,
    SubAgents,
    ToolOutputLimits,
    WarnNearLimits,
)

allowed_commands = [
    'git', 'rg', 'grep', 'find', 'ls', 'cat', 'sed', 'head', 'tail',
    'python', 'uv', 'pytest', 'ruff', 'make',
]

explorer = SubAgent(
    Agent(
        name='explorer',
        description='Explore the codebase and answer questions without modifying anything',
        instructions='Answer with concrete paths and evidence.',
        capabilities=[
            FileSystem('.', read_only=True),
            RepoContext(workspace_dir=Path('.')),
        ],
    )
)

agent = Agent(
    'anthropic:claude-fable-5',
    name='coder',
    instructions='You are a coding agent built on Pydantic AI.',
    capabilities=[
        FileSystem('.'),  # read/write/edit/search, path-traversal safe
        Shell(  # allowlisted commands, LLM API keys stripped from their environment
            cwd='.',
            allowed_commands=allowed_commands,
            denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
        ),
        RepoContext(workspace_dir=Path('.')),  # loads AGENTS.md/CLAUDE.md + repo structure
        Planning(),  # structured task plans the model maintains
        SubAgents(agents=[explorer], agent_folders=None),  # delegate exploration off the main context
        ClearToolResults(max_fraction=0.7),  # clears old tool results near the limit
        WarnNearLimits(max_context_fraction=0.9),  # warns the model before it hits limits
        ToolOutputLimits(),  # bounds oversized tool results
    ],
)
```

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/coder/). While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade; see the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).
