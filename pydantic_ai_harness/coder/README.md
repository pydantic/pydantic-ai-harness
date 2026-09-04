# Coder

`Coder` gives a Pydantic AI agent a complete, opinionated stack for working in a local codebase.
It is a regular [combined capability](https://pydantic.dev/docs/ai/capabilities/custom/#composition-and-middleware-semantics) made from the [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) below, so you can use it as-is or take it apart.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.sandboxes import LocalSandbox
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder('.')])

result = agent.run_sync(
    'Find out why tests/test_parser.py fails and fix the bug it caught.',
    sandbox=LocalSandbox(root=Path.cwd()),  # the run needs a sandbox; this one is the local checkout
)
print(result.output)
#> Found it: `parse()` returned None on empty input instead of raising. Fixed in src/parser.py; tests pass now.
```

Every run needs a sandbox attached: `LocalSandbox(root=...)` works on the checkout itself, and a container- or VM-backed sandbox runs the same agent somewhere isolated. Interfaces that start runs for you accept `sandbox=` too, so pass it to `agent.to_cli_sync(sandbox=...)` or `agent.to_web(sandbox=...)` the same way you pass it to `run()`.

It is literally these capabilities combined, in this order:

- [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/): read, write, edit, and search tools rooted at the workspace inside the run's sandbox
- [`Shell`](https://pydantic.dev/docs/ai/harness/shell/): allowlisted commands run in the run's sandbox, rooted at the workspace (the allowlist is a guardrail, not a security boundary)
- [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/): repository instructions and structure
- [`Planning`](https://pydantic.dev/docs/ai/harness/planning/): a plan the agent creates and keeps current during multi-step work
- [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/): delegation, with a read-only `explorer` sub-agent by default
- [`ClearToolResults`](https://pydantic.dev/docs/ai/harness/compaction/): clears stale tool results at 70% of the model context window
- [`WarnNearLimits`](https://pydantic.dev/docs/ai/harness/compaction/): warns the agent at 90% of the model context window
- [`ToolOutputLimits`](https://pydantic.dev/docs/ai/harness/tool-output-limits/): bounds how much context any single tool result can consume

Pass `subagents=[]` to disable delegation, or supply your own `SubAgent` entries.

`Coder` ships with **no default instructions**: modern models don't need procedural coaching, and each composed capability already contributes its own tool guidance. Pass `instructions='...'` to add your own: identity, tone, or house rules. The exported `coder_agent` separately carries the identity instruction `You are a coding agent built on Pydantic AI.`

Other capabilities pair well with `Coder`; add them alongside it in `capabilities`: core [Web Search](https://pydantic.dev/docs/ai/capabilities/web-search/) and [Web Fetch](https://pydantic.dev/docs/ai/capabilities/web-fetch/) to look up docs and error messages, [Skills](https://pydantic.dev/docs/ai/harness/skills/), [Memory](https://pydantic.dev/docs/ai/harness/memory/), [Conversation Search](https://pydantic.dev/docs/ai/harness/conversation-search/), [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/), and [Dynamic Workflow](https://pydantic.dev/docs/ai/harness/dynamic-workflow/).

The command allowlist is a guardrail against accidents, not a security boundary. Validation checks only the first token, and allowlisted commands such as `python`, `git`, `uv`, and `make` can spawn arbitrary processes, so a model that wants to work around the allowlist can. The sandbox attached to the run is the isolation boundary: `LocalSandbox` isolates nothing, so for untrusted work attach a container- or VM-backed one instead.

## Blown-out equivalent

This is the exact agent the exported `coder_agent` gives you (plus an explicit model), written out block by block:

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import (
    ClearToolResults,
    FileSystem,
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
        FileSystem('.'),  # read/write/edit/search, with textual path checks
        Shell(  # allowlisted commands
            cwd='.',
            allowed_commands=allowed_commands,
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
