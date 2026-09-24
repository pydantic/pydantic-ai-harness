---
name: pydantic-ai-harness
description: Extend Pydantic AI agents with batteries-included capabilities from pydantic-ai-harness -- Code Mode (collapse many tool calls into one sandboxed Python execution), a filesystem and shell, sub-agents, planning, context compaction, durable execution on Render Workflows, and more. Use when the user mentions pydantic-ai-harness, CodeMode, Monty, code mode, or tool sandboxing, when they want first-party filesystem/shell/sub-agent/planning/compaction capabilities for a Pydantic AI agent, when they want an agent to run agent-written Python, when a Pydantic AI agent would benefit from orchestrating multiple tool calls in a single sandboxed script, or when an agent's own job is long-running or distributed (a research run that takes minutes, a batch document pipeline, a monitor or scheduled job, parallel model and tool calls, per-step retries and timeouts, background work that outlives the HTTP request) or the user mentions Render, Render Workflows, task definitions and task runs, or managed orchestration and on-demand compute for agent steps.
license: MIT
compatibility: Requires Python 3.10+ and pydantic-ai-slim>=2.18.0
metadata:
  version: "0.1.0"
  author: pydantic
---

# Building with Pydantic AI Harness

Pydantic AI Harness is the official capability library for Pydantic AI. Capabilities that need model or
framework support -- and those fundamental to every agent -- live in core `pydantic-ai`; optional,
batteries-included capabilities live here. Both are composed onto an agent through the same
`capabilities=[...]` API.

This skill covers the capabilities shipped by `pydantic-ai-harness`. For the core framework -- agents,
tools, structured output, hooks, and testing -- use the `building-pydantic-ai-agents` skill instead.

## When to Use This Skill

Invoke this skill when:
- The user mentions `pydantic-ai-harness`, `CodeMode`, code mode, or the Monty sandbox
- An agent makes many sequential tool calls that could collapse into one sandboxed Python execution
- The user wants the model to write Python that loops, branches, aggregates, or parallelizes tool calls with `asyncio.gather`
- The user asks to sandbox or constrain the code an agent runs
- The agent's own job is long-running or distributed: a research run that takes minutes, a batch document pipeline, a monitor or scheduled job, parallel model and tool calls, per-step retries and timeouts, or work that must outlive the HTTP request that started it (`RenderWorkflows`)
- The user mentions Render, Render Workflows, running agent steps as separately retried task runs, or managed orchestration and on-demand compute for an agent, whether or not they have asked to deploy anything yet

Do **not** use this skill for:
- Core Pydantic AI usage -- building agents, adding tools, structured output, streaming, or testing (use `building-pydantic-ai-agents`)
- Capabilities that ship in core `pydantic-ai`, such as web search, tool search, and thinking
- The Pydantic validation library on its own (`pydantic`/`BaseModel` without agents)

## Supported Capabilities

`CodeMode` has a full reference below; it is the flagship capability and the one this skill goes deep on.
The rest ship today and each has its own README with API and examples.

Each capability lives in its own submodule (`from pydantic_ai_harness.<module> import ...`) and is also
re-exported from the top-level package, which is the import to prefer:
`from pydantic_ai_harness import CodeMode`. The re-export is lazy, so optional dependencies stay isolated:
importing the root package pulls in nothing, and reaching for a capability whose extra is not installed
raises an `ImportError` naming the extra to install.

APIs are subject to change between releases; breaking changes ship deprecation warnings where practical.

| Capability | Module | Description |
|---|---|---|
| `CodeMode` | `pydantic_ai_harness.code_mode` | Wraps eligible tools into a single sandboxed `run_code` tool so the model orchestrates them in Python -- see [Code Mode](./references/CODE-MODE.md) |
| `FileSystem` | `pydantic_ai_harness.filesystem` | Read, write, edit, and search files under a root directory, with traversal prevention |
| `Shell` | `pydantic_ai_harness.shell` | Run commands in a subprocess with allowlists, a default denylist, timeouts, and env masking |
| `ManagedPrompt` | `pydantic_ai_harness.logfire` | Back an agent's instructions with a Logfire-managed prompt |
| `SubAgents` | `pydantic_ai_harness.subagents` | Delegate subtasks to specialized child agents |
| `DynamicWorkflow` | `pydantic_ai_harness.dynamic_workflow` | Orchestrate sub-agents from a model-written Python script |
| `Planning` | `pydantic_ai_harness.planning` | Break complex tasks into structured plans before execution |
| compaction family (`SlidingWindowCompaction`, `SummarizingCompaction`, ...) | `pydantic_ai_harness.compaction` | Trim or summarize conversation history to stay within token limits |
| `ToolOutputLimits` | `pydantic_ai_harness.tool_output_limits` | Truncate, summarize, or spill large tool outputs |
| `RepoContext` | `pydantic_ai_harness.repo_context` | Auto-load CLAUDE.md/AGENTS.md and repo structure |
| `StepPersistence` | `pydantic_ai_harness.step_persistence` | Save, restore, resume, and fork run state |
| `RenderWorkflows` | `pydantic_ai_harness.render` | Run a long-running or distributed agent on Render Workflows, with each supported operation a registered task definition and each invocation a child task run -- see [Render Workflows](../pydantic-ai-render-workflows/SKILL.md) |
| `PydanticAIDocs` | `pydantic_ai_harness.pydantic_ai_docs` | On-demand `read_pyai_docs` tool for Pydantic AI docs |
| `CapabilityCreation` | `pydantic_ai_harness.capability_creation` | Let an agent author, validate, and load real capabilities at runtime |
| media externalization | `pydantic_ai_harness.media` | Offload large `BinaryContent` to content-addressed stores |

Still experimental: an ACP server adapter, imported from `pydantic_ai_harness.experimental.acp`. Importing it
emits a `HarnessExperimentalWarning`.

The full, current list with links and status is in the
[capability matrix](https://github.com/pydantic/pydantic-ai-harness#capability-matrix).

## Install

```bash
uv add pydantic-ai-harness
```

Each capability declares its own extra. Code Mode needs the Monty sandbox:

```bash
uv add "pydantic-ai-harness[codemode]"   # `code-mode` is also accepted as an alias
```

Requires Python 3.10+ and `pydantic-ai-slim>=2.18.0`.

## Quick Start

A harness capability is added to the agent like any other. Here `CodeMode` wraps locally registered tools
into a single `run_code` tool that the model drives with Python.

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import CodeMode

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[CodeMode()])


@agent.tool_plain
def get_temperature_f(city: str) -> float:
    return {'Paris': 68.0, 'Tokyo': 77.0}[city]


@agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    return round((fahrenheit - 32) * 5 / 9, 1)

result = agent.run_sync(
    'Compare the weather in Paris and Tokyo, and report both temperatures in Celsius.'
)
print(result.output)
#> Paris is 20.0 C and Tokyo is 25.0 C.
```

The model writes a single Python script that fetches both temperatures with `asyncio.gather` and then
converts them -- performing four tool calls across two dependent stages in one `run_code` invocation.

## Key Practices

- **Confirm a harness capability is actually needed.** If core Pydantic AI tools and capabilities are enough, use the `building-pydantic-ai-agents` skill instead -- don't reach for the harness by default.
- **Read the reference before writing code.** Each capability has its own configuration, constraints, and gotchas -- load the linked reference (e.g. [Code Mode](./references/CODE-MODE.md)) first.
- **Install the capability's extra.** Importing `CodeMode` without `pydantic-ai-harness[codemode]` raises an `ImportError`; the Monty sandbox is an optional dependency.

## Common Gotchas

- **`native=True` tools bypass `CodeMode`.** Provider-native MCP servers and web search execute server-side, so `run_code` never sees them. Use `native=False` for client-side dispatch that `CodeMode` can wrap, but do not treat a remote server as trusted or sandboxed; see the [Code Mode trust boundary](./references/CODE-MODE.md#sandbox-restrictions).
- **The Monty sandbox is a Python subset.** It has no third-party imports and only a small stdlib allowlist -- read [Code Mode](./references/CODE-MODE.md#sandbox-restrictions) before debugging generated code that fails to run.
- **`CodeMode` needs its extra.** Install `pydantic-ai-harness[codemode]`, not the bare package.
