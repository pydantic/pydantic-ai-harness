---
name: pydantic-ai-harness
description: >
  Extend Pydantic AI agents with additional capabilities. The supported capability is `Code Mode` which provides
  sandboxed python tool orchestration. Use when the user mentions executing agent written code, tool sandboxing, pydantic-ai-harness, CodeMode, Monty, or when a pydantic-ai agent would benefit from collapsing many tool calls into one sandboxed Python execution.
license: MIT
compatibility: Requires Python 3.10+ and pydantic-ai-slim>=1.80.0
metadata:
  version: "0.1.0"
  author: pydantic
---

# Build with Pydantic AI Harness

Pydantic AI Harness is a capability library for Pydantic AI agents. While core capabilities fundamental to all agents live in `pydantic-ai`, additional capabilities can be found here.

## Supported Capabilities

| Name | Description | Reference |
| --- | --- | --- |
| `code-mode` | Sandboxed python tool orchestration with `CodeMode()`| [Code Mode](./references/CODE-MODE.md) |

## Install

```bash
uv add pydantic-ai-harness
```

Requires Python 3.10+ and `pydantic-ai-slim>=1.80.0`.

## Extending pydantic-ai with pydantic-ai-harness Capabilities

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP  # from the core pydantic-ai package
from pydantic_ai_harness import CodeMode


agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        MCP('https://api.githubcopilot.com/mcp/'),
        CodeMode(),
    ],
)

result = agent.run_sync('Rank the open PRs on pydantic/pydantic-ai-harness by thumbs-up reactions. Which 5 should we merge first?')
print(result.output)
```

The `MCP` capability continues to come from the core `pydantic-ai` package.

An additional capability, `CodeMode` from pydantic-ai-harness, consolidates all tools into a single run_code tool, allowing the model to coordinate multiple tool calls via Python rather than requiring a separate model round‑trip for each call.

## Key Practices

- Confirm the task actually needs a capability from pydantic-ai-harness. If ordinary Pydantic AI tools or capabilities are enough, use the core Pydantic AI skill instead.
- Identify which capability is needed and read the applicable reference document before writing code.