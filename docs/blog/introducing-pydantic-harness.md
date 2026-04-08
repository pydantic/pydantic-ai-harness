# Introducing Pydantic Harness

*Composable capabilities for Pydantic AI agents*

## The problem we're solving

Pydantic AI powers hundreds of thousands of agent applications. As the framework grows, so does the pressure to add features -- memory systems, guardrails, approval workflows, cost tracking, tool management, and more.

But there's a tension. Every addition to core increases the maintenance surface. Every PR needs security review, architectural alignment checks, and testing across all supported providers. And increasingly, we're seeing a flood of low-quality, AI-generated PRs that consume review bandwidth without adding value.

We needed a better model.

## Capabilities: the extension point

With the [capabilities framework](https://ai.pydantic.dev/capabilities/) in Pydantic AI, we built the answer. A capability is a self-contained unit of agent behavior -- an `AbstractCapability` subclass that bundles:

- **Tools** for the agent to use
- **Lifecycle hooks** to intercept and modify model requests, tool calls, and the overall run
- **Instructions** to add to the system prompt
- **Model settings** to configure per-step behavior

Capabilities compose naturally. Attach multiple to an agent, and they layer cleanly:

```python
from pydantic_ai import Agent
from pydantic_harness import InputGuardrail, CostGuard

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        InputGuardrail(guard=lambda text: 'DROP TABLE' not in text),
        CostGuard(max_total_tokens=5000),
    ],
)
```

## Pydantic Harness: the composition layer

**Pydantic Harness** is a new package where pre-made capabilities live. It's separate from pydantic-ai core, which means:

- **Faster iteration**: capabilities can ship without waiting for a core release cycle
- **Lower barrier**: the review process focuses on capability correctness, not framework impact
- **Community-friendly**: contributing a capability here is straightforward compared to modifying core

### What's available today

We're launching with **Guardrails** -- five capability classes that cover input/output validation, cost budgets, tool access control, and async content monitoring:

| Capability | What it does |
|---|---|
| `InputGuardrail` | Validates user input before the agent run starts |
| `OutputGuardrail` | Validates model output after the run completes |
| `CostGuard` | Enforces token budget limits per run |
| `ToolGuard` | Controls per-tool access (block, require approval) |
| `AsyncGuardrail` | Runs a guard alongside model requests (concurrent, blocking, or monitoring mode) |

### What's coming

We have 19 more capabilities planned, from memory and session persistence to stuck-loop detection and adaptive reasoning. Check the [full roadmap](https://github.com/pydantic/pydantic-harness#available-capabilities) in the README.

## Build your own

Pydantic Harness isn't just for us. Anyone can build and publish capability packages.

We've included a [starter template](https://github.com/pydantic/pydantic-harness/tree/main/template) with everything you need: pyproject.toml, test fixtures, CI, and context engineering for AI assistants. Copy it, implement your `AbstractCapability` subclass, and publish to PyPI.

The naming convention: `pydantic-ai-<capability>`. Users register your capability via `custom_capability_types` on `Agent.from_spec()` for YAML/JSON spec support.

## How we build: AICA-powered development

We're dogfooding agent-assisted development on this repository. Issues get labeled, an AI Code Assistant (AICA) generates implementation plans, humans review, and the AICA implements -- all orchestrated through our [ralph loop](https://github.com/pydantic/pydantic-harness/blob/main/docs/processes.md) workflow.

This isn't autonomous AI replacing developers. It's a structured collaboration where:

1. Humans define what needs to be built (issues)
2. The AICA proposes how (plan PRs)
3. Humans review, ask questions, request research
4. The AICA implements and iterates on feedback
5. Humans approve the final result

Every step has human oversight. The AICA can't merge, can't change dependencies, and can't bypass review.

## Get involved

- **Use it**: `pip install pydantic-harness`
- **Build capabilities**: start with the [template](https://github.com/pydantic/pydantic-harness/tree/main/template)
- **Contribute ideas**: [open a capability request](https://github.com/pydantic/pydantic-harness/issues/new?template=capability-request.yml)
- **Follow along**: [github.com/pydantic/pydantic-harness](https://github.com/pydantic/pydantic-harness)

We're excited to see what the community builds.
