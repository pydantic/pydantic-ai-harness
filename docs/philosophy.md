# Philosophy

## The problem

Pydantic AI is a production-grade agents framework. Its reliability matters -- hundreds of thousands of developers depend on it. This demands a strict review process: every PR needs careful security review, code quality verification, and architectural alignment.

But this same rigor creates a bottleneck. The framework gets flooded with low-quality, AI-generated PRs that are expensive to review and often miss the mark. Meanwhile, legitimate contributors with good ideas face long review cycles because the team's review bandwidth is finite.

## The solution

**Capabilities** are the answer. Pydantic AI's capabilities framework (`AbstractCapability`) provides a clean extension point that hooks into the agent graph without touching core code. A capability can provide tools, lifecycle hooks, instructions, and model settings -- everything needed to add sophisticated behavior to an agent.

**Pydantic Harness** is where these capabilities live. It's a separate package with a faster iteration cycle, where:

- The Pydantic AI team publishes pre-made capabilities that require boilerplate to build from scratch
- The community can contribute capabilities with a lower barrier than modifying core
- Anyone can fork the patterns to publish their own capability packages

## Capabilities over PRs

Instead of submitting a PR to pydantic-ai that adds a new feature, ask: can this be a capability?

- **Memory system?** That's a capability
- **Cost tracking?** Capability
- **Approval workflow?** Capability
- **Custom tool filtering?** Capability

If it hooks into the agent lifecycle, it's a capability. Build it here (or in your own package), not as a PR to core.

The only things that belong in pydantic-ai core are changes to the framework itself: new hook points, model providers, performance improvements, bug fixes.

## For package authors

We encourage building and publishing your own capability packages. The convention:

- Package name: `pydantic-ai-<capability>` (e.g. `pydantic-ai-guardrails`)
- Implement `get_serialization_name()` for spec/YAML support
- Implement `from_spec()` if your constructor takes non-serializable arguments
- Register via `custom_capability_types` on `Agent.from_spec()`

See the [`template/`](../template/) directory for a complete starter setup, and [Publishing capability packages](https://ai.pydantic.dev/extensibility/#publishing-capability-packages) in the Pydantic AI docs.
