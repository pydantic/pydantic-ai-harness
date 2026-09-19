---
title: Model Router
description: Choose an agent's model from a named menu using another Pydantic AI model.
---

# Model Router

Choose the model for an agent run from a named menu, using another Pydantic AI model to make the choice.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/model_router/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

Declare each model with a stable key and a description of when it should be used. `default` is the choice used if routing fails:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.model_router import ModelChoice, ModelRouter

router = ModelRouter(
    choices={
        'fast': ModelChoice(
            'openai:gpt-5.6-luna',
            'Lookups, extraction, or a change confined to one place.',
        ),
        'capable': ModelChoice(
            'openai:gpt-5.6-sol',
            'Architecture, security, or a decision that is expensive to get wrong.',
        ),
    },
    router_model='openai:gpt-5.6-luna',
    default='capable',
)

agent = Agent(
    instructions='You are a helpful engineering assistant.',
    capabilities=[router],
)

result = agent.run_sync('Explain why this authentication design is safe.')
print(result.output)
```

This example needs the OpenAI provider group:

```bash
pip/uv-add pydantic-ai-harness 'pydantic-ai-slim[openai]'
```

The router creates an internal agent named `model_router`. Its `output_type` is a `Literal` built from the choice keys. Language models answer through structured output, while typed models can make the same choice without generating text. The router instructions contain each choice description.

## Routing input

The routing input is the normalized message history serialized as JSON. On the first step it also carries the run's new prompt, which Pydantic AI's bootstrap `ModelSelectionContext` does not contain yet -- so a fresh run's very first step is routed from the user's own question rather than defaulting. A router on another provider therefore receives the conversation content used to make the decision.

That input is not bounded. In `'per_step'` mode each routing request grows with the conversation, so a long run eventually spends more on routing than the choice is worth, and a history that outgrows the router's context window fails the request and falls back to `default`. Pair `'per_step'` routing with [compaction](compaction.md): the compacted history is what the router reads, which is usually what you wanted it to read anyway.

## When routing runs

`mode` controls the decision cadence:

| Mode | Router cost | Behavior |
|---|---|---|
| `'once'` | One router model request per run. | Keeps the first choice for every model request in the run. It cannot react to later message history. |
| `'per_step'` | One router model request per logical model request step. | Reconsiders the choice as Pydantic AI exposes more message history, so later steps can use a different model. |

`'once'` is the default. It keeps routing cost and latency fixed. Use `'per_step'` when a run can begin with routine work and later reach a step that needs a different model.

Pydantic AI calls model selectors once per logical step. Provider polling or continuation inside one step stays on the selected model.

## Confidence and failure

Set `confidence_threshold` when constructing `ModelRouter` to reject a reported confidence below that cutoff. For example, with `confidence_threshold=0.8`, a reported value below `0.8` falls back. The threshold has no effect when the router reports no confidence.

When the router response has `provider_details['confidence']`, a value below the threshold selects `default`. TypeSafe reports the confidence for a bare `Literal` under the `response` key. The capability also accepts a numeric confidence directly and uses the least numeric value when a provider reports a mapping without `response`. Reported confidence must be finite and between `0` and `1`; an invalid value selects `default` as a routing error.

A router model that reports no confidence keeps its pick. This lets ordinary language models route without provider-specific metadata.

If the router model raises at request time, returns invalid model behavior after its normal output retries, or returns an unknown key, `ModelRouter` selects `default` and the main run continues. Cancellation is not converted into a fallback.

Configuration mistakes do not fall back. An empty menu, an unknown `default`, a threshold outside `0` to `1`, and a `router_model` Pydantic AI cannot resolve all raise `UserError` from the `ModelRouter` constructor, so a typo surfaces immediately instead of routing every request to `default` for the life of the agent.

## Options

| Option | Default | Behavior |
|---|---:|---|
| `choices` | required | Mapping of keys to `ModelChoice(model, description)`. Keys and descriptions must be non-empty. |
| `router_model` | required | Any model name or `Model` instance accepted by Pydantic AI. |
| `default` | required | Configured choice used for router errors and low confidence. |
| `mode` | `'once'` | Route `'once'` per run or `'per_step'`. |
| `confidence_threshold` | `None` | Minimum reported confidence from `0` to `1`. Missing confidence accepts the pick. |

Both the router and choice models may be model names or configured `Model` instances. Model instances preserve their clients, credentials, base URLs, and instrumentation.

## Observability

Each routing decision emits a `model_router.select` span on the parent run's tracer. It has these attributes:

| Attribute | Meaning |
|---|---|
| `model_router.choice` | Choice actually used after fallback. |
| `model_router.confidence` | Reported confidence, when present. |
| `model_router.fallback_reason` | `none`, `low_confidence`, or `error`. |
| `model_router.mode` | `once` or `per_step`. |
| `model_router.run_step` | Logical request step being routed. |
| `model_router.error.type` | Exception class when the default handled a router failure. |

The internal agent name `model_router` also lets Logfire group its requests, token use, cost, and latency separately from the main agent.

## Composition and execution constraints

`ModelRouter` implements Pydantic AI's existing [`get_model()`](/ai/capabilities/custom/#selecting-the-model) hook. A model passed directly to `run(model=...)`, through a run spec, or through `agent.override(model=...)` takes precedence and skips capability routing. `ModelRouter` takes precedence over the model passed to the `Agent` constructor. When several capabilities contribute a model, Pydantic AI uses the last contribution.

The router request shares the parent run's `RunUsage` and usage limits, so its tokens, cost, and requests count toward the run limits and totals. It reserves one request for the pending parent model call.

Dynamic model selection is not supported by Pydantic AI's durable execution capabilities. Pass an explicit registered model for a durable run. Resuming a suspended provider request also requires the exact model to be supplied explicitly.

`ModelRouter` is configured in Python and does not publish an AgentSpec entry. Its choice and router fields can contain live `Model` instances, and the `Literal` output type depends on the runtime choice keys.

## Why this is in Harness

Pydantic AI core owns the selection mechanism: `ModelSelectionContext`, `ModelSelector`, the `get_model()` hook, and `SelectModel`. `ModelRouter` composes those primitives with an internal typed agent, a named menu, decision cadence, confidence fallback, error fallback, and telemetry. Those are routing policy and reusable agent behavior, so the composition belongs in Harness. Moving it into core would duplicate `SelectModel` with one opinionated way to implement its callable.

## Pydantic AI references

- [Selecting the model](/ai/capabilities/custom/#selecting-the-model)
- [Select Model](/ai/capabilities/select-model/)
- [Capabilities](/ai/capabilities/overview/)
- [Instrumentation](/ai/capabilities/instrumentation/)

## API reference

::: pydantic_ai_harness.model_router.ModelRouter

::: pydantic_ai_harness.model_router.ModelChoice
