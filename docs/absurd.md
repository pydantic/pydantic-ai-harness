---
title: Absurd Durability
description: Make Pydantic AI agents durable with the pydantic-ai-absurd integration.
---

# Absurd Durability

`AbsurdDurability` makes a Pydantic AI agent durable when it runs inside an
Absurd task. Install the optional dependency and add the capability to an
agent when its model calls and tool calls should resume from Absurd
checkpoints after a worker restart.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/absurd/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
uv add "pydantic-ai-harness[absurd]"
```

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import AbsurdDurability

agent = Agent(
    'openai:gpt-5.6-sol',
    name='worker',
    capabilities=[AbsurdDurability()],
)
```

Import `AbsurdDurability` from either `pydantic_ai_harness` or
`pydantic_ai_harness.absurd`. Import `AbsurdParallelExecutionMode` only from
`pydantic_ai_harness.absurd`. Both are the exact objects from
[`pydantic-ai-absurd`](https://pypi.org/project/pydantic-ai-absurd/).
The implementation, task integration, and checkpoint semantics all come from
that package. Harness does not wrap or duplicate the durability runtime.
