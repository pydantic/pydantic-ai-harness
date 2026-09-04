---
title: Background Tools
description: Run selected tools concurrently -- the agent gets an immediate acknowledgment and receives the result as a follow-up message.
---

# Background Tools

`BackgroundTools` lets selected tools run in the background while the agent continues without
waiting. Use it when the model can work on something else until the result is ready.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/background_tools/)

Install the OpenAI provider before running this example:

```bash
pip install "pydantic-ai-slim[openai]" pydantic-ai-harness
```

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent('openai:gpt-5.6-sol', capabilities=[BackgroundTools()])

@agent.tool_plain(metadata={'background': True})
async def slow_research(query: str) -> str:
    """Research a topic thoroughly. Runs in the background."""
    await asyncio.sleep(60)  # stand-in for a long-running job
    return f'Research findings for {query!r}'
```

By default any tool with `metadata={'background': True}` runs in the background. The agent's instructions are augmented automatically so the model knows it shouldn't block waiting for the result.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Selecting which tools run in the background

`BackgroundTools(tools=...)` accepts the standard [`ToolSelector`](/ai/api/pydantic-ai/tools/#pydantic_ai.tools.ToolSelector):

```python
from pydantic_ai_harness import BackgroundTools

# By metadata key (default)
BackgroundTools()                                 # tools with metadata={'background': True}
BackgroundTools(tools={'background': True})       # explicit form
BackgroundTools(tools={'kind': 'research'})       # custom metadata key

# By name
BackgroundTools(tools=['slow_research', 'deep_dig'])

# By predicate
BackgroundTools(tools=lambda ctx, td: td.name.startswith('research_'))
```

### Marking tools in bulk

Combine with [`SetToolMetadata`](/ai/capabilities/set-tool-metadata/) or `FunctionToolset.with_metadata(...)` to mark several tools as background without touching individual definitions:

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai_harness import BackgroundTools

async def deep_research(query: str) -> str:
    return f'Research findings for {query!r}'

async def crawl_site(url: str) -> str:
    return f'Crawled {url}'

research_tools = FunctionToolset([deep_research, crawl_site]).with_metadata(background=True)
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[research_tools],
    capabilities=[BackgroundTools()],
)
```

## Result delivery

If the run remains active, a finished background tool produces a follow-up message:

- On success: `Background tool 'X' (task <id>) completed.\nResult: <return value>`
- On failure: `Background tool 'X' (task <id>) failed: <error>`

The task ID matches the acknowledgment. The follow-up is user content, not another tool return.
`ToolReturn.return_value` and `ToolReturn.content` remain model-visible, including multimodal
content. Application-only `ToolReturn.metadata` and deferred tool names from `ToolReturn.tools` are
not carried into the follow-up. Retries and deferred calls are reported as text failures. Expected
tool errors include
their message. Unexpected exceptions are logged for the application, while the model sees only
their type. Raised exceptions become failure results. Cancelling one background tool does not
cancel its siblings; call `ctx.cancel()` when a background tool needs to stop the run and all live
background tasks.

## Execution behavior

Normal completion waits for background tasks and delivers their follow-ups. Concurrent runs track
their tasks separately. If a run pauses for [deferred tools](/ai/tools-toolsets/deferred-tools/) or
ends through cancellation, a usage limit, or an error, live tasks are cancelled and their results
are dropped. Run cleanup waits for their async tasks to finish, so async tools must propagate
cancellation. Suppressing cancellation can keep cleanup open.

!!! warning
    Python cannot stop a synchronous tool's worker thread, so it may continue after the cancelled
    run returns.

    A synchronous background tool runs concurrently with the agent. Make mutable dependencies and
    other shared state it uses thread-safe.

## Limitations

- **Streaming**: `run_stream()` waits for live background tasks before it returns, then drops their results because it does not take the extra model turn required for delivery. Use `agent.run()` or a driven `agent.iter()` loop when result delivery is required.
- **Realtime**: Realtime sessions already execute tools concurrently. Selected tools stay on the
  realtime session's native tool-result path, so their original result content is preserved and
  they do not return the background acknowledgment.
- **Result hooks and tracing**: The follow-up user message does not pass through tool-result or
  tool-error hooks. A wrap-based capability nested inside `BackgroundTools` can inspect the handler
  outcome; capabilities outside it observe the immediate acknowledgment. Screen and bound the result
  inside the tool when enforcement must not depend on ordering.

## Durable execution

`BackgroundTools` works with Temporal durable execution. A replay rebuilds the run-local background
task while Temporal restores the tool handler from workflow history.

With DBOS, ordinary function tools are not automatically durable steps. Delegate the durable work
inside a background tool to an explicit DBOS step.

A tool handler running inside a durable activity or task must not call `ctx.enqueue()`. Replay
restores the handler's return value, not messages enqueued while the handler ran.

## API

```python {test="skip"}
BackgroundTools(
    tools: ToolSelector = {'background': True},
)
```

## Agent spec (YAML/JSON)

Install Agent spec support before using this example:

```bash
pip install "pydantic-ai-slim[spec]" pydantic-ai-harness
```

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
capabilities:
  - BackgroundTools: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent.from_file('agent.yaml', custom_capability_types=[BackgroundTools])
```

## Further reading

- [Pydantic AI message history -- injecting messages mid-run](/ai/core-concepts/message-history/#injecting-messages-mid-run) -- the underlying primitive
- [Pydantic AI capabilities](/ai/capabilities/overview/)
