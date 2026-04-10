# Tool Error Recovery Capability Plan

## Goal

Provide a reusable `AbstractCapability` subclass that catches unhandled tool execution errors and recovers gracefully, preventing agent run crashes and enabling the model to self-correct.

| Hook used | Purpose |
|---|---|
| `wrap_tool_execute` | Retry logic (re-invokes the tool handler on failure) |
| `on_tool_execute_error` | Inform and fallback strategies (post-failure recovery) |

## Design Decisions

### Three configurable strategies

- **`'inform'`** (default): Return a descriptive error message to the model so it can adjust its approach. This mirrors Open SWE's `ToolErrorMiddleware` which returns errors as tool messages.
- **`('retry', N)`**: Retry the tool call up to *N* times before falling back to `inform`. This mirrors Mastra's retryable error pattern. Implemented in `wrap_tool_execute` since it needs to re-invoke the handler.
- **`('fallback', value)`**: Return a static value on error. Useful for optional/non-critical tools where a default is acceptable.

### Per-tool configuration via `tool_strategies`

A `tool_strategies: dict[str, Strategy]` field allows different strategies per tool name. Any tool not listed uses `default_strategy`. This enables fine-grained control: retry flaky APIs, use fallbacks for optional lookups, and inform for everything else.

### Retry implemented in `wrap_tool_execute`, not `on_tool_execute_error`

`on_tool_execute_error` fires after the tool has already failed and cannot re-invoke it. Only `wrap_tool_execute` has access to the handler and can call it multiple times. When all retries are exhausted, the wrapper falls back to `inform` (returning the error message as the tool result).

### Per-run state isolation via `for_run()`

Retry counts are tracked per-run in `_retry_counts`. The `for_run()` method returns a fresh instance for each agent run, preventing state leakage between runs (same pattern as `StuckLoopDetection`).

### `include_traceback` option

By default, error messages sent to the model contain only the exception type and message. Setting `include_traceback=True` adds the full Python traceback, useful for debugging but wasteful on tokens in production.

### Convenience constructors

`retry(max_retries=3)` and `fallback(value=None)` provide readable shorthand for tuple strategy construction, with validation at creation time.

### Spec-serializable

The capability only takes simple configuration (strings, tuples, dicts, bools), so `get_serialization_name()` returns `'ToolErrorRecovery'` for YAML/JSON spec support.

## Prior Art

- **Open SWE** (`ToolErrorMiddleware`): Catches exceptions during tool calls and returns errors as `ToolMessage` with `status="error"` so the LLM can self-correct.
- **Mastra**: Emits retryable error events with `retryDelay` and queues follow-up messages to the agent.
- **LangGraph**: Tool errors returned to state for agent inspection.
- **smolagents**: `ToolError` shown to agent for approach adjustment.

## Future Work (out of scope for this PR)

- **Exponential backoff** for retry delays (useful for rate-limited APIs).
- **Custom error formatters** via a callback for domain-specific error messages.
- **Error budgets** to abort the run after too many total tool failures across all tools.
- **Integration with `StuckLoopDetection`** to detect retry loops.

## References

- Harness issue #61: Tool Error Recovery capability
- pydantic-ai `AbstractCapability.on_tool_execute_error` and `wrap_tool_execute` hooks
