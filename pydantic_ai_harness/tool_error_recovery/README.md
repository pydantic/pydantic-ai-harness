# Tool Error Recovery

Recover from tool-call errors -- retry, report to the model, fall back, or propagate -- instead of crashing the agent run.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/tool_error_recovery/)

## Motivation

Agentic tool calling can fail for many reasons, and adding error handling and retries to every tool is cumbersome. For third-party tools and MCPs, it's often impossible. `ToolErrorRecovery` is a standardized capability for AI agents to recover from errors during tool calls. Similar in spirit to [FastAPI's exception handlers](https://fastapi.tiangolo.com/tutorial/handling-errors/#install-custom-exception-handlers), but for agents.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_harness.tool_error_recovery import ToolErrorRecovery

agent = Agent('openai:gpt-5', capabilities=[ToolErrorRecovery()])
```

Out of the box, connection errors are retried, bugs are propagated, and everything else is reported to the model, preventing a crash of the agent run. Continue reading to learn how to customize the capability.

## Features

The `ToolErrorRecovery` capability supports four different recovery outcomes:

| Outcome       | Effect                                                                       |
| ------------- | ---------------------------------------------------------------------------- |
| **retry**     | Re-attempt the tool call a number of times (invisible to the model)          |
| **inform**    | Report a terminal failure to the model via `ToolFailed` (`outcome='failed'`) |
| **fallback**  | Return a substitute value as the tool result                                 |
| **propagate** | Re-raise the exception, will crash the run unless caught by another wrapper  |

## Error Classification

`ToolErrorRecovery` intercepts tool execution errors and applies a per-error reaction. A `classify` callable you supply inspects each failure and returns the intended recovery outcome:

```python
def classify(ctx: RunContext[Any], call: ToolCallPart, error: BaseException) -> RecoveryOutcome:
    if isinstance(error, SomeException):
        return RecoveryOutcome.inform(...)
    # and so on
```

Notes:

- You can classify on a tool's name and other properties of the tool call, on the exception type, or both.
- The `ctx: RunContext` parameter is included for cases in which your classification logic additionally depends on the run context.
- To not interfere with existing error handling, the following exceptions from the Pydantic AI control flow are never classified: `SkipToolExecution`, `CallDeferred`, `ApprovalRequired`, `ModelRetry`, `ToolRetryError`, `ToolFailed`, and `ToolFailedError`. Note that this list is hardcoded for now and would require updating if Pydantic AI changes their control flow exceptions.
- Special care must be taken when classifying an exception which is a subclass, be careful about the order of `isinstance(...)` calls, or use `type(...) is` instead.

## Default Behaviour

As mentioned above, control flow exceptions are never classified. Our default classifier is:

```python
def classify(ctx: RunContext[Any], call: ToolCallPart, error: BaseException) -> RecoveryOutcome:
    if isinstance(error, DEFAULT_BUG_TYPES):
        return RecoveryOutcome.propagate()
    if isinstance(error, HookTimeoutError):
        return RecoveryOutcome.inform()
    if isinstance(error, DEFAULT_TRANSIENT_TYPES):
        return RecoveryOutcome.retry(3)
    return RecoveryOutcome.inform()
```

The classification function is the key piece required to construct your own RecoveryPolicy via `RecoveryPolicy(classify=classify)`. A fully custom example could look like this:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness.tool_error_recovery import RecoveryPolicy, ToolErrorRecovery

def classify(...):
    # your classification logic

agent = Agent('openai:gpt-5', capabilities=[
    ToolErrorRecovery(
        policy=RecoveryPolicy(
            classify=classify,
            format_error=my_formatter,
            max_message_len=450,
            include_traceback=False,
            logger=my_logger,
        ),
        max_recoveries=10,
        per_tool_recoveries={"my_tool_name": 5},
    )
])
```
