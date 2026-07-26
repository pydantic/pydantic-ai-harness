---
title: Input, Output & Tool Guardrails
description: Validate the user prompt before it reaches the model, the tool calls the model makes, and the output before it reaches the caller, with allow/block/replace/retry/approve verdicts and optional parallel execution.
---

# Input, Output & Tool Guardrails

Guardrails put a validation layer on the three edges of an agent run: the prompt on its way *in* to the model, the tool calls the model makes along the way, and the output on its way *out* to the caller. Reach for them when unstructured input or output must be screened before it is acted on -- a prompt-injection attempt you never want to send, PII you must redact, an off-topic request you want to refuse cheaply, or an answer that must cite its sources before you show it. Without a guardrail the framework sends whatever the user typed and returns whatever the model produced, verbatim; a guardrail interposes a callable you control that gets the final say.

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

Agents take unstructured input from users and return unstructured output to callers. On its own the framework does not reason about "this is unsafe to send" or "this is unsafe to show" -- a prompt-injection attempt reaches the model as-is, and any output the model produces is returned untouched. You need a place to inspect the value and decide what happens next.

## The solution

Three capabilities -- `InputGuard`, `OutputGuard`, and `ToolGuard` -- each wrap a `guard` callable you supply. The guard inspects a value and returns one of five outcomes. For the run's two edges:

| Outcome | `InputGuard` | `OutputGuard` |
|---|---|---|
| **allow** | send the prompt to the model | return the output to the caller |
| **block** | skip the model call; a refusal message becomes the response | raise `OutputBlocked` |
| **replace** | rewrite the prompt sent to the model (redaction) | substitute a sanitized output |
| **retry** | -- (not valid for input) | send the output back to the model to try again |
| **approve** | -- (not valid for input) | -- (not valid for output) |

`ToolGuard` uses the same outcomes on both sides of a tool call; see [Tool calls](#tool-calls). The asymmetry between input `block` and output `block` is deliberate. Blocking the input spends no tokens, so a graceful refusal is almost always right. Blocking the output means the model already produced something you do not want exposed, so raising forces the caller to decide what to do next.

Both `InputGuard`, `OutputGuard`, and their supporting types are top-level exports:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import GuardResult, InputGuard, OutputGuard


def no_secrets(prompt: str) -> bool:
    return 'api_key' not in prompt.lower()


def no_pii(output: object) -> GuardResult:
    if 'SSN' in str(output):
        return GuardResult.block('The response contained personal data.')
    return GuardResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuard(guard=no_secrets),
        OutputGuard(guard=no_pii),
    ],
)
```

A guard returns a bare `bool` (`True` = allow, `False` = block) for the simple case, or a `GuardResult` for the richer outcomes. Guards may also be async -- return an awaitable `bool`/`GuardResult`, for example to call a moderation API.

`OutputGuard` receives the output unchanged -- no automatic stringification. For a string output the guard reads it directly; for a typed (Pydantic model) output the guard gets the model instance, so pick the serialization that fits the check (read a field, or call `output.model_dump_json()` for JSON text). This avoids the trap of `str(MyModel(...))` producing a `MyModel(field=...)` repr that hides field contents from regex-based checks.

## `GuardResult`

Construct a `GuardResult` with its classmethods, not the raw fields:

```python
from pydantic_ai_harness import GuardResult

GuardResult.allow()                 # let the value through
GuardResult.block('reason')         # refuse; `reason` is optional (a default is used otherwise)
GuardResult.replace(cleaned_value)  # substitute a sanitized value and continue
GuardResult.retry('instruction')    # ask the model to redo the output or the tool call
GuardResult.approve()               # ToolGuard arguments only: defer the call for human approval
```

The block/retry message is produced at the moment the guard decides, so it can carry the guard's own reasoning rather than a string frozen at construction time.

## Redaction (`replace`)

Return `GuardResult.replace(value)` to sanitize rather than refuse. `InputGuard` rewrites the prompt sent to the model; `OutputGuard` substitutes the output returned to the caller.

```python
def scrub_emails(text: str) -> GuardResult:
    cleaned = EMAIL_RE.sub('[email]', text)
    return GuardResult.replace(cleaned) if cleaned != text else GuardResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuard(guard=scrub_emails),   # strip PII before it reaches the model
        OutputGuard(guard=scrub_emails),  # strip PII before it reaches the caller
    ],
)
```

Input redaction requires sequential mode -- it is incompatible with `parallel=True`, since a parallel guard runs alongside a model call that has already started with the original prompt.

## Retry (`retry`)

`OutputGuard` can send a bad output back to the model instead of blocking it. Return `GuardResult.retry(instruction)` -- the instruction is the retry prompt the model sees. This reuses pydantic-ai's normal retry machinery and counts against the run's output-retry budget.

```python
def must_cite_sources(output: object) -> GuardResult:
    if not has_citations(output):
        return GuardResult.retry('Include at least one source citation.')
    return GuardResult.allow()


OutputGuard(guard=must_cite_sources)
```

## Accessing run context

A guard may take a `RunContext` as its first parameter when it needs run state -- `deps` for tenant- or role-aware policy, message history for conversation-aware checks. The parameter is detected from the signature, so prompt-only guards need not declare it:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness import InputGuard


def tenant_policy(ctx: RunContext[MyDeps], prompt: str) -> bool:
    return ctx.deps.tier == 'pro' or 'advanced-feature' not in prompt


InputGuard(guard=tenant_policy)
```

## Parallel input guards

A slow guard (an LLM classifier, a network call) run sequentially adds its latency to every turn. Set `parallel=True` to run the guard concurrently with the model call instead, overlapping the two so the guard adds no latency on the pass path. The model call is cancelled the moment the guard reports a violation.

```python
InputGuard(guard=slow_async_classifier, parallel=True)
```

Parallel mode trades tokens for latency: sequential mode never calls the model when the guard blocks, but parallel mode has already started the model call -- if the guard trips only after the model has responded, those tokens were spent. For fast local checks (regex, keyword lookup) sequential is the better default. `replace` is not available under `parallel=True`.

## Hard-fail path

`block` is the graceful path. To make the caller see an exception instead, raise from the guard:

```python
from pydantic_ai_harness import InputBlocked


def strict_guard(prompt: str) -> bool:
    if contains_credentials(prompt):
        raise InputBlocked('credentials detected')
    return True
```

Any exception raised by the guard propagates as-is -- use `InputBlocked` / `OutputBlocked` / `ToolBlocked` from this module, or your own exception types. `ToolBlocked` carries the `tool_name` and an optional `reason`.

## Tool calls

`ToolGuard` inspects both sides of a tool call: `guard` sees the validated arguments before the tool runs, `result_guard` sees what it returned before the model does.

```python
from pathlib import Path

import httpx
from pydantic_ai import Agent
from pydantic_ai_harness import GuardResult, ToolCallInfo, ToolGuard, ToolResultInfo

WORKSPACE = Path('/workspace')


def stay_in_the_workspace(call: ToolCallInfo) -> GuardResult:
    if call.name == 'write_file':
        # `resolve()` before the containment check: a prefix test on the raw
        # string accepts `/workspace/../etc/passwd`.
        target = Path(str(call.args['path'])).resolve()
        if not target.is_relative_to(WORKSPACE):
            return GuardResult.block(f'{target} is outside the workspace.')
    return GuardResult.allow()


def scrub_secrets(info: ToolResultInfo) -> GuardResult:
    # Text only. `str()` on a structured result or a `ToolReturn` yields a repr, and
    # replacing it with that string would change the result's type -- but only on the
    # calls where the pattern happened to match.
    if not isinstance(info.result, str):
        return GuardResult.allow()
    cleaned = SECRET_RE.sub('[redacted]', info.result)
    return GuardResult.replace(cleaned) if cleaned != info.result else GuardResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[ToolGuard(guard=stay_in_the_workspace, result_guard=scrub_secrets)],
)


@agent.tool_plain
def write_file(path: str, content: str) -> str:
    Path(path).write_text(content)
    return f'wrote {path}'


@agent.tool_plain
def fetch_page(url: str) -> str:
    return httpx.get(url).text
```

The outcomes map onto Pydantic AI control flow rather than a parallel mechanism:

| Outcome | `guard` (arguments) | `result_guard` (result) |
|---|---|---|
| **allow** | run the tool | return the result unchanged |
| **block** | skip execution; the refusal message becomes the tool result (`SkipToolExecution`) | the refusal message replaces the result |
| **replace** | run the tool with substituted arguments (a mapping); the call recorded in the message history keeps the model's original arguments | substitute a sanitized result |
| **retry** | ask the model to redo the call (`ModelRetry`) | ask the model to redo the call (`ModelRetry`); the tool has already run once, so its side effects have happened and the retry runs it again |
| **approve** | defer the call for human approval (`ApprovalRequired`) | -- (the tool has already run) |

`block` is graceful on both stages: the agent sees the refusal text where it expected a tool result, so it can explain the refusal or try another approach. To fail the run instead, raise `ToolBlocked` from the guard.

### Human in the loop

Pydantic AI already owns the approval round trip: a call raising `ApprovalRequired` is held back, the run finishes with a `DeferredToolRequests` output, and you resume it with the human's answers. `ToolGuard` plugs into that rather than inventing a second mechanism, which means approvals a guard asks for and tools marked `requires_approval=True` arrive in the same place.

```python
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai_harness import GuardResult, ToolCallInfo, ToolGuard


def confirm_production(call: ToolCallInfo) -> GuardResult:
    if call.args.get('env') == 'prod':
        return GuardResult.approve()
    return GuardResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[ToolGuard(guard=confirm_production)],
    output_type=[str, DeferredToolRequests],
)


@agent.tool_plain
def deploy(env: str) -> str:
    return f'deployed to {env}'


deferred = await agent.run('deploy the new build')
if isinstance(deferred.output, DeferredToolRequests):
    approvals = {
        call.tool_call_id: True if operator_says_yes(call) else ToolDenied('not on a Friday')
        for call in deferred.output.approvals
    }
    final = await agent.run(
        message_history=deferred.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals=approvals),
    )
```

A denial reaches the model as the tool's result, so the agent can explain itself or try something else. On the resumed run the guard is evaluated again, and `approve` becomes a no-op for a call the human already cleared -- every other verdict still applies, so a policy that has since changed its mind can still block an approved call.

Two shapes of approval, and which to reach for:

| | Deferred (`GuardResult.approve()`) | In-process (an async guard) |
|---|---|---|
| The run | ends, then resumes from `message_history` | stays open; the tool call awaits |
| Fits | HTTP APIs, queues, durable execution, anything that cannot hold a process open | CLIs, TUIs, desktop apps, a websocket to an operator |
| Human answer | `DeferredToolResults` (`True`, `ToolDenied`, or `ToolApproved(override_args=...)`) | whatever the guard returns |

The in-process shape needs nothing extra -- a guard may be async, so it can await the human directly:

```python
async def ask_the_operator(call: ToolCallInfo) -> GuardResult:
    if await operator_approves(call.name, call.args):
        return GuardResult.allow()
    return GuardResult.block('The operator declined this action.')


ToolGuard(guard=ask_the_operator)
```

Pydantic AI also offers approval without a guard at all: `requires_approval=True` on a tool, or `ApprovalRequiredToolset` for a synchronous predicate over a whole toolset. Reach for `ToolGuard` when the decision is async, needs `deps`, or should sit alongside the other verdicts.

Two fields narrow what a guard sees:

```python
ToolGuard(
    guard=stay_in_the_workspace,
    tools=['write_file', 'run_shell'],  # guard only these; None (default) guards every tool
    hidden=['delete_everything'],       # withhold these from the model entirely
)
```

`hidden` is not a blocklist with a nicer name. A hidden tool is dropped from the definitions sent to the model, so it costs no tokens and the model never attempts it; a blocked tool stays visible and the model learns it was refused. Hiding takes a static list of names -- for policy that depends on `deps` or on the arguments, use `guard`.

### What a tool guard does not see

Three kinds of call never reach the execution hooks, so neither `guard` nor `result_guard` is consulted for them:

- **Output tools**, which produce the agent's structured output. Screen that with `OutputGuard`.
- **External and deferred tools**, which the run hands back to your application in `DeferredToolRequests` instead of executing. Pydantic AI rejects them before any execution hook runs, so a guard cannot vet the arguments -- your application is the thing that executes them, and the check belongs there. `hidden` *does* cover them, since it works on the tool definitions.
- **Provider-side builtin tools** such as web search, which run inside the provider and come back as builtin call/return parts rather than tool executions.

A `ToolGuard` is a control over tools this run executes. For the ones it does not, `hidden` is the lever that still applies.

## Streaming

`OutputGuard` inspects the **final** output only -- during `run_stream()` partial chunks reach the caller before the guard runs, so a `block` or `replace` verdict cannot un-send content already streamed. Use `run()` / `run_sync()` when the output must be screened before any of it is exposed. `GuardResult.retry()` is **not** supported under `run_stream()` and surfaces there as `UnexpectedModelBehavior`. `InputGuard` (including `parallel=True`) works the same in streamed and non-streamed runs.

## Tracing

`replace` and `block` are recorded as spans on the active OpenTelemetry tracer, so a redaction or refusal shows up in [Logfire](https://pydantic.dev/logfire) traces (`guardrail redacted input`, `guardrail blocked output`, and so on) with `guardrail.*` attributes. Content attributes -- the original/replacement values for a redaction and the refusal `message` for a block -- are attached **only** when `RunContext.trace_include_content` is enabled, since these can quote the very content the guard exists to keep out of traces.

Tool spans add a `guardrail.tool` attribute naming the tool, and `approve` records `guardrail deferred tool args` with no content attributes at all.

`OutputGuard` positions its block/redact spans so they are always captured by an enclosing `Instrumentation` span regardless of capability order, while `InputGuard` runs innermost so any capability that morphs messages (a prompt rewriter, a context manager) runs first and the guard sees the final prompt the model will receive. `ToolGuard` also runs innermost, which puts it last among argument hooks (it sees the arguments every other capability has finished modifying) and first among result hooks (it sees the raw tool result, before a capability such as [`OverflowingToolOutput`](overflowing-tool-output.md) truncates or offloads it).

## Relationship to `pydantic-ai-shields`

[`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields) provides opinionated implementations on top of these primitives (prompt-injection detectors, PII scrubbers, keyword blocklists). Use the guardrails here when you want to plug in your own validation logic; reach for shields when you need a batteries-included detector.

## API

```python
InputGuard(
    guard,              # Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]]
    parallel=False,     # run concurrently with the model call
)

OutputGuard(
    guard,              # Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]]
)

ToolGuard(
    guard=None,         # inspects a ToolCallInfo before the tool runs
    result_guard=None,  # inspects a ToolResultInfo after it runs
    tools=None,         # Sequence[str] | None -- restrict both guards to these tool names
    hidden=(),          # Sequence[str] -- withhold these tools from the model entirely
)
```

The guard callable takes the inspected value -- the prompt for `InputGuard`, the output for `OutputGuard`, a `ToolCallInfo` or `ToolResultInfo` for `ToolGuard` -- optionally preceded by a `RunContext`. `InputGuardFunc`, `OutputGuardFunc`, `ToolGuardFunc`, and `ToolResultGuardFunc` are the exported signature aliases; `GuardrailError` is the base for `InputBlocked`, `OutputBlocked`, and `ToolBlocked`.

Source: [`pydantic_ai_harness/guardrails/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/guardrails/).
