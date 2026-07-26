---
title: Input & Output Guardrails
description: Validate the user prompt before it reaches the model and the model output before it reaches the caller, with allow/block/replace/retry verdicts and optional parallel execution.
---

# Input & Output Guardrails

Guardrails put a validation layer on the two edges of an agent run: the prompt on its way *in* to the model, and the output on its way *out* to the caller. Reach for them when unstructured input or output must be screened before it is acted on -- a prompt-injection attempt you never want to send, PII you must redact, an off-topic request you want to refuse cheaply, or an answer that must cite its sources before you show it. Without a guardrail the framework sends whatever the user typed and returns whatever the model produced, verbatim; a guardrail interposes a callable you control that gets the final say.

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

Agents take unstructured input from users and return unstructured output to callers. On its own the framework does not reason about "this is unsafe to send" or "this is unsafe to show" -- a prompt-injection attempt reaches the model as-is, and any output the model produces is returned untouched. You need a place to inspect the value and decide what happens next.

## The solution

Two capabilities -- `InputGuard` and `OutputGuard` -- each wrap a `guard` callable you supply. The guard inspects a value (the prompt, or the output) and returns one of four outcomes:

| Outcome | `InputGuard` | `OutputGuard` |
|---|---|---|
| **allow** | send the prompt to the model | return the output to the caller |
| **block** | skip the model call; a refusal message becomes the response | raise `OutputBlocked` |
| **replace** | rewrite the prompt sent to the model (redaction) | substitute a sanitized output |
| **retry** | -- (not valid for input) | send the output back to the model to try again |

The asymmetry between input `block` and output `block` is deliberate. Blocking the input spends no tokens, so a graceful refusal is almost always right. Blocking the output means the model already produced something you do not want exposed, so raising forces the caller to decide what to do next.

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

## Several guards at once

A `guard` takes one callable or a sequence of them. In a chain each runs in
order, and what happens next depends on the verdict:

| Verdict | Effect on the chain |
|---|---|
| `allow` | move on to the next guard |
| `replace` | the rest of the chain inspects the substituted value |
| `block` / `retry` | the chain ends there, since neither leaves a value to judge |

`replace` threading forward is what makes order meaningful: put a redactor
first and everything after it sees the cleaned text.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import InputGuard
from pydantic_ai_harness.guardrails.detectors import blocked_keywords, redact_secrets

agent = Agent(
    'openai:gpt-5.4',
    capabilities=[InputGuard(guard=[redact_secrets, blocked_keywords(['internal-only'])])],
)
```

One `InputGuard` holding three checks is not the same as three `InputGuard`
capabilities: the chain is one place in the capability list, one ordering
decision, and one set of spans, and only the chain threads a redaction into the
check that follows it.

An empty sequence is refused. A guardrail that inspects nothing reads as
configured and behaves as absent, which is worth an error rather than a quiet
pass.

## Ready-made detectors

`pydantic_ai_harness.guardrails.detectors` holds checks you would otherwise
write again: they are plain functions returning a `GuardResult`, so they drop
into a chain beside your own.

```python
from pydantic_ai_harness.guardrails import detectors

detectors.redact_secrets  # rewrites API keys, tokens, and whole private-key blocks out of text
detectors.redact_personal_data  # rewrites emails, card numbers, IBANs, US SSNs
detectors.blocked_keywords(['internal-only'])  # refuses text containing any of them
```

The two redactors rewrite rather than refuse, which is the useful default:
an agent that quoted a key back has still done the work, and blocking the answer
loses it while leaving the key in the message history either way.

Each is `secrets()` and `personal_data()` called with the defaults. Use those factories directly when you need to narrow or extend them:

```python
from pydantic_ai_harness.guardrails.detectors import secrets

secrets(only=['aws_access_key', 'private_key'])
secrets(extra={'internal_ticket': r'INT-\d{4}'})
secrets(placeholder='***')  # the default is `[redacted:{name}]`, which says what it removed
```

Detectors read text, so they suit a prompt directly. An agent output may be a
model instance, and substituting a scrubbed string for one would change its
type, so `for_text` makes you say what should happen:

```python
from pydantic_ai_harness import OutputGuard
from pydantic_ai_harness.guardrails.detectors import for_text, redact_secrets

OutputGuard(guard=for_text(redact_secrets))  # raises on a non-string output
OutputGuard(guard=for_text(redact_secrets, on_other='allow'))  # skips it deliberately
```

**What these do not do.** A regex finds a credential because credentials have a
shape. It does not find a prompt injection, which is ordinary language, and it
does not understand context, so a redactor will sometimes take a string that
only looks like a key. They are one cheap layer, not the answer.

## `GuardResult`

Construct a `GuardResult` with its classmethods, not the raw fields:

```python
from pydantic_ai_harness import GuardResult

GuardResult.allow()                 # let the value through
GuardResult.block('reason')         # refuse; `reason` is optional (a default is used otherwise)
GuardResult.replace(cleaned_value)  # substitute a sanitized value and continue
GuardResult.retry('instruction')    # OutputGuard only: ask the model to redo the output
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

Any exception raised by the guard propagates as-is -- use `InputBlocked` / `OutputBlocked` from this module, or your own exception types.

## Streaming

`OutputGuard` inspects the **final** output only -- during `run_stream()` partial chunks reach the caller before the guard runs, so a `block` or `replace` verdict cannot un-send content already streamed. Use `run()` / `run_sync()` when the output must be screened before any of it is exposed. `GuardResult.retry()` is **not** supported under `run_stream()` and surfaces there as `UnexpectedModelBehavior`. `InputGuard` (including `parallel=True`) works the same in streamed and non-streamed runs.

## Tracing

`replace` and `block` are recorded as spans on the active OpenTelemetry tracer, so a redaction or refusal shows up in [Logfire](https://pydantic.dev/logfire) traces (`guardrail redacted input`, `guardrail blocked output`, and so on) with `guardrail.*` attributes. Content attributes -- the original/replacement values for a redaction and the refusal `message` for a block -- are attached **only** when `RunContext.trace_include_content` is enabled, since these can quote the very content the guard exists to keep out of traces.

`OutputGuard` positions its block/redact spans so they are always captured by an enclosing `Instrumentation` span regardless of capability order, while `InputGuard` runs innermost so any capability that morphs messages (a prompt rewriter, a context manager) runs first and the guard sees the final prompt the model will receive.

## Relationship to `pydantic-ai-shields`

`pydantic-ai-shields` ships each detector as its own capability. The equivalents here are functions instead, which is what lets several run as one chain, share a redaction, and sit beside a guard you wrote.

Two things it has that this deliberately does not. Its `PromptInjection` matches phrases like "ignore previous instructions": injection is ordinary language, so a pattern list catches the examples and misses the attack while flagging a pasted log, and a check that reads as protection without being it is worse than none. Its `NoRefusals` blocks the model from declining, which is a decision about what an agent may say rather than a guardrail on data, and not one to make a default.

## API

```python
InputGuard(
    guard,              # one guard, or a sequence run in order
    parallel=False,     # run concurrently with the model call
)

OutputGuard(
    guard,              # one guard, or a sequence run in order
)
```

The guard callable takes the inspected value -- the prompt for `InputGuard`, the output for `OutputGuard` -- optionally preceded by a `RunContext`. `InputGuardFunc` and `OutputGuardFunc` are the exported signature aliases; `GuardrailError` is the base for `InputBlocked` and `OutputBlocked`.

Source: [`pydantic_ai_harness/guardrails/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/guardrails/).
