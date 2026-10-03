# Retry Policy

`RetryPolicy` retries transient tool failures: rate limits, timeouts, connection drops, and provider errors. Use it when an agent's tools call external services that fail intermittently. It retries only the tools you explicitly mark idempotent, with exponential backoff between attempts, so a retried failure slows a run down instead of failing it.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/retry_policy/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Usage

`RetryPolicy` needs no extra beyond the base package:

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.retry_policy import RetryPolicy

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        RetryPolicy(
            allow_idempotent_retries=True,
            idempotent_tools=frozenset({'web_search'}),  # retry only this tool
            max_retries=3,
            backoff_factor=0.5,  # delays of 0.5s, 1s, 2s, capped by max_backoff
        )
    ],
)
```

The example sets the two flags that enable retries. With the defaults (`allow_idempotent_retries=False` and an empty `idempotent_tools`), no tool call is retried: see [Why retries are opt-in](#why-retries-are-opt-in) below.

## Why retries are opt-in

Retrying a tool call re-runs its handler. For a tool with side effects (a file write, a charge, a sent message), the second run can apply the side effect again. `RetryPolicy` therefore assumes every tool has side effects: by default, a retryable failure is surfaced on the first attempt and no retry is attempted.

To retry a tool, mark it idempotent:

1. Set `allow_idempotent_retries=True` on the capability.
2. List the tool's name in `idempotent_tools`, or set `idempotent: True` in its `tool_overrides` entry.

Only tools marked that way are retried, up to their `max_retries`. A retryable failure on any other tool is surfaced immediately, with `on_failure` called first if you set it.

## Which failures trigger a retry

A tool call is retried when the exception from its handler matches any of:

- an instance of one of `retryable_exceptions` (default: `TimeoutError`, `ConnectionError`, `OSError`)
- an HTTP error whose status code is in `retryable_status_codes` (default: `429`, `500`, `502`, `503`, `504`); the code is read from the exception's `status_code` attribute, or from `status_code` on its `.response`
- an error whose `error_type` attribute is `rate_limit`, `timeout`, or `server_error`

Any other exception propagates immediately, unmodified.

## Backoff

Between attempts, the capability sleeps for `backoff_factor * 2**attempt` seconds (with the default `backoff_factor=0.5`: 0.5s, 1s, 2s) plus jitter of up to 25% of the delay, in either direction. Every delay is capped at `max_backoff` (default `30.0` seconds) with a minimum of `0.01` seconds unless `max_backoff` is smaller, so the backoff is bounded no matter how many attempts elapse. Both `backoff_factor` and `max_backoff` must be finite positive numbers; a zero, negative, or non-finite value raises `ValueError` at construction, in the top-level fields and in per-tool `tool_overrides` entries alike.

Each retry also logs a warning naming the tool, the attempt, and the delay.

## Options

| Field | Default | Purpose |
|---|---|---|
| `max_retries` | `3` | Retry attempts after the first; `0` disables retries for that tool |
| `backoff_factor` | `0.5` | Base delay in seconds; doubles with each attempt |
| `max_backoff` | `30.0` | Upper bound on the delay between attempts, in seconds |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | HTTP status codes that trigger a retry |
| `retryable_exceptions` | `(TimeoutError, ConnectionError, OSError)` | Exception types that trigger a retry |
| `allow_idempotent_retries` | `False` | Gate for retries after the handler has run; see [Why retries are opt-in](#why-retries-are-opt-in) |
| `idempotent_tools` | `frozenset()` | Tool names that are safe to retry after the handler has run |
| `tool_overrides` | `{}` | Per-tool retry counts, backoff, retryable errors, `idempotent`, `on_retry`, and `on_failure`; the global idempotency gate is not overridable |
| `on_retry` | `None` | Called as `on_retry(tool_name, attempt, exc)` before each retry |
| `on_failure` | `None` | Called as `on_failure(tool_name, exc)` when a retryable failure is surfaced |

## Per-tool overrides

`tool_overrides` maps a tool name to a dict of field values that replace the top-level ones for that tool alone:

```python
RetryPolicy(
    allow_idempotent_retries=True,
    tool_overrides={
        'web_search': {'max_retries': 5, 'backoff_factor': 1.0, 'idempotent': True},
        'shell': {'max_retries': 0},  # shell commands are rarely idempotent
    },
)
```

`idempotent: True` in an override marks that tool safe to retry without listing it in `idempotent_tools`.

Multiple policies nest independently and can multiply retry attempts for overlapping tools. Prefer one policy with per-tool overrides.
