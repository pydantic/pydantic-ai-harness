---
title: Audit Log
description: Record a structured, redacted trail of every tool call and run outcome to a pluggable sink for audit, cost attribution, and eval datasets.
---

# Audit Log

`AuditLog` records a structured, redacted trail of every tool call and run outcome to a pluggable sink -- what each tool was called with and what it returned, kept as a durable record you can query for audit and compliance, attribute cost against, or mine into evaluation datasets.

It is the content complement to two things Pydantic AI already gives you: OpenTelemetry spans (observability, emitted to a spans backend and gated behind `trace_include_content`) and [step persistence](step-persistence.md) (boundary events and resumable snapshots, which deliberately omit tool arguments and results). `AuditLog` fills the space between them -- durable, queryable, redacted content -- without standing up a spans backend.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/audit_log/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_harness.audit_log import AuditLog, InMemoryAuditSink

sink = InMemoryAuditSink()
agent = Agent('openai:gpt-5', capabilities=[AuditLog(sink=sink, agent_name='librarian')])

result = await agent.run('look up the release notes')

for call in await sink.list_tool_calls(run_id=result.run_id):
    print(call.tool_name, call.arguments, call.result)
run = await sink.get_run(run_id=result.run_id)
print(run.outcome, run.total_tokens)
```

## What it records

- A `ToolCallRecord` per tool call: `tool_name`, redacted `arguments` (size-bounded JSON), `result` or `error`, `started_at` / `ended_at`, and the `conversation_id` / `parent_run_id` / `agent_name` identity stack.
- A `RunAuditRecord` per run: `outcome` (`completed` / `failed`), `error` on failure, and `input_tokens` / `output_tokens` / `total_tokens` read from `RunContext.usage`.

Both records carry the same `run_id` / `conversation_id` / `parent_run_id` identity as [step persistence](step-persistence.md), so audit records join against runs and step events.

## Sinks

`AuditSink` is a small async protocol (`record_tool_call`, `record_run`, `list_tool_calls`, `get_run`). Three stdlib reference sinks ship here:

- `InMemoryAuditSink` -- process-local, for tests and single-process runs.
- `JsonlAuditSink(path)` -- append-only JSON Lines, one record per line.
- `SqliteAuditSink(database=...)` or `SqliteAuditSink(connection=...)` -- a `tool_calls` table and a `runs` table in one SQLite file.

To send records to a warehouse, an object store, or any other backend, implement the four `AuditSink` methods and pass it via `sink=`. Pass a `sink_resolver` to choose the sink per run from the [`RunContext`](/ai/agents/#run-context) (e.g. a per-tenant sink), mirroring `Memory.store_resolver`.

## Redaction

Each argument passes through a `redactor` -- `(arg_name, value) -> value` -- before it is serialized. The default, `default_secret_redactor`, replaces values whose key names a secret (`api_key`, `token`, `password`, `authorization`, ...) with `***`. Supply your own `redactor` for a different policy, and set `max_value_chars` to bound each argument value, and the result and error text.

```python
def keep_only_ids(key: str, value: object) -> object:
    return value if key.endswith('_id') else '***'


AuditLog(sink=sink, redactor=keep_only_ids, max_value_chars=500)
```

## Lineage across delegation

When an orchestrator's tool calls a delegate's `Agent.run(...)`, the delegate's records carry `parent_run_id` set to the orchestrator's `run_id`, so a tree of delegated runs reconstructs from the records alone. The link is inferred within the process and needs no manual threading.

## Composition with step persistence

`AuditLog` and [step persistence](step-persistence.md) compose on one agent: boundary events and resumable snapshots there, redacted content here. They read the same identity, so their records line up.

## What it does not do

- It does not emit OpenTelemetry spans. Pydantic AI's `Instrumentation` capability already spans agent runs and tool calls; `AuditLog` is a durable record you can query, not a trace.
- It does not resume runs. Use [step persistence](step-persistence.md) for continuable snapshots.
- It does not clean up old records. Retention is the caller's responsibility.
- It redacts arguments by key only -- results and error text carry no keys, so the `redactor` does not see them. Both are size-bounded by `max_value_chars`; keep secrets out of return values, or wrap the sink to scrub them.

## Related

- [Capabilities overview](index.md)
- [Step Persistence](step-persistence.md)
- [Memory](memory.md)

## API reference

::: pydantic_ai_harness.audit_log.AuditLog
