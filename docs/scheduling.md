---
title: Scheduling
description: Let an agent schedule work -- cron, intervals, one-shots -- and run it on time, delivering results to a callback you own.
---

# Scheduling

Let an agent schedule work, and run that work on time.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/scheduling/)

> The API may change between releases. Breaking changes ship deprecation warnings where practical.

`Scheduling` gives the model tools to create and manage schedules. `ScheduleRunner` executes them: when a schedule is due, it runs the agent with the schedule's prompt and hands the outcome to your callback.

## Usage

Install the extra (only cron expressions need it; interval and one-shot schedules work without it):

```bash
pip install 'pydantic-ai-harness[scheduling]'
```

Attach the capability, then run the runner:

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness.scheduling import ScheduleResult, ScheduleRunner, Scheduling, SqliteScheduleStore

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Scheduling(store=SqliteScheduleStore('schedules.db'))],
)


def show(result: ScheduleResult) -> None:
    print(f'{result.schedule.name}: {result.status}')


async def main():
    await agent.run('Every weekday at 9am, summarize our open pull requests.')

    runner = ScheduleRunner(agent, deps=None, on_result=show)
    await runner.run_until_stopped()


asyncio.run(main())
```

The model calls `create_schedule` with a name, a prompt, and a schedule string. The runner claims due schedules once per `tick_interval` (default 60 seconds) and executes each one as a fresh, isolated agent run.

You can also create schedules from code: build a `Schedule` and `add` it to the store the runner reads.

## Schedule forms

| Form | Example | Runs |
|---|---|---|
| `every <N><m\|h\|d>` | `every 30m` | repeatedly, on a fixed interval |
| `in <N><m\|h\|d>` | `in 2h` | once, that far from now |
| ISO 8601 datetime | `2026-09-01T09:00` | once, at that time |
| Five-field cron | `0 9 * * MON-FRI` | repeatedly, per the expression |

Naive datetimes and cron expressions are interpreted in the schedule's IANA timezone (`Scheduling(timezone=...)`, UTC by default). Cron occurrences keep their wall-clock hour across DST transitions.

## Execution semantics

- **At most once.** A schedule's next occurrence is advanced and saved before the agent runs, so a crash mid-run skips an occurrence instead of running it twice.
- **No automatic retry.** A failed run records `last_error` and the schedule keeps its next occurrence; for recurring work, the next occurrence is the retry.
- **No overlap.** A schedule still running when it comes due again is skipped, never run concurrently with itself.
- **One runner per store.** At-most-once and no-overlap hold within a single runner process. Two runners sharing a store can both claim the same occurrence. The store carries no per-user scoping: every schedule belongs to the agent's principal, so give each tenant its own store.
- **No backlog replay.** A recurring schedule overdue beyond `misfire_grace` (default 10 minutes) runs once now and continues from the next future occurrence. An overdue one-shot is recorded as `missed` instead of running hours late. Resuming a paused schedule continues from its next future occurrence.
- **Bounded runs.** `max_runs` counts attempts; when reached, the schedule completes. `run_timeout` and per-schedule or runner-wide `usage_limits` cap each run's wall-clock time and spend.
- Empty output is a success, not an error.

## Delivering results

Every outcome is stored on the schedule (`last_status`, `last_output`, `last_error`) and passed to `on_result`, sync or async. A callback that raises is recorded as `last_delivery_error`; it never fails the run.

`deliver_to` is an opaque hint. The harness does not interpret it; your callback routes on it:

```python
from pydantic_ai_harness.scheduling import ScheduleResult


async def deliver(result: ScheduleResult) -> None:
    target = result.schedule.deliver_to or 'stdout'
    print(f'[{target}] {result.schedule.name}: {result.output}')
```

## Scheduling tools inside scheduled runs

Inside a scheduled run, this capability's scheduling tools and instructions are absent. The application can still expose other tools that write to the same store, so this guard only prevents access through `Scheduling` itself.

## Custom stores and concurrent writers

`Schedule.version` is store bookkeeping for optimistic concurrency control. A `ScheduleStore.save()` implementation replaces a record only when the supplied version matches the stored version, persists it with `version + 1`, raises `ScheduleConflictError` for a stale version, and raises `ValueError` for an unknown id. Writers re-read and retry after conflicts so updates apply to current state.

Use one runner per store. Versioned saves protect concurrent tool and runner updates within that design, but do not coordinate claims across runner processes.

## Driving the runner from outside

`tick()` claims and executes everything due, then returns. Call it from system cron, a workflow engine, or serverless infrastructure instead of keeping `run_until_stopped()` alive:

Invocations must not overlap for a given store; schedule the external trigger accordingly.

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness.scheduling import ScheduleRunner, Scheduling, SqliteScheduleStore

store = SqliteScheduleStore('schedules.db')
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Scheduling(store=store)])


async def main():
    await ScheduleRunner(agent, deps=None, store=store).tick()


asyncio.run(main())
```

Durability capabilities on the agent compose transparently: scheduled runs execute like any other run.

## Agent spec

```yaml
capabilities:
  - Scheduling: {backend: sqlite, database: schedules.db, timezone: Europe/Berlin}
```

Load it with `Agent.from_spec(..., custom_capability_types=[Scheduling])`.

## See also

- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Subagents](subagents.md) -- another capability that runs prompts as fresh, isolated agent runs

## API reference

::: pydantic_ai_harness.scheduling.Scheduling

::: pydantic_ai_harness.scheduling.ScheduleRunner
