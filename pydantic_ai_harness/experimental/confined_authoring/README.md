# ConfinedAuthoring

> [!WARNING]
> **Experimental.** This capability lives under `pydantic_ai_harness.experimental` and may
> change or be removed in any release, without a deprecation period. Import it from the
> experimental path -- there is no top-level export:
>
> ```python
> from pydantic_ai_harness.experimental.confined_authoring import ConfinedAuthoring
> ```
>
> Importing any experimental capability emits a `HarnessExperimentalWarning`. Silence **all**
> harness experimental warnings with a single filter (no per-capability lines needed):
>
> ```python
> import warnings
> from pydantic_ai_harness.experimental import HarnessExperimentalWarning
>
> warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
> ```

Let an agent author, validate, and persist its own **sandboxed** tools -- and call them on
its next run.

## Installation

This capability runs authored slots in the [Monty](https://github.com/pydantic/monty) sandbox,
so it needs the same extra `CodeMode` uses (no new extra is added):

```bash
uv add "pydantic-ai-harness[code-mode]"
```

## The problem

An agent often discovers, mid-task, that it wants a tool its host does not have. Two existing
options each give up something:

- [`RuntimeAuthoring`](../authoring/) lets the agent write a real capability, but that
  capability runs arbitrary Python **in the host process**. It fits when the agent already
  runs shell commands and edits files -- the same trust boundary -- but not when the authoring
  model is untrusted, when authored tools must be isolated per tenant, or when the host wants
  the authored tool to reach only a narrow set of host functions.
- [`CodeMode`](../../code_mode/) runs model-written Python in a Monty sandbox, but its
  `run_code` tool is ephemeral: nothing the agent writes survives the call, so it cannot grow
  a durable tool.

`ConfinedAuthoring` fills the gap those two leave: authored tools that are **typed and
validated**, **sandboxed**, **persistent**, and reach the host only through a **capability-scoped,
default-deny allowlist** of injected functions.

## The solution

`ConfinedAuthoring` exposes three tools to the model:

- `author_tool_slot(name, description, code, parameters, uses, returns)` -- author a tool. `code`
  is a Monty script (a subset of Python): it reads each declared parameter as a bound variable,
  calls the injected functions it lists in `uses`, and its final expression becomes the tool's
  return value. The slot is validated immediately and, on success, becomes callable on the next
  run.
- `list_tool_slots()` -- list authored slots with their status and any validation error.
- `disable_tool_slot(name)` -- stop serving a slot.

The host declares the pool of `InjectedFunction`s that authored slots may call. A slot reaches
the host only through the subset of that pool it declares in `uses`. Nothing else is available
inside the sandbox: no import, filesystem, environment, clock, subprocess, or network.

```python
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.experimental.confined_authoring import ConfinedAuthoring, InjectedFunction


async def http_get(ctx: RunContext[None], kwargs: dict[str, object]) -> object:
    # Real host code -- the only reach an authored slot has out of the sandbox.
    return {'status': 200, 'url': kwargs['url']}


authoring = ConfinedAuthoring[None](
    directory=Path('.slots'),
    functions=[
        InjectedFunction(
            name='http_get',
            call=http_get,
            parameters={'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
            returns={'type': 'object'},
            description='Fetch a URL and return the response.',
        )
    ],
)
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[authoring])
```

The model can now author, for example, a `check_site` tool whose Monty code is:

```python
resp = await http_get(url=url)
{'up': resp['status'] == 200, 'url': url}
```

with `parameters=[{'name': 'url', 'type': 'string'}]`, `uses=['http_get']`, and
`returns='object'`. On the next run, `check_site` is a real tool the agent (or another agent
over the same store) can call.

## The four properties

1. **Typed + validated slots.** A slot's parameters are a small JSON-schema subset (`string`,
   `integer`, `number`, `boolean`, `array`, `object`); the calling model's arguments are
   schema-validated. Before a slot is served it passes a static check: Monty's type-checker
   against the parameter and injected-function stubs (wrong argument types, calls to undeclared
   names, and an async result used without `await`), a missing-`await` scan for a discarded
   coroutine, and -- when a return type is declared -- a check that the final expression matches
   it.
2. **Sandboxed execution.** Each slot runs in a Monty sandbox via the shared
   `MontyExecutor`, the same execution loop `CodeMode` and `DynamicWorkflow` use.
3. **Persistence + health contract.** Slots persist to a `slots.json` manifest and reload each
   run. A slot moves `draft -> validated -> active`, and `last_error` stays truthful: a slot
   whose used function was removed since authoring stops being served and records why.
4. **Capability-scoped injected functions.** The host owns the pool; each slot narrows to a
   subset (`uses`). Default-deny: nothing outside that subset -- and no ambient host access --
   is reachable.

## Activation boundary

An authored slot is live on the **next** `agent.run(...)`, not the run that authored it, because
Pydantic AI resolves a run's tools once at the start. Unlike `RuntimeAuthoring`, there is
nothing to thread through: `ConfinedAuthoring` serves its own slots, so the toolset reloads the
manifest at the start of each run.

```python
history = None
while not done:
    result = await agent.run(next_prompt, message_history=history)
    history = result.all_messages()
```

Because slots persist to `directory/slots.json`, a fresh process picks them up by constructing a
new `ConfinedAuthoring` over the same `directory` and `functions`.

## Serving without authoring

`ConfinedAuthoringToolset` serves a store's slots on its own, without the authoring tools. A
host can let one agent author slots and a separate, least-privilege agent run them over a shared
`SlotStore`:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.experimental.confined_authoring import ConfinedAuthoringToolset, SlotStore

store = SlotStore(directory=Path('.slots'), functions=[...])
serving = Agent('anthropic:claude-sonnet-4-6', toolsets=[ConfinedAuthoringToolset(store=store)])
```

## Threat models this fits

- **Untrusted or adversarial authoring model.** The authored code never runs host Python; its
  only side effects are the injected functions the host chose.
- **Multi-tenant isolation.** Give each tenant its own `directory` and `functions` pool, so one
  tenant's slots cannot see another's functions or manifest.
- **Deliberately least-privilege agents.** Remove broad tools (shell, file writes) and let the
  injected-function allowlist be the only escape hatch. A slot can do exactly what its `uses`
  functions allow, and no more.

Across these, a default `max_duration_secs` cap (overridable via `resource_limits`) bounds a slot's
in-sandbox compute, so a hostile slot cannot hang the host indefinitely with a pure-CPU loop.

One caveat for the untrusted-author and multi-tenant cases: the `uses` allowlist and validation
constrain what a slot can *execute*, not the metadata the model *reads*. A slot's `description`
becomes the tool's model-visible `ToolDefinition.description`, and authored slots activate on the
next run without a separate host-promotion step, so an untrusted author sharing a store could plant
prompt-injection text that later agents see. Single-tenant use (where the author is the trusted
operator) is unaffected. A shared-store deployment with untrusted authors should gate promotion
behind a trusted host or keep host-authored tool descriptions separate from authored text -- a
first-class promotion gate is tracked as a follow-up.

## Scope

Only `tool` slots exist today. The slot record carries a `kind` field so `hook` and
`instruction` slots can be added without a manifest migration; those kinds are not implemented
yet. Return-shape is checked statically at authoring time and guarded again at execution when a
return type is declared.
