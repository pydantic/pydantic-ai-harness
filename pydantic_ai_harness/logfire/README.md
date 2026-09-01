# Logfire-backed capabilities

Drive agent configuration from [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/),
so you can iterate on it from the Logfire UI -- versioned, labelled, and rolled out -- without redeploying.

Two capabilities cover the two managed surfaces:

- [`AgentControl`](#agentcontrol) -- the whole agent config (`agent__<name>`): instructions, model,
  model settings, and the LLM-facing definitions of the agent's tools, as one variable matching
  Logfire's Agent Control UI
- [`ManagedPrompt`](#managedprompt) -- one managed prompt (`prompt__<name>`), shareable across agents
  (legacy: a prompt-only `AgentControl` now covers this)

Each capability takes an optional `name` that selects its backing variable. **When you omit it, the
name defaults to the agent's own `name`** -- so `AgentControl()` on an
`Agent(..., name='checkout_assistant')` resolves `agent__checkout_assistant`:

```python
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import AgentControl

agent = Agent(
    'openai:gpt-5',
    name='checkout_assistant',
    tools=[...],
    capabilities=[AgentControl(label='production')],  # -> agent__checkout_assistant
)
```

The agent must then have a `name` (pass `name=` or let pydantic-ai infer it from the assignment);
a nameless capability on a nameless agent raises a clear error. The derived name is normalized
exactly the way the Logfire UI normalizes observed agent names (lowercased, non-alphanumeric runs
collapsed to `_`), so the variable the SDK resolves is the variable the UI creates when you manage
an observed agent. Pass an explicit `name` to decouple the variable from the agent's name, or to
share one variable across agents. A nameless capability derives its variable -- and can source the
model for a model-less agent -- from the agent's `name` the first time it's needed in a run, so
naming the agent is all it takes.

Normalization is lossy, so agent names that differ only in punctuation or case land on one variable:
`checkout-assistant`, `Checkout Assistant`, and `checkout_assistant` all resolve
`agent__checkout_assistant`. Two such agents share a single managed config -- including when they
live in different services that report to the same Logfire project. Pass an explicit `name` to keep
them apart.

They share one contract: **the code-defined agent is the fallback.** Every managed value is a
patch on what's written in code -- absent fields keep their code values, and a missing, invalid,
or unreachable remote value degrades to exactly the agent the developer wrote, never a crashed
run. Values resolve **once per run** and the resolved label + version ride as baggage on every
span of the run, so traces always show which version produced which behavior. Each capability's
`resolved` exposes the active run's `ResolvedVariable`; pass it to `resolution_reason` (exported
from `pydantic_ai_harness.logfire`) to read *why* it resolved the way it did (e.g. a
`'code_default'` fallback), across logfire SDK versions.

**Auto-create on first use:** when the backing variable doesn't exist in Logfire yet, it is
created in the background on first use -- with the payload's JSON schema and description -- so the
Logfire UI becomes the editing surface without a manual create step. Creation happens off the run's
thread, is attempted at most once per process per variable and Logfire instance, and never blocks or
fails the run. Because the variable it writes is persistent and visible to everyone with access to
the project, the outcome is reported into that same Logfire project: a log record on success, and a
log record plus a `UserWarning` on failure. Opt out per capability with `auto_create=False`.
Inside a durable workflow or flow, managed values remain readable but auto-create and baseline
publishing are skipped with one warning: their background threads and remote writes are not replay-safe.
Create the variable in the Logfire UI, or run the SDK outside the workflow once.

`AgentControl` additionally publishes the variable's `example` as an `AgentConfig`-shaped snapshot of
the code-side agent (instructions, model, effective settings, and each tool's description and
parameter descriptions) -- the baseline the Logfire UI's override editor and optimizer diff managed
values against. Each component is recorded at the point before `AgentControl` replaces it, so managed
settings, renamed tools, and managed instructions are not mistaken for code. It publishes in the
background after the first model request that exposes a new
snapshot. An unchanged baseline causes no write; a changed one is published at most once per process. The
provider's current complete definition is copied and only `example` is replaced, preserving labels,
rollout, overrides, and other metadata. The snapshot is taken from the triggering request, so
for instructions or a toolset that vary with `deps`, run input, or the step within a run, it is one
point-in-time sample rather than a description of the agent. An agent that never reaches a model
request never publishes a baseline.

Its `instructions` list one entry per code-defined instruction block -- the agent's own text and each
toolset's, but not `AgentControl`'s managed contribution -- each carrying the `id` that addresses it and a
`dynamic` flag. That is what the UI needs to offer an override per block: the joined prompt a trace
records has no seams in it, so a baseline built from telemetry alone could only ever be copied
wholesale, and copying it wholesale is exactly [the mistake](#where-your-base-prompt-lives) of
sending the agent's own text to the model twice with a frozen `Today is <date>` in the middle of it.

An `example` describes the code rather than being a value to apply -- nothing resolves it -- which is
what lets publishing default to on: a failed or stale snapshot cannot change a run. Publishing needs
a token with `project:write_variables`; a missing scope, network failure, or missing variable warns
once per process and leaves the run untouched. Pass `publish_baseline=False` to disable it, for example
when the process intentionally uses a read-only variables token.

Install the extra:

```bash
pip install 'pydantic-ai-harness[logfire]'
```

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire/)

## `ManagedPrompt`

`ManagedPrompt` is legacy; use a prompt-only [`AgentControl`](#agentcontrol) for new setups.

Back an agent's instructions with a Logfire-managed
[Prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/).

### The problem

Prompts are critical to agent behavior, but iterating on them through the normal
edit -> review -> deploy loop is slow, and you can't easily A/B test a change or roll it
back the moment it misbehaves in production.

### The solution

`ManagedPrompt` declares the backing managed variable for you and resolves it **once per
run**, feeding the value into the agent's instructions. The resolution happens inside the
run's `wrap_run` hook using the
[`ResolvedVariable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as a context manager that stays open for the whole run -- so the selected label and version
are attached as baggage to every child span of the agent run. You get a direct correlation
between a run's behavior and the exact prompt version that produced it, plus instant
iteration and rollback from the Logfire UI.

### Usage

Pass the prompt name and a default value. The name `support_agent` is declared as the managed
variable `prompt__support_agent` -- the naming Logfire's Prompt management uses (hyphens in a
name become underscores). The default keeps the agent working until a remote value is published.
Omit `name` to default it to the agent's own `name` (`prompt__<agent name>`); a `default` is still
required.

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt

logfire.configure()

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are a helpful customer support agent. Be friendly and concise.',
            label='production',
        )
    ],
)

result = agent.run_sync('My order never arrived.')
print(result.output)
```

### Targeting

For deterministic A/B assignment (the same user always sees the same label), pass a
`targeting_key`. It can be a static string or a callable that derives the key from the
[`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) -- handy
when the key lives in your agent's `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt


@dataclass
class Deps:
    user_id: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are a helpful customer support agent.',
            targeting_key=lambda ctx: ctx.deps.user_id,
        ),
    ],
)
```

Pass `attributes` (or a callable returning them) for condition-based targeting rules.
When `label` is omitted, the variable's rollout and targeting rules pick the label;
when both `targeting_key` and `attributes` are omitted, Logfire falls back to its own
targeting context and then to the active trace id.

### Templating with deps

By default the resolved prompt is used verbatim. Pass `render_template=True` to render it as a
Handlebars template against the agent's `deps` -- the same mechanism as
[`TemplateStr`](https://ai.pydantic.dev/api/#pydantic_ai.TemplateStr) -- so `{{field}}` is filled
from `deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt


@dataclass
class Deps:
    customer_name: str


agent = Agent(
    'openai:gpt-5',
    deps_type=Deps,
    capabilities=[
        ManagedPrompt(
            'support_agent',
            default='You are helping {{customer_name}}. Be friendly and concise.',
            render_template=True,
        ),
    ],
)
```

Rendering requires `pydantic-handlebars` (install `pydantic-ai-slim[spec]`). It is off by default.

### Prompt-cache trade-off

The resolved value lands in the agent's **system instructions**. Provider prompt caches (Anthropic,
OpenAI, etc.) key strictly by prefix -- `tools -> system -> messages` -- so any change to the system
block invalidates the cached prefix for the affected runs.

| Mode | Cache impact |
| --- | --- |
| Pinned `label='production'`, no rollout split | **Cache-stable.** The value only changes on a deliberate prompt rollout, which is the same cost as a redeploy. |
| Percentage rollout across labels (no `label=`) | Different runs land on different labels -> splits the cache into one lane per label. |
| `targeting_key` per user/tenant with multiple labels in play | Cache lanes per assigned label; deterministic per key but still N lanes overall. |
| Mid-traffic label flip in the Logfire UI | One-shot cold-invalidation for everyone on that label. |

In short: pinning a `label` keeps the cache hot; using managed values as an A/B platform is opt-in
cache cost. If you don't need rollouts, `label='production'` is the recommended default. The same
applies to `AgentControl`'s `instructions` section -- and to its `tool_definitions`, which sit even
earlier in the cached prefix.

### Using your own variable

Declaring the same name more than once is fine -- each `ManagedPrompt` builds its own backing
variable, so sharing a prompt across several agents just works. Pass an existing
[`logfire.variables.Variable`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
as the first argument instead of a name when you want to declare the variable yourself --
for example a `template_var`, or one registered for `variables_push`:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness import ManagedPrompt

logfire.configure()

support_prompt = logfire.var(
    name='prompt__support_agent',
    type=str,
    default='You are a helpful customer support agent. Be friendly and concise.',
)

agent = Agent('openai:gpt-5', capabilities=[ManagedPrompt(support_prompt, label='production')])
```

When `name` is a prompt name, pass `logfire_instance=` to declare the variable on a specific
Logfire instance instead of the module-level default.

### Notes

- The prompt resolves to a `str`. By default it's used verbatim; set `render_template=True`
  to render `{{...}}` against `deps` (see [Templating with deps](#templating-with-deps)).
- Resolution is isolated per run via a context variable, so a single capability instance
  is safe to share across concurrent runs.
- `ManagedPrompt.resolved` exposes the active run's `ResolvedVariable` (value, label, version,
  reason) for inspection -- e.g. from inside a tool.
- The capability runs outermost (wrapping `Instrumentation`) so the resolved variable's baggage
  covers the agent run span as well as its children. On recent Logfire versions both the
  selected label and the version are propagated as separate baggage attributes.
- Resolution happens **once per run**. A label flip or rollout change that lands in Logfire
  mid-run is not picked up until the next run starts -- the trade-off for run-stable
  instructions and a single baggage scope across all child spans.
- For Logfire-side targeting that lives outside the agent (e.g. set once per request handler),
  use Logfire's
  [`targeting_context`](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
  in an outer scope; `ManagedPrompt` only needs `targeting_key`/`attributes` when the key
  comes from the agent's `RunContext`.
- `prompt__` is reserved for Logfire's first-party Prompt management, so `ManagedPrompt` can't
  auto-create its variable through the generic write API -- it warns once; create the prompt via
  the Logfire UI's Prompts flow.

## `AgentControl`

Back a whole agent's configuration -- instructions, model, model settings, and the LLM-facing
definitions of its tools -- with one `agent__<name>` variable, the same variable Logfire's
Agent Control UI edits.

### The problem

An agent's behavior is spread across knobs that all live in code: the instructions, the model and
its sampling settings, and the tool descriptions that make up half of what the model sees. Tuning
any of them -- or letting Logfire's optimizer propose the tuning -- takes a redeploy per tweak, and
tuning them *together* (a new prompt that only works with a smarter model) takes coordinated
redeploys.

### The solution

`AgentControl` resolves one `AgentConfig` value per run with **presence semantics**: each section
that is present -- `instructions`, `model`, `settings`, `tool_definitions` -- is managed from
Logfire, and each absent section keeps the code-defined behavior. The whole config versions and
rolls out as one unit, so a prompt change and the model change it depends on land atomically.
Removing a section in Logfire is a deliberate revert-to-code.

### Usage

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import AgentControl

logfire.configure()


def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny.'


agent = Agent(
    'openai:gpt-5',
    name='checkout_assistant',
    tools=[get_weather],
    capabilities=[
        AgentControl(  # -> agent__checkout_assistant
            instructions='You are a concise checkout assistant.',
            label='production',
        )
    ],
)
```

The variable holds an `AgentConfig`:

```json
{
  "instructions": [
    "Always confirm the order total.",
    {"id": "agent", "instructions": "You are a concise checkout assistant."},
    {"id": "toolset:legacy_crm", "instructions": null}
  ],
  "model": "openai:gpt-5",
  "settings": {
    "temperature": 0.4,
    "max_tokens": 2048,
    "thinking": "high",
    "provider_options": {
      "anthropic": {"thinking": {"type": "enabled", "budget_tokens": 16384}}
    }
  },
  "tool_definitions": [
    {
      "name": "get_weather",
      "new_name": "lookup_weather",
      "description": "Look up the current weather for a city.",
      "parameter_descriptions": {"city": "City name, e.g. 'London'"}
    }
  ]
}
```

- `instructions` is a list of blocks, and each entry either **adds** text or **swaps out** one of the
  blocks the agent already assembles. An entry with no `id` adds; an entry with an `id` replaces that
  block's text, or drops it with `null`. A bare string is the shorthand for a single added block, so
  `"instructions": "Be brief."` still means what it always did. See
  [Where your base prompt lives](#where-your-base-prompt-lives).
  Text supports `{{...}}` runtime placeholders, which pass through verbatim unless
  `render_template=True` renders them against `deps` (like `ManagedPrompt`).
- `model` is a pydantic-ai model string. It's a first-class field, not a setting: pydantic-ai keeps
  the model id separate from `ModelSettings` (which has no `model` key), so there's no collision
  putting them side by side.
- `settings` keys are the canonical, cross-framework ones (they match
  `pydantic_ai.settings.ModelSettings`), with a nested `provider_options` escape hatch for
  provider-specific settings (`provider_options.openai.reasoning_effort` lowers to the
  `openai_reasoning_effort` model setting, and a provider-specific value wins over its canonical
  counterpart). `thinking` accepts `true`/`false` or an effort level (`'minimal'` ... `'xhigh'`),
  exactly like the unified `thinking` model setting.
- `tool_definitions` is a list too, each entry naming the tool it patches by its original (code-side)
  `name`; every other field is optional and unset fields keep the tool's own definition.

`''` is never accepted where a string carries meaning -- `model`, a block's text, `new_name`. Omission
and `null` already mean "leave this to code", so an empty string is only ever a half-filled field, and
`"model": ""` in particular is not "no model": Pydantic AI raises `Unknown model:` on every request the
agent makes, and the config around it is valid, so nothing downstream would catch it.

Each instruction block is limited to 65,536 characters. An oversized bare `instructions` value drops
that section; an oversized list entry drops only itself. Both warn once per process, and valid sibling
entries and other config sections continue to apply.

### Where your base prompt lives

Instructions are a **composition point**, not a single field. Pydantic AI assembles them from every
source that contributes one -- the agent's own literal, `@agent.instructions` functions injecting
today's date or the signed-in user, each toolset and MCP server, each capability, a skill catalog --
and a managed section that replaced the lot would silence all of them. So `instructions` works in two
ways, and picking the wrong one is the difference between a prompt that reads well and one sent twice.

**An entry with no `id` adds a block.** This is all a capability can do on its own: pydantic-ai appends
every contribution to the agent's own, so an `Agent(instructions='CODE')` with a managed
`'MANAGED'` sends the model `"CODE\n\nMANAGED"`.

**An entry with an `id` swaps out an existing block** -- replacing its text, or dropping it with
`null`. This is what reaches text no capability owns, the agent's own literal included. The keys come
from pydantic-ai's [`InstructionPart.id`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.InstructionPart.id):

| `id` | Addresses |
| --- | --- |
| `agent` | the agent's own literal `instructions` |
| `toolset:<id>` | everything a toolset with that `id` contributes |
| `capability:<id>` | everything a capability with that `id` contributes |
| `agent:<name>` | one `@agent.instructions(name=...)` part |
| `capability:<id>:<name>` | one `@capability.instructions(name=...)` part |

Blocks pydantic-ai cannot key have no entry that reaches them: a callable passed to
`Agent(instructions=...)`, anything from `run(instructions=...)`, a toolset with no `id` of its own. An
`id` that matches nothing in this deployment is inert rather than an error, so one config can be applied
across services that don't all install the same toolsets.

That gives three places instructions come from, and two ways a published config can act on each:

| Where | Managed from Logfire? | What a published config does to it |
| --- | --- | --- |
| `Agent(instructions=...)` | Only by `id` | Adds to it by default; `{"id": "agent", ...}` replaces or drops it. |
| `AgentControl(instructions=...)` | It *is* the code-side default | **Supersedes** it -- the capability contributes the published value or the default, never both. |
| The published `instructions` in Logfire | Yes | It is the value. |

An `agent` override is the sharper tool, but the capability default is still the better home for a
prompt you *intend* to manage: it supersedes rather than shadows, so there is only ever one copy of the
text and no way for the two to drift.

```python
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import AgentControl

agent = Agent(
    'openai:gpt-5',
    name='checkout_assistant',
    capabilities=[AgentControl(instructions='You are a concise checkout assistant.')],
)
```

`instructions=` is shorthand for `default=AgentConfig(instructions=...)` and behaves identically; pass
the two together and you get a `UserError` rather than a silent precedence rule. Use the long form
when other sections need code-side defaults too. The other sections have no shorthand because they
have no such trap: a managed `model`, `settings`, or `tool_definitions` supersedes the agent's own, so
`Agent(model=...)`, `Agent(model_settings=...)`, and a tool's own docstring stay the natural code-side
homes.

> **The mistake to avoid.** Copying the agent's observed system prompt into a managed value as *added*
> text -- a prompt lifted out of a trace, or the whole `example` snapshot -- while the same text stays
> in `Agent(instructions=...)` sends every block of it to the model **twice**, and pins any dynamic
> block (`Today is 2026-07-29.`) to whatever it said at the moment it was copied. Take the text over by
> `id`, or move it to the capability; don't re-add it.

Added blocks compose the way a capability's instructions always have. Pydantic AI groups static
instruction text ahead of dynamic text (so providers can cache the stable prefix) and preserves source
order within each group, which puts them after the agent's own literal and `@agent.instructions` text
and before dynamic toolset instructions. An override leaves its block's position and its `dynamic` flag
alone -- replacing a dynamic block does not make it static -- so no override moves the cacheable prefix.
Only `agent.override(instructions=...)` replaces the lot, and a capability can't reach it.

### Notes

- **Instructions add unless an entry names what it replaces:** an entry with no `id` is appended to the
  agent's own, so a base prompt you want to manage belongs on the capability
  (`AgentControl(instructions=...)`), which a published config supersedes. An entry *with* an `id` swaps
  out that block instead. See [Where your base prompt lives](#where-your-base-prompt-lives).
- **Overrides apply to the assembled prompt:** they are applied in `before_model_request`, the last
  point at which every contribution exists, which is what lets an entry reach a toolset's or an MCP
  server's text and not just the agent's. They land on
  [`instruction_parts`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.ModelRequestParameters.instruction_parts),
  the source of truth for what the model is sent, so the rewritten prompt is what message history and
  traces show too.
- **Two entries naming the same thing keep the first,** with a warning -- the same rule as a colliding
  tool rename, so a hand-edited value stays predictable instead of depending on JSON ordering.
- **Tool definitions are overlays, never code:** the tool itself -- its implementation and its
  parameter schema structure -- stays exactly as written in code. Only the LLM-facing spec (name,
  description, parameter description strings) is remotely patchable, so a remote value can never
  drift from the validator the tool actually runs against.
- **Renames round-trip:** `new_name` changes the name the model is shown; a call to the renamed
  tool routes back to the original implementation, and `ctx.tool_name` inside the tool is the
  original name. A rename that collides with a name another tool already advertises is dropped
  with a warning (other patches still apply) rather than breaking the run.
- An override naming a tool that no longer exists is inert -- that's the drift case (the tool was
  removed or renamed in code), and the Logfire UI is where it becomes visible. An instruction entry
  whose `id` matches nothing behaves the same way.
- **Model precedence:** the managed `model` is sourced during model selection via the capability's
  `get_model` hook, so it slots in with the right precedence -- a call-site `run(model=...)` beats
  it, it beats the agent's constructor model, and a fully model-less agent (named or nameless) can be
  driven entirely from Logfire. Model selection evaluates static or callable targeting inputs once,
  and the authoritative run resolution reuses that exact result. An unknown managed model warns and
  keeps the code model instead of failing the run.
- **Settings precedence:** managed settings merge **over** the agent's constructor `model_settings`
  and **under** per-run `model_settings=`, so run arguments always win.
- **Adoption reporting:** for the run's duration, `logfire.managed.applied_sections` baggage names
  the sections the capability applied (e.g. `instructions,settings`), which the Logfire UI reads to
  distinguish a wired-up managed agent from one whose config resolves but isn't applied. `model` is
  reported when present even if a call-site `run(model=...)` outranked it that run.
- **One stored schema, two writers:** the variable can be created by this SDK or by the Logfire
  Agent Control UI, and whichever gets there first stores the schema that the UI edits against and
  that Logfire validates every later version of the value against. Both sides therefore write the
  same hand-maintained schema, exported as `AGENT_CONFIG_JSON_SCHEMA`. It stays permissive on
  purpose: `AgentConfig` ignores keys it doesn't know so a value written by a newer UI degrades to
  the sections an older SDK understands instead of failing, and a stored schema that rejected those
  keys would break that by refusing the write.
- **Forward compatibility covers values, not just keys:** a `thinking` effort level or a
  `service_tier` that a newer Pydantic AI accepts and this SDK has never heard of drops just that
  setting; a tool override that doesn't validate (a missing `name`, an empty `new_name`) drops just that
  override; an instruction entry that doesn't validate (an empty text, an entry naming neither what nor
  where) drops just that block -- each with a warning naming the offending value, emitted once per
  process so a per-run resolution can't turn it into noise. Everything else in the config still
  applies. A malformed settings key drops independently, and a wrong instructions, settings, or
  tool-definitions container drops only that section. Each list entry is a unit of degradation for
  the same reason: it addresses exactly one
  thing, so dropping it costs exactly that thing. The
  alternative isn't stricter, it's blunter: an `AgentConfig` that fails validation falls back to the
  code-defined agent *whole*, so one unfamiliar enum value would silently un-manage the instructions,
  the model, and every tool override with it.
- `AgentControl.resolved` exposes the active run's `ResolvedVariable`, and resolution is isolated
  per run, exactly like `ManagedPrompt`.
