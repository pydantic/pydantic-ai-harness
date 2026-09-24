# Jev Capability Composer

> [!NOTE]
> Nothing here is re-exported at the top level -- import from the submodule:
>
> ```python
> from pydantic_ai_harness.jev import JevCapabilityComposer
> ```
>
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

Let [Jev](https://typesafe.ai) pick the model, thinking effort, and capabilities of each run, from a menu and an allowlist you define.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/jev/)

## The problem

A general-purpose agent carries every tool and runs on one model for every request. "Commit this" pays for the same frontier model, the same thinking budget, and the same thirty tool definitions as "redesign the auth layer". Asking a language model to size the request first costs about as much as the request itself.

## The solution

`JevCapabilityComposer` asks [Jev](https://pydantic.dev/docs/ai/models/typesafe/) once, before a run starts. Jev is a classifier rather than a language model: it answers typed questions with a confidence, in a few hundred milliseconds. One request asks three things:

- **model**: which entry of your `models` menu should handle the request
- **thinking**: `low`, `medium`, or `high` reasoning effort
- **capabilities**: for each entry of the catalog, whether the request needs it

The run then goes ahead on the picked model and thinking effort, with the picked capabilities added. Everything else the agent was configured with still applies -- its instructions, output type, tools, guardrails, persistence, and limits -- because the picks join the run rather than replacing it. When Jev's confidence in the model pick is below `confidence_threshold`, the run keeps the capabilities Jev picked but uses `unsure_model`: the last entry of `models` unless you name one, so order the menu from cheapest to strongest. When Jev picks no capabilities, the run is left as the agent configured it.

The default threshold of 0.4 comes from a hand-labelled check of the example menu below. The example sets `unsure_model='medium'` from the same check: Jev is rarely unsure of a clearly worded request, and mostly unsure of vague ones ("add a cache", "finish the TODOs"), which usually need a scoped change rather than the strongest model. On 40 vague prompts, falling back to `medium` put 44 of 60 picks on the labelled tier against 25 for `max`, at the cost of 2 clearly worded prompts in 330 running a tier too low. `scripts/jev_eval.py` reruns the check; recalibrate against prompts of your own before relying on either setting.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.jev import JevCapabilityComposer
from pydantic_ai_harness.subagents import ModelOption

agent = Agent(
    'openai-codex:gpt-6-sol',
    capabilities=[
        JevCapabilityComposer(
            models={
                'fast': ModelOption('openai-codex:gpt-6-luna', description='Answering a question, or one command or trivial edit that needs no investigation'),
                'medium': ModelOption('openai-codex:gpt-6-sol', description='A focused code change with a clear cause or spec, in one or a few files'),
                'max': ModelOption('openai-codex:gpt-6-astra', description='Open-ended work: an unknown root cause, a design decision, or changes across many files'),
            },
            unsure_model='medium',
        )
    ],
)

result = agent.run_sync('commit this with a sensible message')
print(result.output)
```

Jev runs through Pydantic AI's `TypeSafeModel`. Install `pydantic-ai-harness[jev]`, which brings in `pydantic-ai-slim[typesafe]`, and set `TYPESAFE_API_KEY`; see [TypeSafe (Jev)](https://pydantic.dev/docs/ai/models/typesafe/). The models on the menu need their own providers: the example's `openai-codex:` models need the `openai` extra of `pydantic-ai-slim` and a `codex login`. To use another picker, pass `jev_model=` any model that can fill a structured output. A picker that reports no confidence is trusted as given.

## What Jev is asked

The questions are an ordinary output type. The composer builds one per model menu and catalog, using Pydantic AI's `Choices` for the options only known at construction and a `Thinking` enum with a docstring per member:

```python {test="skip" lint="skip"}
class Composition(BaseModel):
    """Compose an agent to handle this request: its model, reasoning effort, and capabilities."""

    model: Annotated[str, Choices({'fast': 'Quick answers ...', 'medium': 'Everyday coding ...', 'max': 'Hard reasoning ...'})] = Field(
        description='Which model should handle this request?'
    )
    thinking: Thinking = Field(description='How much reasoning effort does this request need?')
    capabilities: list[Annotated[str, Choices({'filesystem': 'Read, search, and edit files ...', ...})]] = Field(
        description='Does handling this request need this capability?'
    )
```

`TypeSafeModel` asks the two pick-one fields as Choice questions and fans the list out into one yes/no per catalog entry, all in a single request. Jev can only answer with an option it was offered, so there is no invented model or capability to reject.

The descriptions are what Jev decides from. Give each `ModelOption` a `description` that says what kind of request it is for; without one, Jev sees only the model name. Describe a tier by the work a request takes (whether it needs investigation first, how far the change reaches) rather than by how capable the model is: Jev tells the tiers apart with more confidence that way, so fewer prompts escalate to the strongest model.

## The catalog is an allowlist

`catalog` maps a key to a `ComposableCapability`: a capability class, the arguments to build it with (passed to its `from_spec`, as in an `AgentSpec` entry), and a description Jev reads. Only catalog entries can be added to a run, and each run that picks one gets a fresh instance.

When no `catalog` is given, the composer calls `default_catalog()` as it is constructed. Every entry needs no third-party API key, and the ones that need configuration take a default:

| Key | Capability | Configured with | Included when |
|---|---|---|---|
| `filesystem` | [FileSystem](../filesystem/) | defaults | always |
| `shell` | [Shell](../shell/) | defaults | always |
| `planning` | [Planning](../planning/) | defaults | always |
| `repo_context` | [Repo Context](../repo_context/) | the working directory | always |
| `pydantic_ai_docs` | [Pydantic AI Docs](../pydantic_ai_docs/) | defaults | always |
| `web_search` | Pydantic AI's `WebSearch` | DuckDuckGo fallback when the `duckduckgo` extra is installed | always |
| `skills` | [Skills](../skills/) | `SKILLS_DIRECTORY` (`.agents/skills`) | that directory exists and the `skills` extra is installed |
| `code_mode` | [Code Mode](../code_mode/) | defaults | the `code-mode` extra is installed |
| `web_fetch` | Pydantic AI's `WebFetch` | local fetcher fallback | the `web-fetch` extra is installed |

Both web entries use the model's native tool when it has one. `web_fetch` is only offered with its local fallback, because native URL fetching is missing on common models, OpenAI's among them. Without the `duckduckgo` extra, `web_search` is native only, and a run whose model has no native web search raises `UserError` when it picks it. The `jev` extra installs both web extras.

Some capabilities are left out on purpose:

- **Need a key or an external service**: `ExaSearch`, `YouSearch`, `ModalSandbox`, `BrowserUse`, `LocalStack`, `Macroscope`. They build without arguments, then call a paid service or need a CLI or container.
- **No default could be right**: `Advisor` needs a model, `AskUser` an answerer, `ConversationSearch` a history source, `SubAgents` its agents, and `Memory`'s persistent stores are objects, so a capability built per run would get an empty in-memory store.
- **Bundles of entries already here**: `Coder` and `Researcher` combine `FileSystem`, `Shell`, and web search, so picking one next to those entries would register the same tools twice.
- **Shape the run, not the task**: compaction, spend limits, persistence, and guardrails belong on the agent, where they cover every run whatever Jev picks.

Add any of them yourself. `ComposableCapability.of` describes an entry from the first line of the capability's docstring unless you pass `description=`. This one also needs the `exa` extra and an `EXA_API_KEY`:

```python
from pydantic_ai_harness.exa import ExaSearch
from pydantic_ai_harness.jev import ComposableCapability, JevCapabilityComposer, default_catalog

composer = JevCapabilityComposer(
    models={'fast': 'openai-codex:gpt-6-luna'},
    catalog={
        **default_catalog(),
        'exa': ComposableCapability.of(ExaSearch, description='Research a topic across many web sources'),
    },
)
```

A docstring says what a capability is. Jev decides better from what a request would need it for, which is why the default entries carry descriptions written for that. Say which requests need an entry, not only what it does: on 200 labelled prompts, describing `shell` as running tests and git got it onto 25% of scoped code changes, and adding "any request that changes code needs this to check the change works" got it onto 89%. An entry with a broad description gets picked for requests that don't need it; `skills` is scoped to requests that name a skill for that reason.

## Models and thinking

`models` takes the same entries as the [sub-agents](../subagents/) model menu: a model ID, a `Model`, or a `ModelOption`. The picked entry becomes the run's model, unless the run was given one with `Agent.run(model=...)`, which takes precedence. The picked effort goes on the run as `ModelSettings(thinking=...)`, and a `ModelOption.settings` overrides it, so an entry that must always think hard can say so. A model whose profile does not support thinking ignores the setting.

## What changes and what doesn't

The picks apply to the whole run: every model request in it uses the picked model and has the picked capabilities. The agent's instructions, output type, message history, dependencies, tools, and other capabilities are unchanged. A picked entry whose class the agent already has is not added a second time, since both would register the same tools, and the agent's configuration of it is the one that applies. When the run gets that class some other way -- passed to `Agent.run(capabilities=...)`, or picked by a second composer -- the pick's tools give way to it on each request: the run's own configuration still wins, so a pick cannot widen what the run was given.

Jev's request counts toward the run's usage and its `usage_limits`. It is a separate request made through its own agent, so a [spend limits](../spend/) capability on your agent does not price it.

Jev reads the text of the run's prompt, or of the latest user prompt in `message_history` when the run is given none, before any model request is made. The agent's [input guardrails](../guardrails/) screen that text first: when one blocks the prompt, Jev is not asked and the guardrail blocks the run as usual, and when one redacts it, Jev reads the redacted text. Their guards therefore run twice per run, once before Jev and once on the first model request, which matters for a slow or paid guard. Only guardrails on the agent are seen; put an `InputGuardrail` on the agent rather than passing it to `Agent.run` if it must cover Jev. Other capabilities that rewrite model requests do not cover what is sent to Jev.

Each run asks Jev again, so a conversation's model and capabilities can change from one turn to the next. The composer is not built for durable execution: a worker that re-derives a run's capabilities would ask Jev again, and could get a different answer.

## Telemetry

Each decision is a `jev_capability_composer compose` span on the run's tracer, with:

| Attribute | Value |
|---|---|
| `jev_composer.model`, `jev_composer.thinking`, `jev_composer.capabilities` | what Jev picked |
| `jev_composer.confidence.<field>` | Jev's confidence per field |
| `jev_composer.action` | `compose`, `escalate` (unsure of the model, so the run uses `unsure_model`), `fallthrough` (no capabilities picked), or `blocked` (an input guardrail blocked the prompt, so Jev was not asked) |
| `jev_composer.run_model` | the `models` key the run uses, when the picks were applied |
| `jev_composer.prompt` | the text Jev read, after any redaction, only when the run includes content in traces; never recorded for a blocked prompt |

Jev's request appears as its own agent span under it. When the picks are applied, the run emits a `CapabilitiesComposedEvent` into its event stream as it starts; its `escalated` field says whether `unsure_model` replaced Jev's pick.

Watch the escalation and fall-through rates as well as the picks. A composer that escalates on most prompts is running everything on `unsure_model`, and one that falls through on most is costing a Jev request per run and changing nothing; tune `confidence_threshold` and the descriptions against labelled prompts of your own, then pin the Jev version you tuned against with `jev_model='typesafe:jev-1.13.0'`.
