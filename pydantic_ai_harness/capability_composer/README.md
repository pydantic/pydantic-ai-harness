# Capability Composer

> [!NOTE]
> Nothing here is re-exported at the top level -- import from the submodule:
>
> ```python
> from pydantic_ai_harness.capability_composer import CapabilityComposer
> ```
>
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

Let a picker model, [Jev](https://typesafe.ai) by default, compose a sub-agent for each prompt -- its model, thinking effort, and capabilities, from a menu and an allowlist you define -- and hand it the turn.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/capability_composer/)

## The problem

A general-purpose agent carries every tool and runs on one model for every request. "Commit this" pays for the same frontier model, the same thinking budget, and the same thirty tool definitions as "redesign the auth layer". Asking a language model to size the request first costs about as much as the request itself.

## The solution

`CapabilityComposer` asks a picker model once, on a run's first model request. The default picker is [Jev](https://pydantic.dev/docs/ai/models/typesafe/), a classifier rather than a language model: it answers typed questions with a confidence, in a few hundred milliseconds. One request asks three things:

- **model**: which entry of your `models` menu should handle the request
- **thinking**: `low`, `medium`, or `high` reasoning effort
- **capabilities**: for each entry of the catalog, whether the request needs it

The composer builds a sub-agent on the picked model and thinking effort, with the picked capabilities, runs it on the conversation, and returns its answer as the model response: the agent's own model is not called for that turn. The sub-agent is an independent run, like a [sub-agent](../subagents/) delegation, so only what [the sub-agent gets](#what-the-sub-agent-gets) crosses over. When the picker's confidence in the model pick is below `confidence_threshold`, the sub-agent keeps the capabilities the picker picked but runs on `unsure_model`: the last entry of `models` unless you name one, so order the menu from cheapest to strongest. When the picker picks no capabilities, the agent's own model handles the turn as usual.

The default threshold of 0.4 comes from a hand-labelled check of the example menu below. The example sets `unsure_model='medium'` from the same check: Jev is rarely unsure of a clearly worded request, and mostly unsure of vague ones ("add a cache", "finish the TODOs"), which usually need a scoped change rather than the strongest model. On 40 vague prompts, falling back to `medium` put 44 of 60 picks on the labelled tier against 25 for `max`, at the cost of 2 clearly worded prompts in 330 running a tier too low. `scripts/capability_composer_eval.py` reruns the check; recalibrate against prompts of your own before relying on either setting.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.capability_composer import CapabilityComposer
from pydantic_ai_harness.subagents import ModelOption

agent = Agent(
    'openai-codex:gpt-6-sol',
    capabilities=[
        CapabilityComposer(
            models={
                'fast': ModelOption('openai-codex:gpt-6-luna', description='Answering a question, or one command or trivial edit that needs no investigation'),
                'medium': ModelOption('openai-codex:gpt-6-sol', description='A focused code change with a clear cause or spec, in one or a few files'),
                'max': ModelOption('openai-codex:gpt-6-astra', description='Open-ended work: an unknown root cause, a design decision, or changes across many files'),
            },
            unsure_model='medium',
            instructions='You are a coding agent working in this repository.',
        )
    ],
)

result = agent.run_sync('commit this with a sensible message')
print(result.output)
```

Jev runs through Pydantic AI's `TypeSafeModel`. Install `pydantic-ai-harness[capability-composer]`, which brings in `pydantic-ai-slim[typesafe]`, and set `TYPESAFE_API_KEY`; see [TypeSafe (Jev)](https://pydantic.dev/docs/ai/models/typesafe/). The models on the menu need their own providers: the example's `openai-codex:` models need the `openai` extra of `pydantic-ai-slim` and a `codex login`.

## Other pickers

`picker_model=` takes any model that can fill a structured output, such as a small language model:

```python {test="skip"}
CapabilityComposer(models=..., picker_model='openai-codex:gpt-6-luna')
```

The questions are the same output type, filled through the model's output tool. What changes:

- **No escalation.** A language model reports no confidence, so its picks are trusted as given: `confidence_threshold` and `unsure_model` never apply.
- **Latency and cost.** Every composed turn waits for the picker before the sub-agent starts. Jev answers in about 0.3s; `gpt-6-luna` took 2 to 3 seconds, billed for the whole schema each time.
- **Different judgement.** On the 240 labelled prompts in `scripts/capability_composer_eval.py`, `gpt-6-luna` picked the labelled model tier for 88% of clearly worded coding prompts against Jev's 98%, and 19 of its 20 misses ran a tier too high. It picked every capability a prompt needed more often: 98 to 100% per tier against Jev's 87 to 91% for medium and max. Both fell through on every chat prompt and matched about 60% of vague ones.

Rerun the check with `--picker-model` before switching pickers.

## What the picker is asked

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

The descriptions are what the picker decides from. Give each `ModelOption` a `description` that says what kind of request it is for; without one, the picker sees only the model name. Describe a tier by the work a request takes (whether it needs investigation first, how far the change reaches) rather than by how capable the model is: the picker tells the tiers apart with more confidence that way, so fewer prompts escalate to the strongest model.

## The catalog is an allowlist

`catalog` maps a key to a `ComposableCapability`: a capability class, the arguments to build it with (passed to its `from_spec`, as in an `AgentSpec` entry), and a description the picker reads. Only catalog entries and `shared_capabilities` can be on a sub-agent, and each sub-agent that picks an entry gets a fresh instance.

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
- **Shape the run, not the task**: compaction, spend limits, and guardrails belong in `shared_capabilities`, where they cover every sub-agent whatever the picker picks.

Add any of them yourself. `ComposableCapability.of` describes an entry from the first line of the capability's docstring unless you pass `description=`. This one also needs the `exa` extra and an `EXA_API_KEY`:

```python
from pydantic_ai_harness.exa import ExaSearch
from pydantic_ai_harness.capability_composer import ComposableCapability, CapabilityComposer, default_catalog

composer = CapabilityComposer(
    models={'fast': 'openai-codex:gpt-6-luna'},
    catalog={
        **default_catalog(),
        'exa': ComposableCapability.of(ExaSearch, description='Research a topic across many web sources'),
    },
)
```

A docstring says what a capability is. The picker decides better from what a request would need it for, which is why the default entries carry descriptions written for that. Say which requests need an entry, not only what it does: on 200 labelled prompts, describing `shell` as running tests and git got it onto 25% of scoped code changes, and adding "any request that changes code needs this to check the change works" got it onto 89%. An entry with a broad description gets picked for requests that don't need it; `skills` is scoped to requests that name a skill for that reason.

## Models and thinking

`models` takes the same entries as the [sub-agents](../subagents/) model menu: a model ID, a `Model`, or a `ModelOption`. The picked entry becomes the sub-agent's model. The picked effort goes on it as `ModelSettings(thinking=...)`, and a `ModelOption.settings` overrides it, so an entry that must always think hard can say so. A model whose profile does not support thinking ignores the setting.

## What the sub-agent gets

- **The conversation**: the messages the agent's model would have been sent, so it can follow up on earlier turns. The prompt is as the capabilities before the composer left it, attachments included.
- **`deps`**, and the run's **usage and `usage_limits`**, so its requests count toward the run.
- **`instructions`**: the composer's, not the agent's. `system_prompt` parts already in the conversation stay in it.
- **`shared_capabilities`**: put guardrails, approval guards, and anything else every sub-agent needs here. A pick of the same class as one of them is left out, so they keep their configuration. Each is set up for the sub-agent's run as for any run, so one that keeps per-run state, such as [spend limits](../spend/), counts within that sub-agent run.
- **`event_stream_handler`**: receives the sub-agent's events, such as its tool calls, which the agent's own event stream does not carry.

It gets nothing else of the agent's: not its tools, capabilities, or output type. The agent's history records one request and the sub-agent's answer, under the sub-agent's model name, so later turns see what earlier sub-agents said rather than the tool calls they made.

Some runs are not composed:

- **Structured output**: the sub-agent answers in text, so when the run's output type cannot take text (`int`, a model, `PromptedOutput` or `NativeOutput`), the agent's own model handles it and the picker is not asked. An output type that includes `str` is composed.
- **After the first request**: only a run's first model request is composed. A prompt with no text, or a run resumed from a tool result, is left to the agent.

Approval inside the sub-agent has to happen while it runs, with an async `ToolGuardrail` guard in `shared_capabilities` that asks the user (see [human in the loop](../guardrails/#human-in-the-loop)). Deferred approval (`GuardrailResult.approve()`, `requires_approval=True`) ends a run with `DeferredToolRequests`, which the sub-agent's text output cannot carry back.

The picker's request and the sub-agent's count toward the run's usage and `usage_limits`. Core also counts the agent's own request step that the answer stands in for, so a composed turn counts one request more than the models served. A spend limits capability on the agent prices neither, since they are separate runs; add one to `shared_capabilities` for the sub-agent.

## Guardrails and other request capabilities

The composer sits innermost in the model request, inside every [`InputGuardrail`](../guardrails/) -- including one passed to `Agent.run(capabilities=...)` -- and after capabilities that rewrite the request. A prompt a guardrail blocks never reaches the picker, and one it redacts reaches the picker and the sub-agent redacted. An `InputGuardrail` with `parallel=True` runs its guard alongside the model request, which here means alongside the picker and the sub-agent, so they may read the prompt before the guard decides, as the agent's model would; a block still cancels them.

Each run asks the picker again, so a conversation's model and capabilities can change from one turn to the next. The composer is not built for durable execution: a worker replaying a run would ask the picker again, and could get a different answer.

## Telemetry

Each decision is a `capability_composer compose` span on the run's tracer, with:

| Attribute | Value |
|---|---|
| `capability_composer.model`, `capability_composer.thinking`, `capability_composer.capabilities` | what the picker picked |
| `capability_composer.confidence.<field>` | the picker's confidence per field |
| `capability_composer.action` | `compose`, `escalate` (unsure of the model, so the sub-agent runs on `unsure_model`), or `fallthrough` (no capabilities picked, so the agent's model handles the turn) |
| `capability_composer.run_model` | the `models` key the sub-agent runs on, unless it fell through |
| `capability_composer.prompt` | the text the picker read, only when the run includes content in traces |

The picker's request and the sub-agent's run appear as their own agent spans under the run's model request. When a sub-agent is composed, a `CapabilitiesComposedEvent` goes into the agent's event stream before it starts; its `escalated` field says whether `unsure_model` replaced the picker's pick. A run that is not composed (structured output, no prompt text) has no span.

Watch the escalation and fall-through rates as well as the picks. A composer that escalates on most prompts is running everything on `unsure_model`, and one that falls through on most is costing a picker request per run and changing nothing; tune `confidence_threshold` and the descriptions against labelled prompts of your own, then pin the Jev version you tuned against with `picker_model='typesafe:jev-1.13.0'`.
