# Harness CLI

A terminal client for the [Coder](../coder/) harness. `harness -p "..."` runs one prompt against `Coder` in the current directory and prints the response.

This is the first slice of a larger client. The interactive session, streamed rendering, config file, and slash commands land one plan item at a time; the plan is [`agent_docs/harness-cli-plan.md`](https://github.com/pydantic/pydantic-ai-harness/blob/main/agent_docs/harness-cli-plan.md). `harness` is a placeholder command name.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/cli/). While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade; see the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install

```bash
pip install 'pydantic-ai-harness[cli]'
```

The `cli` extra pulls Termflow, the Markdown renderer for the terminal, on Python 3.11+. The one-shot mode below does not use it yet and runs on 3.10.

## Usage

```bash
harness -p "Explain what tests/test_parser.py covers."
harness -p "Add a --verbose flag to scripts/build.py." --model openai:gpt-5.6-sol
```

`--model` takes any name Pydantic AI's [`infer_model`](https://pydantic.dev/docs/ai/models/overview/) accepts and defaults to `anthropic:claude-fable-5`. The provider's API key comes from the environment as usual; a missing key exits with the provider's own message.

The agent is `Coder` rooted at the current directory: it can read and edit files under it and run the commands on `Coder`'s allowlist. That allowlist is a guardrail against accidents, not a security boundary; see the [Coder docs](../coder/) for the exact composition and how to sandbox it.

## Testing against the CLI agent

`cli_agent` is exported and model-less, so tests swap in a model with `Agent.override` and drive `main` with an explicit argument list:

```python
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.cli import cli_agent, main

with cli_agent.override(model=TestModel(call_tools=[], custom_output_text='Hello.')):
    main(['-p', 'hi'])
#> Hello.
```

## Telemetry

The CLI adds no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) already covers the run: `logfire.instrument_pydantic_ai()` before `main()` traces every model and tool call.
