# Harness CLI

A terminal client for the [Coder](../coder/) harness. `harness -p "..."` runs one prompt against `Coder` in the current directory and renders the run as it happens: streamed Markdown for the model's text, one line per tool call and result.

This is an early slice of a larger client. The interactive session, config file, and slash commands land one plan item at a time; the plan is [`agent_docs/harness-cli-plan.md`](https://github.com/pydantic/pydantic-ai-harness/blob/main/agent_docs/harness-cli-plan.md). `harness` is a placeholder command name.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/cli/). While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade; see the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install

```bash
pip install 'pydantic-ai-harness[cli]'
```

The CLI needs Python 3.11 or newer. It renders with [Termflow](https://pypi.org/project/termflow-md/), which the `cli` extra installs on 3.11+; on 3.10 importing `pydantic_ai_harness.cli` raises an `ImportError` that says so.

## Usage

```bash
harness -p "Explain what tests/test_parser.py covers."
harness -p "Add a --verbose flag to scripts/build.py." --model openai:gpt-5.6-sol
```

`--model` takes any name Pydantic AI's [`infer_model`](https://pydantic.dev/docs/ai/models/overview/) accepts and defaults to `anthropic:claude-fable-5`. The provider's API key comes from the environment as usual; a missing key exits with the provider's own message.

The agent is `Coder` rooted at the current directory: it can read and edit files under it and run the commands on `Coder`'s allowlist. That allowlist is a guardrail against accidents, not a security boundary; see the [Coder docs](../coder/) for the exact composition and how to sandbox it.

## What you see

`CliBridge` is the capability that renders the run. It subscribes to the run's event stream and writes to the terminal as events arrive:

- Model text streams through Termflow as Markdown: headings, emphasis, lists, code blocks with syntax highlighting, and tables render as the text completes each line.
- Each tool call prints as `> tool_name {"arg": ...}` and its result as `< tool_name` followed by the first line of the return value. When a result spans several lines, the line ends with `(+N lines)`; every line is cut to the terminal width. A tool that asks the model to retry prints as `! tool_name` with the retry reason.
- Thinking parts and capability events (a file read, a plan update) are not rendered yet; the plan schedules them.

The bridge is an ordinary capability, so you can put it on your own agent. Wire it last so it observes every other capability's events:

```python
from pydantic_ai import Agent

from pydantic_ai_harness.cli import CliBridge
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder(), CliBridge()])
```

`CliBridge(output=..., width=..., style=...)` chooses the stream to write to (`sys.stdout` when unset, resolved at the start of each run), the column width (detected from the terminal when unset), and the Termflow `RenderStyle` palette. Each run renders on a fresh bridge, so nothing carries over between runs.

## Testing against the CLI agent

`cli_agent` is exported and model-less, so tests swap in a model with `Agent.override` and drive `main` with an explicit argument list:

```python
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.cli import cli_agent, main

with cli_agent.override(model=TestModel(call_tools=[], custom_output_text='Hello.')):
    main(['-p', 'hi'])
#> Hello.
```

To assert on a rendered transcript, give `CliBridge` an `io.StringIO` as `output` and a fixed `width`, then strip the ANSI styling with `termflow.ansi.visible` before comparing.

## Telemetry

The CLI adds no spans of its own, and neither does `CliBridge`: rendering records no decision that core's spans do not already show. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) covers the run: `logfire.instrument_pydantic_ai()` before `main()` traces every model and tool call.
