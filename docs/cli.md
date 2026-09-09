---
title: Harness CLI
description: A terminal client for the Coder harness. An interactive session (or one prompt with -p) against Coder in the current directory, rendered as the run happens.
---

# Harness CLI

A terminal client for the [Coder](coder.md) harness. `harness` starts an interactive session with `Coder` in the current directory, and `harness -p "..."` runs one prompt and exits. Either way the run renders as it happens: streamed Markdown for the model's text, one line per tool call and result.

This is an early slice of a larger client. Slash commands, sessions, and a richer line editor land one plan item at a time; the plan is [`agent_docs/harness-cli-plan.md`](https://github.com/pydantic/pydantic-ai-harness/blob/main/agent_docs/harness-cli-plan.md). `harness` is a placeholder command name.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/cli/).

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
pip install 'pydantic-ai-harness[cli]'
```

The CLI needs Python 3.11 or newer. It renders with [Termflow](https://pypi.org/project/termflow-md/), which the `cli` extra installs on 3.11+; on 3.10 importing `pydantic_ai_harness.cli` raises an `ImportError` that says so.

## Usage

```bash
harness
harness --model openai:gpt-5.6-sol
harness -p "Explain what tests/test_parser.py covers."
```

`--model` takes any name Pydantic AI's [`infer_model`](/ai/models/overview/) accepts. It overrides the config file's `model`, which defaults to `anthropic:claude-fable-5`. The provider's API key comes from the environment as usual; a missing key exits with the provider's own message.

The agent is `Coder` rooted at the current directory: it can read and edit files under it and run the commands on `Coder`'s allowlist. That allowlist is a guardrail against accidents, not a security boundary; see the [Coder docs](coder.md) for the exact composition and how to sandbox it.

### The session

Without `-p`, `harness` shows a `harness>` prompt and reads one line per prompt. The conversation carries across prompts, so a follow-up sees everything before it.

- **Enter** sends the line. Blank lines are ignored.
- **Typing during a run** steers it: the line is enqueued into the run and reaches the model at its next request (or turns into one more request if the run was about to finish). The session confirms with `(steer queued: ...)` when the line is taken.
- **Ctrl+C** cancels the run in flight through core's `AgentRun.cancel`: the model request or tool call is torn down, `(cancelled)` prints, and what completed before the cancel stays in the conversation. When nothing is running, Ctrl+C starts a fresh prompt line.
- **Ctrl+D** ends the session.

Line editing is what the terminal itself provides (Backspace, Ctrl+U, Ctrl+W); history search, completion, and paste handling are later plan items. With `-p`, the session's stdin is not read, so piped input is neither a prompt nor a steer.

## Configuration

Settings live in `~/.pydantic-ai-harness/config.json`. A missing file means the defaults; an invalid one exits with the path and what is wrong with it. Every key is optional:

```json
{
  "model": "anthropic:claude-fable-5",
  "theme": {"palette": "default", "code_style": "monokai"},
  "show_thinking": false,
  "yolo": false
}
```

- **`model`** is the model for every run unless `--model` overrides it.
- **`theme.palette`** picks the Termflow colors for Markdown, tool lines, and status notes: `default`, `dracula`, `gruvbox`, or `nord`.
- **`theme.code_style`** is the [Pygments style](https://pygments.org/styles/) for code blocks. An unknown name falls back to `monokai`.
- **`show_thinking`** prints the model's thinking parts, dimmed, as they stream. They are hidden by default.
- **`yolo`** approves every shell command and file change without asking. `--yolo` on the command line turns it on for one invocation.

The file is read when `harness` starts and again at the start of each run, so a change made during a session applies to the next prompt. From Python, `Config.load()` and `Config.save()` read and write the same file (or a path you pass), and `Config.default_path()` is where it lives.

## What you see

`CliBridge` is the capability that renders the run. It subscribes to the run's event stream and writes to the terminal as events arrive:

- Model text streams through Termflow as Markdown: headings, emphasis, lists, code blocks with syntax highlighting, and tables render as the text completes each line.
- Each tool call prints as `> tool_name {"arg": ...}` and its result as `< tool_name` followed by the first line of the return value. When a result spans several lines, the line ends with `(+N lines)`; every line is cut to the terminal width. A tool that asks the model to retry prints as `! tool_name` with the retry reason.
- Thinking parts print dimmed, as plain text, when `show_thinking` is on.
- A file change renders from the `file_system.*` events `FileSystem` emits: the unified diff of what `write_file`, `edit_file`, or `create_directory` is about to do (additions green, removals red, hunk headers cyan, cut at 8192 characters with a dimmed `(diff truncated)` note) before the change is put to the approver, and then the tool's own result line. A `search_files` or `find_files` result line is replaced by a dimmed match count, `3 matches for 'pattern' in src`, with `, truncated` when the search hit its cap.
- A shell command renders from the `shell.*` events `Shell` emits rather than from the tool's return value: `$ command` as the process starts, each output line as it arrives (stderr dimmed), and a dimmed `exit 0 (0.3s)`, `timed out (...)`, or `stopped (...)` summary, with `output truncated` added when the model saw only the tail. The generic `< run_command` line is skipped for these calls. Other capability events (a file edit, a plan update) are not rendered yet; the plan schedules them.

## Approving commands and file changes

Before `Shell` runs a command it emits a `ShellCommandRequestEvent`, and before `FileSystem` writes, edits, or creates a directory it emits a `FileChangeRequestEvent` carrying the diff; the bridge puts each to the run's approver. In a session that is a prompt on the terminal, after the diff:

```text
? run git push origin main [y/N]
? edit src/app.py [y/N]
```

`y` or `yes` lets it proceed; anything else, including the end of input, declines it, and the model is told `[Command was not run: declined by the user]` or `['src/app.py' was not edited: declined by the user]`. The answer is read ahead of the steer queue, so it is never forwarded to the model as a message.

`--yolo` (or `"yolo": true` in the config file) approves everything without asking. One-shot mode (`-p`) has no terminal to ask, so without `--yolo` every command and file change is declined with a reason that says so.

The approver is a run-time dependency, not part of the bridge: `Repl` passes `CliDeps(approver=...)` to each run and the bridge reads it from `ctx.deps`. `Approver` is a protocol, `async (event, *, description) -> Verdict`, so a policy engine can answer instead of a person. `allow_all`, `DeclineAll(reason=...)`, and `TerminalApprover(answers=..., output=...)` are the three that ship; `Repl(approver=...)` installs one for a session, and `CliBridge(approver=...)` pins one on the bridge for an agent that does not use `CliDeps`. A bridge with neither declines every request and says why.

The session's own lines, the prompt and the dimmed `(cancelled)` and `(steer queued: ...)` notes, come from `Repl`, not the bridge.

The bridge is an ordinary capability, so you can put it on your own agent. Wire it last so it observes every other capability's events:

```python
from pydantic_ai import Agent

from pydantic_ai_harness.cli import CliBridge
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder(), CliBridge()])
```

`CliBridge(output=..., width=..., config=...)` chooses the stream to write to (`sys.stdout` when unset, resolved at the start of each run), the column width (detected from the terminal when unset), and the `Config` whose `theme` and `show_thinking` it renders with (read from the config file at the start of each run when unset, so a bridge on your own agent follows the same theme as `harness`). Each run renders on a fresh bridge, so nothing carries over between runs.

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

To drive a whole session without a terminal, build a `Repl` over your agent and feed it `Lines`: `push` a line to type it, `push(None)` for Ctrl+D, and call `interrupt()` for Ctrl+C. `Repl.run()` is the session, `Repl.run_once(prompt)` is `-p`, and `Repl.history` is the conversation so far.

```python
import asyncio
import io

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.cli import CliBridge, Lines, Repl


async def main() -> None:
    buffer = io.StringIO()
    agent = Agent(capabilities=[CliBridge(output=buffer, width=80)])
    lines = Lines()
    lines.push('hi')
    lines.push(None)
    repl = Repl(agent=agent, model=TestModel(custom_output_text='Hello.'), lines=lines, output=buffer)
    await repl.run()
    print(buffer.getvalue())
    #> harness> Hello.
    #> harness>


asyncio.run(main())
```

## Telemetry

The CLI adds no spans of its own, and neither does `CliBridge` or `Repl`: rendering and prompt reading record no decision that core's spans do not already show, and a cancel or steer is visible in core's run span through `RunCancelled` and `EnqueuedMessagesEvent`. Core's [instrumentation](/ai/capabilities/instrumentation/) covers the run: `logfire.instrument_pydantic_ai()` before `main()` traces every model and tool call.

::: pydantic_ai_harness.cli.CliBridge

::: pydantic_ai_harness.cli.Config

::: pydantic_ai_harness.cli.Theme

::: pydantic_ai_harness.cli.Repl

::: pydantic_ai_harness.cli.Lines
