# CLAI plugins

A plugin is a Python file that teaches CLAI new tricks: react when something
happens, add a `/command`, give the agent a tool, or change how a tool's output
is shown on screen.

Everything a plugin can do goes through one object, the `PluginHost`. There is no
global registry to import and no magic file to name. You get a `host`, you tell it
what you want, you're done.

## On-demand authoring help

The default CLAI agent exposes `read_clai_customization_guide`. When you ask for
customization, its instructions tell it to read this guide first. Only the short
hint and tool description are present initially; the bundled text is read on tool
invocation. Reading it needs neither a checkout nor a network connection.
This tool needs no arguments and ignores extra arguments supplied by a model.
Other tools keep their existing validation.

The [bundled guide](src/pydantic_clai2/customization.md) includes examples for
commands, hooks, settings, renderers, custom Termflow menus, and custom model
launchers. It also names current limits: PluginHost does not register providers,
replace the prompt editor, or alter the built-in model catalog. Those need a
custom agent launcher or a source change, as explained in the guide.

Custom agents can opt in with `customization_guide()` from
`pydantic_clai2.customization`. The tool only returns documentation; it does not
write files, activate plugins, or grant permission to execute generated code.
Keep the bundled guide aligned with this contract when changing plugin APIs.

## Default code rendering

The default stream renderer buffers fenced code until the fence closes or the
text part ends, then highlights the whole block. This preserves multiline lexer
context. Unlabelled and Markdown fences stay literal; unknown languages use
plain text. Long code lines wrap to the terminal width. Prose still streams line
by line. A plugin renderer that handles a text event replaces this default
rendering for that event.

## Themes

```text
/theme
/theme light
/theme system
/theme pydantic
```

`/theme` opens a searchable picker with colour samples and a current-choice mark.
Enter saves and applies; Esc or Ctrl-C keeps the current choice. `/theme NAME`
selects directly, with Tab completion. `pydantic` is the default dark-terminal
brand palette; `light` uses darker accents and pale surfaces; `system` uses your
terminal's foreground and background without fixed UI colours. System diffs keep
plain `+` and `-` markers. CLAI does not change or detect the terminal background.

The setting is `display.theme`, shared with `/set` and persisted in SQLite.
The project field is `theme`; a project override returns at next start. Changes
apply immediately to the prompt, completions, status, menus, Markdown, thinking,
and built-in tool output, not text already printed in scrollback. The early
startup animation retains brand colours. Shell ANSI colours and hard-coded
plugin styles are not rewritten. Fenced code uses Monokai for `pydantic`,
Friendly for `light`, and terminal ANSI colours for `system`. Theme selection
makes no model requests and emits no telemetry.

Read roles when you render, not when the plugin imports:

```python
from rich.text import Text
from pydantic_ai_harness.filesystem import FileWrittenEvent

from pydantic_clai2 import theme
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.render(FileWrittenEvent)
    def show_write(event: FileWrittenEvent) -> Text:
        return Text(f'wrote {event.path}', style=theme.current().accent)
```

`theme.current()` exposes `accent`, `info`, `warning`, `error`, `muted`, and
`thinking`, plus Markdown and diff colours. Roles are scoped to the shell and
follow setting changes in its tasks and menu workers. Existing uppercase colour
constants remain the default brand values; use `current()` to follow selection.
Raw ANSI uses `theme.sgr(colour)`, including its 16-colour fallback. Custom Termflow
menus use `markdown_style()` when built. `theme.current().syntax` supplies the Rich
Syntax name `monokai`, `friendly`, or `ansi_dark` for renderer integrations.
The built-in fenced-code renderer uses this field too.

## Tool retries

CLAI defaults to three retries per tool call. `/set run.tool_retries N` changes
that default for subsequent turns; `N` must be a non-negative integer, with `0`
disabling retries. Explicit retry limits on a tool or toolset take precedence.
Output-validation and HTTP transport retry budgets are unchanged.

## Credentials

CLAI's `/login openai-codex` and the vllm and openrouter connections store tokens
in the configured keyring backend, not plugin settings. Large token bundles use
multiple entries to fit Windows Credential Manager's size limit. When no keyring
backend exists, credentials go to a per-account `0600` file under the user's CLAI config
directory instead. None of this changes plugin APIs. See
[Codex authentication](README.md#codex-authentication) for storage and security
details.

## Where plugins live

Plugins are trusted Python code. Drop-in files execute automatically at startup;
this directory is an executable startup configuration, not a sandbox. The default
Coder runs as your OS user and can modify it, just as it can modify your shell
startup files. Use a separate OS identity or sandbox for untrusted agent work.

Two ways to install one:

1. Drop a `.py` file (or a package folder) into
   `$XDG_CONFIG_HOME/pydantic-clai2/plugins/` (default `~/.config/pydantic-clai2/plugins/`).
   Its name is the file name without `.py`.
2. Point CLAI at anything importable, from the shell or from inside CLAI:

   ```sh
   clai2 plugins add notify my_package.notify
   /plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": true}'
   ```

No restart needed when you do it from inside CLAI. A plugin you add or enable is
active for the next prompt; one you disable or remove is gone for the next
prompt. From the shell, `clai2 plugins ...` only saves; it loads on the next
start. Plugins in the folder and the ones you added by name are managed the
same way. Explicit module declarations take precedence over a same-named drop-in
file. Reloading a disabled plugin is rejected; enable it first. Drop-in entry
modules are compiled from current source. Installed modules use `importlib.reload`,
which can retain globals removed from source; initialize plugin state explicitly.

Plugins are trusted code running as you. Only install what you trust.

## The built-in plugins

The coding tools are a plugin too, and so are asking you multiple-choice
questions mid-run, reading the repository's instruction file, and keeping the
conversation inside the context window. Native notifications, MCP configuration,
and update notices are plugins too. These plugins are marked
`(built-in)` and enabled unless you say otherwise:

| Id | Backed by | Settings | Does |
|---|---|---|---|
| `coder` | `pydantic_ai_harness.coder:Coder` | `{"unrestricted_filesystem": true, "repo_context": false}` | the file and shell tools |
| `ask_user` | `pydantic_clai2.ask_user_menu:activate` | `{}` | the `ask_user_question` tool: multiple-choice questions answered from the terminal |
| `repo_context` | `pydantic_clai2.repo_context` | `{}` | reads `CLAUDE.md` or `AGENTS.md` from the launch directory into the instructions |
| `persistence` | `pydantic_clai2.sessions` | `{}` | Harness step checkpoints for interrupted session recovery |
| `compaction` | `pydantic_clai2.compaction` | `{}` | automatic summarisation with a truncation fallback, `/compact`, and the context warning |
| `notifications` | `pydantic_clai2.notifications` | `{}` | native desktop alerts for completed or failed turns and questions awaiting an answer |
| `mcp` | `pydantic_clai2.mcp` | `{}` | `/mcp status` and explicit approval to load MCP servers; no servers connected by default |
| `updates` | `pydantic_clai2.updates` | `{}` | background PyPI update notices, with manual update guidance |

### `updates`: update notices

```text
/plugins disable updates
/plugins enable updates
```

`updates` checks public PyPI once per activation in a background task. A newer
stable release with an unyanked file compatible with your Python produces a
notice containing both versions and installation-specific guidance. It never
runs an installer. Source/direct installs keep their original workflow; uv tool
receipts, pipx environments, and pip metadata select their own guidance. Other
uv environments get conditional uvx advice rather than a guessed project/tool
command. See [update methods](README.md#update-notices) for the full table.

The request has a five-second deadline and a 1 MiB limit, without retries.
Offline errors, 404s (including an unpublished package), malformed responses,
and missing local version metadata are silent. PEP 440 comparison ignores
remote pre-release, development, local, empty, and fully yanked releases.
It does not use custom indexes or proxy environment settings. No prompts,
credentials, or installed version are sent. No model calls or capability
telemetry spans are added: this is terminal maintenance, not an agent operation.

Startup does not await the check. The worker awaits `host.notify` if a turn or
menu owns output. Disabling persists and cancels both a pending check and a
pending notice. `session_end` cancels and awaits the task, including on exit or
reload; enabling or reloading starts a new check. There is no process-global
worker, recurring poll, or persistent cache.

### Optional harness capabilities

`/plugins` and `/plugins list` also include every other public harness capability,
including each compaction strategy and guardrail. These entries start disabled.
`Coder`, `AskUser`, and `RepoContext` use the integrated entries above instead of
appearing twice. Deprecated aliases, toolsets, stores, and the ACP server adapter
are not separate capabilities.

Press Space to enable an entry. Its preview shows the import path and any load
error. Listing disabled entries does not import their modules or require their
optional packages. Some capabilities need an extra installed in CLAI's Python
environment, credentials, or constructor settings before they can load. Supply
JSON constructor settings by replacing the declaration under the same id:

```text
/plugins add sliding_window_compaction pydantic_ai_harness.compaction:SlidingWindowCompaction '{"max_messages": 40}'
```

For callbacks, stores, or other Python objects, use a plugin module that builds
the capability and calls `host.add(...)`, registered under that id. The menu does
not construct these objects or install dependencies. Avoid enabling overlapping
tool providers together, such as `filesystem` or `shell` alongside `coder`.
Removing an optional built-in restores its disabled declaration; enable and
disable choices persist between launches. Project, drop-in, and saved declarations
retain their usual precedence over built-ins.

`/plugins disable coder` gives you a chat-only CLAI (a writing or research setup
with `ExaSearch` instead, say); `/plugins enable coder` brings the tools back;
`/plugins remove coder` cannot forget a built-in, so it resets it to its
defaults. `/plugins disable repo_context` stops the instruction file from being
read. To run a built-in with different options, add your own declaration under
the same name and it takes the built-in's place:

```text
/plugins add coder pydantic_ai_harness.coder:Coder '{"unrestricted_filesystem": false, "repo_context": false}'
/plugins add repo_context pydantic_clai2.repo_context '{"walk_up": true}'
```

Keep `"repo_context": false` on a replacement `coder`: `Coder` bundles its own
`RepoContext`, and with the `repo_context` plugin also on, the instruction file
would reach the model twice.

`repo_context` wraps harness `RepoContext` with the launch directory as the
workspace and its default filenames. Its settings:

| Key | Default | Does |
|---|---|---|
| `walk_up` | `false` | also load instruction files from every directory between the workspace and your home directory |
| `inventory_tool` | `false` | give the agent `inventory_agent_context`, which maps the repo's `.claude`, `.agents`, `.codex`, and `.grok` assets |
| `nested_traversal` | `false` | when the agent reads or lists a directory, tell it about that directory's instruction file |
| `nested_inject` | `"pointer"` | what nested traversal adds: `"pointer"` (one line naming the file) or `"contents"` |

A repository can declare plugins too, in `.clai/settings.json`; they show as
`(project)` and rank just above the built-ins. They start off, because a
repository must not run code as you just because you opened it: CLAI names the
ones waiting at startup, and `/plugins enable NAME` approves one. See
[Project settings](README.md#project-settings).

`compaction` directly registers harness `FallbackCompaction` with
`max_fraction=threshold`; harness owns the automatic trigger. `/compact` runs the
same chain unconditionally. Only `ModelAPIError`, `FallbackExceptionGroup`, and
`UsageLimitExceeded` cause summarisation to fall back to truncation; other exceptions
propagate. `/plugins disable compaction` turns automatic compaction,
`/compact`, and its context warning off; a declaration under the same name
changes its settings (`strategy`, `threshold`, `protected_tokens`,
`context_window`, `summarization_model`; see the README):

```text
/plugins add compaction pydantic_clai2.compaction '{"threshold": 0.7, "context_window": 200000}'
```

### `notifications`: native desktop alerts

```text
/plugins disable notifications
/plugins enable notifications
/plugins remove notifications
```

`notifications` is enabled by default and has no settings. Enable and disable
choices persist; removing it restores the enabled default. It observes `turn_end`
for completed and failed turns, and `AskUserRequestedEvent` before the answerer
runs. Cancelled turns do not notify. Disabling notifications leaves `ask_user`
available; replacing its answerer still notifies if it uses harness `AskUser`.
No new lifecycle hooks are introduced.

Notifications use the title `CLAI2` and fixed status text. Prompts, answers, tool
arguments, paths, and error details are excluded to keep conversation content out
of desktop banners and notification history. They are sent even when the terminal
is focused, without requesting a sound.

macOS uses `/usr/bin/osascript`; the sender may appear as Script Editor. Allow
notifications for that sender in System Settings if needed. Focus mode and OS
notification settings can hide them. Linux uses `/usr/bin/notify-send` when
installed and requires a desktop notification service. Other platforms do nothing. Remote
sessions notify the machine running CLAI, not your local client.

Delivery awaits an async subprocess with a two-second timeout, no shell
interpolation, and terminal input/output disconnected. Both utilities use
absolute system paths rather than `PATH` lookup. The child receives only
`DISPLAY`, `WAYLAND_DISPLAY`, `DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, and
`XAUTHORITY` when set; provider credentials and dynamic-loader settings are not inherited. Missing utilities, nonzero
exits, and timeouts are ignored rather than failing the turn or question. A stalled
utility can delay the next prompt or question menu by up to two seconds. Timeout or
cancellation kills and reaps the child. There are no background workers to survive
unloading and no additional telemetry; core already traces the event hooks when
instrumentation is enabled.

### `mcp`: explicitly approved MCP servers

```text
/mcp status
/mcp load
/mcp load --approve
/plugins disable mcp
```

The built-in `mcp` plugin is enabled by default and needs no extra installation.
It does not read project config or connect at startup. `/mcp` and `/mcp status`
show the path and loaded server count without connecting; this is not a health
check. `/mcp load` names the file and warns about execution and environment access.
After reviewing it, `/mcp load --approve` loads the `.mcp.json` in the directory
where the plugin activated. It does not search parent directories or save approval.
Reloading, disabling, or restarting drops this in-memory approval; it cannot
approve another repository's `.mcp.json`. `/reload` also resets it.

Core's `load_mcp_toolsets` parses `mcpServers` and prefixes tools with server
names. It supports stdio `command`, `args`, `env`, and `cwd`, plus HTTP `url` and
`headers`. If an entry has both `command` and `url`, `command` wins.
Environment references support `${VAR}` and `${VAR:-default}`.
Relative paths use the process directory, not the config file's parent.
URLs ending in `/sse` select SSE; others select Streamable HTTP. Unknown keys
are ignored: `disabled` does not skip servers, and `type` does not select transport.
See the [runnable server example](README.md#mcp-servers) and
[core MCP reference](https://pydantic.dev/docs/ai/mcp/client/).

Each successful load replaces the toolsets in a reusable `Capability`.
Config-loading errors hide field values and leave the previous toolsets unchanged.
Check file access, JSON shape, and missing environment variables locally. Core owns connections
per turn; the plugin disables stdio keep-alive. Normal completion, model failure,
and CLAI's Esc/Ctrl-C `Task.cancel()` path close subprocesses. Unloading uses the
ordinary `PluginLoader` lifecycle to discard the host, commands, and capabilities.
No extra telemetry is emitted because core already instruments the tool calls.

Stdio server stderr goes to owner-only files in a private temporary directory,
not over the editor. `/mcp load --approve` and `/mcp status` show its path.
Files are named `server-N.log` by configuration order and append across turns.
Each load gets a new directory. Logs can contain sensitive server output; they
remain after exit for diagnosis, so remove them when no longer needed.

> **Outer cancellation limitation.** In Pydantic AI 2.44.0 and 2.46.0, cancelling
> an outer AnyIO scope during an MCP tool call can leave the stdio subprocess alive
> after the run unwinds. Reloading or disabling this plugin does not recover
> from that leak. Embedded callers must not assume this cancellation path is safe.
> The core-only regression is retained as a strict expected failure, tracked in
> [pydantic-ai issue #8548](https://github.com/pydantic/pydantic-ai/issues/8548).

```text
/plugins add mcp pydantic_clai2.mcp '{"config_path": "/absolute/path/to/.mcp.json"}'
/plugins remove mcp
```

`config_path` is the only setting, a string or `null` with a default of `null`.
An explicit absolute path opts into loading that file on every plugin activation.
Relative paths are rejected so a saved setting cannot trust a different file
based on which repository you open. The absolute setting is global but keeps
pointing to the same file. Prefer absolute `cwd` and script paths too.
`/plugins remove mcp` restores the default unconfigured built-in.

> **Trust includes later edits.** MCP config can execute commands as you and
> expand environment variables, including credentials. Persistent `config_path`
> trusts later edits and symlink-target changes on future activations. This is
> not a sandbox. Use environment references for secrets instead of embedding
> them in plugin settings or command history.

### `ask_user`: questions answered from the terminal

The second built-in, `ask_user` (`pydantic_clai2.ask_user_menu:activate`), gives
the model the harness's `AskUser` capability: one tool, `ask_user_question`, for
asking you one to ten multiple-choice questions when the task is ambiguous. Each
question appears inline below the conversation, leaving terminal scrollback
available. The title says `question 2 of 3` when there are several. Up/Down moves;
Enter or an option's number selects it. For multi-select questions, Enter or a
number toggles an option; the `Done` row submits once at least one is selected.
Space is not required. The highlighted option's description appears below the
choices. Esc, Ctrl-C, or Ctrl-D declines the whole request, which tells the
model you declined and lets the run continue. Streaming output is flushed and
the editor paused before questions open. It returns with the same draft, and
what you picked is printed to the transcript afterwards.
`/plugins disable ask_user` takes the tool away.

The capability does not know it is in a terminal. It hands an `AskUserRequest`
to an `Answerer` (one async callable returning an `AskUserResponse`) and waits.
To answer questions somewhere else, a web page or a chat bridge, say, replace
the built-in with your own plugin under the same name that constructs `AskUser`
with a different answerer:

```python
from pydantic_ai_harness.ask_user import AskUser, AskUserAnswer, AskUserRequest, AskUserResponse

from pydantic_clai2.plugins import PluginHost


async def ask_over_http(request: AskUserRequest) -> AskUserResponse:
    # POST request.questions to your front end, keyed by request.id, and wait
    # for the reply; return AskUserResponse(cancelled=True) if the user dismisses it.
    picks = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
    return AskUserResponse(answers=tuple(picks))


def activate(host: PluginHost[None]) -> None:
    host.add(AskUser(answerer=ask_over_http))
```

```text
/plugins add ask_user my_ask_user
```

The request and its response are also emitted as `AskUserRequestedEvent` and
`AskUserAnsweredEvent`, so a plugin that only wants to watch (log the question,
show a "waiting for you" state) registers `@host.on(EventClass)` or
`@host.render(EventClass)` without being the answerer.

## Browser mode compatibility

```python
from pydantic_ai import ModelRequestContext, RunContext
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.on('before_model_request')
    async def observe(ctx: RunContext[None], request: ModelRequestContext) -> ModelRequestContext:
        host.console.print(f'Model request {ctx.run_step}')
        return request
```

This plugin works in both interfaces. Install the `pydantic-clai2[web]` extra
and start `clai2 --web` to serve core's `Agent.to_web()` UI on
<http://127.0.0.1:7932>. See [browser setup and limits](README.md#browser-chat).

| Plugin surface | Browser behavior |
| --- | --- |
| `host.add(...)`, core hooks, typed core events | Contribute to core agent runs, including per-run capability factories. |
| `session_start`, `session_end` | Run once per ASGI lifespan, not once per browser chat. Plugins close after requests drain, before the agent closes and signal handling exits. |
| `turn_start`, `turn_end` | Registration raises and prevents startup. Use core run hooks for cross-interface guards. |
| `host.full_screen()` | Raises when called; terminal widgets have no browser equivalent. |
| Slash commands and `host.render(...)` | May register, but the browser does not dispatch or display them. Core renders its own stream. |
| `host.console`, `host.notify(...)` | Write to the server's terminal, not the browser. No interactive terminal owns that output. |
| `host.conversation`, `host.status` | Detached defaults, not the browser's history or status. Use core run context for run data. |

The terminal `ask_user`, `persistence`, `notifications`, and `updates` modules
are omitted even if enabled in the store. Preferences are not rewritten; the
next terminal launch still uses them. Omission matches the shipped module,
not just the plugin ID: replacing one with a custom plugin does not bypass
compatibility checks. There is no browser answerer for `ask_user` or CLAI SQLite
saving and resume.

Agent runs share one execution slot per server so browser tabs do not execute
coding tools concurrently. A real HTTP/stdio regression covers browser
disconnection during an MCP call: core's streaming runner closes the process.
The separate [outer-cancellation limitation](#mcp-explicitly-approved-mcp-servers)
affects direct `Agent.run()` embeddings.

Other enabled plugin import or activation failures abort startup. Project
plugins still require approval, and store overrides retain their precedence.
Use the terminal to change plugin settings, then restart the server. Core owns
the run loop, tool retries, model settings, and streaming; browser mode does not
simulate shell turn hooks or add telemetry beyond core and plugin spans.

## Managing plugins

`/plugins` on its own opens a full-screen menu, the same kind Code Puppy uses
for `/agent` and `/mcp`:

```text
 Plugins

 [x] coder         pydantic_ai_harness.coder:Coder   | coder
 [x] notify        ~/.config/pydantic-clai2/plugins  | source  ~/.config/.../notify.py
 [ ] audit         my_package.audit                  | state   enabled, loaded
                                                     | adds    2 commands, 1 hook, 0 tools
                                                     | error   none

 Up/Down move - Space enable/disable - R reload - D remove - Enter/Q close
```

The left side lists every plugin with `[x]` for on and `[ ]` for off. The right
side shows details for the highlighted one: where it came from, whether it
loaded, what it registered, and the last error if loading failed. Every key
acts immediately; there is no save step, so Enter, Q, Esc, and Ctrl-C all just close.
Closing returns to the prompt without printing the plugin list. Use `/plugins list`
to print it.
Adding a plugin needs a name and a module, so that stays a typed command.

With arguments `/plugins` is a plain command, and `clai2 plugins ...` outside
CLAI does the same thing:

| Command | Does |
|---|---|
| `/plugins list` | show every plugin and whether it is on |
| `/plugins add NAME module[:attr] [JSON]` | save it and load it now |
| `/plugins remove NAME` | forget an installed declaration; persistently disable a drop-in (delete its file yourself to remove it); reset a built-in or project-declared plugin to its declaration |
| `/plugins enable NAME` / `disable NAME` | load or unload, remembered across restarts |
| `/plugins reload NAME` | re-import the file and load it again (for editing a plugin while CLAI runs) |
| `/reload` | reload CLAI's own Python modules for development and rebuild the shell without restarting the process |

`/reload` takes no arguments. It uses `importlib.reload`, preserves the conversation,
agent, selected model, and active settings, and reactivates enabled plugins against
the refreshed shell types. Each loaded plugin receives `session_end` before reload
and `session_start` when loaded again. Plugin hosts and their registrations are
recreated, but installed module globals not overwritten by the new source can
survive. Initialize mutable state in `activate`. Disabled and unapproved project
plugins stay off. Failed imports or shell rebuilds restore the
previous module bindings and report the error, but cannot undo import-time side
effects. Restart for changes to startup code, import dependencies, or agent
construction. Third-party dependencies are not recursively reloaded.

What "load" and "unload" mean for your plugin:

- Load runs `activate(host)` and then fires `session_start` for that plugin, so a
  plugin loaded mid-session sees the same first event as one loaded at start.
- Unload fires `session_end` for that plugin, then drops everything it
  registered: handlers, commands, tools, renderers. Nothing else is touched.
- Both only happen between prompts, never while the agent is running.
- Drop-in entry modules load from current source. Installed entry modules use
  `importlib.reload`, which retains globals absent from the new source. Initialize
  plugin state explicitly on activation.

## The first plugin: give the agent web search

Pydantic AI Harness ships capabilities that are plugins as they are. `ExaSearch`
adds `web_search` and `get_page` tools backed by [Exa](https://exa.ai). Install
the extra and set the key, then add the class by name:

```sh
pip install 'pydantic-ai-harness[exa]'
export EXA_API_KEY=...
```

```text
/plugins add exa pydantic_ai_harness.exa:ExaSearch '{"num_results": 8}'
```

The JSON is passed to the constructor, so any keyword `ExaSearch` accepts that
JSON can express works here; an option that takes a Python object, like
`client`, needs the file form below. Ask the agent something that needs the web on the next prompt and it has
the tools. `/plugins disable exa` takes them away again.

The same thing as a plugin file, `~/.config/pydantic-clai2/plugins/search.py`,
which is the shape to start from when you want more than one capability, or a
`/command`, or a hook alongside it:

```python
from pydantic_ai_harness.exa import ExaSearch

from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost) -> None:
    host.add(ExaSearch(num_results=8, text_summary=True))
```

A plugin module exposes one function, `activate(host)`. CLAI calls it once when
the plugin loads. Everything you register inside it stays until the plugin is
unloaded or CLAI quits. Any Pydantic AI capability goes through `host.add`, so
`YouSearch` from `pydantic_ai_harness.youdotcom`, core's `WebSearch`, or one you
wrote yourself all work the same way.

`module:attr` may also name a function that takes a host, if you prefer a name
other than `activate`.

## Reacting to a moment

Plugins are not only for tools. This one rings the terminal bell when a turn
finishes, so you can tab away during a long run:

```python
from pydantic_clai2.plugins import PluginHost, TurnEnd


def activate(host: PluginHost) -> None:
    @host.on('turn_end')
    async def ping(event: TurnEnd) -> None:
        host.console.bell()
```

## What you can register

### React to a moment: `@host.on(name)`

`name` is a string. Your editor autocompletes it, and the type checker knows what
`event` your handler receives for each name. A typo is an error, not silence.
`host.on` is always used as a decorator.

Four names belong to CLAI itself. They fire outside the agent run, in the shell:

| Name | When | Event fields | Can change things? |
|---|---|---|---|
| `session_start` | CLAI has started, before the first prompt | `agent`, `settings` | no |
| `session_end` | CLAI is quitting | `reason`: `exit`, `eof`, or `error` | no |
| `turn_start` | you pressed Enter on a prompt | `text` | yes: edit `event.text`, or `event.cancel()` |
| `turn_end` | the turn finished, failed, or was interrupted | `text`, `outcome`, `result`, `error` | no |

Ctrl-C during an agent run keeps the prompt and captured partial messages in
conversation history for the next turn. Cancellation still reaches the running
tools for cleanup; it does not undo completed side effects or retry the run.
A prompt cancelled by `turn_start` never starts an agent run and is not retained.

Every other name is a Pydantic AI lifecycle hook, spelled exactly as on core's
`Hooks().on`, with the same handler signature. The ones people reach for:

| Name | When |
|---|---|
| `before_run` / `after_run` | an agent run starts / finishes |
| `before_model_request` | just before the model is called; you can edit the request |
| `before_tool_execute` | a tool is about to run; raise to stop it |
| `after_tool_execute` | a tool has returned |
| `tool_execute_error` | a tool raised |
| `event` | every stream event; prefer `@host.on(EventClass)` below |

The full list and every signature are in the
[hooks reference](https://pydantic.dev/docs/ai/core-concepts/hooks/). Core's
`wrap_*` and `on_*_error` methods drop their prefix here (`run`, `tool_execute`,
`run_error`), matching `Hooks().on`.

### React to a typed event: `@host.on(EventClass)`

Tools and capabilities emit typed events (a shell started, a file was written).
Pass the class instead of a string:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness.filesystem import FileWrittenEvent
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.on(FileWrittenEvent)
    async def log_write(ctx: RunContext[None], event: FileWrittenEvent) -> None:
        host.console.print(f'wrote {event.path}')
```

Some events let you say no. If the event has a `cancel()` method, calling it stops
the action before it happens (for example `FileChangeRequestEvent`).

### Add a `/command`: `host.commands.register(...)`

```python
from pydantic_clai2.commands import Command

host.commands.register(
    Command(
        name='greet',
        description='Say hello',
        handler=lambda args: 'Hello ' + (' '.join(args) or 'there'),
    )
)
```

The handler gets the arguments as a list of strings and returns the text to show.
It may be `async`. Add `complete=` to offer Tab suggestions. Names must be unique;
clashing with a built-in is an error at startup, not a silent override.

### Give the agent tools or instructions: `host.add(capability)`

```python
from pydantic_ai.capabilities import Capability

tools = Capability(instructions='Prefer British spelling.')


@tools.tool_plain
def word_count(text: str) -> int:
    return len(text.split())


host.add(tools)
```

`host.add` also accepts a function that takes a `RunContext` and returns a
capability (or `None`), for tools that should only exist in some runs.

### Draw an event yourself: `@host.render(EventClass)`

Built-in tool rendering shows one summary line per call by default, clipped to
the terminal width and followed by a blank line. In the default theme, tool names are pink; arguments
and bullet markers are muted grey. Shell output and completion details, grep results, and file
diffs are hidden from the terminal, not from the model. Set
`/set display.tool_output true` to restore detailed output; `display.shell_lines`
and `display.grep_lines` then control preview lengths (20 lines each by default).
This setting does not suppress plugin renderers or interactive questions.

CLAI shows unknown tool calls as `● tool_name`, with the name in the accent colour. To show something
better, return a Rich renderable (a `str` is fine). Return `None` to say "not mine,
use the default".

```python
from pydantic_ai_harness.filesystem import FileWrittenEvent
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost[None]) -> None:
    @host.render(FileWrittenEvent)
    def show_write(event: FileWrittenEvent) -> str:
        return f'wrote {event.path}'
```

CLAI flushes any streaming text before it prints what you return, so your output
never lands in the middle of a paragraph.

### Take the whole screen mid-run: `async with host.full_screen()`

An interactive widget opened from inside a tool call (including the built-in
inline `ask_user` questions) has to wait for streamed text to finish and the
editor and status row to get out of the way. `host.full_screen()` flushes pending output, suspends the
editor's input reader, and restores the editor and its draft when the block exits.
The editor remains active during agent turns: users can draft and queue messages,
but turns and slash commands execute sequentially. While work or turn lifecycle
hooks are active, a `Working` label and spinner appear in the editor's top border,
without adding an input row or changing the draft. The indicator uses the editor's refresh cycle, adds no
background task, and is hidden while a full-screen interface owns the terminal. Pending message previews appear
above the editor in execution order (`Follow-up:` for messages, `Command:` for
slash commands), and disappear when consumed. The preview is read-only; clipping
and flattening multiline text for display do not change the submitted text.
Esc in the live editor cancels active work, including turn lifecycle hooks,
without clearing the draft or requesting exit. While a plugin owns the screen,
its menu retains control of Esc.
Slash-command handlers already
run with the editor suspended; tool-driven widgets must take the screen explicitly:

```python
from pydantic_clai2.plugins import PluginHost


async def choose(host: PluginHost[None]) -> str:
    async with host.full_screen():
        return await show_my_menu()
```

When no editor or stream is active, taking the screen is a no-op. It only settles
the screen; drawing, and restoring the terminal afterwards, is the widget's job. One
widget owns the screen at a time: a second `full_screen()` (from a parallel tool
call, say) waits for the first block to exit. Do not nest it inside itself.

### Background terminal notices: `await host.notify(message)`

```python
import asyncio
from rich.console import Console
from pydantic_clai2.plugins import PluginHost

host = PluginHost(name='example', console=Console(), settings={})
asyncio.run(host.notify('A newer release is available.'))
```

In the shell, `notify` waits until turns and menus release output, then prints
literal text above the editor without changing its draft. This is terminal text,
not a native OS notification. A standalone host prints immediately. Use it from a
plugin-owned background task, not from a turn hook or command that must finish
before the notice can appear. Cancel and await
that task at `session_end`; cancellation discards a waiting notice. For model
events, continue to use `host.render` instead.

### Read your settings: `host.settings(Model)`

The JSON passed to `plugins add` is validated against a model you define:

```python
from pydantic import BaseModel


class NotifySettings(BaseModel):
    sound: bool = True


settings = host.settings(NotifySettings)
```

Bad or missing values fail at startup with a message naming your plugin.

### Reach the conversation and the status row: `host.conversation`, `host.status`

`host.conversation` is the retained history: `messages` is a snapshot,
`await commit_messages(...)` persists and swaps it between turns, and `resolved_model()` is the
model the next prompt will use. `host.status` is the footer's state; set
`context_alert` to paint the context figure in the warning colour. The built-in
`compaction` plugin uses both. A host built outside the shell gets an in-memory
`Transcript` and a detached `Status`, so tests need no special case.

## Rules that keep plugins predictable

- Handlers are `async`. There is no sync variant of anything.
- CLAI host observers return `None`. Core hooks keep their exact core return
  contracts: for example `before_model_request` must return its `ModelRequestContext`.
  For cancelable host events, edit the event or call `event.cancel()`.
- Raising in `turn_start` prevents the turn. Raising in core `before_tool_execute`
  fails the agent run, not only that tool. Use core's documented tool-denial
  mechanisms when the model should recover instead of ending the run, such as
  `pydantic_ai.exceptions.SkipToolExecution` for skipping an individual tool.
- Startup plugins load in alphabetical ID order. CLAI host handlers run in
  activation order, while `/plugins list` remains alphabetical. Core hook ordering
  follows core composition, including reverse order for `after_*` hooks. The first renderer that returns something wins. A
  plugin loaded later goes to the end of the line.
- Anything you print, print through `host.console`, so it stays in step with
  streaming output.

## Testing a plugin

`PluginHost` is an ordinary object. Build one with a `Console` writing to a
`StringIO`, call your `activate`, then call the handlers you registered with
hand-made events. No terminal, no model, no network.

```python
from io import StringIO

from rich.console import Console

from pydantic_clai2.plugins import PluginHost, TurnEnd

host = PluginHost(name='notify', console=Console(file=StringIO()), settings={})
activate(host)
for handler in host.handlers:
    await handler(TurnEnd(text='hi', outcome='completed'))
```

A plugin that reads the history gets a `Transcript` by default; pass
`conversation=Transcript(messages=[...], model=TestModel())` to seed it.

## vllm connection

Open `/add_model`, choose `vllm`, then enter a trusted HTTP(S) server root or `/v1` URL, and optionally a token. CLAI queries `/v1/models` and opens a searchable model picker. HTTP sends tokens unencrypted; use HTTPS outside trusted local networks.

## openrouter connection

Open `/add_model`, choose `openrouter`, then choose **Sign in with browser** or **Enter API key**. Browser sign-in opens OpenRouter's [PKCE authorization flow](https://openrouter.ai/docs/use-cases/oauth-pkce) and receives an authorization code on a temporary loopback listener. CLAI exchanges the code for a user-controlled API key over HTTPS. If the browser cannot reach CLAI (for example over SSH), paste the final callback URL or authorization code into the terminal. If no browser opens, open the printed authorization URL manually. Login times out after five minutes; Ctrl-C cancels it. You can revoke the generated key on OpenRouter.

Manual entry still accepts a key from https://openrouter.ai/keys in a masked prompt. After either method, select a model from the live catalog. CLAI validates the key with `/api/v1/key` before fetching `/api/v1/models`. Cancelling before model selection leaves the saved connection unchanged.

The connection is saved in the configured Python keyring backend after selection; backend security depends on your keyring configuration. Tokens are not stored in SQLite or command history. The selected model persists across restarts. Select the provider again to browse its live models or reconfigure the saved connection. Discovery is explicit and has a 20-second network timeout; redirects are not followed. Agent inference uses Pydantic AI core.


## Saved API keys

Open `/keys` to browse and manage saved API keys in a full-screen menu.
Use A to add, Enter to replace a value, R to rename, and D to delete with
confirmation. Values are masked and are not shown in previews. Changes save
immediately; Esc or Ctrl-C closes the menu. Errors appear inside the menu.

The compatibility command `/set api_key` prompts for a name and masked API key. Names are trimmed
and uppercased automatically, so `my_vllm_key` becomes `MY_VLLM_KEY`. Use letters,
numbers, and underscores, starting with a letter or underscore. Saving an existing
name asks before replacing it. Ctrl-C or Ctrl-D cancels without saving. Do not put
the secret on the command line.

When saved keys exist, vLLM's token prompt and OpenRouter's **Enter API key** flow
show a searchable list of names. Choose one, enter a different key privately, or
choose **No API key** for vLLM. Esc closes the picker without connecting. Browser
login flows are unchanged. Select keys only for endpoints you trust.

Named keys use the existing credential backend, separate from provider logins and
SQLite settings. If no OS keyring exists, CLAI warns that it saved them in the
per-user `0600` plaintext file `credentials-api-keys.json` in its config directory.
Key values never appear in the picker or confirmation. Names are labels, not
exported environment variables. Selecting a saved key stores a reference, not a copy. Discovery and each new
turn resolve its current value. Replacing a key updates connections that reference
it. Deleting it makes those connections fail until you restore the same name or
reconfigure them. Keys referenced by saved connections cannot be renamed. A cross-process lock
serializes key changes and connection saves so concurrent CLAI sessions do not
overwrite each other's key edits. The lock file contains no credentials.

Existing connections with inline credentials, manually entered connection keys,
and browser logins remain unchanged. To switch an existing connection to a
reference, reconfigure it through `/add_model` and select a saved key. Changes do not
alter an already running request or revoke credentials at the provider.

### Adding and selecting models

`/add_model` opens the provider catalog, connection setup, and per-model settings.
`/add_model PROVIDER:NAME` adds a model directly. Adding a model also selects it
for the next prompt. `/model` is a flat picker of added models, and `/model NAME`
selects one without the menu. Its Tab suggestions contain only added models.
The list persists across sessions. The currently configured model is retained
when upgrading; `/set model NAME` also saves the model in this list.

### Persisting conversation changes

Use `await host.conversation.commit_messages(messages)` for between-turn history
changes. It commits to storage before replacing the live history, and rejects
changes while an operation is running. `replace_messages(...)` remains an
in-memory compatibility API; it does not save by itself. `Transcript` implements
`commit_messages` without disk IO for headless plugin tests.

`host.conversation.step_store` is the configured Harness `StepStore`, or `None`
for an in-memory host. The built-in `persistence` plugin binds
`StepPersistence(capture_frontier=True)` to it. Do not register a second recorder
for the same store and run. Removing the plugin removes step capture on subsequent
turns; conversation-head saving is owned by `Session` and continues independently.

Harness exports `SnapshotSaved` from `pydantic_ai_harness.step_persistence`.
Subscribe through `@host.on(SnapshotSaved)` to observe committed checkpoints.
It carries `persistence_run_id`, `conversation_id`, `step_index`, and `state`.
This is a notification, not the durable source of truth or permission to replay a
tool. A durable replay may notify again. An observer failure cannot roll back the
already committed snapshot. No new CLAI lifecycle hooks are introduced.

Session naming is a shell-owned background service over Harness's `SessionNamer`.
It never writes into the agent transcript or loads plugin code. `/resume` does
not fire plugin load/unload hooks or restore previous plugin approvals. Cross-project
resume keeps the current working directory and the saved conversation's original
project grouping. The
project/session browser is a dedicated Termflow widget: unlike a single-pane
`MenuBuilder`, it has two independently navigable panes and two-line cards. Its
pure frame and scripted-key tests follow the same headless menu conventions.
The selected project stays highlighted while browsing sessions. The focused pane
is labeled **SELECT PROJECT** or **SELECT SESSION**, with matching key hints.

The resume transcript preview displays at most 24,000 characters of the newest-first
text, with a truncation notice for longer histories. Search is Unicode
case-insensitive and includes text instructions in multimodal prompts.
