# Harness CLI plan: Code Puppy feature parity on capabilities and events

Status: plan, iteration 0. Owner: Mike (mpfaffenberger). Loop driver: this file.
Branch: `feat/experimental-cli` (even with `main` at 8e863b5b when written).

## Rules the loop obeys

1. Every agentic behavior is a `Capability` (core `AbstractCapability`). The CLI host owns only
   terminal rendering, the line editor, slash-command dispatch, config persistence, and provider
   auth. The host never reaches into a capability's internals; it subscribes to events.
2. Every UI update is driven by typed events: core `AgentStreamEvent`s for model output and tool
   calls, `CapabilityEvent`s for everything a capability does. No message bus, no callbacks.
3. Decisions (approve, block, rewrite, answer) are `dispatch='immediate'` events with the core
   `cancel(reason)` shape. Notifications dispatch on the stream. See
   `pydantic-ai-notes/features/harness/2026-09-08 plan - a round of capability events for every
   harness capability.md` for the payload conventions; this file does not repeat them.
4. A missing capability or missing event is a PR on its own branch in a fresh worktree:
   `git worktree add ../harness-<slug> -b puppy/<slug> main`. Open the PR as a draft, then continue
   with the next item. The CLI consumes the branch via a local `uv` source until the PR merges.
5. One item per loop iteration. Pick the first unchecked item in "Execution order", do it, tick it,
   record any non-obvious decision under "Decisions", and stop. Re-read this file at the start of
   every iteration; it is the only state the loop trusts.
6. Never use `Any`. Never use em-dashes. Follow `AGENTS.md`, `agent_docs/capability-authoring.md`,
   and the docs parity rule (README next to code plus `docs/<name>.md`).

## Where things live

- CLI package: `pydantic_ai_harness/cli/` (the `experimental` tier is retired; ACP is the only
  capability left there). Console script under a code name in `pyproject.toml`, optional extra
  `cli` pulling `termflow` (approved as a dependency on 2026-09-08).
- Code name: placeholder `harness` until Mike picks one. The team sync left naming open; do not
  block on it.
- An earlier 480-line skeleton (`_app.py`, `_auth.py`, `_terminal.py`, `tests/experimental/test_cli.py`)
  is in `git stash@{0}` on `main`. Mine it for the provider-auth bits; do not restore it wholesale.
- Bridge capability: `CliBridge` in `pydantic_ai_harness/cli/_bridge.py`, wired last in the
  capability list so it observes every other capability's events via `@on_event`, and answers
  immediate decision events with the terminal prompts.

## Feature inventory

Every unique Code Puppy feature, where it lands, and what event work it needs. "Host" means CLI
code that is not a capability. "Core" means pydantic-ai owns it and the CLI just uses it.

### Agent loop and model output

| Code Puppy feature | Source | Lands in | Events | State |
|---|---|---|---|---|
| Default coding agent, system prompt, tool set | `agents/agent_code_puppy.py`, `_builder.py` | `Coder` composition | families of its members | exists |
| Streamed text, thinking, tool-call deltas; smooth stream; non-streaming fallback render | `event_stream_handler.py`, `smooth_stream.py`, `_non_streaming_render.py` | Host, consuming core `AgentStreamEvent` | core | host work |
| Thinking display filter, suppress thinking / info messages | `on_thinking_display_filter`, config flags | Host render policy | none | host work |
| Steer: inject user text mid-run | `_steer_processor.py`, `steer_metadata.py` | Core `AgentRun.enqueue` / `EnqueuedMessagesEvent` | core | host wiring |
| Cancel / pause / interrupt (Ctrl+C, key listeners) | `_run_signals.py`, `_key_listeners.py`, `pause_controller.py` | Core `AgentRun.cancel`; host keymap | core | host wiring |
| `agent_run_start` / `end` / `result` / `cancel` / `exception` hooks | `callbacks.py` | Core `wrap_run` on `CliBridge`; core run events | core | host wiring |
| `transform_model_messages`, `prepare_model_prompt`, `get_model_system_prompt` | `callbacks.py` | Core `before_model_request` and `instructions` on capabilities | core | nothing to build |
| `user_prompt_submit` (mutate prompt before run) | `callbacks.py` | `guardrails.InputGuard` | `GuardDecisionEvent` (wave 2) | events needed |
| Multiple system messages, per-model prompt overlays | README "Multiple System Messages" | `Coder` instructions plus model profile | none | verify then drop |
| Run stats, token usage, status bar | `run_stats.py`, `token_usage.py`, `status_display.py` | `spend` + `compaction.ReportContextUsage` | `SpendRecordedEvent`, `ContextUsageEvent` | merged (#714) |
| Retry profiles, retry checkpoint, HTTP retry | `retry_profiles.py`, `http_retry.py` | Core model retries, `FallbackModel`, provider `RetryConfig` | core | host config |
| Round-robin model distribution | `round_robin_model.py` | Core `FallbackModel` or a tiny `Model` wrapper in `cli/_models.py` | none | host work |
| Pydantic patches | `pydantic_patches.py` | Audit each patch; upstream to core or drop | none | audit |

### Tools the model calls

| Code Puppy feature | Source | Lands in | Events | State |
|---|---|---|---|---|
| `list_files`, `read_file`, `grep` | `tools/file_operations.py` | `FileSystem` | `DirectoryListedEvent`, `FileReadEvent` merged (#712); `FilesSearchedEvent` | round 2 needed |
| `edit_file`, `create_file`, `replace_in_file`, `delete_snippet`, `delete_file`, `apply_patch` | `tools/file_modifications.py`, `apply_patch.py` | `FileSystem` | `FileWrittenEvent` merged; `FileChangeRequestEvent` (immediate), `FileEditedEvent` with bounded diff, `DirectoryCreatedEvent` | round 2 needed |
| File permission prompt, yolo mode, `fs_access` sandbox | `file_permission_state.py`, `fs_access.py` | `FileSystem` root plus `FileChangeRequestEvent`; yolo = host auto-approves | same | round 2 needed |
| Undo last file change (`/undo`) | `undo_manager.py` | `FileSystem(snapshot_writes=True)` plus `revert_last()`; `FileRevertedEvent` | new | new option |
| `agent_run_shell_command`, background, kill / background chords, inactivity timeout | `tools/command_runner.py`, `shell_backgrounding.py` | `Shell` port | `ShellCommandRequestEvent` (immediate), `Start`, `OutputLine`, `End` | in progress, `puppy/shell-events` |
| Dangerous command guard + allowlist | `config.py`, `command_runner.py` | `Shell` deny policy plus `guardrails.ToolGuard` | `GuardDecisionEvent` | wave 2 |
| `invoke_agent`, `invoke_agent_with_model`, `list_agents`, recursion limit, subagent usage metrics | `tools/subagent_invocation.py`, `_subagent_recursion.py` | `SubAgents` | `DelegationStartEvent`, `DelegationEndEvent` | events needed |
| `ask_user_question` (interactive TUI question) | `tools/ask_user_question/` | New capability `AskUser` | `UserQuestionEvent` (immediate, `answer(...)`) | new capability |
| `agent_share_your_reasoning` | `tools/agent_tools.py` | Drop; thinking parts plus `Planning` cover it | none | decide |
| `load_image_for_analysis`, attachments, clipboard image paste | `tools/image_tools.py`, `command_line/attachments.py` | `media` capability plus host attachment picker | `MediaExternalizedEvent` (wave 3) | exists |
| Browser tools, QA Kitten agent | `tools/browser/`, `agent_qa_kitten.py` | `playwright` / `browser_use` | `BrowserNavigatedEvent` (wave 3) | exists |
| Web retriever agent | `agent_web_retriever.py` | `researcher` + `exa` / `youdotcom` | none | exists |
| Model judge agent | `agent_model_judge.py` | `trajectory_judge` | `TrajectoryJudgedEvent` (wave 2) | exists |
| Planning agent, `/plan` | `agent_planning.py` | `planning` | plan events merged (#714) | exists |
| Universal constructor (`/uc`, model writes its own tools) | `tools/universal_constructor.py` | `runtime_authoring` / `capability_creation` | `CapabilityCreatedEvent` (wave 3) | exists |
| Kennel memory | `kennel_provider.py` | `memory` + `conversation_search` | `MemoryChangedEvent`, `MemoryQueriedEvent`, `ConversationSearchedEvent` | wave 2 |
| Tool output limit (`tool_output_limit_chars`) | `_output_limits.py` | `tool_output_limits` / `overflowing_tool_output` | `ToolOutputLimitedEvent` (fold into #729) | events needed |
| Pre / post tool call hooks (`pre_tool_call`, `post_tool_call`, `fail_closed`) | `callbacks.py` | Core `before_tool_execute` / `after_tool_execute`; guards | core | nothing to build |
| Claude-Code-compatible hook engine (`PreToolUse` shell hooks, exit-code protocol, matchers) | `hook_engine/` | New capability `CommandHooks` | `HookRanEvent(event, matcher, exit_code, action)` | new capability |

### Context, history, and sessions

| Code Puppy feature | Source | Lands in | Events | State |
|---|---|---|---|---|
| Auto compaction, `/compact`, summarization model, protected tokens, strategy choice | `agents/_compaction.py`, config | `compaction` (`SummarizingCompaction`, `SlidingWindowCompaction`, `TieredCompaction`, `ClearToolResults`, `WarnNearLimits`) | core `CompactionStart/EndEvent` via #713 (blocked on core #7801); `ContextLimitWarnedEvent` | events queued |
| `/truncate` (drop N oldest turns) | `agents/_history.py` | Host calls the compaction capability's public API; add a strategy only if none fits | same | verify |
| Autosave, sessions, `/session`, `/autosave_load`, `/quick-resume`, session browser, format migration | `session_storage.py`, `session_lifecycle.py`, `command_line/session_*` | `step_persistence` (save / continue / fork) plus host session index | `StepSavedEvent`, `RunRestoredEvent` | wave 2 |
| `/dump_context`, `/load_context` | `core_commands.py` | Host, over `step_persistence` snapshots | same | host work |
| Agent rules: `AGENTS.md` search order, size cap | README "Agent Rules" | `repo_context` | `RepoNoteEnqueuedEvent` (wave 3) | exists |
| Skills catalog, `/skills`, skill activation | `skill_provider.py`, `tools/skills_tools.py` | `skills` | `SkillsLoadedEvent`, `SkillActivatedEvent` | events needed |
| Transcript guard, message queue, bus, frontend emitter | `messaging/` | Replaced by the event stream; ACP already consumes it | none | delete |

### Agents and models as data

| Code Puppy feature | Source | Lands in | Events | State |
|---|---|---|---|---|
| Agent catalog: Python agents, JSON agents, `/agent`, agent creator | `agent_manager.py`, `json_agent.py`, `agent_creator_agent.py` | `SubAgents` disk loader (Markdown + frontmatter) for definitions; host `/agent` switches the top-level composition | `AgentsLoadedEvent` | extend loader |
| Model registry, models.dev catalog, `/add_model`, `/refresh_models`, custom OpenAI types, timeouts | `model_factory.py`, `models_dev_parser.py`, `add_model_*` | Host `cli/_models.py` over core `infer_model` and providers | none | host work |
| Per-run model selection, per-agent pinning (`/model`, `/pin_model`, `/unpin`, `model_select` hook) | `model_switching.py`, `callbacks.py` | New capability `SelectModel` (swaps `ctx.model` in `before_model_request`) | `ModelSelectedEvent(model, reason)` | new capability |
| Model settings menu (temperature, seed, top_p, max tokens, context length) | `model_settings_*` | Core `ModelSettings`; host menu | none | host work |
| Provider auth: Claude OAuth, ChatGPT Codex, Gemini Code Assist, secret store, private inference | `claude_oauth_transport.py`, `chatgpt_codex_client.py`, `gemini_code_assist.py`, `secret_store*.py` | Host `cli/_auth.py`; propose core provider PRs where a flow is generic | none | host work, core PRs |
| MCP servers: config, `/mcp` commands, catalog install, health, circuit breaker, per-agent bindings, `pre_mcp_autostart` | `mcp_/`, `command_line/mcp/` | New capability `McpServers` wrapping core `MCPServer` toolsets | `McpServerStartedEvent`, `McpServerFailedEvent`, `McpServerStoppedEvent` | new capability |
| Plugins (`plugins/`, trust, callbacks registry) | `plugins/`, `callbacks.py` | Replaced by capabilities; host loads extra capabilities from config via core capability specs | none | delete |
| Logfire / observability toggle | `observability.py` | `logfire` capability plus core instrumentation | none | exists |
| Durable execution (DBOS) | README "Durable Execution" | Core durable exec plus `step_persistence` / `aws_lambda` | none | exists |

### Host chrome (not capabilities, listed for completeness)

Line editor, Ctrl+X chords, external `$EDITOR`, paste handling, completers (files, models, agents,
skills, commands), bottom bar and spinner, splash and figlet banner, onboarding wizard, colors and
theme menus, `/help` overlay and catalog, `/set` and `/show` config, `/cd`, `/clear`, `/exit`,
`/tools`, `/tutorial`, custom prompt-template commands (`~/.<name>/commands/*.md`,
`/generate-pr-description`), version check, diagnostics and error log, i18n, shell passthrough
(`!cmd`), headless / `-p` one-shot mode, CLI args and `handle_cli_args`.

Decisions for v1: i18n is dropped (YAGNI); everything else is host work rendered with Termflow.
Theme hooks (`termflow_style`, `highlighter`, `prompt_text_color`) become a single host `Theme`
dataclass loaded from config, not an extension point.

## New capabilities to propose (each its own worktree and PR)

| Capability | Module | Shape | Why not host code |
|---|---|---|---|
| `AskUser` | `pydantic_ai_harness/ask_user/` | Tool `ask_user_question(questions)`; emits `UserQuestionEvent` (immediate) that a host answers inline; falls back to a deferred tool request when no host answers. Implements open issue #42 (Aditya, 2026-03, never started); PR must reference it | The model calls it; the host only renders |
| `CommandHooks` | `pydantic_ai_harness/command_hooks/` | `before_tool_execute` / `after_tool_execute` run configured shell commands with Claude-Code-compatible env and exit-code protocol; matchers by tool name | Governs tool execution, not rendering |
| `McpServers` | `pydantic_ai_harness/mcp_servers/` | Loads a server config file, exposes core `MCPServer` toolsets, supervises lifecycle with retry and circuit breaker, per-agent binding filter | Provides tools; lifecycle is run state |
| `SelectModel` | `pydantic_ai_harness/select_model/` | `before_model_request` swaps the model from a `selector(ctx)` returning a `Model` or `None`; pinning is one selector | Changes what the run does |
| `FileSystem` undo option | existing package | `snapshot_writes` flag, `revert_last()`, `FileRevertedEvent` | Owned by the writer |

Before opening any of these, check `pydantic-ai-notes` and open/closed PRs in both repos for prior
art, per the repo guide. If core already has the primitive (for example a deferred human-in-the-loop
tool shape that fits `AskUser`), use it and record the finding.

## Execution order

Tick the box when the item is merged or, for CLI host items, when the acceptance check passes on
the branch. Each item names its acceptance check.

### Phase 0: skeleton that runs

- [x] 0.1 `pydantic_ai_harness/cli/` package, `pyproject.toml` console script and `cli` extra
      with `termflow`, `docs/cli.md` + package `README.md`. Acceptance: `uv run harness -p "hi"`
      with `TestModel` prints a response; `tests/cli/` passes. (on `feat/experimental-cli`,
      no PR yet)
- [x] 0.2 `CliBridge` capability: subscribes to core stream events, renders text and tool calls via
      Termflow. Acceptance: a transcript test drives `Agent(capabilities=[Coder(), CliBridge()])`
      with `TestModel` and asserts rendered output. (on `feat/experimental-cli` at a7e221bb,
      no PR yet)
- [x] 0.3 REPL loop: line editor, `-p` one-shot mode, Ctrl+C cancel via `AgentRun.cancel`, steer
      via `AgentRun.enqueue`. Acceptance: tests for cancel and steer. (on `feat/experimental-cli`
      at c164b433, no PR yet; the line editor is the terminal's own until 3.6, see Decisions)
- [x] 0.4 Config file and `Theme`; model selection from config via core `infer_model`. Acceptance:
      config round-trip test. (on `feat/experimental-cli` at 73172635, no PR yet)

### Phase 1: the transcript (events the CLI needs to be usable)

- [x] 1.1 `shell` events PR (`puppy/shell-events`, worktree `../harness-shell`). Finish, open draft
      PR, bridge renders `ShellCommandStartEvent` / `ShellOutputLineEvent` / `ShellCommandEndEvent`
      and answers `ShellCommandRequestEvent` with an approval prompt (yolo auto-approves). PR #845
      (draft, CI green, pydanty labelled); bridge half merged into `feat/experimental-cli`.
- [x] 1.1b Work pydanty's review of #845 (`gh pr view 845 --comments`; the label
      `pydanty:is-working` clears when it lands), address the findings in `../harness-shell`,
      merge the fixes into `feat/experimental-cli`, then mark #845 ready for review. Six required
      findings fixed at 5013ee2c; the blocking one is the floor (1.1c), answered on the PR.
- [x] 1.1c Floor on `pydantic-ai` 2.40.0 (issue #850, branch `puppy/floor-2-40`, worktree
      `../harness-floor`): `@agent.on_event` is 2.40.0+ and six merged docs pages plus shell
      use it. Dependency PR, needs `dependencies:approved`. Unblocks pydanty on #845. PR #851
      (draft, pydanty labelled); waiting on a maintainer for `dependencies:approved`.
- [x] 1.1d When #851 merges: rebase `puppy/shell-events` on `main`, answer pydanty's floor
      finding on #845 with the merged PR, mark #845 ready for review, and merge `main` into
      `feat/experimental-cli` so the CLI lock picks up the floor. #851 merged at 16:07Z
      (by Mike); the branch already carried the `main` merge (6a3791c5), so no rebase was
      needed. Replied on #845 that the floor finding is answered on both sides and marked
      it ready for review. `main` merged into `feat/experimental-cli` at eed149a0; the
      `cli` extra conflict resolved to the 2.40.0 floor plus `termflow`, and the lock now
      resolves pydantic-ai-slim 2.42.0.
- [ ] 1.1e Work the #845 re-review pass. Round one: the 18:13Z pydantic required finding
      fixed at aa50d07c; the Macroscope round it triggered (three findings, two high) fixed
      at 89dfdfdf and replied to. The CI failure on 89dfdfdf (the `TestKillAfterLeaderExit`
      test on every 3.11/3.12 cell) is root-caused and fixed at 42e985f9 (see journal: the
      GHA hosted-compute supervisor tears a `setsid` session down when its leader exits;
      the test now pre-probes and skips where the scenario cannot be demonstrated); a
      one-line coverage pragma landed at 95c7db06 and CI is fully green (16/16 test cells,
      coverage 100, check, correctness). A fresh pydantic run on 95c7db06 was dispatched
      at 20:49Z with the correct trigger label `pydanty:review-lite` (earlier attempts
      used the wrong label and were ignored, see journal). Macroscope verified the
      docs-echo fix (18:40Z); the veto and try/finally threads still await its
      verification. When the pydantic verdict lands: fix what is real, reply per finding,
      and merge the fixes into `feat/experimental-cli`.
- [x] 1.2 `filesystem` round 2 PR (`puppy/filesystem-change-events`, worktree
      `../harness-filesystem-change-events`): `FileChangeRequestEvent`, `FileEditedEvent`,
      `DirectoryCreatedEvent`, `FilesSearchedEvent`. Bridge renders diffs and search counts,
      answers the request event. PR #853 (draft, CI green including coverage, pydanty labelled);
      bridge half merged into `feat/experimental-cli` at 9cd783c7.
- [x] 1.2b Work pydanty's review of #853 (`gh pr view 853 --comments`; the label
      `pydanty:is-working` clears when it lands), address the findings in
      `../harness-filesystem-change-events`, merge the fixes into `feat/experimental-cli`, then
      mark #853 ready for review. One blocking and seven required findings fixed at 0b0932f4,
      merged into `feat/experimental-cli` at f32cb9cd, CI green, marked ready; the floor
      finding is answered with #851 as on #845. A second pydanty pass was requested at 23:32Z
      and had not landed when this iteration ended: check it at the start of the next one.
      The `FileEditedEvent` subclass question for Douwe is in the PR body.
- [ ] 1.2e Work the third pydanty pass on #853 (landed 18:53Z: zero blocking, one
      required, incomplete review -- but it analyzed stale head 4f8e1c43, an ancestor that
      predates the f5b9134d header-cut fix, so the required is already fixed and the
      verdict untrustworthy). Make `FileChangeRequestEvent.cancelled` monotonic/read-only
      the way #845's was just fixed (private field + property, `cancel()` the only setter,
      a `LiftCancel`-style test, docs parity), push, and re-apply the trigger label
      `pydanty:review-lite` (not `pydantic-ai:review-lite`) so a fresh run analyzes the
      real head. Merge the fix into `feat/experimental-cli`.
- [ ] 1.2c When #853 merges: merge `main` into `feat/experimental-cli` and drop the
      `puppy/filesystem-change-events` worktree.
- [x] 1.2d Work pydanty's second pass on #853 (landed 00:04Z: one blocking, the floor finding
      again, answered with #851; five required) plus the Macroscope and Veria threads it picked
      up at 23:41Z (edit-after-await staleness, `parent_file/child.txt` passing the parent
      check, final-newline-only diffs coming out empty, diff cost on large files). Fix in
      `../harness-filesystem-change-events`, merge into `feat/experimental-cli`, reply per
      finding, re-apply `pydanty:review-lite`. The five required were fixed across c70f442c
      to f5b9134d (the last, oversized diff headers cut at the bound); the blocking one
      cleared when #851 merged. The re-applied pass landed 16:59Z green on f5b9134d: zero
      blocking, zero required, one informational note with a skip verdict (a mutation
      window inherent to the by-name design, documented in the security model). Fixes
      merged into `feat/experimental-cli` at ffda2f55.
- [x] 1.3 `subagents` events PR (`puppy/subagents-events`, worktree
      `../harness-subagents-events`): `DelegationStartEvent`, `DelegationEndEvent`. Bridge renders
      a `>>` start line and a `<<` end line per delegation. PR #855 (draft, CI green, pydanty
      labelled at 23:55Z); bridge half merged into `feat/experimental-cli` at eeed7062.
- [ ] 1.3b Work pydanty's review of #855 (`gh pr view 855 --comments`; the label
      `pydanty:is-working` clears when it lands), address the findings in
      `../harness-subagents-events`, merge the fixes into `feat/experimental-cli`, then mark
      #855 ready for review. The floor finding, if it comes, is answered with #851 as on #845.
      The 17:20Z verdict (one blocking, a child `HookTimeoutError` misread as the delegation
      timeout; one required, `__all__` grouping) was fixed at f5dc6fd6. The re-triggered run
      withheld publication at 17:27:58Z ("PR head changed after review", Mike's `main` merge
      landed a minute later), so no verdict is on the current head 424c9634; the label was
      re-applied at ~18:30Z to start a fresh run there. The fixes are not yet merged into
      `feat/experimental-cli`.
- [ ] 1.3c When #855 merges: merge `main` into `feat/experimental-cli` and drop the
      `puppy/subagents-events` worktree.
- [ ] 1.4 `skills` events PR (`puppy/skills-events`): `SkillsLoadedEvent`, `SkillActivatedEvent`.
- [ ] 1.5 `tool_output_limits` events (`ToolOutputLimitedEvent`), coordinated with #729.
- [ ] 1.6 Consume `SpendRecordedEvent`, `ContextUsageEvent`, planning events in the status bar.
- [ ] 1.7 Compaction: consume #713 events once core #7801 ships; `ContextLimitWarnedEvent` PR.

### Phase 2: new capabilities

- [ ] 2.1 `AskUser` capability PR (`puppy/ask-user`, closes #42). Bridge renders the question TUI.
- [ ] 2.2 `SelectModel` capability PR (`puppy/select-model`); host `/model`, `/pin_model`, `/unpin`.
      Core already ships `pydantic_ai.capabilities.SelectModel` with this shape (see Decisions,
      2026-09-09, 0.2); the item is now host wiring plus a `ModelSelectedEvent` only if needed.
- [ ] 2.3 `FileSystem` undo option PR (`puppy/filesystem-undo`); host `/undo`.
- [ ] 2.4 `CommandHooks` capability PR (`puppy/command-hooks`).
- [ ] 2.5 `McpServers` capability PR (`puppy/mcp-servers`), thin config wrapper only; host `/mcp`
      list / start / stop commands.
- [ ] 2.6 `SubAgents` loader: agent catalog events and host `/agent` switching.

### Phase 3: host parity

- [ ] 3.1 Slash-command registry and `/help`; `/compact`, `/truncate`, `/plan`, `/tools`, `/cd`,
      `/clear`, `/set`, `/show`, `/dump_context`, `/load_context`.
- [ ] 3.2 Sessions over `step_persistence`: autosave, `/session`, `/quick-resume`, browser.
- [ ] 3.3 Model registry: models.dev catalog, `/add_model`, `/refresh_models`, model settings menu.
- [ ] 3.4 Provider auth flows from `stash@{0}` `_auth.py`; core provider PRs where generic.
- [ ] 3.5 Attachments and clipboard images via `media`.
- [ ] 3.6 Custom prompt-template commands, shell passthrough, chords, `$EDITOR`. Includes the
      inline line editor 0.3 deferred: raw mode over Termflow's `read_key`, history, bracketed
      paste (a multi-line paste today is one prompt plus steers), and a prompt that survives
      output written while a run is in flight.
- [ ] 3.9 Session error handling: a run that raises (`UsageLimitExceeded`, a model API error)
      ends the session with a traceback today; print it as a status line and keep the prompt.
- [ ] 3.7 Splash, onboarding, theme menus, version check.
- [ ] 3.8 Partner PR to pydantic-ai `docs/navigation.yml` adding `harness/cli` (the sidebar lives
      there, see `agent_docs/docs-conventions.md`). Do this when `feat/experimental-cli` opens its
      PR against `main`, not before.

### Phase 4: remaining event families (wave 2 and 3 of the events plan)

- [ ] 4.1 `guardrails` `GuardDecisionEvent`; dangerous-command guard as a `ToolGuard`.
- [ ] 4.2 `step_persistence` `StepSavedEvent`, `RunRestoredEvent`.
- [ ] 4.3 `memory`, `conversation_search` events.
- [ ] 4.4 `trajectory_judge`, `prompt_injection_defender` events (deprecate callbacks).
- [ ] 4.5 `code_mode` execution lifecycle events.
- [ ] 4.6 Wave 3 as the bridge needs them: `media`, `playwright`, `runtime_authoring`,
      `repo_context`.

## Decisions

Append-only. Date, item, decision, why.

- 2026-09-09, plan: CLI lives in `pydantic_ai_harness/cli/`, not `experimental/`, because the
  experimental tier is retired per `AGENTS.md`.
- 2026-09-09, plan: i18n dropped for v1. No user has asked; it is pure host cost.
- 2026-09-09, plan: plugins and the callback registry are not ported. Capabilities plus core
  capability specs are the extension model; a plugin is a capability the host loads from config.
- 2026-09-09, plan: `agent_share_your_reasoning` is not ported pending evidence that thinking parts
  plus `Planning` leave a gap.
- 2026-09-09, Mike: `AskUser` is inline immediate event first, deferred fallback second.
  Implements #42. Mike confirmed the three vetoable calls above (cli package location, i18n
  dropped, plugins not ported).
- 2026-09-09, Mike: `McpServers` is a thin config-driven wrapper first; the supervisor (health,
  circuit breaker, catalog install) is a later item, not part of 2.5.
- 2026-09-09, plan: approval answers in the bridge go through a pluggable `Approver` because #340
  `PermissionPolicy` is open and targets the same decisions.
- 2026-09-09, plan: the `agent_docs/index.md` link to this file is parked in a stash named
  "cli-plan: agent_docs/index.md" on `feat/experimental-cli`; re-apply it when committing the plan.
- 2026-09-09, 0.1: the existing `cli` extra (a pass-through for core's `clai`) gains
  `termflow-md>=0.9.1 ; python_full_version >= '3.11'` instead of a new `harness-cli` extra, so
  `pydantic-ai-harness[cli]` means "the CLI" and a 3.10 install still resolves. Nothing imports
  `termflow` until 0.2, so the one-shot mode runs on 3.10; 0.2 adds the version gate with the
  renderer.
- 2026-09-09, 0.1: `cli_agent` is a module-level, model-less `Agent` exported from
  `pydantic_ai_harness.cli` (same shape as `coder_agent`); `--model` is passed at run time and
  tests swap it with `cli_agent.override(model=TestModel(call_tools=[], ...))`. `--model test`
  from the shell builds a bare `TestModel` that calls every `Coder` tool with schema-default
  arguments (it writes a file named `a` and exceeds `start_command`'s retries), so the acceptance
  check is the override test in `tests/cli/`, not a shell invocation with `test`.
- 2026-09-09, 0.1: no `--workspace` flag; the current directory is the workspace, matching Code
  Puppy (`/cd` is item 3.1). `-p` is required until 0.3 adds the interactive session.
- 2026-09-09, 0.1: `UserError` from model resolution or the run is reported through
  `parser.error` (one line, exit 2) so a missing API key does not print a traceback. Other
  exceptions propagate; the stash's blanket `except (ModelAPIError, OSError, ...)` hid too much.
- 2026-09-09, 0.1: `cli` is registered in `tests/test_docs_parity.py` as `_NOT_A_CAPABILITY` and
  `cli.md` as a non-capability page, like `media`. It keeps a README, a docs page, a source link,
  and a top-README link anyway; only the capability-specific checks (H1 registry, autodoc) do not
  apply.
- 2026-09-09, 0.1: no `__main__.py`; the console script is the entry point and `uv run harness`
  works in the checkout. Add `python -m` only if someone asks.
- 2026-09-09, 0.1: `pyproject.toml` and `uv.lock` changed, so the eventual PR from
  `feat/experimental-cli` needs the `dependencies:approved` label (pre-approved per this plan).

- 2026-09-09, 0.2: `CliBridge` listens with `@on_event` on the core stream events
  (`PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FunctionToolCallEvent`,
  `FunctionToolResultEvent`); core auto-enables streaming for `agent.run()` when a capability
  has listeners, so `main` needs no `event_stream_handler`. `wrap_run_event_stream` was
  rejected: the bridge only observes, and a wrapper would make it a global filter.
- 2026-09-09, 0.2: Termflow parses whole lines, so text deltas buffer until a newline and flush on
  `PartEndEvent` (`_MarkdownStream`). Smooth pacing (Termflow's `StreamSmoother`) is host chrome
  for 0.3 or later, not part of the bridge.
- 2026-09-09, 0.2: tool calls and results render as one line each (`> name args`,
  `< name first-line (+N lines)`, `! name retry-reason`), cut to the terminal width with
  Termflow's `truncate_ansi`. The full return value is the model's, not the transcript's; a
  richer view (diffs, grep hits) comes from capability events in phase 1.
- 2026-09-09, 0.2: thinking parts are not rendered. The inventory lists the thinking display
  filter as host render policy, so it lands with config (0.4) rather than as a hard-coded choice
  in the bridge. Pinned by `test_thinking_is_not_rendered`.
- 2026-09-09, 0.2: `CliBridge.output` defaults to `None` and resolves `sys.stdout` inside
  `for_run` (via `replace(self)` re-running `__post_init__`) because `cli_agent` is built at
  import time; a construction-time default would capture stdout before pytest's capture or any
  redirect exists. `for_run` also gives each run fresh stream state.
- 2026-09-09, 0.2: `main` no longer prints `result.output`; the bridge already rendered it.
- 2026-09-09, 0.2: the CLI is Python 3.11+ because Termflow is. `tests/cli/conftest.py` skips on
  a missing `termflow`, and the import error names the `cli` extra. Tests pin the asyncio anyio
  backend like `tests/coder` (the CLI runs under `asyncio.run`).
- 2026-09-09, 0.2: `CliBridge` is `Anonymous` in `tests/test_capability_combine.py`: a renderer
  per output stream; two on one stream print twice, which is the user's mistake to see.
- 2026-09-09, 0.2: finding for 2.2: core ships `pydantic_ai.capabilities.SelectModel` (a
  `selector` returning a model or model ID per request step, with `ModelSelectionContext`). Do
  not build a harness one; 2.2 becomes host wiring over core's, plus a `ModelSelectedEvent` PR
  only if the bridge needs to render the switch.
- 2026-09-09, 0.2: CI runs on `main` pushes and PRs only, so `feat/experimental-cli` gets no CI
  until its PR opens. A focused branch-coverage run on `pydantic_ai_harness/cli/*` is the
  substitute for host items (100% at a7e221bb); repo-wide runs stay off.

- 2026-09-09, 0.3: the session is `Repl` in `pydantic_ai_harness/cli/_repl.py` (not `Session`,
  which 3.2 needs for persisted conversations). It drives `agent.iter()` so it holds the
  `AgentRun` handle; `run.cancel()` from the SIGINT handler surfaces as `RunCancelled` at
  context exit, and `exc.all_messages()` becomes the next prompt's `message_history`, so the
  cancelled turn (tool call included) stays in the conversation. Verified against core 2.38.0.
- 2026-09-09, 0.3: Termflow's `TextInput` is a full-frame form widget (`CURSOR_HOME` plus
  clear-below on every paint), so it cannot be the inline prompt. 0.3 ships the terminal's own
  canonical-mode editing over `sys.stdin.readline`; the raw-mode inline editor moved to 3.6.
  Not a new dependency and not a blocker: the acceptance check is cancel and steer.
- 2026-09-09, 0.3: one stdin reader (`Lines.from_stdin`, a daemon thread feeding an
  `asyncio.Queue` via `call_soon_threadsafe`) serves the whole session. A blocked read cannot be
  cancelled, so a reader-per-prompt would leave a stray thread racing the next one for the
  terminal. The rule that falls out: a line read while a run is in flight is enqueued into it
  (`(steer queued: ...)`), a line read while idle is the next prompt. Piped multi-line stdin
  into the session therefore steers the first run; scripting is `-p`, and paste is 3.6.
- 2026-09-09, 0.3: `-p` does not read stdin at all (`Repl(lines=Lines())`), so piped input keeps
  its meaning for a later stdin-as-attachment item (3.5) and pytest's stdin stub never runs.
- 2026-09-09, 0.3: SIGINT is wired with `loop.add_signal_handler` (CI is Ubuntu, dev is macOS;
  Windows would need `signal.signal`). Ctrl+C while idle writes a newline and re-shows the
  prompt, like the Python REPL; Ctrl+D ends the session. The cancel test sends a real SIGINT
  with `os.kill` so the handler wiring is covered, not just `Repl.interrupt()`.
- 2026-09-09, 0.3: `Repl` and `Lines` are public exports so a host embedding the CLI can drive a
  session without a tty; `submit()` is public too because 3.1's slash commands will call it.
  `AgentRunError`s still propagate and end the session (3.9).
- 2026-09-09, 0.3: a size-less pty reports 0 columns and Termflow's `truncate_ansi` then blanks
  the bridge's tool lines. Real terminals report a size; noted, not handled.

- 2026-09-09, 0.4: the config file is JSON at `~/.pydantic-ai-harness/config.json` (the
  directory the stashed skeleton already used for credentials). TOML would need `tomli-w` to
  write and the round trip is the acceptance check; JSON is stdlib both ways and `Config` is a
  Pydantic `BaseModel` (the file is a boundary, so validation is earned) with `load`, `save`,
  and `default_path`. A missing file is the defaults; an invalid one raises `UserError` with the
  path and the validation detail, which `main` reports through `parser.error`. Pydantic ignores
  unknown keys, so an older CLI reads a newer file.
- 2026-09-09, 0.4: `Theme` is a `kw_only` dataclass nested in `Config` with `palette` (a
  `Literal` over Termflow's named `RenderStyle`s: `default`, `dracula`, `gruvbox`, `nord`) and
  `code_style` (a Pygments style name; Termflow falls back to `monokai` on an unknown one, so
  it is not validated). Code Puppy's `prompt_text_color` is not ported: the prompt is a plain
  string until the inline line editor (3.6) owns the prompt line.
- 2026-09-09, 0.4: `CliBridge.style` became `CliBridge.config: Config | None`, read from the
  file at the start of each run when `None`, the same rule `output` already follows for
  `sys.stdout`. Capabilities are fixed at agent construction and `Agent.override` has no
  `capabilities` (checked on core 2.38.0), so the module-level `cli_agent` cannot be handed a
  theme any other way. `main` reads the same file for the model; two readers of one file, no
  in-memory copy to keep in sync, so 3.1's `/set` only needs `Config.save` and the next run
  picks it up. A `CliBridge` on a user's own agent follows the harness theme for the same
  reason; documented.
- 2026-09-09, 0.4: `--model` beats the file, the file beats `DEFAULT_MODEL`. No env var (YAGNI)
  and no eager `infer_model` in `main`: it would resolve the provider before
  `cli_agent.override(model=TestModel(...))` applies and fail CI on a missing API key. The
  config string reaches core's `infer_model` through `agent.iter(model=...)`, which is the
  "via core `infer_model`" the item asks for.
- 2026-09-09, 0.4: `show_thinking: bool = False` lives on `Config`, not `Theme` (it is policy,
  not color), and closes the thinking-display decision 0.2 deferred here. Thinking parts stream
  as dimmed plain text through `_DimStream`, not Markdown; `_MarkdownStream` and `_DimStream`
  share a `_Stream` protocol so the part hooks stay one code path.
- 2026-09-09, 0.4: `tests/cli/conftest.py` sets `HOME` to `tmp_path` for every CLI test so
  `Config.default_path()` never reaches the developer's real file. `Path.home()` reads `HOME`
  on POSIX; Windows would need `USERPROFILE`, and CI is Ubuntu.

- 2026-09-09, 1.1: the shell PR keeps `start_command` / `check_command` / `stop_command` and
  adds the four events to the existing toolset instead of porting Code Puppy's tool wholesale
  (inactivity timeout, detached background mode). The events are what the CLI needs; the port
  is a separate item if the CLI's daily use asks for it. Helpers moved to `_process.py` to keep
  `_toolset.py` under the size limit; they are public names inside the private module because
  pyright's `reportPrivateUsage` fires on cross-module underscore imports.
- 2026-09-09, 1.1: `ShellOutputLineEvent` only for foreground commands (background output goes to
  files the model polls); a background command's end event fires from `check_command` or
  `stop_command` and carries that call's `tool_call_id`, not the original `start_command`'s.
- 2026-09-09, 1.1: the approver is a run-time dependency. `Repl` passes `CliDeps(approver=...)`
  to `agent.iter(deps=...)` and the bridge reads `ctx.deps`; `cli_agent` is now
  `Agent[CliDeps, str]`. Rejected: a ContextVar (works but is not how core hands run-time state to
  capabilities) and rebuilding `cli_agent` per session (breaks the module-level agent tests use).
  `CliBridge(approver=...)` still exists for a user's own agent without `CliDeps`; with neither,
  every request is declined with a reason naming both options. `Approver` returns a `Verdict`
  (`allowed`, `reason`) rather than a bool so `DeclineAll` can say why.
- 2026-09-09, 1.1: `Lines.ask()` routes the next pushed line to a pending question ahead of
  `read()`, because the steer task drains `read()` for the whole run and would otherwise forward
  the `y` to the model. End of input answers the question and still reaches `read()`. One
  question at a time; a second concurrent `ask()` raises.
- 2026-09-09, 1.1: `-p` mode has no stdin reader by the 0.3 decision, so without `--yolo` every
  request is declined with `one-shot mode has no terminal to ask; run with --yolo to allow`
  instead of hanging. `--yolo` and `Config.yolo` both select `allow_all`.
- 2026-09-09, 1.1: the bridge skips its generic `< tool` result line for any `tool_call_id` an
  end event already summarised, keyed on the event's `tool_call_id`; it keeps the `> tool` call
  line because that fires before the request event. It does not know tool names.
- 2026-09-09, 1.1: core refuses `AgentRun.emit` for capability events (only `CustomEvent`), so the
  bridge test for a host-emitted end event uses a one-hook test capability.
- 2026-09-09, 1.1: the docs snippet style for events is `@agent.on_event` with untyped params,
  matching the planning and spend pages; snippets are ruff-checked, not run.
- 2026-09-09, loop: issuing several `edit` calls against one file in a single tool batch loses
  all but one of them (they apply against the same base). One edit per file per call.

- 2026-09-09, 1.1b: pydanty's blocking finding on #845 (`Agent.on_event` missing on the 2.38.0
  floor) is not fixed on the shell page. `@agent.on_event` landed in core 2.40.0
  (pydantic-ai#8101) and the merged planning, spend, compaction, filesystem, and
  system-reminders pages already use it, so shell keeps the sibling style and the floor bump
  is its own dependency PR (issue #850, item 1.1c). Supersedes nothing; the 1.1 snippet-style
  decision stands.
- 2026-09-09, 1.1b: a rewrite the policy refuses now raises `ModelRetry` prefixed with the
  same `[Command rewritten (...)]` note the success path uses, so the model is not told off
  for a command a listener wrote. Precedence between listeners is documented and pinned:
  cancel beats rewrite in either order (`rewrite()` never clears `cancelled`), last rewrite
  wins. Not changed to raise on rewrite-after-cancel: a listener cannot see the others.
- 2026-09-09, 1.1b: the cancellation test checks group liveness through
  `ps -eo pgid=,stat=` (zombie state `Z`) rather than `killpg(pgid, 0)`, which succeeds
  while an orphan sits unreaped under a non-reaping PID 1 (pydanty's sandbox). Chosen over the
  suggested `/proc` scan because the suite runs on macOS too.
- 2026-09-09, 1.1b: the `../harness-shell` worktree venv has no `pyright`; run the main
  checkout's binary with `--pythonpath .venv/bin/python` against the worktree.
- 2026-09-09, 1.1b: pydanty took 42 minutes on #845 (labelled 21:03Z, verdict 21:45Z), at the
  long end of the 20-57 range. Budget for it before polling.

- 2026-09-09, 1.1c: the floor is 2.40.0 (the release with `@agent.on_event`), not the newest
  2.42.0; the floor names the feature, the lock takes what resolves. `uv.lock` therefore lands
  on 2.42.0 and the `lowest-versions` CI job is the one that exercises 2.40.0 itself.
- 2026-09-09, 1.1c: seven `pydantic-ai-slim` floors move, not the six #850 counted: `aws-lambda`
  (#417) repeats the base floor and was added after #780. Grep `>=2.40.0` before the next bump.
- 2026-09-09, 1.1c: the `[tool.uv]` `openai>=2.45.0` override was already below slim's
  `openai>=3.0.0` at 2.38.0 (3.8.0 at 2.40.0). Its comment only ties `anthropic` to slim's
  floor, and `lowest-direct` still resolves `openai` 3.11.0 because nothing else admits lower,
  so it was left alone and noted on #851. Not filed as an issue; nobody has hit it.
- 2026-09-09, 1.1c: `uv lock --resolution lowest-direct --dry-run` is the local stand-in for the
  `lowest-versions` CI job; it reports the slim and SDK versions without touching `uv.lock`.
- 2026-09-09, 1.1c: the loop does not self-apply `dependencies:approved`. The workflow says a
  maintainer adds it, and a dependency PR approving its own dependency change is the one label
  the loop must not touch. Ticked on the draft plus the pydanty label; 1.1d holds the
  post-merge follow-through.

- 2026-09-09, loop: a "when X merges" item whose X has not merged is not the first unchecked
  item for this iteration; the loop re-checks its gate every iteration and takes the next
  unchecked box. 1.1d was skipped this way because #851 still lacks `dependencies:approved`.
- 2026-09-09, 1.2: `FileEditedEvent` subclasses `FileWrittenEvent` (core registers it under its
  own kind, `file_system.file_edited`, and `@on_event` matches with `isinstance`), so every
  existing `FileWrittenEvent` listener keeps seeing edits and a listener that wants the diff
  filters on the subclass. Chosen over extending `FileWrittenEvent` with `diff` because hosts
  emit that event themselves and a new required field would break them; a defaulted field is
  the neutered-validation smell. The events plan's open question is put to Douwe on #853.
- 2026-09-09, 1.2: the write request fires after the access and parent checks and before the
  descriptor is opened, because `_write_file` opens with `O_CREAT | O_EXCL` and a cancelled
  write must not leave an empty file behind. The edit request fires after the conflict and
  uniqueness checks. The docs say exactly that rather than one rule for both.
- 2026-09-09, 1.2: `kind` is the event envelope's discriminator on `CapabilityEvent`, so the
  search event's field is `search: Literal['find', 'grep']`, not the `kind` the events plan
  wrote. Pyright's `reportIncompatibleVariableOverride` is what caught it; core would have
  silently mis-tagged the payload.
- 2026-09-09, 1.2: `create_directory` on an existing directory emits neither the request nor
  `DirectoryCreatedEvent`: nothing changes, so there is nothing to approve. Its return string
  stays `Created directory: ...` because `test_create_existing_ok` pins it; renaming that is a
  separate change.
- 2026-09-09, 1.2: `search_files`, `find_files`, and `create_directory` took the
  `_x_tool(ctx, ...)` / public `x(...)` / `_x(ctx | None, ...)` split the other tools already
  use; the model-facing docstring moves to the `_tool` method. A direct call outside a run
  (`ctx is None`) asks nobody, consistent with the other events.
- 2026-09-09, 1.2: the open-with-retry loop in `_write_file` moved to a module-level
  `_open_for_write` to stay under ruff's complexity cap. `_toolset.py` is now about 900 lines;
  splitting its pure helpers into `_helpers.py` is deferred because `tests/filesystem` imports
  several of them from `_toolset` (against `AGENTS.md`) and that cleanup deserves its own PR.
- 2026-09-09, 1.2: `main` moved twice during the iteration (#846, #847, #849); #847 changed the
  same lines of `_edit_file` and `_write_file` (CRLF hashes). Rebased before pushing; the
  diff's before-image reads with `newline=''` like #847's canonical view but with lenient
  decoding, since it is for display and the hash check reads strictly through the descriptor.
- 2026-09-09, 1.2: per-worktree setup is `uv sync --all-extras --python 3.12` (the default
  picks 3.14 and `--no-sync` then warns on every command) and pyright from the main checkout,
  `../pydantic-ai-harness/.venv/bin/pyright --pythonpath .venv/bin/python <paths>`.
- 2026-09-09, 1.2: focused coverage on one interpreter reports two misses in `_toolset.py` and
  `test_filesystem.py` that `main` has too (a 3.13+ `ELOOP` branch and a 3.14 monkeypatch);
  CI combines the matrix and passed at 100%. Do not chase those two locally.
- 2026-09-09, loop: `git stash` on a clean worktree is a no-op and the following `git stash pop`
  pops the repository's shared `stash@{0}`, which was the 480-line CLI skeleton this plan says
  to mine for 3.4. It was re-pushed intact (8 files, 482 insertions) as `stash@{0}` with the
  message "WIP on main: d004ad6a CLI skeleton (...); mine for 3.4 provider auth", now based on
  `puppy/filesystem-change-events`. Rule: `git stash list` first, and only `git stash push -m`.
- 2026-09-09, 1.2 bridge: the diff renders before the approver is asked, so the user decides on
  what they can see; additions green, removals red, hunk headers cyan, headers and context
  plain, plus a dimmed `(diff truncated)` note. `FilesSearchedEvent` replaces the generic
  result line with a match count the way a shell end event does. `FileWrittenEvent`,
  `FileEditedEvent`, and `DirectoryCreatedEvent` get no handler: the generic `< tool` line
  already says what happened, and the diff was shown at request time.

- 2026-09-09, 1.2b: pydanty's blocking finding was that `write_file` announced a change and
  then failed its `expected_hash` check, so the request's "only changes that would go ahead"
  promise was false for writes. Fixed in the code, not the docstring: `_announce_write`
  checks the hash against the canonical text before the request and the descriptor re-check
  stays as the guard. Supersedes the 1.2 "docs say exactly that rather than one rule for
  both" decision; one rule now holds for both, and the docs name the race.
- 2026-09-09, 1.2b: `create_directory` pre-checks its collisions with `_nearest_existing`
  (walks up to the first existing ancestor) rather than `resolved.parent` alone, so
  `file.txt/deeper/still` is refused before the announce. The `mkdir` excepts are now race
  guards and carry `# pragma: no cover`, the treatment `AGENTS.md` gives unreachable-by-test
  branches.
- 2026-09-09, 1.2b: the diff is skipped for a direct `write_file` (no run, nobody to show it
  to, and it would read a file the write never needed to read) but not for `edit_file`, which
  already holds the text; guarding both would have cost a `ctx is not None and change is not
  None` dance for no saved I/O.
- 2026-09-09, 1.2b: pydanty took 45 minutes on #853 (labelled 22:38Z, verdict 23:23Z), and
  its `check_bots.sh` watermark must be after the "Review started" ack or every poll re-prints
  it. Replying in one comment that maps each finding to the fix, then re-applying
  `pydanty:review-lite`, is the ritual; the second pass is what a maintainer sees first.

- 2026-09-09, loop: 1.1d and 1.2c were both gated shut at the start of this iteration (#851
  still a draft without `dependencies:approved`, #853 under pydanty's second pass with
  `pydanty:is-working` set), so 1.3 was the first unchecked item, per the earlier loop rule.
- 2026-09-09, 1.3: no `delegation_id`. One delegation is one `delegate_task` call, so the
  `tool_call_id` core stamps on both events is the pair key, unlike shell's `command_id`, which
  exists because a background command spans several tool calls. Pinned by the parallel test.
- 2026-09-09, 1.3: `DelegationStartEvent.model` is the menu key (explicit, or the restricted
  delegate's default), not a resolved model name. The key is the user's own vocabulary and is
  safe to persist; resolving a name would mean `infer_model` on a child that may not have a
  model until run time. `_resolve_model` became `_resolve_model_key` so the event and the run
  read the same value.
- 2026-09-09, 1.3: `DelegationEndEvent.usage` is the child's own `RunUsage` only when its
  accounting is separate (`usage_limits` set, or `forward_usage=False`); the child then runs on
  a fresh `RunUsage` the toolset holds, so usage is reported on every outcome, not just `ok`.
  With shared usage it is `None`: `result.usage()` would be the whole tree's total, which
  misleads more than it informs.
- 2026-09-09, 1.3: the events fire only from a capability-owned tool. Core raises `UserError`
  on a capability event from a tool with no owning capability, and `SubAgentToolset` is a
  public export an existing test registers directly in `Agent(toolsets=[...])`. The toolset
  checks `ctx.tools[name].capability_id` (both public) rather than taking an "I am owned"
  flag whose only correct value depends on the caller. Neither shell nor filesystem guards
  this; they have no direct-registration test. Documented and pinned.
- 2026-09-09, 1.3: a refused delegation (unknown sub-agent, key off the menu, `max_calls`
  exhausted) emits nothing, and a propagating exception ends without an end event, the shell
  rule. `outcome` therefore has five values and no `error`; the parent run's own failure is
  what a host sees for the sixth case.
- 2026-09-09, 1.3: `delegate_task` split three ways (`delegate_task` validates,
  `_run_delegation` sets up and brackets with the events, `_settle` awaits and classifies into
  a frozen `_Ended`) because the emit branches pushed the one method over ruff's complexity
  cap. `_Ended.cause` set means "raise `ModelRetry(output) from cause`" after the end event,
  which keeps the chaining the old inline raises had.
- 2026-09-09, 1.3: text bound is one constant, `MAX_EVENT_TEXT_CHARS` (4096), for `task` and
  `output` alike; a task is a paragraph and an output is a summary, and neither needed its
  own knob.
- 2026-09-09, 1.3: the docs snippet uses `@agent.on_event`, the sibling style, so pydanty will
  raise the 2.38.0 floor finding a third time; it is answered with #851, as on #845 and #853.
- 2026-09-09, 1.3 bridge: "nested invocation panels" is a `>>` line as the child starts and a
  `<<` line as it ends, with the outcome word (omitted on `ok`), the duration, and the text the
  parent got. The child's own stream renders nothing in between: the child run has no bridge
  unless `shared_capabilities` passes one, and the events plan says nested streaming is an
  `event_stream_handler` concern. A failed or contained delegation raises a retry prompt with
  the same text as the end line, so `_on_tool_result` now skips summarised call IDs for retry
  prompts too; nothing before this produced both.
- 2026-09-09, 1.3 bridge: merged `puppy/subagents-events` into `feat/experimental-cli` rather
  than a local `uv` source, as 1.1 and 1.2 did.
- 2026-09-10, 1.1d: merging `main` into `feat/experimental-cli` for the floor conflicts in
  the `cli` extra; the resolution keeps the 2.40.0 floor and `termflow` in the same extra and
  re-resolves the lock (now pydantic-ai-slim 2.42.0). A plain `uv sync` drops `termflow` and
  the CLI tests then skip silently, so sync the CLI worktree with `--extra cli` before
  running `tests/cli`.
- 2026-09-10, 1.2d: the pydantic dogfooding runs were flaky on 2026-09-10 afternoon (several
  "did not produce an accepted result" or silently dropped runs across #845, #853, and #855).
  Re-applying `pydanty:review-lite` is the only remedy; nothing in the PRs needs changing for
  them. Runs that did complete landed within ~25-45 minutes.
- 2026-09-10, 1.2d: the 16:59Z green verdict was not the whole story: CI's 100% coverage gate
  had failed on #853 at 15:22Z on `tests/filesystem/test_events.py:593-594`, the macOS-only skip
  branch of the oversized-name test that never runs on the Linux matrix (220-byte name components
  fit under Linux NAME_MAX, only a 1024-byte PATH_MAX like macOS's trips it). Fixed at 27557f0e
  with `# pragma: no cover`, the repo convention for unreachable except branches. `main` (floor)
  also merged into the #853 branch at 87984c83 and both are in `feat/experimental-cli` at 8d2d5013;
  `pydanty:review-lite` re-applied for the new head.
- 2026-09-10, 1.3b: a child `HookTimeoutError` is a child crash, not a delegation timeout, so
  `_settle` routes child-sourced timeouts (a `HookTimeoutError`, or any `TimeoutError` when no
  budget is set) through the same contain decision as any other crash, rather than the
  re-raise the suggested patch used. `contain_errors` documents what happens to unexpected
  child crashes, and a hook overrunning its budget is one; the suggested re-raise would have
  aborted the parent even with containment on.
- 2026-09-10, 1.1b: the #845 re-review on 6a3791c5 landed 18:13Z: zero blocking, one required
  (`kill_process_group` resolved the group with `os.getpgid` at kill time; once a naturally
  exited leader was reaped that raised `ProcessLookupError` and the sweep was skipped while
  surviving group members stayed alive and unreachable), and the `process-group-kill-sweep`
  charter did not run, so the verdict is an "incomplete review" that asks for a re-run.
  Fixed at aa50d07c: the child is spawned with `start_new_session=True`, so the leader's pid
  is the group id for the group's whole life and the sweep now signals `os.killpg(proc.pid)`
  directly. `TestKillAfterLeaderExit` in `tests/shell/test_events.py` forks a child that
  outlives the shell, waits for the leader to be reaped, and asserts the sweep still reaches
  the child; it fails against the old code. Replied on the PR and re-applied the label.
- 2026-09-10, 1.1b: the aa50d07c push did retrigger Macroscope (it reviewed within a
  minute, where a full pass usually takes ~30). Three findings, all addressed at 89dfdfdf:
  (1) the interactive refusal in `_check_command` echoes the complete command, so a
  credential in a policy-refused rewrite reaches the transcript, contradicting the docs'
  "not shown to the model"; documented the exception in `docs/shell.md` and the README,
  kept the echo for consistency with the other refusal messages. (2) `cancelled` was a
  plain field, so a later listener could lift an earlier veto; it is now a read-only
  property over a private field with `cancel()` as the only setter, docs say the cancel
  is final, new test `test_a_later_listener_cannot_lift_an_earlier_cancel`. (3) the new
  kill test wrapped in try/finally so a failed check sweeps the group anyway. The same
  mutable-veto shape exists in `FileChangeRequestEvent` on #853; noted for that PR's next
  round rather than pushed, to avoid invalidating its in-flight pydantic run again.
- 2026-09-10, 1.1b/1.3b: Macroscope is a check-run bot with no comment trigger; it
  re-evaluates on push and skips heads whose diff is unchanged since its last completed
  review (it says so in the skip title). The @mention comments on #845/#855 are inert. The
  aa50d07c push on #845 is the retrigger for that PR; #855's last real verdict (12:23Z)
  predates the f5dc6fd6 fix, and a fresh push is the only way to move its "diff unchanged"
  skip. If a future #855 fix lands, the re-evaluation happens automatically.
- 2026-09-10, 1.2c: #853 CI is fully green on 87984c83 (26 passed, 8 standard skips, one
  cancelled no-op "evaluate dependency approval"). The pydantic re-review retriggered for
  the new head is in flight.
- 2026-09-10, 1.3b: the #855 pydantic re-trigger from the previous session was stuck,
  and the root cause was a wrong label. The label that dispatches a pydantic/pydanty
  review run is `pydanty:review-lite` (description: "Trigger: freeform AI-decomposed
  review of this PR. No CI precondition. Consumed on dispatch."), NOT
  `pydantic-ai:review-lite`. The previous session (and this one, for an hour) had been
  cycling `pydantic-ai:review-lite` on #845 and #855, which the bot ignores; the stuck
  `pydanty:is-working` on #855 (17:27:58Z, a published-withheld run) and the 24h-old
  `is-working` on PR #589 are bot bookkeeping artifacts, not the clog. Cleared the stale
  #855 `is-working`, applied `pydanty:review-lite` to #845 and #855 at 20:49Z: both
  picked up within 25 seconds. Earlier sessions' notes that "the review label is
  pydantic-ai:review-lite" were wrong. Verdicts expected ~21:15-21:50Z.
- 2026-09-10, 1.1e: root-caused the `TestKillAfterLeaderExit` CI failures (all 3.11/3.12
  cells, none on 3.10/3.13/3.14) with two diagnostic pushes. The GHA host runs steps under
  a `hosted-compute-agent` systemd service that is the step's session and group leader; it
  tears a `setsid` session down (the whole group, in under half a second) the moment its
  leader exits, while orphaned members of the supervisor's own session survive (a no-setsid
  control probe stayed alive). The test lost a race between its post-reap assertion and the
  supervisor's kill, and the race odds are interpreter-speed dependent (asyncio reaping
  latency), which is why 3.11/3.12 failed near-100% and the others passed. The fix
  (42e985f9) keeps the test unchanged and adds a one-second pre-probe that skips it where
  the host destroys the very scenario under test; verified both paths locally and in
  Docker, including a simulated hostile supervisor. Reproduction note: a plain
  `python:3.12-slim` container never shows it (no systemd supervisor).
- 2026-09-10, 1.2e: the third pass landed 18:53Z (33 minutes, run df_run_090c5c00b8474ed6ac20,
  "Reviewed at ... head 4f8e1c438ca8"): zero blocking, one required, incomplete (2/5
  charters did not run). The required is a stale-snapshot artifact: 4f8e1c43 is an
  ancestor three commits below the real head 87984c83 -- it predates f5b9134d, the commit
  that added the very `header[:MAX_EVENT_DIFF_CHARS]` cut the pass "suggests", which is
  pinned by `test_headers_of_an_oversized_name_are_cut_at_the_bound`. The run was
  retriggered for 87984c83 but analyzed an old state; the verdict (and the incomplete
  flag) are untrustworthy. No reply needed on the PR beyond re-applying the label for a
  fresh run; do the `FileChangeRequestEvent` monotonic-veto fix in the same round.

## Open questions for Mike

1. Console script code name. Placeholder is `harness`.
