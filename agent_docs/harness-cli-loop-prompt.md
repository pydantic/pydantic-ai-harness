You are one iteration of a loop building the harness-native CLI. The plan at
`agent_docs/harness-cli-plan.md` is the only state you trust. Do exactly one unchecked item, then stop.

## Setup (every iteration)

1. Work in the worktree `/Users/mpfaffenberger/code/harness-cli` on branch `feat/experimental-cli`.
   Create it if missing: `git worktree add ../harness-cli feat/experimental-cli` from the main
   checkout. Never work in `/Users/mpfaffenberger/code/pydantic-ai-harness` directly; it is shared.
2. Read, in order: `AGENTS.md`, `agent_docs/index.md`, `agent_docs/harness-cli-plan.md`,
   `agent_docs/capability-authoring.md`. Activate the `pydantic-ai-tenets` skill before writing code,
   and `pr-bot-watch` after any push.
3. If `agent_docs/harness-cli-plan.md` is untracked, commit it first (with the `agent_docs/index.md`
   link parked in the stash named "cli-plan: agent_docs/index.md"; re-create the one-line link if
   the stash is gone) and push. That commit is the whole first iteration.
4. Pick the first `- [ ]` item in "Execution order", top to bottom. If a branch or worktree for it
   already exists, resume it instead of starting over.

## Doing the item

Capability or event PR items (anything that names a `puppy/<slug>` branch):

- `git worktree add ../harness-<slug> -b puppy/<slug> main` (or reuse it). Implement per the
  conventions in the plan's "Rules the loop obeys" and the payload conventions it links: one
  `_events.py` per capability, explicit `name=`, `dispatch='immediate'` only for decisions with the
  core `cancel(reason)` shape, bounded payloads with `truncated`, tests through
  `Agent(..., capabilities=[..., Listener()])` with explicit `id=`, README plus `docs/<name>.md`
  in parity, an "Events" section in both, telemetry decided and documented.
- Deprecate existing callbacks with `HarnessDeprecationWarning`; never remove them in the same PR.
- Verify locally, only what you touched:
  `uv run --no-sync ruff format --check .`, `uv run --no-sync ruff check .`,
  `PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run --no-sync pyright <files>`,
  `uv run --no-sync pytest -p no:cacheprovider tests/<capability>`.
  Do not run repository-wide pyright, pytest, or coverage.
- Commit (message says why), push, open a **draft** PR against `main`. Title: imperative sentence,
  identifiers in backticks, no `feat:` prefix. Body links the plan item and any issue it closes
  (`AskUser` closes #42). Then run the `pr-bot-watch` loop until CI is green and pydanty's findings
  are addressed.
- Then merge `puppy/<slug>` into `feat/experimental-cli` in `../harness-cli` and wire the bridge to
  consume the new events (render or answer them), with a transcript test in `tests/cli/`.

Host items (Phase 0 and Phase 3, and the bridge half of any item):

- Work directly in `../harness-cli` on `feat/experimental-cli`. Code lives in
  `pydantic_ai_harness/cli/`, tests in `tests/cli/`, docs in `docs/cli.md` and the package README.
- The host never reaches into a capability's internals. It subscribes to events with `@on_event` on
  `CliBridge`, answers immediate decision events through the pluggable `Approver`, and drives the
  run through `AgentRun.enqueue` and `AgentRun.cancel`. If you find yourself needing a callback or a
  message bus, stop: that is a missing event, and it becomes a PR item you add to the plan.
- Commit and push `feat/experimental-cli` after the item's acceptance check passes.

## Finishing the item

1. Tick the box in `agent_docs/harness-cli-plan.md`. Add the PR number next to it.
2. Append every non-obvious decision you made to the plan's "Decisions" section (date, item,
   decision, why). Supersede, never edit, earlier entries.
3. If you discovered a missing item, add it as a new unchecked box in the right phase.
4. Commit the plan change on `feat/experimental-cli` and push.
5. Stop. Report the item, the PR link if any, and the next unchecked item.

## Hard rules

- Never use em-dashes. Never use `Any`. No `cast()`. Keyword-only arguments beyond the first one or
  two. Dataclasses for value objects, `kw_only=True`. Docstrings use single backticks.
- No dependency changes without an issue link; `pyproject.toml` or `uv.lock` PRs need the
  `dependencies:approved` label (the `termflow` extra in item 0.1 is pre-approved, link the plan).
- Never commit to `main`, never force-push, never rewrite `feat/experimental-cli` history.
- When blocked on a design question the plan does not answer, write it under "Open questions for
  Mike" in the plan, tick nothing, commit the plan, and stop. Do not guess.
- Do not delegate further than one level. If you spawn a subagent for an independent PR item, give
  it this prompt with the item number fixed and tell it not to delegate.
- When every box is ticked, say so and stop.
