# SPEC: disk-loaded sub-agents for `experimental.subagents`

Extend the existing `SubAgents` capability so a repo's markdown agent definitions
(`.agents/agents/*.md` or `.claude/agents/*.md`) become delegatable sub-agents,
coexisting with explicitly-passed `SubAgent` entries.

Scope: Phase 1 is this spec only. No implementation until David's answers land.

---

## Questions for David

Each question is self-contained and carries my recommendation. The first two
(tool + model mapping) are the load-bearing ones -- they decide how a
coding-assistant `.md` file maps onto Loopy/pyai's surface, which differs from
Claude Code's.

### Q1 -- Tool-name mapping (`tools` / `allowed-tools` frontmatter)

A CC agent definition lists tools by CC names: `Read`, `Edit`, `Write`, `Bash`,
`Grep`, `Glob`, etc. These names do **not** exist in pyai or harness. The closest
harness equivalents are differently named:

| CC tool | nearest harness tool | capability |
|---|---|---|
| `Read` | `read_file` | `filesystem` |
| `Edit` | `edit_file` | `filesystem` |
| `Write` | `write_file` | `filesystem` |
| `Grep` / `Glob` | `search_files` / `find_files` | `filesystem` |
| `Bash` | `run_command` | `shell` |

There is no central pyai registry that resolves a string like `"Read"` to a tool,
and Loopy's actual tool surface is its own. So harness cannot ship a correct
default map -- it has to come from the caller (Loopy).

**Recommendation (default):** add a caller-supplied
`tool_resolver: Callable[[Sequence[str]], Sequence[AbstractToolset[AgentDepsT]]] | None = None`.
The loader passes each definition's parsed tool names to the resolver; the
returned toolsets are attached to that disk agent (via `Agent(toolsets=...)`).

- When **no** `tool_resolver` is given: the `tools` frontmatter is parsed and
  recorded but **not applied** -- the disk agent is built tool-less. We emit one
  `HarnessExperimentalWarning` per definition that listed tools but got no
  resolver, naming the dropped tools. (Never silently drop -- CLAUDE.md.)
- **Unknown tool names** (resolver returns nothing for a name): the resolver
  owns that policy. Harness's contract: whatever names the resolver ignores, the
  loader logs once via warning. Default loader behavior = warn-and-skip, not
  raise, so one stray tool name in a repo's `.md` doesn't hard-fail agent
  startup.

**Decide:** (a) accept the `tool_resolver` callback shape above? (b) for unknown
names: warn-and-skip (my rec) vs raise? (c) does Loopy instead want to pass a
flat `tool_map: Mapping[str, toolset]` it owns, and let harness do the lookup +
warn? A map is simpler for Loopy; a callback is more flexible (globs, wildcards
like CC's `Bash(git:*)`). I lean **callback**.

### Q2 -- Model + effort (REDESIGNED per David, signed off)

The `model_aliases`/`default_model` design is dropped. Final design:

- The CC `model:` frontmatter field is **ignored** (along with `color`).
- Disk-loaded sub-agents **inherit the parent run's model** by default. A
  model-less disk `Agent` is built; at delegation the toolset passes
  `model=ctx.model` so it runs on the parent's model.
- The caller may **optionally override**, per agent, both the model and the
  thinking/effort level via `agent_overrides: Mapping[str, AgentOverride]`,
  keyed by agent name. `AgentOverride(model=..., effort=...)`.
- A module-level **minimum-effort floor** (`MINIMUM_EFFORT_FLOOR: ThinkingEffort
  = 'low'`) plus a `clamp_effort(level, floor=...)` helper force every agent the
  capability builds to at least the floor. Both are exported so the orchestrator
  can apply the same floor to its own agents (orchestrator-side application is
  Loopy work).

Effort is wired through pyai's real mechanism: `ModelSettings.thinking`
(`bool | Literal['minimal','low','medium','high','xhigh']`, in
`pydantic_ai.settings`). Each disk agent is built with
`model_settings=ModelSettings(thinking=clamp_effort(override_effort))`. `clamp`:
`None`/`False` -> floor; `True` (provider default) -> unchanged; a concrete level
below the floor -> floor; at/above -> unchanged.

### Q3 -- Precedence when the same agent name appears in multiple sources

Sources, highest precedence first under my recommendation:

1. Explicitly-passed `agents=[SubAgent(...)]` (code beats convention).
2. Project folder (`./.agents/agents/` or `./.claude/agents/`).
3. Home folder (`~/.agents/agents/` or `~/.claude/agents/`).

**Recommendation:** higher precedence wins; the shadowed lower-precedence entry
is **skipped** (not an error). Rationale: a project agent overriding a home
agent, or an explicit code agent overriding a repo file, is the intended
override path -- erroring would make the feature unusable. **But** a duplicate
within the *same* explicitly-passed list stays a hard error (current behavior,
unchanged) -- that's a programmer mistake, not an override. I also log one
warning per shadowed disk definition so the override is visible.

**Decide:** confirm "explicit > project > home, shadow silently-with-warning",
and that same-list explicit duplicates keep raising.

---

## Resolved here (no David input needed, but flagging the calls I made)

### R1 -- Frontmatter parser: hand-rolled, not PyYAML

`pyyaml` is **not** a declared runtime dependency of harness (only
`httpx` + `pydantic-ai-slim` are; PyYAML is present in this venv only
transitively via dev deps). I cannot `uv add` it (CLAUDE.md / CI auto-closes dep
changes). So the loader ships a small, dependency-free parser limited to the
known frontmatter shape:

- A leading `---\n ... \n---` block. Keys: `name`, `description`, `model`,
  `color`, and `tools` **or** `allowed-tools`.
- Scalar values are taken verbatim (string). `tools`/`allowed-tools` accept
  either a comma-separated string (`Read, Edit, Bash`) or a YAML-style block list
  (`- Read` lines) -- both common in the wild.
- Anything after the closing `---` is the markdown body = the agent's
  `instructions`.
- A file with no frontmatter block -> body is the whole file; `name` falls back
  to the filename stem; no model/tools.

This is intentionally narrow. If a repo needs full YAML frontmatter we revisit
with an opt-in extra. Documented as a limitation in the README.

### R2 -- Folder configuration + `.agents`->`.claude` fallback

New field `agent_folders: str | Sequence[Path] | None = None`:

- `None` (default): convention. For each root in `[Path.cwd(), Path.home()]`,
  pick `<root>/.agents/agents/` if `<root>/.agents/` exists, else
  `<root>/.claude/agents/`. Missing folders are skipped silently (a repo with no
  agent dirs just contributes nothing).
- `str` (a leaf folder *name*, e.g. `'reviewers'`): same convention but the leaf
  `agents` is replaced by the given name -> `<root>/.agents/<name>/` (fallback
  `<root>/.claude/<name>/`), both roots.
- `Sequence[Path]` (absolute paths): use exactly those folders, in order, as the
  disk source. No `.agents`/`.claude` convention, no fallback. Earlier path =
  higher precedence (after explicit `agents`).

Within a folder, every `*.md` is a candidate. Non-`.md` files ignored.

### R3 -- `name` / `description` / `color` mapping

- `name`: delegate name (`SubAgent.name`). Frontmatter `name` wins; else the
  filename stem. Feeds the existing dedupe/precedence logic.
- `description`: `SubAgent.description` -> drives the system-prompt listing.
- `color`: **ignored** -- purely cosmetic in CC, no pyai equivalent. Documented
  as ignored (not dropped silently -- README states it).
- body: the agent's `instructions` (`Agent(instructions=body)`).

### R4 -- How disk agents merge into the existing roster

Loader runs inside `__post_init__`, before `_by_name` is built:

1. Build `SubAgent` entries from explicit `self.agents` (unchanged).
2. Discover + parse disk definitions, build an `Agent` per definition, wrap each
   as a `SubAgent`.
3. Merge by precedence (Q3): explicit > project > home / earlier-path. Shadowed
   entries skipped with a warning.
4. Build `_by_name` from the merged list. Same-name collisions that are *not*
   resolved by precedence (i.e. within explicit list) raise as today.

The toolset, instructions, run-controls, and `get_serialization_name() -> None`
are all unchanged -- disk agents are just additional `SubAgent`s.

### R5 -- Per-delegate run controls for disk agents

Disk definitions carry no `usage_limits` / `timeout_seconds` / `max_calls` /
`on_failure`. They inherit the `SubAgents`-level defaults (all `None` today).
Out of scope to encode budgets in frontmatter now; note as possible follow-up.

### R6 -- Deps typing (implementation risk, my call, flagged)

Disk-built agents are constructed deps-free (`Agent(model, instructions=...)` ->
`Agent[None, str]`), since the loader cannot know the parent's `AgentDepsT` at
construction time. At run time the parent still forwards `ctx.deps`; a deps-free
agent ignores them. For pyright-strict generic correctness the merged disk
entries are produced as `SubAgent[AgentDepsT]` via the loader being typed to the
capability's `AgentDepsT` while building `None`-deps agents -- I'll resolve the
exact typing seam in implementation without `Any`/`cast` (likely a typed helper
that returns `SubAgent[AgentDepsT]` and a deps-free `Agent` whose forwarded deps
are unused). Calling out now in case David wants disk agents to instead receive
the parent deps_type explicitly -- default is deps-free.

---

## Public surface added (proposed)

```python
SubAgents(
    agents=(),                 # unchanged: explicit SubAgent entries (highest precedence)
    agent_folders='agents',    # folder-name str (convention, default) | Sequence[Path] | None (disable)  (R2)
    agent_overrides={},        # Mapping[str, AgentOverride(model=..., effort=...)]  (Q2)
    tool_resolver=None,        # Callable[[str], Sequence[AgentToolset[object]] | None]  (Q1)
    # ...all existing fields unchanged (forward_usage, inherit_tools,
    #    shared_capabilities, event_stream_handler, tool_name)
)
```

New exported names: `AgentOverride`, `ToolResolver`, `clamp_effort`,
`MINIMUM_EFFORT_FLOOR`. Loader internals live in `_disk.py` + `_effort.py`.

Disk loading is **on by default** (`agent_folders='agents'`): constructing the
capability auto-loads the conventional `.agents/agents/` + `.claude/agents/`
layout for cwd and home. Pass `agent_folders=None` to disable, or a `Sequence[Path]`
for explicit folders. (David's call: auto-load over opt-in.)

Tools (Q1, signed off): default is to inherit the parent's tools via the
existing `inherit_tools` mechanism (no resolver needed). A `tool_resolver` is the
optional per-name override path (it can honor `Bash(git:*)`-style entries);
unknown names (`None` returned) warn-and-skip. The `tools:`/`allowed-tools`
frontmatter is parsed and applied only when a resolver is supplied.

## Testing plan (Phase 2)

Extend `tests/experimental/subagents/test_subagents.py` (TestModel only):

- frontmatter parse: full block, missing block, list-form vs comma-form tools,
  `allowed-tools` alias, `color` ignored, body-as-instructions.
- folder resolution: `.agents` present -> `agents/`; `.agents` absent ->
  `.claude/agents/`; leaf-name override; explicit absolute paths; missing dirs
  skipped. (Use `tmp_path` + monkeypatched cwd/home.)
- precedence: explicit shadows project shadows home; warning emitted; same-list
  explicit duplicate still raises.
- model resolution: alias hit, direct id, `default_model` fallback, unresolved
  -> `ValueError`.
- tool resolution: resolver applied; no resolver + tools listed -> warning,
  agent tool-less; unknown name -> warn-and-skip.
- end-to-end: a disk agent is delegatable via `delegate_task` through
  `Agent(capabilities=[SubAgents(agent_folders=...)])`.

## Docs (Phase 2)

Update `subagents/README.md`: new "Loading sub-agents from disk" section
covering folder convention + `.claude` fallback, frontmatter keys + the
model/tool mapping contract, precedence, and the documented limitations
(narrow YAML, `color` ignored, deps-free disk agents). Note in CLAUDE-aligned
style: no em-dashes, single backticks, state-the-mechanism.
