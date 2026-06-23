# SPEC: `pydantic_ai_harness.experimental.context`

A capability that lets a coding agent discover and load a repo's accumulated
coding-assistant context engineering (CE): instruction files
(`CLAUDE.md`/`AGENTS.md`) and an inventory of where its CE assets
(skills/subagents/hooks) live. Generic -- no dependency on Loopy. Loopy adopts
it via its own wiring.

---

## Questions for David

Self-contained; each has my recommendation. Only the nested-traversal one
strictly blocks implementation.

### Q1. Nested-on-traversal mechanism (REQUIRES sign-off)

Strategy 3 loads a directory's `CLAUDE.md`/`AGENTS.md` *when the agent lists or
reads that directory*. This couples to the list/read tools. Two axes:

**Axis A -- where the load happens:**
- **(A1, recommended)** An `after_tool_execute` hook keyed on a *configurable
  set of tool names* (default `{'list_directory', 'read_file'}` for harness fs;
  Loopy overrides with `{'list_dir', 'read_file'}`). The hook reads the path
  from a configurable arg key (default `'path'`), finds that dir's CE md, and
  appends it to the tool result, once per dir per run. Self-contained in this
  capability; works against any tool by name; the injected text lands in the
  message tail (cache-safe -- see Cache convention). Cost: must know the tool's
  path arg key.
- **(A2)** Fold it into the filesystem capability so its own tools self-augment.
  Rejected: ties this capability to harness fs, does not work for Loopy's tools,
  violates "stay generic".

**Axis B -- what gets injected:**
- **(B1, recommended default)** Pointer note: append
  `"This directory has a CLAUDE.md (path). Read it if relevant."` -- low token
  cost, model opts in via its existing read tool, no duplication when the model
  then reads it.
- **(B2, opt-in)** Full contents: append the md body inline. Heavier; risks
  duplicating content the model reads anyway. Offer as a config flag
  (`nested_inject='pointer' | 'contents'`).

**My recommendation: A1 + default B1, with a flag to switch to B2.** Ship
nested-traversal **off by default** (see Q2) so adopters opt in explicitly.

Need your call on: A1 vs anything else, and pointer-vs-contents default.

### Q2. Which strategies are default-on?

- Walk-up instruction autoload (Strategy 1): **ON** -- core feature, cache-safe
  (system-instruction injection, read once at run start).
- Asset inventory tool (Strategy 2): **ON** -- only exposes a tool; cheap, no
  passive cost.
- Nested-on-traversal (Strategy 3): **OFF** -- couples to tool names, volatile,
  needs explicit wiring. Opt-in via constructor.

Confirm this posture (esp. nested off-by-default).

### Q3. Public capability class name

Module is `context` (fixed by the task), but `Context` as a class name collides
with `RunContext` / `ModelRequestContext` in pyai vocab. **Recommend
`RepoContext`.** Alternatives: `ProjectContext`, `ContextDiscovery`,
`AgentContextFiles`. Your call -- you care about naming precision and the
overload is real.

---

## Scope

In: (1) home + walk-up `CLAUDE.md`/`AGENTS.md` autoload with defined
precedence/dedup; (2) an inventory tool reporting *where* CE assets live;
(3) nested-on-traversal load (design settled here, off by default). Out:
parsing skill/agent frontmatter or hook bodies (siblings E2/E3 do that); writing
or translating CE; any Loopy-specific wiring.

## Module layout

```
pydantic_ai_harness/experimental/context/
  __init__.py        # warn_experimental('context') + public re-exports
  _capability.py     # RepoContext(AbstractCapability) -- the three strategies
  _loader.py         # walk-up discovery, dedup, precedence, render
  _inventory.py      # asset scan + result models
  _toolset.py        # FunctionToolset exposing the inventory tool
  README.md          # experimental warning box + problem/solution + cache note
tests/experimental/context/
  test_context.py
  conftest.py        # tmp-repo fixtures (optional; can fold into test file)
```

## Public API

`RepoContext[AgentDepsT](AbstractCapability[AgentDepsT])` -- frozen dataclass,
mirroring `Planning`/`SubAgents` shape.

Constructor fields:

| field | type | default | purpose |
|---|---|---|---|
| `workspace_dir` | `Path` | required | deepest dir the agent works in |
| `home_dir` | `Path \| None` | `None` | shallowest dir to stop walk-up at (inclusive). `None` = stop at filesystem-root-bounded `workspace_dir` only (no walk-up) |
| `filenames` | `Sequence[str]` | `('CLAUDE.md', 'AGENTS.md')` | instruction filenames, in within-dir precedence order |
| `autoload_instructions` | `bool` | `True` | Strategy 1 on/off |
| `expose_inventory_tool` | `bool` | `True` | Strategy 2 on/off |
| `inventory_tool_name` | `str` | `'inventory_agent_context'` | tool name |
| `nested_traversal` | `bool` | `False` | Strategy 3 on/off |
| `nested_inject` | `Literal['pointer', 'contents']` | `'pointer'` | Strategy 3 payload (per Q1) |
| `traversal_tool_names` | `frozenset[str]` | `frozenset({'list_directory', 'read_file'})` | tool names that trigger Strategy 3 |
| `traversal_path_arg` | `str` | `'path'` | arg key holding the dir/file path |
| `asset_roots` | `Sequence[str]` | `('.claude', '.agents', '.codex', '.grok')` | dirs scanned by inventory |

Re-exported from `__init__.py`: `RepoContext`, `RepoContextToolset`,
`AgentContextInventory` (+ nested result models), `ContextFile`.

### Strategy 1 -- walk-up instruction autoload

`_loader.py`:
- `discover_instruction_files(workspace_dir, home_dir, filenames) -> list[ContextFile]`:
  walk from `home_dir` down to `workspace_dir` (ancestors first). For each dir,
  collect existing `filenames` in order. `ContextFile = (dir, path, content)`.
- **Precedence (recommended):** ancestor-first, workspace-last. Broadest context
  first, most-specific (closest to the model's recency window) last. Documented
  as the merge order.
- **Dedup:** by resolved real path *and* by content hash. Handles the common
  case where `AGENTS.md` is a symlink to `CLAUDE.md` (load once) and where two
  ancestors share an identical file. First occurrence in precedence order wins.
- Render: each file wrapped in a labeled block, e.g.
  `<context-file path="...">\n{content}\n</context-file>`, joined by precedence.

Injection site: `get_instructions()` -- returns the rendered blocks as static,
**cache-stable** system instructions. Files are read once (at `for_run` /
construction), not per request, so the cached prefix stays byte-identical.
Documented limitation: changes to these files mid-run are not reloaded (they are
treated as static run context).

`for_run` returns a per-run instance that has resolved + cached its
`ContextFile` list and per-run nested-seen set.

### Strategy 2 -- asset inventory tool

`_inventory.py` + `_toolset.py`. `RepoContextToolset` (a `FunctionToolset`)
exposes one tool `inventory_agent_context()` returning a structured
`AgentContextInventory` model (**structured result, not prose** -- the
orchestrator consumes it programmatically to decide what to read/translate).

`AgentContextInventory` shape:
```
roots: list[AssetRoot]
  AssetRoot:
    root: str                 # e.g. ".claude"
    exists: bool
    skills: list[str]         # paths to SKILL.md files found under skills/
    agents: list[str]         # paths to *.md under agents/
    settings: str | None      # path to settings.json if present (hooks live here)
    notes: str | None         # e.g. ".codex uses TOML; .agents mirrors .claude"
```
Locates only -- does not open/parse SKILL.md, agent md, or settings.json.
Scans `asset_roots` relative to `workspace_dir`. Static usage hint added via
`get_instructions` ("call `inventory_agent_context` to map existing CE assets").

### Strategy 3 -- nested-on-traversal (off by default; see Q1)

When `nested_traversal=True`, `after_tool_execute` fires on tools whose name is
in `traversal_tool_names`. It:
1. extracts the path from `args[traversal_path_arg]`; resolves to a directory
   (the dir itself for a list, the file's parent for a read);
2. skips if that dir was already surfaced this run (run-scoped `set`);
3. finds the dir's CE md via the loader;
4. appends a pointer (or contents, per `nested_inject`) to the tool result and
   returns it.

Result handling: if `result` is a `str`, append; otherwise wrap into a string
representation + note (kept minimal -- harness/Loopy list/read tools return
strings). Cache-safe: the augmentation rides in the message tail, never in
system instructions, so it cannot invalidate the cached prefix.

### Cache convention

- Strategy 1 (static, read once) -> system instructions via `get_instructions`.
  Cache-stable by construction; documented.
- Strategy 3 (volatile, depends on which dir was just touched) -> message-tail
  via `after_tool_execute` result append. Never baked into system instructions.
- The README and the `RepoContext` docstring state the tradeoff explicitly and
  note that nested-traversal content is injected post-cache-breakpoint.

## Generic / Loopy adoption notes

Genericity hinges on configurable tool names + path arg key (Strategy 3) and a
caller-supplied `workspace_dir`/`home_dir`. Loopy passes its own
`traversal_tool_names={'list_dir', 'read_file'}`. Nothing imports Loopy. No core
changes required -- all three strategies use existing hooks
(`get_instructions`, `get_toolset`, `after_tool_execute`).

## Test plan (100% branch coverage, `TestModel`, `ALLOW_MODEL_REQUESTS=False`)

Loader (`_loader.py`), via tmp dirs:
- walk-up collects ancestor-first, workspace-last in correct order
- `home_dir=None` -> only workspace dir considered
- both `CLAUDE.md` + `AGENTS.md` present -> both loaded, within-dir order
- symlink `AGENTS.md -> CLAUDE.md` -> deduped to one
- identical content in two ancestors -> deduped by hash
- missing files / empty dirs -> skipped cleanly

Capability through `Agent(capabilities=[RepoContext(...)])`:
- `get_instructions` includes rendered blocks when `autoload_instructions=True`;
  empty/`None` when off or nothing found
- inventory tool registered iff `expose_inventory_tool=True`; result shape for a
  fixture repo with `.claude/skills`, `.agents/agents`, `.codex`, settings.json
- inventory `notes`/`exists`/`settings=None` branches

Nested-traversal (direct `after_tool_execute` / `RunContext` where isolating
through `Agent` is hard):
- appends pointer on first traversal of a dir with CE md
- second traversal of same dir -> no re-append (seen set)
- tool name not in `traversal_tool_names` -> untouched
- dir without CE md -> untouched
- `nested_inject='contents'` -> body appended
- non-str result branch

Spec serialization: `get_serialization_name` returns a stable name (holds only
plain config + paths) -- mirror `Planning`.

## Out of scope (named skips)

- Parsing SKILL.md / agent md / settings.json contents (E2/E3).
- Translating or rewriting CE assets.
- Watching files for mid-run changes (Strategy 1 is read-once by design).
- Any Loopy wiring.

## Docs

README with the standard experimental warning box (copy `subagents/README.md`
structure), problem/solution, the cache tradeoff note, and a config table.
Add the capability to `agent_docs/` index if AICA preflight requires it
(checked during implementation).
