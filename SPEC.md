# SPEC: `pydantic_ai_harness.experimental.authoring`

Runtime capability authoring -- the meta-factory that lets an orchestrator write a
real pydantic-ai capability/hook `.py` at runtime, validate it, and make it usable.
Pairs with the E4 docs capability (docs teach the agent *how* to write a capability;
this capability *loads and runs* what it wrote).

Decision already locked with David: the mechanism is **native** capability authoring
(write a real `.py`, import it, validate it, register it -- full native hook surface,
all of `pydantic_ai/capabilities/abstract.py`). Same trust boundary Loopy already has
(bash + edits). The dormant `pa` Monty-registration system is the safer-sandbox
alternative and is recorded below, but we are going native.

---

## Questions for David

Each is self-contained with my recommendation. Q1 is the one that shapes the whole design.

### Q1 (KEY) -- Activation boundary: what does "usable" mean for an authored capability?

**Finding first (this constrains the answer):** A capability *cannot* be added to a
live, already-executing `agent.run()`. pyai resolves the effective capability set once
at the top of `Agent.iter`, before any graph node runs; `Agent.root_capability` is a
read-only property with no setter (`pydantic_ai/agent/__init__.py` `Agent.__init__`,
`agent/abstract.py` `AbstractAgent.root_capability`). So "inject mid-run" is off the
table by pyai's design -- not a harness limitation.

But pyai *does* support **additive per-run injection**: every run method
(`run`/`run_sync`/`run_stream`/`run_stream_events`/`iter`) takes
`capabilities=[...]`, merged with the agent's configured set for that one run
(`agent/__init__.py` `Agent.iter`: `CombinedCapability([base_capability, *extra_capabilities])`).
So an authored capability is live on the **very next `agent.run()` call** without
rebuilding the `Agent`. If the orchestrator drives a loop of `agent.run(message_history=...)`
iterations (Loopy's `run_workflow` does exactly this -- builds the agent once, loops
`agent.run()` reusing history), the authored capability is live on the **next loop
iteration**.

**Why this is one notch coarser than personas (the asymmetry that matters):** Loopy's
runtime personas are "immediately usable in the same run" only because a persona adds
*no new tools and no new hooks* -- it is an entry in an in-memory `registry` dict that
the single generic `delegate(persona, task)` tool reads at call time and spins into a
fresh sub-`Agent` (`pa/orchestrator.py` `define_persona`/`delegate`). A full capability
has arbitrary lifecycle hooks (`before_model_request`, `wrap_tool_execute`, ...) and may
add new *native tools*; those only exist once the capability chain + toolset are
assembled, which happens at run start. They cannot be dispatched through one generic
tool. So full-hook authored capabilities are **inherently next-`run()`**, never
same-run.

**Recommendation -- Option A:** author -> validate -> persist *now*; make live by
injecting through a shared store into the next `agent.run(capabilities=...)` loop
iteration. No process restart. After authoring, the tool returns a message telling the
model the capability is registered and active on the next iteration (mirrors Loopy's
`define_persona` "usable immediately" message, adjusted for the next-run boundary). This
requires the orchestrator owner to thread the authoring store into each `.run()` call --
a small, documented integration contract (one line, like Loopy threads `personas.json`).

- **Option B (heavier):** persist `.py` only; authored caps picked up when the
  orchestrator *next constructs* its `Agent` (next process / next `make_orchestrator`).
  Simpler harness (no live thread-through) but the agent cannot use what it just authored
  until a rebuild -- worse feedback loop for a self-extending agent.

**I recommend A.** Confirm A vs B. (A subsumes B: A also persists to disk, so a fresh
process still loads them; A just *also* makes them live within the running loop.)

### Q2 -- Validation depth: construct-only, or exercise a hook?

**Recommendation:** validate = (1) import the module, (2) find exactly one
`AbstractCapability` subclass (narrowed via `isinstance(obj, type) and issubclass(obj, AbstractCapability)`),
(3) construct it with no args (or a declared kwargs dict), (4) call the **side-effect-free
static getters** and type-check their returns: `get_instructions()`,
`get_toolset()`, `get_native_tools()`, `get_model_settings()`,
`get_serialization_name()`. Do **not** invoke the async lifecycle hooks
(`before_model_request`, `wrap_tool_execute`, `before_run`, ...): they require a real
`RunContext` + synthetic inputs, are fragile to fabricate, and may have side effects.
The static getters are the cheap, safe surface that catches the common authoring errors
(bad toolset construction, wrong return types) without a live run.

Confirm this middle ground, or do you want either (a) construct-only (no getters), or
(b) a deeper smoke that actually exercises one async hook against a `TestModel`-backed
synthetic `RunContext`?

### Q3 -- Should authoring overwrite / version / be revocable?

**Recommendation (resolve unless you object):** `author_capability(name, code)` upserts
by `name` (re-authoring replaces the prior `.py`, like `save_runtime_persona`'s
read-modify-write upsert). Add a `list_authored_capabilities()` tool and a
`disable_authored_capability(name)` tool (flips a manifest status to `disabled`, mirrors
`pa/registration_tools.py` `disable_registration`). No git-style versioning. Flag only if
you want hard-immutability or version history instead.

---

## Finding: is mid-run capability registration feasible?

**No -- not into a live run; yes -- on the next `agent.run()`.** Detail:

| Tier | Supported? | Mechanism |
|---|---|---|
| Into a live, executing run | **No** | `root_capability` snapshotted at `Agent.iter` start; no setter |
| Next `agent.run()` call (same `Agent` object) | **Yes** | per-run `capabilities=[...]` additive merge (`Agent.iter`) |
| Next process / baseline rebuild | **Yes** | new `Agent(...)` or `Agent.from_spec(..., custom_capability_types=[...])` |

Corroborating prior research: harness `SubAgents` declares
`get_serialization_name() -> None` because it "holds live `Agent` instances"
(`experimental/subagents/_capability.py`) -- live objects aren't spec-serializable, so
authored capabilities (which hold live code) likewise persist as `.py`, not as a spec.
The realistic activation path is therefore per-run injection (Q1 Option A), which sits
between "live mid-run" (impossible) and "next process" (too slow).

---

## What we're building (scope)

A capability `RuntimeAuthoring` (`AbstractCapability` subclass) exposing a toolset:

- `author_capability(name: str, code: str) -> str` -- write `<dir>/<name>.py`, import,
  validate (Q2), upsert into the store + manifest, return a status message.
- `list_authored_capabilities() -> str` -- list registered authored capabilities
  (name, class, status, last validation error).
- `disable_authored_capability(name: str) -> str` -- mark disabled so it stops being
  injected (Q3).

Plus a `CapabilityStore` the orchestrator threads into the next
`agent.run(capabilities=store.load_active())`.

A "hook" in pyai is not a standalone object -- it is a method on a capability. So
"authoring a hook" = authoring a capability that overrides one lifecycle method. One
tool (`author_capability`) covers both; the docs/E4 capability teaches the model to
write a single-method subclass when it only wants one hook. (Resolved, not a David
question.)

---

## Design

### Module layout (follows the experimental template)

```
pydantic_ai_harness/experimental/authoring/
  __init__.py        # warn_experimental('authoring'); re-export RuntimeAuthoring, CapabilityStore, AuthoredCapability
  _capability.py     # RuntimeAuthoring(AbstractCapability[AgentDepsT])
  _toolset.py        # AuthoringToolset(FunctionToolset): the three tools
  _store.py          # CapabilityStore: dir-backed write/import/validate/index; AuthoredCapability record
  _validate.py       # import + subclass-narrow + construct + static-getter checks (Q2)
  README.md          # experimental warning box + problem/solution + integration contract
tests/experimental/authoring/
  __init__.py
  test_authoring.py
```

### `RuntimeAuthoring` capability

- Dataclass `AbstractCapability[AgentDepsT]`.
- Field `dir: Path` -- where authored `.py` + `manifest.json` live (default under a
  state dir; orchestrator passes its own).
- `get_toolset()` returns `AuthoringToolset` over a `CapabilityStore(self.dir)`.
- `get_instructions()` -- static, cache-stable guidance: "You can author new
  capabilities with `author_capability`. A capability is a subclass of
  `AbstractCapability` ...; it becomes active on the next iteration, not the current
  one." (Sets the next-run expectation, mirroring persona guidance.)
- `get_serialization_name() -> None` -- holds a live store / loads live code; not
  spec-serializable (same reasoning as `SubAgents`).
- Exposes the store (property) so the orchestrator can call
  `store.load_active()` and pass it into the next `agent.run(capabilities=...)`.

### `CapabilityStore` (`_store.py`)

Mirrors Loopy's persona persistence (`pa/personas.py` `save_runtime_persona` /
`load_runtime_personas`) but for `.py` files:

- `write(name, code)` -> writes `<dir>/<name>.py`, validates, upserts `manifest.json`.
- `manifest.json` shape: `{"capabilities": [{"name", "module_file", "class_name",
  "status": "active"|"disabled", "last_error": str|None}, ...]}` -- JSON, indent=2,
  read-modify-write upsert keyed by name (exactly the persona pattern). This is the
  UI-visible surface (Loopy's personas.json is loaded by the UI; same idea).
- `load_active()` -> `list[AbstractCapability[...]]` -- re-imports each active module,
  re-validates, constructs, returns instances for per-run injection. Corrupt/failed
  entries are skipped (logged into `last_error`), not raised -- same fail-soft posture as
  `load_runtime_personas`.
- Import via `importlib.util.spec_from_file_location` + `module_from_spec` with a unique
  module name per (name, re-author) so re-authoring re-imports fresh (avoids stale
  `sys.modules` cache).

### Validation (`_validate.py`) -- Q2 default

```
module = import_module_from_path(path)            # ModuleType
caps = [obj for _, obj in inspect.getmembers(module)
        if isinstance(obj, type) and issubclass(obj, AbstractCapability)
        and obj is not AbstractCapability and obj.__module__ == module.__name__]
# exactly one -> construct -> call static getters -> type-check returns
```

`inspect.getmembers` returns `(str, object)` pairs -- `object`, not `Any`. `isinstance`
+ `issubclass` narrow to `type[AbstractCapability]`. The constructed instance is typed
`AbstractCapability[...]`. No `Any` crosses the harness boundary even though the loaded
code is dynamic. This is the core typing discipline the CLAUDE.md demands.

### Immediacy boundary & activation (Q1 Option A)

1. Model calls `author_capability(name, code)` during run *N*.
2. Tool writes + validates + upserts the store. Returns: "registered; active next
   iteration."
3. Orchestrator loop, before run *N+1*: `caps = authoring.store.load_active()` ->
   `agent.run(prompt, message_history=history, capabilities=[*base, *caps])`.
4. Run *N+1* sees the new tools/hooks live.

The single integration line the orchestrator owner must add is threading
`store.load_active()` into the next `.run(capabilities=...)`. Documented in README as the
integration contract. (This is the analogue of Loopy threading `personas.json` through
`make_orchestrator`.)

### Trust boundary & the `pa` Monty alternative (on record)

Native authoring executes arbitrary Python in-process at import + construct + run time --
the same trust boundary Loopy already operates under (it runs bash and edits files). The
**safer-sandbox alternative** is the dormant `pa` registration system in the Loopy tree
(`pa/slots.py`, `pa/registration_tools.py`, `pa/capability.py` `PaRegistrations`,
`pa/registrations.py`, `pa/registration_runtime.py`, `pa/monty_bridge.py`): it wires
type-checked, resource-limited, allowlist-gated `pydantic_monty` snippets into typed hook
"slots" (`instruction`, `compaction`, `before_tool_hook`, `tool`, ...). It trades native
power for sandboxing and is **next-run-only** too (`PaRegistrations` loads its YAML
manifest once in `__post_init__`; a registration written mid-run can't add a tool to the
in-flight run). We are choosing native; this paragraph is the recorded tradeoff so the
decision is auditable.

---

## Resolved by me (not David questions)

- **One tool for capability + hook.** A hook is a capability method; `author_capability`
  writes a full subclass. The E4 docs teach single-method subclasses.
- **Persist as `.py`, not spec.** Authored caps hold live code; like `SubAgents` they
  are not spec-serializable (`get_serialization_name -> None`).
- **Fail-soft loading.** A corrupt/failed authored module is skipped with `last_error`
  recorded, never crashes the loop (matches `load_runtime_personas`).
- **Upsert + disable** rather than versioning (Q3 recommendation; flip if David objects).
- **UI surface = `manifest.json`** (same role as `personas.json`).
- **No `pyproject.toml`/`uv.lock` edits.** Only stdlib (`importlib`, `inspect`, `json`,
  `pathlib`) + pyai.

## Testing plan (100% branch coverage, `TestModel` only)

- author -> module imported, manifest upserted, `load_active()` returns a constructed
  instance.
- the authored instance, injected via `Agent(..., capabilities=[...])` /
  `agent.run(capabilities=[...])`, actually contributes its tool/instruction (end-to-end
  through `Agent` with `TestModel`, per CLAUDE.md testing guidance).
- validation failures: not a subclass, zero subclasses, multiple subclasses, construct
  raises, getter returns wrong type -> each returns an error message + records
  `last_error`, does not raise out of the tool.
- re-author same name -> fresh import (no stale `sys.modules`), manifest upsert.
- disable -> dropped from `load_active()`.
- corrupt manifest / corrupt module on `load_active()` -> skipped, not raised.

## Out of scope (Phase 1)

- The orchestrator-side loop wiring (lives in Loopy, not harness) -- harness only
  provides the capability + store + the documented integration contract.
- The E4 docs capability (separate, paired).
- Any sandboxing (explicitly chose native; `pa` Monty path documented as the alternative).
