# Staleness Tracker

Tell the model when files it read have changed on disk since it read them.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.staleness import StalenessTracker
> ```

Staleness Tracker is a released, non-experimental capability. Pydantic AI Harness is still on 0.x releases, so the API may change between minor releases. See the repository [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## The problem

A model cannot perceive elapsed time or concurrent change. Once it reads a file, it acts on
that snapshot for the rest of the run -- even if another agent, a build step, `git checkout`,
or the user edits the file underneath it. Nothing in the transcript tells it the ground
moved, so it confidently edits, references, or reasons about stale contents.

## The solution

`StalenessTracker` records `(path, mtime, size)` after every file read/write the model makes,
then -- before each model request -- re-stats the tracked set and injects an *ephemeral*
notice naming any file that changed or was deleted since it was last observed:

```
<system-reminder>Files changed on disk since you last read them: src/foo.py, tests/test_foo.py. Re-read before relying on their contents.</system-reminder>
```

- The notice is added in `wrap_model_request`, which runs *after* the durable history is
  persisted, so it reaches the model but is never written to `message_history`. Notices
  never accumulate, and the check re-evaluates fresh every request.
- A `CachePoint` sits immediately *before* the notice, so the cached prefix stays
  byte-identical turn over turn -- only the notice falls outside the cache.
- The re-check is **stat-only** (no hashing) over an LRU-capped set, so it stays cheap even
  on large runs.

The agent's own writes are never flagged as staleness: observation happens in
`after_tool_execute`, *after* the write lands, so a self-write records the post-write
`(mtime, size)` and matches cleanly on the next check.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.staleness import StalenessTracker

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[StalenessTracker()])
result = agent.run_sync('Refactor the auth module.')
```

## Which tools count as file reads/writes

Data-driven, and meant to be changed. The `track` mapping is `tool-name pattern -> where the
path lives in that tool's args`; patterns are matched with `fnmatch`. Each value is either
the name of the arg holding the path, or a callable `(args) -> paths`:

```python
StalenessTracker(
    track={
        'read*': 'file_path',       # any tool whose name starts with `read`
        'open_file': 'filename',    # a host-specific tool + arg name
        'patch': lambda args: args.get('targets', []),  # multi-path call
    }
)
```

The default covers `read` / `read_file` / `write` / `write_file` / `edit` / `apply_patch`,
each reading whichever of `file_path` / `path` the call carries.

For full control, pass `path_extractor`, which replaces `track` entirely and decides for
every tool which paths (if any) a call touched:

```python
def path_extractor(tool_name: str, args: dict) -> list[str]:
    if tool_name in {'grep', 'ls'}:
        return []  # not a read of a specific file
    return [args['path']] if 'path' in args else []

StalenessTracker(path_extractor=path_extractor)
```

## Configuration

```python
StalenessTracker(
    track=...,             # tool-name pattern -> path arg / callable (see above)
    path_extractor=None,   # full escape hatch; replaces `track` when set
    root=None,             # base dir for relative paths (None = process cwd)
    max_tracked=200,       # LRU cap on tracked files (bounds re-stat cost)
    max_listed=10,         # cap per changed/deleted group before `(+N more)`
    cache_ttl='5m',        # TTL for the CachePoint before the notice ('5m' | '1h')
    notice_tag='system-reminder',  # wrapper tag for the notice
)
```

## Deliberate scope

This is the small, focused freshness signal, not a filesystem watcher. Out of scope:

- **Content hashing** -- stat (mtime + size) only; a same-size, same-mtime rewrite is not
  detected (rare, and hashing every tracked file per request is the cost this avoids).
- **Directory watching** -- only files the model actually read/wrote are tracked.
- **Git-status integration** -- no awareness of branches, stashes, or the index.
- **Cross-run persistence** -- the ledger is per-run (`for_run`) and never shared between
  concurrent runs or persisted to disk.

## Design notes

- **Notice keeps firing until the model re-reads.** A changed file stays in the notice on
  every subsequent request until the model reads it again (which refreshes the record). This
  mirrors Hermes's edit-staleness ledger: the honest signal is "still stale", not "told you
  once". Because the notice is ephemeral, repeating it costs nothing in durable history.
- **Delivery channel.** The notice rides the ephemeral message tail behind a `CachePoint`
  (the `Planning` pattern), so it never mutates stored history and never busts the cache --
  matching the cache-stability contract. The `<system-reminder>` wording aligns with the
  `SystemReminders` convention (PR #181); the delivery mechanism deliberately diverges from
  #181's persisted `before_model_request` injection to honor the no-history-mutation rule.

## Further reading

- [`pydantic_ai_harness.staleness` source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/staleness/)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Hooks](https://ai.pydantic.dev/hooks/) -- `after_tool_execute` (observe) and
  `wrap_model_request` (ephemeral inject) are the two surfaces used here
- [What Fable wants from a harness](https://github.com/pydantic/pydantic-ai-notes/blob/main/features/harness-comparison/2026-07-06%20what%20fable%20wants%20from%20a%20harness.md) §3
