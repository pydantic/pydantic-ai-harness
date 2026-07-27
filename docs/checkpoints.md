---
title: Checkpoints
description: Snapshot the project before a mutating tool runs, so any damage a later tool call does is restorable.
---

# Checkpoints

File-level undo for coding agents: snapshot the project before a mutating tool runs, so any
damage a later tool call does is restorable.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.checkpoints import Checkpoints
> ```

Checkpoints is a released, non-experimental capability. Pydantic AI Harness is still on 0.x releases, so the API may change between minor releases. See the repository [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## The problem

An agent editing files can make a change you want to undo -- a bad refactor, a deleted block,
a wrong path taken. Without a safety net, agents stay conservative and users hand-fix
mistakes. The fix five coding harnesses converged on is a shadow git repository: snapshot the
working tree before each file-mutating tool call, independent of the user's own version
control.

## The solution

`Checkpoints` commits the project's files to a git repository that is separate from the
user's `.git` before any tool whose name is in `mutating_tools` executes. Restore a snapshot
with `restore`, or list them with `checkpoints`.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.checkpoints import Checkpoints
from pydantic_ai_harness.filesystem import FileSystem

checkpoints = Checkpoints(project_root='.')
agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[FileSystem(), checkpoints])
await agent.run('Refactor the auth module')

for cp in checkpoints.checkpoints():
    print(cp.id, cp.tool_name, cp.files_changed)

# Undo everything the run did:
checkpoints.restore(checkpoints.checkpoints()[0].id)

# Or restore just one file from a specific checkpoint:
checkpoints.restore(cp.id, paths=['src/auth.py'])
```

## Shadow git mechanics

The snapshots live in a repository at `<state_dir>/checkpoints/<project-slug>/`, used with
`GIT_DIR` pointing at that directory and `GIT_WORK_TREE` pointing at the project root. It has
its own committer identity, gpg signing off, and no hooks, and it is isolated from the user's
git config (`GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` point at `os.devnull` for shadow commands,
so global and system git config never apply).

- **Never touches the user's `.git`.** The shadow repo has its own `GIT_DIR`; git also refuses
  to add a nested `.git` directory, and `info/exclude` lists `.git/` as a second guard.
- **Works in non-git projects.** The shadow repo is `git init`-ed on first use, so the project
  itself does not need to be a git repository.
- **Respects `.gitignore`.** Because the work tree is the project root, `git add -A` reads the
  project's own ignore files, so ignored paths (`node_modules/`, build output) stay out of
  snapshots.
- **Debounced.** If nothing changed since the last checkpoint, no empty commit is made and the
  previous checkpoint is reused.

`state_dir` defaults to `~/.pydantic-ai-harness`. Point it elsewhere for a project-local or
ephemeral store. Keep it outside `project_root` (or add it to the project's `.gitignore`): a
`state_dir` nested under the work tree would be picked up by `git add -A` and snapshotted into
every checkpoint.

## When snapshots are taken

A checkpoint is taken before a tool whose name is in `mutating_tools` (defaults cover the
harness `FileSystem` toolset plus write/edit/patch/create/delete/move names common to other
coding harnesses). Set `snapshot_before_bash=True` to also snapshot before shell tools
(`bash_tools`); it is off by default because shell commands are often read-only.

```python
Checkpoints(mutating_tools={'write_file', 'edit_file', 'apply_patch'}, snapshot_before_bash=True)
```

Each checkpoint records the tool it was taken before and its run id in the shadow commit
message. `files_changed` reports what changed since the previous checkpoint (for the first
checkpoint, everything it captured).

## Restore semantics

`restore` is a `git checkout` from the shadow repo. It overwrites files that existed at the
checkpoint and re-creates ones deleted since. It does **not** remove files created after the
checkpoint -- restoring is additive, not a hard reset. Pass `paths=[...]` to restore only
specific files.

## Exposing restore to the model

By default the model cannot restore checkpoints -- that is usually a human or application
action. Set `expose_tool=True` to add `list_checkpoints` and `restore_checkpoint` tools to the
agent.

```python
Checkpoints(expose_tool=True)
```

## Scope

- **Files only.** This is the file-level slice of undo. Conversation rewind and forking pair
  with the branchable session-history track ([harness issue #321](https://github.com/pydantic/pydantic-ai-harness/issues/321)).
- **Best-effort.** A shadow-git failure emits a `CheckpointWarning` and lets the run continue
  rather than aborting the agent. Escalate to an error in dev/CI with
  `warnings.filterwarnings('error', category=CheckpointWarning)`.
- **Last-writer-wins across concurrent runs.** Runs against the same project share one shadow
  repo; each snapshot commits the whole work tree as it looks at that instant.

## Further reading

- [`pydantic_ai_harness.checkpoints` source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/checkpoints/)
- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Pydantic AI hooks](/ai/core-concepts/hooks/) -- the snapshot is taken before each mutating tool call
