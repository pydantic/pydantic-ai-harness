# Coder

`Coder` gives a Pydantic AI agent tools and guidance for investigating, editing, and testing a codebase.
It works in the run's workspace, on this machine or in a sandbox. It is a regular combined capability made from [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/), [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/), and the [context management](https://pydantic.dev/docs/ai/harness/compaction/) capabilities, so you can use it whole or take it apart.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

Install the Coder extra to include ripgrep (`rg`), which backs the `list_files` and `grep` tools:

uv:

```bash
uv add "pydantic-ai-harness[coder]"
```

pip:

```bash
pip install "pydantic-ai-harness[coder]"
```

The extra installs `ripgrep==14.1.0` except on Android, where `rg` must be supplied separately on `PATH`.
Add a provider extra such as `[coder,anthropic]` when needed.
`Coder` edits files and runs commands in the run's [workspace](https://pydantic.dev/docs/ai/workspace/), and does not pick one for you: nothing touches your machine unless you attach `LocalWorkspace` from `pydantic_ai.capabilities`. A run without a workspace fails at its start.

<!-- Keep this blown-out example in sync across docs/coder.md, docs/index.md, README.md, pydantic_ai_harness/coder/README.md, and examples/coding_agent.py. -->

```python
import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[LocalWorkspace('.', env={'PATH': os.environ['PATH'], 'HOME': os.environ['HOME']}), Coder()],
)
```

```python
result = agent.run_sync('Investigate the failing parser test, fix the cause, and run focused checks.')
print(result.output)
```

The workspace's working directory, here the current directory, is the project: file paths resolve from it and commands start in it. To run the same agent in isolation, a sandbox capability (Modal, E2B, Daytona, or Sprites) replaces `LocalWorkspace`, and Coder's tools edit files and run commands in the sandbox instead. Commands run without an allowlist, and the file tools' path restrictions do not apply to them.

Interfaces that start runs for you work the same way: [`agent.to_cli_sync()`](https://pydantic.dev/docs/ai/cli/) and [`agent.to_web()`](https://pydantic.dev/docs/ai/web/) run in the attached workspace.

The exported `pydantic_ai_harness.coder:coder_agent` is the same agent, model-less and named `coder`, working in the directory it is imported from.
Use it with the Pydantic AI CLI:

```bash
uvx --with "pydantic-ai-harness[coder]" clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-sonnet-5
```

### The command environment

Commands in a local workspace inherit nothing from the agent process. The example passes `PATH` so commands find your tools (`rg`, `git`, `uv`, anything under Homebrew or `~/.local/bin`) and `HOME` for tools that keep their configuration there; add anything else your commands need to `env`. Passing `os.environ` would hand the model's commands every secret in the process, LLM API keys included.

## Sharing a workspace

A workspace outlives the run that used it, so the next run can pick up where the last one left off, in the same files.

To continue the conversation, pass its messages. The latest response records which workspace it ran in, and `LocalWorkspace` continues in that directory (it declines a record naming any other directory, so message history cannot point the agent somewhere else on your machine):

```python
result = agent.run_sync('Add a --verbose flag to the CLI.')
result = agent.run_sync('Document the new flag in the README.', message_history=result.all_messages())
```

To hand the work to a different agent, or to a later run with a fresh conversation, pass the workspace itself. The reviewer below has no workspace of its own, so it works exactly where it is told:

```python
reviewer = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[Coder(instructions='Review the uncommitted change and run the tests. Do not edit files.')],
)
review = reviewer.run_sync('Review the change.', workspace=result.workspace)
```

With a sandbox this is how several agents collaborate in one isolated environment. Sub-agents need nothing extra: [`SubAgents`](https://pydantic.dev/docs/ai/harness/subagents/) runs each delegate in the parent run's workspace.

## Composition

`Coder()` is these capabilities, in this order:

1. A `Capability` carrying the default instructions, plus any `instructions=` you pass.
2. [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/)`(content_hashes=False, max_read_chars=60000, tools=FILE_TOOL_NAMES)`, where
   `FILE_TOOL_NAMES` is `read_file`, `write_file`, `edit_file`, `list_files`, and `grep`. Its `root_dir` is the workspace's working directory.
3. [`Shell`](https://pydantic.dev/docs/ai/harness/shell/)`(denied_commands=[], allow_interactive=True, default_timeout=270, tools=['shell'])`.
4. [`RepoContext`](https://pydantic.dev/docs/ai/harness/repo-context/)`(expose_inventory_tool=False)` for repository instructions and structure.
   Pass `repo_context=False` to leave it out when the agent already binds its own `RepoContext`, so the
   instruction files are not loaded twice.

Then the plumbing, which the agent never calls directly:

5. [`ClearToolResults`](https://pydantic.dev/docs/ai/harness/compaction/)`(max_fraction=0.7)` and [`WarnNearLimits`](https://pydantic.dev/docs/ai/harness/compaction/)`(max_context_fraction=0.9)`.
6. A private [`ToolOutputLimits`](https://pydantic.dev/docs/ai/harness/tool-output-limits/) specialization that truncates any tool result over 64,000 characters
   without adding a spill-retrieval tool. Its stable ID, `coder_tool_output_limits`, lets durability
   capabilities bind its inherited operations without colliding with a separately configured `ToolOutputLimits`.
7. [`RepairToolArguments`](../repair_tool_arguments/) repairs malformed JSON tool arguments before normal validation (see below).

Every tool comes from `FileSystem` or `Shell`; those pages document each one in full. Build the same
agent from the pieces to change any setting, for example to keep content hashes, add `list_directory`,
or allowlist commands.

## Six tools

| Tool | Behavior |
| --- | --- |
| `read_file(path, offset=0, limit=None)` | Zero-based line offset, one-based displayed line numbers, up to 2,000 lines or 60,000 characters of complete lines; the continuation hint names the exact next offset, and a line too long for the window is named and skippable. No hash header. |
| `write_file(path, content)` | Create a file in an existing directory, or replace one. No `expected_hash`. |
| `edit_file(path, old_text, new_text)` or `edit_file(path, replacements=[...])` | Exact replacements, each matching once; a batch is checked in memory and written only if every replacement matches. |
| `list_files(path='.', glob=None)` | `rg --files`, sorted by path, respecting ignore files and skipping hidden files. |
| `grep(pattern, ...)` | Ripgrep search with `path`, `glob`, `file_type`, `ignore_case`, `literal`, and `context` (0 to 20). |
| `shell(command, mode='foreground', timeout=270)` | Unrestricted commands rooted at the workspace that outlive the run. |

Results are bounded by `FileSystem`'s caps (2,000 lines or 60,000 characters per `read_file`, 1,000 lines or files per search or listing) and Coder's 64,000-character
tool-output limit; a truncation marker means more output was omitted, so narrow the search rather than
assuming it was complete. A `read_file` window stays under the output limit, so paging by `offset` never skips lines. Use `shell` for `mkdir`, `find`, process inspection, and `kill`. File writes
keep the standalone filesystem's protected-path rules (`.git`, `.env`, keys, and secrets); shell can bypass
these rules. Coder does not include planning, delegation, or the run-scoped `run_command` family.

## Filesystem scope

File tools are scoped to the workspace's working directory by default. For trusted local use,
`Coder(unrestricted_filesystem=True)` sets `FileSystem(root_dir='/',
protected_patterns=[])`: relative paths still resolve from the working directory, and absolute paths anywhere in
the run's workspace are accepted, such as `/tmp/example.py`. OS permissions and file-change event listeners still apply. This permits
modifying secrets and repository metadata: use it only when you trust the agent and its inputs. Shell commands
were already unrestricted.

## Long-running commands

`shell` is the [`Shell`](https://pydantic.dev/docs/ai/harness/shell/) capability's persistent tool. Foreground waits at most 270 seconds
(or a smaller positive `timeout`) and then returns handles for the same running process; background returns
them immediately. Both end with a PID, an absolute output log path, and an absolute JSON status path (inside the workspace) whose
`exit_code` is `null` while the command runs; foreground puts the last 16,000 bytes of output before them. Commands outlive the agent run, so servers keep running; there
is no completion notification or automatic wake-up after a final response. The Shell page covers the
supervisor, cleanup, and the `CommandStartedEvent`, `CommandOutputEvent`, and `CommandFinishedEvent` progress
events a UI can subscribe to.

The default instructions tell the agent to finish required work before giving a final response: do other
useful work, then poll status and output until completion or a genuine blocker.
Servers may remain running after startup and readiness are verified. Commands get only the environment
the workspace passes (see [The command environment](#the-command-environment)); host files remain
accessible to commands in a local workspace.

## Instructions

The default instructions keep engineering guidance brief: autonomous investigation and completion,
focused changes and verification, and pragmatic DRY, YAGNI, SOLID, and the Zen of Python.
Tool descriptions supply tool usage; `RepoContext` supplies repository instructions and structure.
`Coder(instructions='...')` appends project-specific guidance rather than replacing defaults.
Use it for additional policy, such as file-size limits or a preferred verification workflow.

## Tool argument repair

`Coder` composes [`RepairToolArguments`](../repair_tool_arguments/), which uses `json-repair` for malformed JSON before Pydantic AI validates the tool schema.
Valid JSON and already-parsed arguments pass through unchanged. Missing fields and invalid types still
follow normal validation and retry behavior. Repair applies to tools added alongside Coder too.
If the repair parser raises a value or recursion error, original arguments go through normal validation.

Repair is heuristic: malformed input can be ambiguous, and inferred strings may differ from the model's
intent. It does not supply a schema to the repair library or bypass exact edit matching.
Each attempt emits a `repair_tool_arguments` span through `ctx.tracer`, without arguments or file
contents. Other Coder operations rely on core tool spans and on the events its `FileSystem` and `Shell`
capabilities emit.

## Upgrading

This release makes the workspace the single place that decides where an agent works. Removed arguments are still accepted, emit a `HarnessDeprecationWarning` naming the fix, and are ignored.

- **Attach a workspace.** `Coder`, `FileSystem`, `Shell`, `RepoContext`, and `Macroscope` fail at run start without one, as do `Skills`, `PydanticAIDocs` (with a local checkout), and `ToolOutputLimits` (when it can spill) unless given their own `workspace=` or store. Add `LocalWorkspace('.', env=...)` to the agent's capabilities, as in [Usage](#usage).
- **Set the directory on the workspace.** `Coder('dir')`, `Shell(cwd=)`, `FileSystem(cwd=)`, `Macroscope(cwd=)`, and `RepoContext(workspace_dir=)` are ignored; use `LocalWorkspace('./dir')`.
- **Pass the command environment.** A local workspace used to give commands the host's `PATH`, `HOME`, `LANG`, and `TMPDIR`; now they get the workspace's `env` plus `Shell(env=)`. Pass `env={'PATH': ..., 'HOME': ...}` to `LocalWorkspace` (see [The command environment](#the-command-environment)).
- **`FileSystem(root_dir=)`** defaults to the working directory and resolves relative values from it. It must contain the working directory, symlinks that lead outside it are refused, and `root_dir='/'` turns the checks off.
- **Harness files moved into the working directory.** Tool-output spills and Shell background-job files are under `.pydantic-ai-harness/` (git-ignored) instead of `$TMPDIR`. `ToolOutputLimits(store=LocalFileStore())` keeps spills on this machine.
- **Skills** are read from the workspace at run start and loaded with a `load_skill` tool. [`Skills(workspace=LocalWorkspaceBackend('/app'))`](https://pydantic.dev/docs/ai/harness/skills/) reads them from somewhere else.
- **Capability Creation** runs only when the workspace is a writable `LocalWorkspace`.

## Benchmarking

See the [Terminal-Bench 2.1 playbook](TERMINAL_BENCH.md) for running Coder
inside Harbor, pinning the adapter and harness, and inspecting trial results.

See the [source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/coder/).
