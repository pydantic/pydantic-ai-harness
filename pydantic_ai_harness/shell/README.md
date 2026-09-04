# Shell

Give an agent the ability to run shell commands inside the run's sandbox, with
allow/deny controls and managed background processes.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/shell/)

## The problem

Agents frequently need to run a build, a test suite, a linter, or a quick
`grep`. Wiring up command execution -- streaming output, timeouts, truncation,
killing runaway processes, and cleaning up background jobs at the end of a run --
is fiddly boilerplate that every agent reinvents.

## The solution

`Shell` exposes command-execution tools rooted at a working directory, with
configurable allow/deny lists and automatic cleanup of background processes
when the agent run ends. Commands run inside the sandbox attached to the run, so
every run needs one; without it the first tool call raises an error that says
how to attach one.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.sandboxes import LocalSandbox
from pydantic_ai_harness import Shell

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Shell(cwd='workspace', allowed_commands=['ls', 'cat', 'rg'])],
)

result = agent.run_sync(
    'List the Python files and summarize the largest one.',
    sandbox=LocalSandbox(root=Path.cwd()),  # the agent process's own filesystem
)
print(result.output)
```

`cwd` is a sandbox path: absolute, or relative to the sandbox working directory.
`~` is not expanded. `LocalSandbox` runs commands on the agent process's own
machine and isolates nothing; for untrusted work attach a container- or
VM-backed sandbox instead.

## Tools

| Tool | Purpose |
|---|---|
| `run_command` | Run a command synchronously and return labelled stdout/stderr plus exit code. Honors a per-call or default timeout. |
| `start_command` | Launch a long-running command (server, watcher) in the background; returns an ID. |
| `check_command` | Report the status and accumulated output of a background command. |
| `stop_command` | Terminate a background command and return its final output. |

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. `max_output_chars` limits the text sent to the model after
execution; the **tail** is kept (the head is dropped), so errors, stack traces,
and the `[stderr]` section -- which all land at the end -- survive truncation.
Background command status and exit metadata follow the captured output so they
remain in the retained tail. `LocalSandbox` also has a separate 10 MiB safety
ceiling on command output.

Each command starts in the configured `cwd`; `cd` affects only that command. The
omitted foreground timeout uses the configured default, and an override must be
positive and at most `max_timeout`. For longer work, use `start_command`.

## Command controls

| Field | Effect |
|---|---|
| `allowed_commands` | If non-empty, only these executables may run (allowlist). |
| `denied_commands` | These executables are always rejected (denylist). |
| `denied_operators` | Shell operators (e.g. `>`, `>>`, `|`) that are rejected when present. |
| `allow_interactive` | If `False` (default), commands that expect a TTY (`vi`, `sudo`, `ssh`, ...) are blocked. |

`allowed_commands` and `denied_commands` are mutually exclusive -- set one, not
both. `denied_commands` defaults to a list of destructive commands (`rm`,
`rmdir`, `mkfs`, `dd`, `format`, `shutdown`, `reboot`, `halt`, `poweroff`,
`init`); pass an empty list to disable. The executable name is extracted with
`shlex`, so arguments don't bypass the check.

An empty `allowed_commands` collection does not select allowlist mode. The
configured `denied_commands` remain active; when omitted, this is the built-in
denylist. Pass `denied_commands=[]` to disable command-name filtering.

A denied or blocked command surfaces to the model as a `ModelRetry` (the model
can retry with an allowed command) rather than aborting the run. So does every
other failure the model can act on: a working directory an earlier command
deleted or replaced with a file, and a command holding a NUL byte or a character
that cannot be encoded for the operating system. A sandbox that has gone away
aborts the run instead, since the model cannot recover from it. Bare backend
programming failures also abort the run.

> **These checks are best-effort, not a security boundary.** `allowed_commands`
> is a guardrail against accidents, not a security boundary. Validation checks
> only the first token, and allowlisted commands such as `python`, `git`, `uv`,
> and `make` can spawn arbitrary processes. A model that wants to work around
> the allowlist can. The sandbox is the isolation boundary: for untrusted work
> attach a container- or VM-backed one rather than `LocalSandbox`.

## Environment control

`env` sets the environment commands run with; `denied_env_patterns` removes
names from it by glob before it is handed to the sandbox.

| Field | Effect |
|---|---|
| `env` | Explicit environment for commands. `None` (the default) leaves the sandbox's own environment in place. |
| `denied_env_patterns` | Glob patterns (`fnmatch`) for variable names removed from `env`. Mirrors `denied_commands`. |

`denied_env_patterns` requires an explicit `env` mapping and filters that mapping
only; it cannot remove a variable the sandbox itself provides.
`LocalSandbox` runs commands on the agent process's own machine, but passes only
its fixed `PATH`, `HOME`, `LANG`, and `TMPDIR` variables plus the explicit `env`.
It is not an isolation boundary: commands can still access other host files.
A container- or VM-backed sandbox starts from its own image instead.

```python
import os

from pydantic_ai_harness import LLM_API_KEY_ENV_PATTERNS, Shell

# A fixed environment.
Shell(cwd='.', env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/home/agent'})

# Or one assembled from your own, minus provider credentials.
Shell(cwd='.', env=dict(os.environ), denied_env_patterns=LLM_API_KEY_ENV_PATTERNS)
```

`LLM_API_KEY_ENV_PATTERNS` covers common provider prefixes (`ANTHROPIC_*`,
`GATEWAY_*`, `GEMINI_*`, `GOOGLE_*`, `OPENAI_*`, `OPENROUTER_*`) plus
`PYDANTIC_AI_GATEWAY_API_KEY`. It targets LLM credentials only: it does not
cover other secrets (a `LOGFIRE_TOKEN`, a GitHub token, cloud credentials), and
its prefixes are coarse, so `GOOGLE_*` also strips non-credential vars like
`GOOGLE_APPLICATION_CREDENTIALS`. Treat it as a starting point and add your own
patterns.

How `env` combines with what the sandbox already provides is the sandbox backend's
decision: `LocalSandbox` adds yours to its own fixed `PATH`, `HOME`, `LANG` and
`TMPDIR`, others may replace it outright. Supply what a command needs rather
than assuming the agent process's environment reaches it.

## Background processes

`start_command` writes stdout/stderr to temp files in the sandbox and returns a
short ID. Use `check_command(command_id)` to poll and `stop_command(command_id)`
to terminate and collect final output. Each process is started with `setsid`, so
the whole process group can be signalled -- `SIGTERM`, escalating to `SIGKILL`
after a grace period.

On run end, the toolset's `__aexit__` terminates every still-running background
process and deletes its temp files. The agent runtime enters toolsets via an
`AsyncExitStack`, so this cleanup runs whether the run succeeds or raises -- an
agent that forgets to call `stop_command` won't leak processes.

## Configuration

```python
Shell(
    cwd='.',                       # str | Path -- sandbox working directory
    allowed_commands=[],           # allowlist (mutually exclusive with denied)
    denied_commands=[...],         # denylist (defaults to destructive commands)
    denied_operators=[],           # blocked shell operators
    default_timeout=30.0,          # seconds, per run_command
    max_timeout=600.0,              # maximum seconds, per run_command
    max_output_chars=50_000,       # output cap returned to the model
    allow_interactive=False,       # allow TTY-style commands
    env=None,                      # explicit env (None = the sandbox's own)
    denied_env_patterns=[],        # glob patterns removed from `env`
)
```

## Agent spec (YAML/JSON)

`Shell` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Shell:
      cwd: ./workspace
      allowed_commands: ['ls', 'cat', 'rg', 'pytest']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Shell

agent = Agent.from_file('agent.yaml', custom_capability_types=[Shell])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`Shell`.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
