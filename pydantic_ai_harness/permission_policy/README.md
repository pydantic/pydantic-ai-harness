# Permission Policy

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.permission_policy import PermissionPolicy
> ```

Permission Policy is a released, non-experimental capability. Pydantic AI Harness is still on 0.x releases, so the API may change between minor releases. See the repository [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

An allow / ask / deny rule engine over tool calls, evaluated as each tool runs -- with the
hard-won shell-command mechanics (compound-command splitting, a conservative-parse gate,
wrapper stripping, a read-only safelist, and a flag denylist) that every coding harness has
had to re-derive.

## The problem

Pydantic AI core gives you `ApprovalRequired` and deferred tools -- the *mechanism* to pause a
tool call for human approval. What it does not give you is a *policy*: a way to say "let the
model run `git status` freely, ask me before `git push`, and never let it `rm -rf`". Building
that safely is deceptively hard, because the dangerous surface is the shell:

- `git status && rm -rf /` -- a compound command where one segment is fine and one is fatal.
- `echo $(rm -rf /)` -- the danger hides inside a command substitution.
- `timeout 5 env LD_PRELOAD=evil.so ls` -- wrappers and env assignments smuggle execution
  past a naive `ls`-looks-safe check.
- `git status-evil` -- a prefix match that isn't a word-boundary match.

## The solution

`PermissionPolicy` is a capability that evaluates every tool call in the `wrap_tool_execute`
hook and resolves it to one of three verdicts:

| Verdict | Effect |
|---|---|
| **allow** | the tool runs |
| **deny** | the tool does not run; a message goes back to the model explaining whether re-requesting with justification can help, or the action is never allowed |
| **ask** | the call is routed through Pydantic AI's deferred-approval machinery (`ApprovalRequired`) for a human -- or an `on_ask` handler -- to resolve |

### Rules: ordered, last-match-wins

Rules are an **ordered list; the last matching rule wins** (the model
[opencode](https://github.com/sst/opencode) uses -- specificity comes from position, not an
action-precedence lattice). Each rule is a `(tool-name glob, optional argument matcher,
verdict)`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.permission_policy import PermissionPolicy, Rule

policy = PermissionPolicy(
    rules=[
        Rule('deny', tool='run_command', command='git'),          # deny all git ...
        Rule('allow', tool='run_command', command='git status'),  # ... except git status
        Rule('deny', tool='delete_file'),                         # bare-name deny (any args)
        Rule('ask', tool='write_file', args=lambda a: a['path'].startswith('/etc')),
    ],
)
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[policy])
```

The argument matcher is, in precedence order: `command` (a shell prefix, see below), then
`args` (an arbitrary predicate over the argument dict), then neither (matches any arguments).

### Shell commands: the safe mechanics

For shell-class tools (by default `run_command`, `bash`, `shell`, ...; configurable via
`shell_tools`/`command_arg`), the `command` matcher and the built-in safety analysis operate
on the command with:

- **Compound-command splitting.** The command is split at `&&`, `||`, `;`, `|` and **every
  segment must independently pass**; one bad segment poisons the whole call.
- **A conservative-parse gate.** If the command contains anything we can't prove is a plain
  list of commands -- subshells `(...)`, command substitution `$(...)`/backticks, variable
  expansion `$VAR`, redirection `>`/`<`, background `&`, globs, brace expansion, env-assignment
  prefixes -- the whole call **degrades to `ask`. It never guess-allows.** (This is
  [Codex CLI](https://github.com/openai/codex)'s reject-if-complex posture.)
- **Word-boundary prefix rules.** `git status` matches `git status -sb` but never
  `git status-evil` or `git push`.
- **Wrapper stripping.** `timeout`, `nice`, `env`, `xargs`, `sudo`, ... are peeled to the inner
  command before matching (`timeout 5 git status` is `git status`). Wrapper shapes that could
  smuggle execution -- `env FOO=...` (LD_PRELOAD), `sudo -u`, options on `timeout` -- degrade
  to `ask` instead of peeling.
- **A read-only safelist** auto-allows known-safe commands (`ls`, `cat`, `grep`, `git status`,
  `find` without `-exec`, `rg` without `--pre`, `sed -n {N}p`, ...), and **a flag denylist**
  overrides allows for their dangerous variants (`git -C`/`-p`, `find -delete`/`-exec`,
  `rg --pre`, `rm -rf`, arbitrary interpreters like `bash -c`).

The safelist and denylists are adopted, with attribution, from Codex CLI's command-safety
module; see `_safelist.py` and the [OSS mining notes][mining].

### How the two channels combine

A call has two verdict sources: your **rules** (last-match-wins) and the **built-in
command-safety analysis**. They are merged **most-restrictive-wins** (`deny > ask > allow`).
This is what makes the flag denylist *override* a broad allow: `Rule('allow', command='git')`
plus the command `git -C /etc status` resolves to `ask`, because the built-in analysis flags
`-C` and the more restrictive verdict wins. To opt out of the built-in analysis entirely and
govern shell tools by your rules alone, set `analyze_shell_commands=False`.

## `ask`: the deferred-approval integration

An `ask` verdict raises
[`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired], so it flows through Pydantic
AI's standard deferred-tool machinery. You resolve it in one of two ways:

1. **Externally** (real human-in-the-loop): add `DeferredToolRequests` to the agent's
   `output_type` (or use a `HandleDeferredToolCalls` capability). The run pauses and returns
   the pending approvals; you approve/deny and resume with `DeferredToolResults`.
2. **Inline**, via an `on_ask` handler on the policy:

   ```python
   def on_ask(ctx, request):
       if request.command and request.command.startswith('git push'):
           return True                         # approve
       return 'pushing is disabled in CI'      # deny with this message (or return False)

   policy = PermissionPolicy(rules=[Rule('ask', tool='run_command')], on_ask=on_ask)
   ```

   `on_ask` returns `True` to approve, `False` to deny with a default message, or a string to
   deny with that message. It may be sync or async. The policy only resolves the asks **it**
   raised, so it composes cleanly with other deferred-tool handlers.

If an `ask` verdict fires and neither of these is configured, the run raises the standard
"`DeferredToolRequests` is not among output types" `UserError` -- the same contract as core's
`requires_approval=True` tools.

## Escalation protocol

When `add_escalation_note=True` (the default), guarded tools' descriptions gain a short note
telling the model it may restate a denied call once with a brief justification, and not to
retry commands reported as never allowed. A denied call's result message says the same. This
is Codex CLI's prompt-taught escalation, minus the persistence channel.

## Options

| Option | Default | Purpose |
|---|---|---|
| `rules` | `[]` | Ordered allow/ask/deny rules; last match wins. |
| `default_verdict` | `'ask'` | Verdict when no rule matches and command-safety has no opinion. Set `'allow'` for allowlist-by-exception, `'deny'` to block-by-default. |
| `shell_tools` | `run_command`, `bash`, ... | Tool names whose command argument is analyzed. |
| `command_arg` | `'command'` | The argument holding the shell command. |
| `analyze_shell_commands` | `True` | Run the built-in command-safety analysis for shell tools. |
| `deny_removes_tool` | `False` | Remove a tool denied *regardless of arguments* from the model's toolset entirely (Claude Code bare-name-deny). |
| `add_escalation_note` | `True` | Append the escalation note to guarded tools' descriptions. |
| `on_ask` | `None` | Resolve `ask` verdicts inline instead of surfacing them as `DeferredToolRequests`. |

## Deliberate scope (so these don't read as oversights)

- **`ask` ships the real deferred-approval integration**, not a deny-fallback -- it raises
  `ApprovalRequired` and resolves through core's machinery, with `on_ask` as an inline
  convenience.
- **Rule persistence is out of scope for v1.** The model cannot propose "always allow this
  prefix" rules yet. When that lands, proposals must be vetted against
  `BANNED_PREFIX_SUGGESTIONS` (exported here, adopted from Codex) -- the rule-persistence
  channel needs its own denylist, distinct from the execution channel. opencode's
  doom-loop-to-session-rule pattern is the follow-up shape.
- **Standard-library parsing only.** The analyzer uses a quote-aware scanner plus `shlex`, no
  new dependency. Because the conservative gate degrades anything it cannot prove plain, a
  full bash AST parser would only *narrow* what degrades to `ask` -- it would never turn a
  wrong-allow into a right-allow.
- **An optional LLM guardian** (auto-approver for `ask` verdicts) is a natural extension of
  `on_ask` and is left for a follow-up.

## Further reading

- [`pydantic_ai_harness.permission_policy` source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/permission_policy/)

[mining]: https://github.com/pydantic/pydantic-ai-notes/blob/main/features/harness-comparison/2026-07-08%20oss%20implementation%20mining%20-%20oh-my-pi%2C%20codex%2C%20opencode.md
