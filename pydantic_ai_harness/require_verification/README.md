# Require Verification

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/require_verification/)

`RequireVerification` redirects a coding agent that tries to finish after editing code
without running a recognized check. It keeps a per-run evidence ledger:

- a successful `write_file` or `edit_file` call makes older evidence stale;
- common test, lint, type-check, and build commands record fresh pass/fail evidence;
- an attempted completion without a fresh pass gets a verification nudge;
- redirects are bounded by `max_attempts` so a missing tool or broken environment
  cannot trap the run in a loop.

The capability is opt-in. Leave it off chat or research agents that do not edit code.

> The API may change between releases. Where practical, breaking changes ship with
> a deprecation warning.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.require_verification import RequireVerification
from pydantic_ai_harness.shell import Shell

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        FileSystem(root_dir='.'),
        Shell(cwd='.'),
        RequireVerification(),
    ],
)
```

By default, Markdown, MDX, and reStructuredText edits are exempt. Override
`exempt_path_patterns` when documentation in your project has a different shape.

## Custom tools

Name custom mutation and verification tools explicitly:

```python
from pydantic_ai_harness.require_verification import RequireVerification

RequireVerification(
    mutating_tools=('write_file', 'edit_file', 'apply_workspace_patch'),
    verification_tools=('run_project_checks',),
)
```

A custom verification tool counts as passing when it returns successfully. Raise an
exception on failure so Pydantic AI does not call `after_tool_execute` for that result.

`run_command` receives additional handling. Commands containing common test, lint,
type-check, or build tools are classified automatically, and the Harness shell result's
exit-code or timeout markers are treated as failures. Common formatter and fix commands
also count as edits, so checks run before them become stale.

The guard only observes tools named in its configuration and recognized shell command
forms. If another tool can modify source files, add its name to `mutating_tools`.

## Scope

- Verification shell commands must be standalone. Chained, piped, background, and
  command-substitution forms do not count because their final shell status may hide a
  failed check.
- Any non-exempt edit invalidates all earlier evidence. The guard does not try to infer
  a test's dependency closure.
- After `max_attempts` redirects, the run may finish without passing evidence. The
  capability does not add a machine-readable verification verdict to the run result.
