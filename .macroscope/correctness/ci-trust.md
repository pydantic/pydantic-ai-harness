---
include:
  - ".github/**"
  - "Makefile"
  - "integration_tests/**"
  - "scripts/**"
---

Treat checked-out pull-request content and every command, import, script, or
task it invokes as untrusted. Do not expose repository or environment secrets
to that execution. Check both trusted and fork pull-request behavior and the
aggregate required check. For path-gated jobs, start from the executed command
and verify that the filter covers its task-runner or script entry point and all
dependency, configuration, image, and workflow inputs that can change it.
