## Summary

<!-- Brief description of the changes -->
<!-- If this crosses a command/parser, process/container, network, resource-
lifecycle, output-limit, or CI trust boundary, add "Boundary notes" covering:
the downstream contract and focused reproduction; cleanup-failure and
non-default-configuration evidence; final limit accounting; and, for CI, the
event/ref -> checked-out code -> credentials -> executable steps map plus
conditional-job path inputs. -->

## Linked Issue

<!-- REQUIRED: Every PR must have a linked issue. Open one first if it doesn't exist. -->
<!-- Use: Fixes #... or Closes #... -->

Fixes #

## Checklist

- [ ] Linked issue exists and is referenced above
- [ ] Tests added/updated for new behavior
- [ ] `make lint && make typecheck && make test` passes locally (don't stress about CI -- we'll help)
- [ ] No changes to `pyproject.toml` or `uv.lock` (dependency changes require a separate issue)
- [ ] Docstrings use single backticks (not RST double backticks)
