---
include:
  - "pydantic_ai_harness/**/*.py"
  - "tests/**/*.py"
  - "integration_tests/**/*.py"
---

For user or model input passed to another parser, verify guards against the
syntax that parser accepts, including aliases, abbreviations, normalization,
separators, and repeated options. For external resources, report failed cleanup
and preserve tracked identity until cleanup succeeds. Trace non-default
addresses, endpoints, paths, and credentials through every consumer. Measure
documented limits on the final returned value, including markers and metadata.
