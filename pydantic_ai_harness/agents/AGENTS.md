# Pydantic AI Harness - Agents

This module exposes prepackaged agents that users can just import, or install in their own agents via:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.agents import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder('.')])

result = agent.run_sync('Find out why tests/test_parser.py fails and fix the bug it caught.')
print(result.output)
#> Found it: `parse()` returned None on empty input instead of raising. Fixed in src/parser.py; tests pass now.
```
