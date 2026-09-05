---
title: GitHub
description: Give an agent scoped access to GitHub's hosted MCP tools with read-only defaults and approval-gated writes.
---

# GitHub

`GitHub` lets an agent read or change one GitHub repository or the repositories owned by one organization.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/github/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[github]" "pydantic-ai-slim[openai]"
```

Replace `openai` with the extra for your model provider if needed.

## Set up GitHub

Create a fine-grained GitHub personal access token for the target repository with only the permissions the agent
needs, then set these environment variables:

```bash
export GITHUB_TOKEN='your-token'
export GITHUB_REPOSITORY='pydantic/pydantic-ai'
export OPENAI_API_KEY='your-model-key'
```

The example uses the token directly and does not open a browser. For OAuth, the application hosting MCP must complete
GitHub's OAuth 2.1 flow and pass a configured MCP client through `client=`. The host may open a browser during that
flow; this integration does not start it.

## Review a pull request

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.github import GitHub

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions='Review the requested pull request. Cite file paths and line numbers for every finding.',
    capabilities=[
        GitHub(
            repository=os.environ['GITHUB_REPOSITORY'],
            auth=os.environ['GITHUB_TOKEN'],
            toolsets=('repos', 'pull_requests'),
        )
    ],
)

result = agent.run_sync('Review pull request #123')
print(result.output)
```

The repository also includes this as [`examples/github_pr_review.py`](../examples/github_pr_review.py).

## What to ask

- Read files, commits, branches, releases, issues, and pull requests.
- Search code, commits, issues, and pull requests within the configured scope.
- Review a pull request and cite relevant files.
- With write access, create branches; create or update issues, files, and pull requests.

## Operational constraints

- Pass exactly one of `repository='owner/repo'` or `organization='owner'`. Organization scope permits repository tools
  only when the repository owner matches that organization.
- `access='read'` is the default. The built-in remote transport sends GitHub's `X-MCP-Readonly: true` header, and the
  integration also hides tools without an explicit true MCP `readOnlyHint`.
- Set `access='write'` to expose mutations and configure the agent with
  `output_type=[str, DeferredToolRequests]`. Mutations then return `DeferredToolRequests` until the caller approves or
  denies each tool-call ID and resumes with `DeferredToolResults`. Set `require_approval=False` only if the application
  enforces an equivalent approval policy.
- Supported `toolsets` are `repos`, `issues`, and `pull_requests`. Searches containing the uppercase token `OR`,
  including quoted literal uses, are rejected. `repo:`, `org:`, or `user:` qualifiers that do not exactly match the
  configured scope are also rejected, even when they appear as literal text. Tools with an opaque secondary target are
  hidden; named secondary targets outside the scope are rejected.
- The PAT or OAuth access token, together with GitHub organization policy, remains GitHub's authorization boundary.
  Scope checks do not provide strict response isolation: GitHub can include linked public resources or other resources
  the token can read. Restricting the token to the same target and permissions reduces private-data exposure.
- The default endpoint is GitHub's hosted MCP server at `https://api.githubcopilot.com/mcp/`. GitHub Enterprise Cloud
  with data residency can use its tenant `copilot-api` URL through `url=`. GitHub Enterprise Server requires a
  caller-configured local MCP client because GitHub does not host the remote server there.

## API reference

::: pydantic_ai_harness.github.GitHub
