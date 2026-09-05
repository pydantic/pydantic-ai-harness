# Atlassian

Give an agent site-scoped access to Jira, with optional Confluence, Jira Service Management, and Bitbucket tools.

## Installation

```bash
uv add "pydantic-ai-harness[atlassian]" "pydantic-ai-slim[openai]"
```

## Setup

Your organization needs a paid Jira, Confluence, Service Collection, or Teamwork Collection Cloud subscription and a
verified business domain. An organization administrator must enable Rovo, allow Rovo MCP, and give the account you sign
in with access to the site and data you want the agent to use.

Find the site's `cloudId` by opening `https://<your-site>.atlassian.net/_edge/tenant_info`, then set it with your model
credential:

```bash
export ATLASSIAN_CLOUD_ID='the-cloudId-from-tenant_info'
export OPENAI_API_KEY='your-openai-api-key'
```

Pydantic AI starts Atlassian's OAuth 2.1 flow through its MCP client. The first run opens a browser for Atlassian
sign-in and site consent. The default OAuth token store is in memory, so a new process may ask you to sign in again.

## Example

```python
import os

from pydantic_ai import Agent
from pydantic_ai_harness.atlassian import Atlassian

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[Atlassian(cloud_id=os.environ['ATLASSIAN_CLOUD_ID'])],
)

result = agent.run_sync('Summarize my unresolved Jira work and identify the oldest item')
print(result.output)
```

## What the agent can do

- Read and search Jira work items, projects, boards, sprints, comments, transitions, and versions.
- Read selected Confluence pages, spaces, comments, attachments, tasks, and permissions when `products` includes
  `'confluence'`.
- Read selected Jira Service Management alerts, schedules, and teams when `products` includes
  `'jira_service_management'`.
- Read selected Bitbucket repositories, branches, commits, pull requests, and pipelines when `products` includes
  `'bitbucket'` and the workspace is linked to an Atlassian organization.
- Create or update selected Jira, Confluence, JSM, and Bitbucket records after you set `access='read_write'` on
  `Atlassian` and approve each requested write.

## Operational constraints

- The default is Jira-only and exposes reviewed read and search tools. Set `products=(...)` to add the other product
  families.
- `Atlassian` requires `access='read_write'` for writes and `access='destructive'` for Jira permanent deletes, then
  requests approval for each mutation by default. `require_approval=False` is for callers that supply another approval
  policy or intentionally allow unattended changes.
- Direct `AtlassianToolset` use controls tool exposure and site scope but does not request approval. Wrap it with
  `.approval_required()` when mutations need approval.
- Every product call must use the configured `cloud_id`; Harness checks it before the request reaches Atlassian.
- OAuth consent is site-scoped. API-token credentials are not site-scoped by Atlassian, so the `cloud_id` check remains
  important. Use `authorization_token=` only for an Atlassian service-account API key sent as a Bearer token. Personal
  API tokens require Basic authentication on a preconfigured FastMCP client passed with `client=`. JSM tools require
  one of these API-token mechanisms and do not support OAuth 2.1.
- In a multi-user service, create a separate `Atlassian` capability or `AtlassianToolset` for each user. Do not share an
  authenticated MCP client between users.
- Atlassian organization permission groups and the signed-in account's product permissions can further reduce the
  available tools.
- For two sites on one agent, wrap each `Atlassian` capability in Pydantic AI's `PrefixTools` so their tool names do not
  collide.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/atlassian/)

See Atlassian's [Rovo MCP tools](https://support.atlassian.com/atlassian-ai-gateway/docs/supported-tools/),
[OAuth setup](https://support.atlassian.com/atlassian-ai-gateway/docs/configure-oauth-2-1/), and
[API-token setup](https://support.atlassian.com/atlassian-ai-gateway/docs/configure-authentication-via-api-token/).

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).
