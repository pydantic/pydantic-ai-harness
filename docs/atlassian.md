# Atlassian

Use Jira, Confluence, and other Atlassian tools across your accessible sites. `Atlassian` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[atlassian]" "pydantic-ai-slim[openai]"
```

Set `ATLASSIAN_API_KEY` to an Atlassian service-account API key, or pass `auth=...`. When neither is supplied, the connection starts browser OAuth. `auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://support.atlassian.com/atlassian-ai-gateway/docs/get-started-with-the-atlassian-remote-mcp-server/).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.atlassian import Atlassian

agent = Agent('openai:gpt-5.6-sol', capabilities=[Atlassian()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

The connection uses `https://mcp.atlassian.com/v2/mcp?tools=all`, Atlassian's flat tool catalog. Existing user permissions and organization settings determine access to sites and products. A personal API token uses `auth=httpx.BasicAuth(email, token)`; the environment variable is for a service-account bearer key. V2 OAuth requires a new sign-in when migrating from V1. See [token authentication](https://support.atlassian.com/atlassian-ai-gateway/docs/configure-authentication-via-api-token/).

## Tool selection and approval

For application-level filtering or approval, compose the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.atlassian import Atlassian

capability = Atlassian()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](tool-output-limits.md).

## Connection customization

Pass `client` to use a configured FastMCP client or transport, including custom OAuth token storage and MCP handlers. That client owns its URL, authentication, and server configuration; configure those on it instead of the capability.

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/atlassian/)
