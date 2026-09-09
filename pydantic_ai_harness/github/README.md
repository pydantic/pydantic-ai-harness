# GitHub

Read and change GitHub repositories, issues, pull requests, and other accessible resources. `GitHub` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[github]" "pydantic-ai-slim[openai]"
```

Set `GITHUB_TOKEN` to a GitHub personal access token, or pass `auth=...`. `auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.github import GitHub

agent = Agent('openai:gpt-5.6-sol', capabilities=[GitHub()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

`toolsets=['repos', 'issues', 'actions']` sends GitHub's native `X-MCP-Toolsets` header. Omit it to keep server defaults. `read_only=True` sends `X-MCP-Readonly: true`. `url` can select a GitHub Enterprise Cloud endpoint. Configure repository access through the token or GitHub App permissions; this capability does not interpret search syntax or enforce a repository boundary. For additional headers or host-configured OAuth, pass a configured MCP client.

## Tool selection and approval

For application-level filtering or approval, compose the existing [toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.github import GitHub

capability = GitHub()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

Pass `client` to use a configured FastMCP client or transport, including custom OAuth token storage and MCP handlers. That client owns its URL, authentication, and server configuration; configure those on it instead of the capability. With a custom client, `read_only=True` filters annotations rather than configuring the remote server.

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/github/)
