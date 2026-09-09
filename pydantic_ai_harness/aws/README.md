# AWS

Use AWS knowledge and account tools through its managed MCP server. `AWS` connects an agent to the provider's hosted MCP server. By default it exposes the tools the server offers, including write tools. Provider credentials and server settings determine what those tools may access.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Install and connect

```bash
uv add "pydantic-ai-harness[aws]" "pydantic-ai-slim[openai]"
```

`auth` accepts an `httpx.Auth` for caller-managed authentication. See the [provider setup](https://docs.aws.amazon.com/agent-toolkit/latest/userguide/getting-started-aws-mcp-server.html).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.aws import AWS

agent = Agent('openai:gpt-5.6-sol', capabilities=[AWS()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Provider settings

`AWS()` uses browser OAuth with the Virginia endpoint. `region='eu-central-1'` selects the Frankfurt MCP endpoint; it does not constrain the regions used by tool calls. Existing IAM permissions determine access. OAuth requires the AWS sign-in permissions described in the [OAuth guide](https://docs.aws.amazon.com/agent-toolkit/latest/userguide/oauth-authentication.html).

For SigV4 or a named AWS profile, configure the [official AWS MCP proxy](https://github.com/aws/mcp-proxy-for-aws) and pass its MCP transport through `client`. The proxy owns credential discovery and signing.

## Tool selection and approval

`read_only=True` keeps only tools explicitly marked `readOnlyHint: true`; unmarked tools are omitted. This can leave no tools when a server does not annotate its read operations. Credentials remain the access-control boundary.

For application-level filtering or approval, compose the existing [toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example, this requires approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.aws import AWS

capability = AWS()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the resulting requests using the [deferred tools workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). Output limits can be composed with [Tool Output Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

Pass `client` to use a configured FastMCP client or transport, including custom OAuth token storage and MCP handlers. That client owns its URL, authentication, and server configuration; configure those on it instead of the capability. `read_only=True` applies the same annotation filter to custom clients.

`include_instructions` controls whether server instructions reach the model. Keep authenticated connections separate for different users. To combine connections with overlapping tool names, give them distinct IDs and compose [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/aws/)
