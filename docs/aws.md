---
title: AWS
description: Connect a Pydantic AI agent to the managed AWS MCP Server with explicit account, Region, and access scope.
---

# AWS

`AWS` lets an agent use AWS documentation and, with authentication, operate one real AWS account through the managed AWS MCP Server.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/aws/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[aws]" "pydantic-ai-slim[openai]"
```

## Provider setup

Public AWS knowledge needs no credentials, environment variables, or browser sign-in.

For the interactive OAuth flow shown here, attach the `AWSMCPSignInOAuthAccessPolicy` managed policy, or a custom
policy that grants `signin:AuthorizeOAuth2Access` and `signin:CreateOAuth2Token`, to the IAM user or role that will sign
in. Configure a FastMCP
`StreamableHttpTransport('https://aws-mcp.us-east-1.api.aws/mcp', auth='oauth')`, then pass it as `managed_transport` with
`authentication='oauth'`. FastMCP handles OAuth and opens a browser on the first request. Configure encrypted token
storage for persistent or multi-user applications. This interactive flow does not use AWS access-key environment
variables. Other caller-owned OAuth transports retain their own credential, token, and browser lifecycle.

For SigV4, install AWS CLI 2.32.0 or later. Before using `aws login`, attach the
`SignInLocalDevelopmentAccess` managed policy to the IAM user, role, or group. Do not use root credentials for agent
operations; use a least-privilege IAM role or user instead.
The command opens the default browser. On a headless host, use `aws login --remote` and finish sign-in on a
browser-enabled device. Then authenticate and verify the selected identity:

```bash
aws login
export AWS_PROFILE=default
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export AWS_REGION=us-west-2
export OPENAI_API_KEY='<your OpenAI API key>'
```

The MCP Proxy for AWS handles the AWS credential chain, refresh, and SigV4 request signing. Pass its
`StdioTransport` as `managed_transport` with `authentication='sigv4'`. The example passes the selected profile and
operation Region to the proxy. Replace the `OPENAI_API_KEY` value before running it.

## Example

This example asks for an account inventory and requires approval before any non-read tool runs:

```python
import json
import os

from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai_harness.aws import AWS

account_id = os.environ['AWS_ACCOUNT_ID']
region = os.environ['AWS_REGION']
transport = StdioTransport(
    command='uvx',
    args=[
        'mcp-proxy-for-aws-cli==1.6.5',
        'https://aws-mcp.us-east-1.api.aws/mcp',
        '--profile',
        os.environ['AWS_PROFILE'],
        '--metadata',
        f'AWS_REGION={region}',
    ],
    keep_alive=False,
)
agent = Agent(
    'openai:gpt-5.6-sol',
    output_type=[str, DeferredToolRequests],
    capabilities=[
        AWS(
            account_id=account_id,
            region=region,
            authentication='sigv4',
            managed_transport=transport,
            access='approval_required',
        )
    ],
)

result = agent.run_sync('List the S3 buckets in this account and report each bucket Region.')
if isinstance(result.output, DeferredToolRequests):
    approvals = {}
    for call in result.output.approvals:
        details = json.dumps(call.args, indent=2, sort_keys=True)
        prompt = f'Approve {call.tool_name} for AWS account {account_id} in {region}?\n{details}\n[y/N] '
        approvals[call.tool_call_id] = True if input(prompt) == 'y' else ToolDenied('denied')
    result = agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals=approvals),
    )
print(result.output)
```

## What the agent can do

- Search and read current AWS documentation.
- List AWS Regions and check regional service or feature availability.
- With OAuth or SigV4, inspect resources and run AWS API workflows allowed by IAM.
- With authenticated access, generate Amazon S3 presigned upload or download URLs and poll long-running tasks.

## Operational constraints

- `account_id` and `region` declare the model-facing scope. IAM and the authenticated transport enforce actual access.
- The managed endpoints are `us-east-1` and `eu-central-1`. `endpoint_region` selects the unauthenticated endpoint;
  an authenticated transport selects its endpoint in its URL or proxy arguments.
- `access='read_only'` is the default. It hides every tool not explicitly marked `readOnlyHint=true`.
- `access='approval_required'` exposes non-read tools through Pydantic AI's deferred approval flow. Denied calls do not
  run. Each new non-read call, including a model-initiated retry, requires approval.
- `access='unrestricted'` removes the Harness approval gate; IAM still applies.
- Managed tool names repeat across scopes. Wrap each `AWS` instance in Pydantic AI's `PrefixTools` with a unique prefix
  when one agent uses multiple accounts or target Regions.
- `max_output_bytes` and `max_output_lines` cap each managed tool result before it enters model context and history.
  FastMCP still receives the full response. Ask for a narrower or paginated result when the response is truncated.
- Presigned URLs are temporary bearer credentials that enter model context and message history. Use short expiries,
  redact them from logs and traces, and avoid generating them in conversations whose history is retained.
- The caller owns transport timeout, retry, cancellation, and cleanup behavior. Cancellation does not roll back an AWS
  side effect. The single-run example uses `keep_alive=False` so the proxy exits after the toolset disconnects.
- A supplied transport must point to the managed AWS MCP Server. Use [LocalStack](localstack.md) for emulated AWS.
- Without SigV4 `AWS_REGION` metadata, AWS operations default to `us-east-1`.
- The server is GA with no additional service fee. Normal AWS resource and data-transfer charges still apply.
- Agent specs support only the unauthenticated knowledge path because authenticated transports carry runtime identity.

## API reference

::: pydantic_ai_harness.aws.AWS
