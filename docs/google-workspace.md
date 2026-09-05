# Google Workspace

Google Workspace lets an agent read selected Gmail, Calendar, Drive, Docs, Sheets, Slides, Chat, and People data, with explicit opt-in for changes.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/google_workspace/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install

```bash
uv add "pydantic-ai-harness[google-workspace]" "pydantic-ai-slim[openai]"
```

## Provider setup

In Google Cloud:

1. Join the Google Workspace Developer Preview Program, then enable the Workspace API and MCP service for each product you will select.
2. If you select Chat, configure the Google Chat API app and turn off **Enable interactive features**.
3. Use your application's OAuth flow to obtain a bearer token with the read-only scopes for each selected product. Add mutation scopes only when using `read_only=False`. Each value below starts with `https://www.googleapis.com/auth/`.

| Product | Read-only scope suffixes | Additional mutation scope suffixes |
|---|---|---|
| Gmail | `gmail.readonly` | `gmail.compose`, `gmail.modify` |
| Drive | `drive.readonly` | `drive.file` |
| Docs | `drive.readonly`, `documents.readonly` | `drive.file`, `documents` |
| Sheets | `drive.readonly`, `spreadsheets.readonly` | `drive.file`, `spreadsheets` |
| Slides | `drive.readonly`, `presentations.readonly` | `drive.file`, `presentations` |
| Calendar | `calendar.calendarlist.readonly`, `calendar.events.freebusy`, `calendar.events.readonly` | `calendar.events` |
| Chat | `chat.spaces.readonly`, `chat.memberships.readonly`, `chat.messages.readonly` | `chat.messages.create`, `chat.users.readstate` |
| People | `directory.readonly`, `userinfo.profile`, `contacts.readonly` | None |

Set these environment variables:

- `GOOGLE_ACCESS_TOKEN`: a caller-managed Google OAuth bearer token.
- `OPENAI_API_KEY`: the key used by the example model.

The capability does not open a browser, mint tokens, or refresh tokens. Your application owns OAuth, including its pre-registered client, redirect URI, scope ceiling, storage, and refresh policy.

Google requires screening prompts sent to and responses returned from Workspace MCP servers for malicious content and prompt injection. Use [Model Armor or another documented screening solution accepted by your users](https://developers.google.com/workspace/guides/configure-mcp-security). Static `instructions=` text is model guidance, not a screening control. When Model Armor logging is enabled, it logs the entire payload, which can expose sensitive content in logs.

## Example

Save this as `workspace_agent.py` after setting the two environment variables above:

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness.google_workspace import GoogleWorkspace

agent = Agent('openai:gpt-5.6-sol', capabilities=[GoogleWorkspace()])


async def main() -> None:
    result = await agent.run('Summarize unread project mail and list my meetings today.')
    print(result.output)


asyncio.run(main())
```

Run it with `uv run python workspace_agent.py`.

## What you can ask

- "Summarize unread project mail and list my meetings today."
- With Drive, Docs, and Sheets selected: "Find the launch plan in Drive, read the linked Doc, and show the budget values from its Sheet."
- With Slides and Drive selected: "Read the quarterly Slides presentation and list the files I can share with the team."
- With Chat and People selected: "Search Chat for the release decision and find the email address for its author in People."

## Operational constraints

- Gmail and Calendar are selected by default. Pass `services=('drive', 'docs', 'sheets', 'slides', 'chat', 'people')` to select other products.
- Tool names are prefixed by service. Use `allowed_tools=('gmail_search_threads', 'calendar_list_events')` for an exact allowlist.
- `read_only=True` is the default. It filters exposed tools, but it does not reduce the bearer token's scopes. Issue the token with the narrowest scopes your application permits. `read_only=False` requires an explicit `allowed_tools` list. To require approval for every exposed operation, including writes, create the workspace with `read_only=False` and an `allowed_tools` list, then pass `workspace.get_toolset().approval_required()` through `toolsets`, `workspace.get_instructions()` through `instructions`, and include `DeferredToolRequests` in `output_type`. Follow the [deferred tools guide](/ai/deferred-tools/) to approve and resume the run.
- Pass a token with `GOOGLE_ACCESS_TOKEN` or `access_token=`. Hosted applications can instead pass a caller-owned MCP client or `MCPToolset` for each selected service through `clients=`. One MCP client or `MCPToolset` represents one authenticated identity. For concurrent runs, use `@agent.toolset(per_run_step=False)` with credentials or clients from `deps` to create a fresh `GoogleWorkspace` and toolset per run. Also pass the text returned by a configured workspace's `get_instructions()` as Agent instructions. See [per-user authentication](/ai/mcp/client/#per-user-authentication). If one run uses multiple identities for the same service, wrap each workspace toolset in an outer `.prefixed('alice')` or `.prefixed('bob')` label because the service prefixes alone collide.
- Automatic local OAuth is unavailable because the upstream MCP client currently replaces an explicit scope ceiling during discovery. Track [modelcontextprotocol/python-sdk#2317](https://github.com/modelcontextprotocol/python-sdk/issues/2317).
- Workspace content can contain instructions aimed at the model. Keep mutation tools narrow and review proposed changes before approval.

Google's [Workspace MCP configuration guide](https://developers.google.com/workspace/guides/configure-mcp-servers) lists the provider setup and current tool catalog.

## API reference

::: pydantic_ai_harness.google_workspace.GoogleWorkspace
