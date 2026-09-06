# Slack

Give an agent access to Slack, or let people message it in Slack.

- [Run an agent in Slack](#run-an-agent-in-slack): reply to direct messages and channel mentions.
- [Use Slack from Python](#use-slack-from-python): read Slack from your own script or application.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](../../docs/index.md#version-policy).

## Run an agent in Slack

You need an OpenAI API key and permission to create a Slack app in your workspace.
Slack currently allows its AI tools only for internal apps and apps published in
its Marketplace. See [Slack's requirements](https://docs.slack.dev/ai/slack-mcp-server/#app-identity).

### 1. Install

```bash
uv init slack-agent
cd slack-agent
uv add "pydantic-ai-harness[slack]" "pydantic-ai-slim[openai]" uvicorn
```

Download [slack_agent.py](https://github.com/pydantic/pydantic-ai-harness/blob/main/examples/slack_agent.py)
and save it in this directory as `slack_agent.py`.

### 2. Create your Slack app

Open [Your Apps](https://api.slack.com/apps), choose **Create New App**, then
**From scratch**. Name it `Pydantic AI Agent` and select your workspace.

In the app settings:

1. Under **Agents**, enable **Slack Model Context Protocol (MCP) Server**. This lets the agent use Slack's tools.
2. Under **OAuth & Permissions**, add these **Bot Token Scopes**:

   ```text
   app_mentions:read
   assistant:write
   channels:history
   chat:write
   groups:history
   im:history
   ```

3. Add these **User Token Scopes** so the agent can find and read conversations:

   ```text
   channels:history
   groups:history
   im:history
   search:read.public
   search:read.private
   ```

4. Under **App Home**, enable the **Messages** tab and allow users to send messages.
5. Under **Basic Information**, find the **Client ID**, **Client Secret**, and **Signing Secret** for the next step.

For other actions, such as sending messages as the user or reading files, add the
[corresponding permissions](https://docs.slack.dev/ai/slack-mcp-server/#oauth-scopes-needed-on-user-token-for-different-tools).
Users must authorize again after you change permissions.

### 3. Start the server

Replace the values below. `agent.example` must be your server's public HTTPS address.
For local testing, use an HTTPS tunnel to port `8000`, such as
[ngrok](https://docs.slack.dev/ai/slack-mcp-server/developing/#add-a-redirect-url).

```bash
export OPENAI_API_KEY='your-openai-api-key'
export SLACK_CLIENT_ID='your-client-id'
export SLACK_CLIENT_SECRET='your-client-secret'
export SLACK_SIGNING_SECRET='your-signing-secret'
export SLACK_USER_SCOPES='channels:history,groups:history,im:history,search:read.public,search:read.private'
export SLACK_REDIRECT_URI='https://agent.example/slack/oauth_redirect'
export SLACK_INSTALLATION_DIR="$PWD/slack-data"
```

In Slack's **OAuth & Permissions**, add the `SLACK_REDIRECT_URI` value under
**Redirect URLs** and save it. Then start the server:

```bash
uv run uvicorn slack_agent:build_app --factory --host 0.0.0.0 --port 8000 --workers 1
```

The example creates a private `slack-data` directory for Slack credentials.
Keep it out of source control.

### 4. Connect Slack

Forward your public HTTPS address to port `8000` on the running server.
Keep these three paths available:

| Path | Purpose |
| --- | --- |
| `/slack/events` | Receive messages from Slack |
| `/slack/install` | Let each user connect their account |
| `/slack/oauth_redirect` | Finish connecting the account |

In Slack's **Event Subscriptions**:

1. Enable events.
2. Set **Request URL** to `https://agent.example/slack/events` and wait for **Verified**.
3. Under **Subscribe to bot events**, add `app_mention`, `message.channels`, `message.groups`, and `message.im`.
4. Save changes.

Open `https://agent.example/slack/install` and approve the requested permissions.
Share that link with everyone who will use the agent; each person must connect
their own account. Put the link in the app description so users can find it later.

Send the app a direct message. To use it in a channel, invite the app to the
channel and mention it, for example: `@Pydantic AI Agent summarize this thread`.

### 5. Keep it running

Deploy `slack_agent.py`, `pyproject.toml`, and `uv.lock` to a server or container
host with a public HTTPS address and a persistent disk.

- Install with `uv sync --locked` and use the same start command above.
- Set the environment variables in your host's settings.
- Point `SLACK_INSTALLATION_DIR` to a private directory on the persistent disk. It stores connected accounts and pending sign-ins.
- Run one instance with one worker. Restarting clears which channel threads the agent follows; mention it again to resume.
- If the public address changes, update both Slack URLs and `SLACK_REDIRECT_URI`.

## Use Slack from Python

Install `pydantic-ai-harness[slack]` and `pydantic-ai-slim[openai]` as above.
Use a **User OAuth Token**, not a bot token. For your own internal app, enable
MCP and add the user permissions in step 2, then install the app from
**OAuth & Permissions** and copy its **User OAuth Token**.

```bash
export OPENAI_API_KEY='your-openai-api-key'
export SLACK_USER_TOKEN='your-user-oauth-token'
```

Save this as `ask_slack.py`:

```python {names="defined"}
import os

from pydantic_ai import Agent
from pydantic_ai_harness.slack import Slack

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[Slack(token=os.environ['SLACK_USER_TOKEN'])],
)
print(agent.run_sync('Summarize the recent discussion in #support').output)
```

```bash
uv run python ask_slack.py
```

Replace `#support` with a channel you can access. This path does not need the
Slack server or `register_slack`.

## Customize your agent

Edit `build_agent()` in `slack_agent.py` to change its instructions, model, or tools.
Keep its output as text. `register_slack(bolt, agent)` adds Slack access using
whoever sent the message; do not also add `Slack` to that agent, directly or through a factory.

If your code needs to know who sent the message, use `current_slack_context()`
during the run. To build your own dependencies, pass
`deps_factory` to `register_slack`; it can be a normal or an `async` function
and receives a `SlackContext` containing the user, workspace, channel, and message IDs.

For a Python application with multiple users, create `Slack(token=...)` for each
run using [Pydantic AI's capability factories](https://pydantic.dev/docs/ai/capabilities/overview/).

## What to expect

- Direct messages get direct replies. Replies to a thread stay in that thread.
- After a channel mention, the agent answers other people's replies in that thread too. It remembers up to 1,024 recently active threads; older ones need another mention. There is no stop-following command.
- Each message starts a fresh agent run. Earlier discussion can be read from Slack, but model chat history is not saved.
- If you add memory, it is separate for each person in a thread. Unthreaded direct messages do not share it.
- Group direct messages and messages from external Slack Connect users are ignored.
- Slack tools use the sender's permissions. Review what your agent may share back into a channel.
- Replies arrive after the run finishes. Duplicate Slack deliveries can trigger duplicate runs; interrupted runs are not resumed.

With Pydantic AI core 2.38.0, cancelling a run during a Slack tool call can leave
its connection open. This known core issue still needs a fix before release.
