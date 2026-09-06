# Slack

Put your existing Pydantic AI agent in Slack. People can send it direct messages
or mention it in a channel. The integration runs your agent and posts its answer.

Your agent keeps its model, instructions, and tools. For each message, it also
gets access to Slack using the sender's permissions.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](../../docs/index.md#version-policy).

## Connect your agent

Install in your existing Python project:

```bash
uv add "pydantic-ai-harness[slack]" uvicorn
```

`register_slack` connects your agent to your configured Slack app.
The complete setup is shown in [Serve it](#serve-it):

```python
from pydantic_ai_harness.slack import register_slack

register_slack(bolt, agent)
```

- `agent`: your Pydantic AI agent, returning text.
- `bolt`: an asynchronous Slack `AsyncApp`, configured with [Bolt](https://docs.slack.dev/tools/bolt-python/concepts/authenticating-oauth/), Slack's Python library for receiving messages and connecting user accounts.

Registration handles direct messages, channel mentions, and follow-up replies.
It supplies the sender's Slack tools automatically; do not also add a `Slack`
capability to this agent, directly or through a factory.

## Serve it

Add `slack_server.py` to your project. Replace `from my_app import agent` with
the import for your existing agent. If it needs dependencies, use
[`deps_factory`](#agents-with-dependencies) in the registration call:

```python {names="defined"}
import os
from pathlib import Path

from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_sdk.oauth.installation_store.file import FileInstallationStore
from slack_sdk.oauth.state_store import FileOAuthStateStore

from pydantic_ai_harness.slack import register_slack
from my_app import agent

storage = Path(os.environ['SLACK_INSTALLATION_DIR'])
storage.mkdir(mode=0o700, parents=True, exist_ok=True)
storage.chmod(0o700)
client_id = os.environ['SLACK_CLIENT_ID']

oauth = AsyncOAuthSettings(
    client_id=client_id,
    client_secret=os.environ['SLACK_CLIENT_SECRET'],
    redirect_uri=os.environ['SLACK_REDIRECT_URI'],
    scopes=['app_mentions:read', 'assistant:write', 'channels:history', 'chat:write', 'groups:history', 'im:history'],
    user_scopes=['channels:history', 'groups:history', 'im:history', 'search:read.public', 'search:read.private'],
    installation_store=FileInstallationStore(base_dir=str(storage), client_id=client_id),
    state_store=FileOAuthStateStore(
        expiration_seconds=600,
        base_dir=str(storage / 'oauth-state'),
        client_id=client_id,
    ),
)
bolt = AsyncApp(oauth_settings=oauth, signing_secret=os.environ['SLACK_SIGNING_SECRET'])
register_slack(bolt, agent)
app = AsyncSlackRequestHandler(bolt, path='/slack/events')
```

If you already have a configured Bolt app, use it directly. Only the last two
lines are needed to connect your agent and expose the web app.

Set these environment variables alongside your agent's existing configuration:

```bash
export SLACK_CLIENT_ID='your-client-id'
export SLACK_CLIENT_SECRET='your-client-secret'
export SLACK_SIGNING_SECRET='your-signing-secret'
export SLACK_REDIRECT_URI='https://your-domain/slack/oauth_redirect'
export SLACK_INSTALLATION_DIR="$PWD/slack-data"
```

Start your server:

```bash
uv run uvicorn slack_server:app --host 0.0.0.0 --port 8000 --workers 1
```

Deploy your project to a server or container host. Keep one instance running,
forward public HTTPS requests to port `8000`, and put `SLACK_INSTALLATION_DIR`
on persistent storage. That directory contains user credentials; keep it private
and out of source control.

## Connect Slack to your server

Create an app in [Slack's app settings](https://api.slack.com/apps), or use your
existing app. It must be an internal or Marketplace app with **Slack Model
Context Protocol (MCP) Server** enabled under **Agents**.
See [Slack's requirements](https://docs.slack.dev/ai/slack-mcp-server/#app-identity).

1. Copy **Client ID**, **Client Secret**, and **Signing Secret** from **Basic Information** into the environment variables above.
2. In **OAuth & Permissions**, add `scopes` from `slack_server.py` as **Bot Token Scopes** and `user_scopes` as **User Token Scopes**. Add `SLACK_REDIRECT_URI` under **Redirect URLs**.
3. With your server running, enable **Event Subscriptions**. Set the request URL to `https://your-domain/slack/events` and wait for **Verified**. Subscribe to `app_mention`, `message.channels`, `message.groups`, and `message.im`.
4. Enable the **Messages** tab in **App Home** and allow users to send messages.
5. Open `https://your-domain/slack/install` to connect your account. Share this link with everyone using the agent; each person must authorize it.

Keep `/slack/events`, `/slack/install`, and `/slack/oauth_redirect` available at
your public address. For local testing, you can use an
[HTTPS tunnel](https://docs.slack.dev/ai/slack-mcp-server/developing/#add-a-redirect-url).

Users can now message the app. In a channel, invite the app and mention it.
The permissions above allow reading conversations; add
[other permissions](https://docs.slack.dev/ai/slack-mcp-server/#oauth-scopes-needed-on-user-token-for-different-tools)
if your agent needs more Slack actions. Users must authorize again after a change.

## Agents with dependencies

If your agent uses `deps`, supply them for each message with `deps_factory`:

```python
from my_app import agent, Deps
from pydantic_ai_harness.slack import SlackContext, register_slack

def make_deps(context: SlackContext) -> Deps:
    return Deps(user_id=context.user_id)

register_slack(bolt, agent, deps_factory=make_deps)
```

Use your own dependency type and fields. The factory can be `async`, for example
to load an account from your database. `SlackContext` includes the workspace,
user, channel, message, and thread IDs. Tools can also read it during a run with
`current_slack_context()`.

## Conversation behavior

- Direct messages get direct replies; replies to a thread stay in that thread.
- After a channel mention, the agent answers subsequent human replies in that thread. It remembers up to 1,024 recently active threads. After eviction or a restart, mention it again. There is no stop-following command.
- Each message starts without saved model chat history. The agent can read earlier discussion from Slack.
- If you add memory, it is separate for each person in a thread. Unthreaded direct messages do not share it.
- Group direct messages and messages from external Slack Connect users are ignored.
- Durable execution capabilities are not supported.
- Replies arrive when the run finishes. Duplicate deliveries can trigger duplicate runs; interrupted runs are not resumed.
- Slack access follows the sender's permissions. Your agent remains responsible for what it shares in a channel.

With Pydantic AI core 2.38.0, cancelling a run during a Slack tool call can leave
its connection open. This known core issue still needs a fix before release.

## Use Slack tools without hosting

`Slack(token=user_token)` gives an agent Slack tools without receiving messages
from Slack. Use this only when you need tool access in your own application:

```python
from pydantic_ai_harness.slack import Slack

result = agent.run_sync('Read the discussion in #support', capabilities=[Slack(token=user_token)])
```

Use a user OAuth token from an MCP-enabled Slack app, not a bot token.
