# Slack

Put a Pydantic AI agent in a Slack thread. `SlackChat` lets the agent post
progress, show a checklist, ask a question, and send files while it works, and
tells it when to do each. `SlackApprovals` stops a dangerous tool until a person
clicks Approve. `SlackBot` wires both to a working bot in a few lines, over
whichever transport suits where you deploy it.

Slack is a front door here, not the agent's only reach. The same agent takes
whatever other toolsets you give it, so it can read a Linear ticket, open a pull
request, and report back in the thread it was asked in.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

## Installation

```bash
uv add "pydantic-ai-harness[slack,anthropic]"
```

The `slack` extra pulls in `slack-sdk` and `slack-bolt`. Only `SlackBot`
needs Bolt, and it lives in its own module, so importing anything else from
this package does not require it.

## Quick start

```python {title="slack_agent.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls

from pydantic_ai_harness.slack import (
    FileConversationStore,
    SlackBot,
    SlackApprovals,
    SlackChat,
    SlackInteractions,
    SlackThread,
)

interactions = SlackInteractions()

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    deps_type=SlackThread,
    capabilities=[
        SlackChat(interactions=interactions, file_root='./workspace'),
        HandleDeferredToolCalls(handler=SlackApprovals(interactions)),
    ],
)


@agent.tool_plain(requires_approval=True)
def merge_pull_request(number: int) -> str:
    """Merge a pull request. Asks a person in Slack first."""
    return f'Merged #{number}'


SlackBot(
    agent,
    interactions=interactions,
    store=FileConversationStore('~/.slack-agent'),
    allowed_user_ids=['U01ABC2DEF3'],
).run()
```

Set `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`, run the script, invite the bot to a
channel, and mention it.

## How Slack reaches the bot

Two transports, and only the last line of your program differs.

```python {test="skip"}
bot.run()                                   # Socket Mode
app.mount('/slack/events', bot.http_app())  # Events API
```

| | Socket Mode | Events API |
| --- | --- | --- |
| Needs | `SLACK_APP_TOKEN` | `SLACK_SIGNING_SECRET` and a public HTTPS URL |
| Connection | your process dials out | Slack posts to you |
| Suits | a laptop, a container, anything that stays up | Lambda, Cloud Run, anything that scales to zero |

`http_app()` returns an ASGI app, so it mounts on FastAPI, Starlette, or
anything else ASGI. Bolt checks the signature on every request and answers
Slack's setup challenge, and it replies before running the agent, so Slack's
three-second deadline is met however long the turn takes. Give Slack the URL you
mounted it at as the **Request URL**, and turn Socket Mode off in the manifest.

A redelivered event is ignored rather than run twice, which for an agent with
write access is the difference between one pull request and two.

Neither credential is needed to construct a `SlackBot`. `run()` asks for the app
token and `http_app()` asks for the signing secret, so you only configure the one
you use.

## Setting up the Slack app

1. Go to <https://api.slack.com/apps> and create an app from the manifest below.
2. Under **Basic Information -> App-Level Tokens**, generate a token with the
   `connections:write` scope. That is `SLACK_APP_TOKEN` (`xapp-`).
3. Under **Install App**, install to the workspace and copy the Bot User OAuth
   Token. That is `SLACK_BOT_TOKEN` (`xoxb-`).
4. Invite the bot to a channel with `/invite @your-bot`, then mention it.

```json
{
  "display_information": { "name": "My Agent" },
  "features": {
    "bot_user": { "display_name": "My Agent", "always_online": true },
    "agent_view": { "agent_description": "Does the work, in the thread" },
    "app_home": { "messages_tab_enabled": true, "messages_tab_read_only_enabled": false }
  },
  "oauth_config": {
    "scopes": {
      "bot": [
        "app_mentions:read",
        "assistant:write",
        "channels:history",
        "chat:write",
        "files:write",
        "groups:history",
        "im:history",
        "im:read",
        "im:write",
        "mpim:history"
      ]
    }
  },
  "settings": {
    "event_subscriptions": {
      "bot_events": ["app_mention", "message.im", "message.mpim", "message.channels", "message.groups"]
    },
    "interactivity": { "is_enabled": true },
    "socket_mode_enabled": true
  }
}
```

That manifest sets up Socket Mode, so Slack connects to you and this runs on a
laptop behind a firewall with no public URL. For the Events API instead, set
`socket_mode_enabled` to `false` and give `event_subscriptions` a `request_url`
pointing at wherever you mounted `http_app()`. `assistant:write` is what makes `set_status` show a
working-state line; without it that tool reports it is unavailable and the run
carries on. `app_home` is what lets people DM the bot: without
`messages_tab_enabled` Slack answers "Sending messages to this app has been
turned off". Use `agent_view` rather than the older `assistant_view`, which new
apps can no longer set.

## What the agent can do in the thread

`SlackChat` registers three tools always, and two more when configured:

| Tool | Registered | What it does |
| --- | --- | --- |
| `post_message` | always | Says something mid-run without ending the turn |
| `post_plan` | always | Posts a checklist, and edits it in place when given the `plan_id` it returned |
| `set_status` | always | Sets the short working-state line |
| `ask_user` | with `interactions=` | Asks a multiple-choice question and waits for a click |
| `upload_file` | with `file_root=` | Sends a file from that directory into the thread |

It also ships the instructions that make those tools worth having. A model with
`post_message` and no guidance says nothing until it finishes, which is the one
thing a Slack agent exists not to do. Pass `instructions=` to replace the
default, or `instructions=''` to add none and say it yourself.

`SlackChatToolset` is the same tools without the guidance, for an agent whose
instructions already cover it.

`ask_user` and `upload_file` are opt-in on purpose. An agent that should never
block on a person should not be given a tool that blocks on a person, and there
is no safe directory to judge a model-supplied path against until you name one.

`post_plan` keeps no state of its own: it hands the model a `plan_id` and edits
that message when the model passes it back. The id is signed for the run that
posted it, so a model cannot edit another message whose timestamp it happened to
see, and a second turn in the thread posts a fresh checklist rather than
overwriting the one before it.

Paths outside `file_root` are refused, as are directories and paths that do not
exist. `post_message` refuses text over 3500 characters rather than truncating
it, because a reader cannot see what was cut.

## Approvals

`ask_user` is the model choosing to ask. Approvals are you deciding it does not
get a choice:

```python
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset


def merge_pull_request(number: int) -> str:
    return f'Merged #{number}'


github = ApprovalRequiredToolset(
    FunctionToolset([merge_pull_request]),
    approval_required_func=lambda ctx, tool_def, args: tool_def.name.startswith('merge'),
)
```

Every call the wrapper flags is posted as a question with Approve and Deny
buttons, and the run continues the moment someone answers. Only the person who
started the run can answer, unless you name a group:

```python
SlackApprovals(interactions, allowed_user_ids=['U01REVIEWER'])
```

A prompt nobody answers is denied, and so is a call that does not fit in the
2900 characters Slack can show in one block. Neither is truncated: approving half
a call is approving something nobody read. An agent with write access to real
systems should not act because a question timed out, or because the part that
mattered scrolled off the end. A tool that genuinely needs a large payload should
take a file path instead. The default wait is ten minutes;
change it with `SlackInteractions(timeout_seconds=...)`. Options are capped at
75 characters, Slack's own button limit, and are refused rather than shortened
so two buttons can never read the same.

Prompts for one thread are posted one at a time, so two questions never leave
competing sets of buttons in the same conversation.

## Conversation history

History is keyed by workspace, channel, and thread root, so each thread is its
own conversation and a reply picks up where the last turn left off.

`InMemoryConversationStore` is the default and is lost on restart.
`FileConversationStore` writes one JSON file per conversation, through a
temporary file it then moves into place, so a crash does not leave a truncated
history. It creates the directory and its files for the owner only, since they
hold whole conversations, but a directory that already exists keeps the
permissions it has. Point it at a private path: anyone who can write there can
put words in the agent's history. For anything shared between processes,
implement `ConversationStore` against your database.

## Building the app yourself

`SlackBot` is convenience, not a requirement. To keep control of the Bolt app
-- HTTP mode, OAuth across workspaces, your own listeners -- build it yourself
and call the same pieces:

```python {title="my_slack_app.py"}
from collections.abc import Mapping

from pydantic_ai import Agent
from slack_bolt.app.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from pydantic_ai_harness.slack import InMemoryConversationStore, SlackThread

app = AsyncApp()
store = InMemoryConversationStore()
agent = Agent('anthropic:claude-sonnet-4-6', deps_type=SlackThread)


@app.event('app_mention')
async def on_mention(
    event: Mapping[str, object], client: AsyncWebClient, context: Mapping[str, object]
) -> None:
    thread = SlackThread(
        client=client,
        channel_id=str(event['channel']),
        thread_ts=str(event.get('thread_ts') or event['ts']),
        user_id=str(event['user']),
        team_id=str(context['team_id']),
    )
    history = await store.load(thread.key)
    result = await agent.run(
        str(event['text']),
        deps=thread,
        conversation_id=thread.key,
        message_history=list(history) or None,
    )
    await client.chat_postMessage(
        channel=thread.channel_id, thread_ts=thread.thread_ts, text=result.output
    )
    await store.save(thread.key, result.all_messages())
```

`team_id` comes off the listener's `context` rather than the event body, which is
where Bolt puts it reliably. Post before you save, so a failed post does not leave
the next turn building on a reply nobody saw.

Route button clicks back with `SlackInteractions.resolve(block_id=..., value=...,
user_id=...)`, reading all three off the Slack action payload. Every button these
prompts post carries an `action_id` starting with `PROMPT_ACTION_PREFIX`, which
is what your action listener should match on.

## Who can use the bot

`allowed_user_ids` is checked before a run starts. Without it, anyone who can
reach the bot can spend its tokens and invoke its tools, and a warning says so at
startup. Set it, or set `SLACK_ALLOWED_USER_IDS` to a comma-separated list.

This is separate from who may approve. A person can be allowed to ask the agent
for things without being allowed to approve a production deploy.

## Known limits

- The agent's `deps` type must be `SlackThread`. Composing it into a larger deps
  object is not supported yet.
- A follow-up message in a thread waits for the current run to finish rather
  than interrupting it, and there is no stop button.
- Replies longer than 3500 characters are split at that boundary, without regard
  for code fences.
- Cancelling a run while a prompt is open leaves its buttons in the thread; they
  no longer resolve to anything.
- One `SlackBot` serves one workspace.
