# Slack

Put a Pydantic AI agent in a Slack thread. `SlackChatToolset` lets the agent
post progress, show a checklist, ask a question, and send files while it works.
`SlackApprovals` stops a dangerous tool until a person clicks Approve.
`SlackAgent` wires both to a Socket Mode app, so a working bot is a few lines
and no public web server.

Slack is a front door here, not the agent's only reach. The same agent takes
whatever other toolsets you give it, so it can read a Linear ticket, open a pull
request, and report back in the thread it was asked in.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)

## Installation

```bash
uv add "pydantic-ai-harness[slack,anthropic]"
```

The `slack` extra pulls in `slack-bolt`, which only `SlackAgent` needs.
Everything else in this package talks to Slack through a small protocol and has
no dependency beyond Pydantic AI, so you can use it with your own client.

## Quick start

```python {title="slack_agent.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls

from pydantic_ai_harness.slack import (
    FileConversationStore,
    SlackApprovals,
    SlackChatToolset,
    SlackInteractions,
    SlackThread,
)
from pydantic_ai_harness.slack.app import SlackAgent

interactions = SlackInteractions()

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    deps_type=SlackThread,
    instructions=(
        'You work in Slack. Post a plan before multi-step work and keep it updated. '
        'Keep replies short; put anything long in a file and send that instead.'
    ),
    toolsets=[SlackChatToolset(interactions=interactions, file_root='./workspace')],
    capabilities=[HandleDeferredToolCalls(handler=SlackApprovals(interactions))],
)


@agent.tool_plain(requires_approval=True)
def merge_pull_request(number: int) -> str:
    """Merge a pull request. Asks a person in Slack first."""
    return f'Merged #{number}'


SlackAgent(
    agent,
    interactions=interactions,
    store=FileConversationStore('~/.slack-agent'),
    allowed_user_ids=['U01ABC2DEF3'],
).run()
```

Set `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`, run the script, invite the bot to a
channel, and mention it.

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
    "assistant_view": { "assistant_description": "Does the work, in the thread" }
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

Socket Mode means Slack connects to you, so this runs on a laptop behind a
firewall with no public URL. `assistant:write` is what makes `set_status` show a
working-state line; without it that tool reports it is unavailable and the run
carries on.

## What the agent can do in the thread

`SlackChatToolset` registers three tools always, and two more when configured:

| Tool | Registered | What it does |
| --- | --- | --- |
| `post_message` | always | Says something mid-run without ending the turn |
| `post_plan` | always | Posts a checklist and edits the same message as steps progress |
| `set_status` | always | Sets the short working-state line |
| `ask_user` | with `interactions=` | Asks a multiple-choice question and waits for a click |
| `upload_file` | with `file_root=` | Sends a file from that directory into the thread |

`ask_user` and `upload_file` are opt-in on purpose. An agent that should never
block on a person should not be given a tool that blocks on a person, and there
is no safe directory to judge a model-supplied path against until you name one.

Paths outside `file_root` are refused, as are directories and paths that do not
exist. `post_message` refuses text over the limit rather than truncating it,
because a reader cannot see what was cut.

## Approvals

`ask_user` is the model choosing to ask. Approvals are you deciding it does not
get a choice:

```python {test="skip"}
from pydantic_ai.toolsets import ApprovalRequiredToolset

github = ApprovalRequiredToolset(
    my_github_toolset,
    approval_required_func=lambda ctx, tool_def, args: tool_def.name.startswith('merge'),
)
```

Every call the wrapper flags is posted as a question with Approve and Deny
buttons, and the run continues the moment someone answers. Only the person who
started the run can answer, unless you name a group:

```python {test="skip"}
SlackApprovals(interactions, allowed_user_ids=['U01REVIEWER'])
```

A prompt nobody answers is denied. An agent with write access to real systems
should not act because a question timed out. The default wait is ten minutes;
change it with `SlackInteractions(timeout_seconds=...)`.

Prompts for one thread are posted one at a time, so two questions never leave
competing sets of buttons in the same conversation.

## Conversation history

History is keyed by workspace, channel, and thread root, so each thread is its
own conversation and a reply picks up where the last turn left off.

`InMemoryConversationStore` is the default and is lost on restart.
`FileConversationStore` writes one JSON file per conversation, via a temporary
file and a rename so a crash does not leave a truncated history. For anything
shared between processes, implement `ConversationStore` against your database.

## Building the app yourself

`SlackAgent` is convenience, not a requirement. To keep control of the Bolt app
-- HTTP mode, OAuth across workspaces, your own listeners -- build it yourself
and call the same pieces:

```python {test="skip"}
from pydantic_ai_harness.slack import SlackThread, conversation_key

thread = SlackThread(
    client=client,
    channel_id=event['channel'],
    thread_ts=event.get('thread_ts') or event['ts'],
    user_id=event['user'],
    team_id=event.get('team'),
)
history = await store.load(thread.key)
result = await agent.run(text, deps=thread, conversation_id=thread.key, message_history=list(history) or None)
await store.save(thread.key, result.all_messages())
```

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
- One `SlackAgent` serves one workspace.
