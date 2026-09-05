"""A ready-made Slack app that serves a Pydantic AI agent, over Socket Mode or HTTP."""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Collection, Mapping
from typing import Generic, Protocol, TypeVar
from weakref import WeakValueDictionary

import anyio
from pydantic import AliasPath, BaseModel, Field, TypeAdapter, ValidationError
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_harness.slack._capability import SlackChat
from pydantic_ai_harness.slack._client import SlackClient
from pydantic_ai_harness.slack._interactions import PROMPT_ACTION_PREFIX
from pydantic_ai_harness.slack._store import ConversationStore, InMemoryConversationStore
from pydantic_ai_harness.slack._thread import SlackThread, bind_thread
from pydantic_ai_harness.slack._toolset import MAX_MESSAGE_CHARS

try:
    from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_bolt.app.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'slack-bolt is required for SlackBot. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error

logger = logging.getLogger(__name__)

BotDepsT = TypeVar('BotDepsT')
"""The deps type the served agent takes. Its own variable because `BotDepsT` is
contravariant, and `SlackBot` builds deps as well as passing them."""


class _Ack(Protocol):
    async def __call__(self) -> object: ...  # pragma: no cover


def _string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


class _PromptClick(BaseModel):
    """The three fields that identify a button click, pulled out of Slack's payload.

    Slack nests them, and everything else in the payload is ignored. Declaring
    the shape means a payload that is not a prompt click fails validation rather
    than being routed with a field missing.
    """

    block_id: str = Field(validation_alias=AliasPath('actions', 0, 'block_id'))
    value: str = Field(validation_alias=AliasPath('actions', 0, 'value'))
    user_id: str = Field(validation_alias=AliasPath('user', 'id'))


_CLICK_ADAPTER = TypeAdapter(_PromptClick)


def _prompt_click(body: Mapping[str, object]) -> _PromptClick | None:
    """The click this payload describes, or `None` when it does not describe one."""
    try:
        return _CLICK_ADAPTER.validate_python(body)
    except ValidationError:
        return None


DEFAULT_ERROR_REPLY = 'Something went wrong handling that message. The details are in the logs.'

_REMEMBERED_EVENTS = 1000
"""How many delivered event ids to remember, so a Slack retry does not start a
second run. Slack retries three times, minutes apart at most, so this is far more
history than it takes to recognise one."""


class SlackBot(Generic[BotDepsT]):
    """Serve one Pydantic AI agent as a Slack bot.

    DMs and channel mentions start a run, history is kept per thread, and the
    agent's text output is posted back into the thread.

    Any agent works, whatever its `deps` type. The thread a run is answering is
    bound around the run rather than passed as `deps`, so
    [`SlackChat`][pydantic_ai_harness.slack.SlackChat] posts to the right place
    and the agent keeps the deps it already had.

    Button clicks route themselves: the bot finds the `SlackChat` on the agent
    and hands clicks to it, so `ask_user` and approvals work with nothing wired
    up by hand.

    Slack can reach the bot two ways, and only the last line differs:

    ```python {test="skip"}
    bot = SlackBot(agent)

    bot.run()                                   # Socket Mode, no public URL
    app.mount('/slack/events', bot.http_app())  # Events API, any ASGI server
    ```

    Socket Mode needs `SLACK_APP_TOKEN` and opens an outbound connection, so it
    suits anything that can hold one. The Events API needs `SLACK_SIGNING_SECRET`
    and a public HTTPS endpoint, and suits anything that cannot -- a Lambda, or a
    container that scales to zero.

    Attributes:
        app: The underlying Bolt app. Register extra listeners on it if you need
            them, or hand it to any of Bolt's other adapters.
    """

    def __init__(
        self,
        agent: AbstractAgent[BotDepsT, str],
        *,
        deps: BotDepsT = None,
        deps_factory: Callable[[SlackThread], BotDepsT] | None = None,
        store: ConversationStore | None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        signing_secret: str | None = None,
        allowed_user_ids: Collection[str] | None = None,
        error_reply: str = DEFAULT_ERROR_REPLY,
    ) -> None:
        """Configure the app without connecting.

        Args:
            agent: Any text-output agent. Its deps type does not matter.
            deps: Passed to the agent as `deps` for every thread. Agents that
                take no deps can leave it unset.
            deps_factory: Builds the deps for one run from the
                [`SlackThread`][pydantic_ai_harness.slack.SlackThread] it is
                answering, for deps that differ per channel or per person. Kept
                separate from `deps` rather than accepting either in one argument,
                so a deps object that happens to be callable is not mistaken for a
                factory.
            store: Where thread history lives. Defaults to process memory, which
                is lost on restart.
            bot_token: Bot token (`xoxb-`). Defaults to `SLACK_BOT_TOKEN`.
            app_token: App-level token (`xapp-`) with `connections:write`.
                Defaults to `SLACK_APP_TOKEN`. Only [`run`][pydantic_ai_harness.slack.SlackBot.run]
                needs it, so an Events API deployment can leave it unset.
            signing_secret: Secret Slack signs its HTTP requests with. Defaults to
                `SLACK_SIGNING_SECRET`. Only
                [`http_app`][pydantic_ai_harness.slack.SlackBot.http_app] needs
                it, so a Socket Mode deployment can leave it unset.
            allowed_user_ids: Slack user ids allowed to start runs. Defaults to
                `SLACK_ALLOWED_USER_IDS`, comma separated. When neither is set,
                anyone who can reach the bot can spend its tokens and invoke its
                tools, and a warning says so at startup.
            error_reply: Posted in the thread when a run raises.

        Raises:
            ValueError: If `bot_token` is missing, both `deps` and `deps_factory`
                were given, or `allowed_user_ids` is a string rather than a
                collection of ids.
        """
        self._agent = agent
        if deps is not None and deps_factory is not None:
            raise ValueError('Pass deps or deps_factory, not both.')
        self._deps = deps
        self._deps_factory = deps_factory
        self._store = store if store is not None else InMemoryConversationStore()
        self._error_reply = error_reply
        self._chat = _find_chat(agent.root_capability)

        bot = bot_token or os.environ.get('SLACK_BOT_TOKEN')
        if not bot:
            raise ValueError('bot_token is required; pass it or set SLACK_BOT_TOKEN')
        # Each transport needs one of these and neither needs both, so they are
        # checked where they are used rather than here.
        self._app_token = app_token or os.environ.get('SLACK_APP_TOKEN')
        self._signing_secret = signing_secret or os.environ.get('SLACK_SIGNING_SECRET')

        if isinstance(allowed_user_ids, str):
            raise ValueError('allowed_user_ids must be a collection of user ids, not a string')
        if allowed_user_ids is None:
            configured = os.environ.get('SLACK_ALLOWED_USER_IDS', '')
            allowed_user_ids = [entry.strip() for entry in configured.split(',') if entry.strip()]
        self._allowed_user_ids = frozenset(allowed_user_ids)
        if not self._allowed_user_ids:
            logger.warning(
                'No allowed user ids configured, so anyone who can reach this bot can run the agent. '
                'Pass allowed_user_ids or set SLACK_ALLOWED_USER_IDS to restrict it.'
            )

        # Weak, so a thread's lock is collected once no turn holds it. A busy
        # workspace would otherwise accumulate one lock per thread forever.
        self._locks: WeakValueDictionary[str, anyio.Lock] = WeakValueDictionary()
        # Slack retries an event it thinks was not delivered, and a retried
        # mention would start a second run with the same side effects.
        self._handled_events: OrderedDict[str, None] = OrderedDict()
        # The capability posts through its own client. Handing it this token
        # means an agent set up for Slack does not need SLACK_BOT_TOKEN as well
        # when the bot was given its token directly.
        if self._chat is not None and self._chat.client is None and self._chat.token is None:
            self._chat.token = bot
        self.app = AsyncApp(token=bot, signing_secret=self._signing_secret)
        self._register_listeners()

    def _deps_for(self, thread: SlackThread) -> BotDepsT:
        """The deps this run gets, built for this thread when a factory was given."""
        if self._deps_factory is not None:
            return self._deps_factory(thread)
        return self._deps

    def _register_listeners(self) -> None:
        # Bolt ships no type information for its decorators, so registration is
        # done by explicit call and the untyped members are silenced one by one.
        # Everything the listeners hand on is narrowed before it leaves here.
        self.app.event('app_mention')(self._on_event)  # pyright: ignore[reportUnknownMemberType]
        self.app.event('message')(self._on_direct_message)  # pyright: ignore[reportUnknownMemberType]
        prompt_clicks = re.compile(f'^{re.escape(PROMPT_ACTION_PREFIX)}')
        self.app.action(prompt_clicks)(self._on_prompt_click)  # pyright: ignore[reportUnknownMemberType]

    async def _on_event(
        self,
        event: Mapping[str, object],
        client: AsyncWebClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
    ) -> None:
        """Pull the envelope fields off a Bolt listener's arguments and run a turn."""
        await self.handle_message(
            event,
            client,
            bot_user_id=_string(context, 'bot_user_id'),
            team_id=_string(context, 'team_id'),
            event_id=_string(body, 'event_id'),
        )

    async def _on_direct_message(
        self,
        event: Mapping[str, object],
        client: AsyncWebClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
    ) -> None:
        # Channel messages reach this listener too, and `app_mention` already
        # covers those. Only a direct or group DM starts a run without a mention.
        if event.get('channel_type') in ('im', 'mpim'):
            await self._on_event(event, client, context, body)

    async def _on_prompt_click(self, ack: _Ack, body: Mapping[str, object]) -> None:
        await ack()
        click = _prompt_click(body)
        if click is None or not self.resolve_prompt(block_id=click.block_id, value=click.value, user_id=click.user_id):
            # Slack shows nothing for a click that changes nothing, so the
            # operator needs a record of prompts that expired or were clicked by
            # someone who could not answer them.
            logger.info('Ignoring a Slack prompt click that resolved nothing')

    def resolve_prompt(self, *, block_id: str, value: str, user_id: str) -> bool:
        """Route a button click from a prompt back to the run waiting on it.

        Call it from your own action handler when you build the Bolt app
        yourself. Returns `False` when the click changed nothing, which covers an
        expired prompt, a repeat click, a person not allowed to answer, and an
        agent with no `SlackChat` that asks anything.
        """
        if self._chat is None:
            return False
        return self._chat.resolve_prompt(block_id=block_id, value=value, user_id=user_id)

    async def handle_message(
        self,
        event: Mapping[str, object],
        client: AsyncWebClient,
        *,
        bot_user_id: str | None = None,
        team_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """Run the agent for one inbound Slack message and post its reply.

        Call this from your own Bolt listeners when you build the app yourself.
        Messages the bot itself sent, messages with a subtype, and messages from
        users outside the allowlist are ignored.

        `team_id` names the workspace and keeps history separate across a
        multi-workspace install. Bolt puts it on the listener's `context`, which
        is more reliable than the event body.

        `event_id` comes off the Slack envelope. Passing it means a redelivery of
        the same event is ignored rather than starting a second run, which for an
        agent with write access means doing the work twice.
        """
        if event_id is not None and self._already_handled(event_id):
            logger.info('Ignoring a repeat delivery of Slack event %s', event_id)
            return
        if event.get('bot_id') is not None or event.get('subtype') is not None:
            return
        user_id = _string(event, 'user')
        channel_id = _string(event, 'channel')
        text = _string(event, 'text')
        timestamp = _string(event, 'ts')
        if user_id is None or channel_id is None or text is None or timestamp is None:
            return
        if user_id == bot_user_id:
            return
        if self._allowed_user_ids and user_id not in self._allowed_user_ids:
            logger.info('Ignoring Slack message from %s, who is not on the allowlist', user_id)
            return

        # Strip only this bot's own mention. Removing every mention would delete
        # the names of people the message refers to, which the agent needs.
        prompt = text.replace(f'<@{bot_user_id}>', '').strip() if bot_user_id else text.strip()
        if not prompt:
            return

        thread_ts = _string(event, 'thread_ts')
        thread = SlackThread(
            channel_id=channel_id,
            thread_ts=thread_ts or timestamp,
            user_id=user_id,
            team_id=team_id or _string(event, 'team'),
        )

        # One run at a time per thread, so a follow-up queues behind the run it is
        # adding to rather than racing it for the same history.
        lock = self._locks.setdefault(thread.key, anyio.Lock())
        async with lock:
            await self._run_turn(client, thread, prompt)

    def _already_handled(self, event_id: str) -> bool:
        """True when this event has been delivered before. Records it either way."""
        if event_id in self._handled_events:
            return True
        self._handled_events[event_id] = None
        if len(self._handled_events) > _REMEMBERED_EVENTS:
            self._handled_events.popitem(last=False)
        return False

    async def _run_turn(self, client: SlackClient, thread: SlackThread, prompt: str) -> None:
        try:
            history = await self._store.load(thread.key)
            # Bound rather than passed as deps, so the agent's own deps are
            # untouched and `SlackChat` still knows where it is talking.
            with bind_thread(thread):
                result = await self._agent.run(
                    prompt,
                    deps=self._deps_for(thread),
                    conversation_id=thread.key,
                    message_history=list(history) or None,
                )
            if result.output.strip():
                await self._post(client, thread, result.output)
            # Saved only once the reply is out. Saving first would leave the next
            # turn building on an answer nobody in the thread ever saw.
            await self._store.save(thread.key, result.all_messages())
        except Exception:
            logger.exception('Slack agent run failed in %s', thread.key)
            try:
                await self._post(client, thread, self._error_reply)
            except Exception:
                logger.exception('Could not post the error reply in %s', thread.key)

    async def _post(self, client: SlackClient, thread: SlackThread, text: str) -> None:
        for start in range(0, len(text), MAX_MESSAGE_CHARS):
            await client.chat_postMessage(
                channel=thread.channel_id,
                thread_ts=thread.thread_ts,
                text=text[start : start + MAX_MESSAGE_CHARS],
            )

    def http_app(self, path: str = '/slack/events') -> AsyncSlackRequestHandler:
        """An ASGI app serving Slack's Events API, to mount on your own server.

        Mount it at `path`, and give Slack that same URL as the request URL. Bolt
        verifies the signature on every request and answers Slack's setup
        challenge, so there is nothing to write.

        ```python {test="skip"}
        app = FastAPI()
        app.mount('/slack/events', bot.http_app())
        ```

        Bolt replies before running the listener, so Slack's three-second
        deadline is met however long the agent takes.

        Raises:
            ValueError: If no signing secret was configured.
        """
        if not self._signing_secret:
            raise ValueError(
                'signing_secret is required to serve the Events API; pass it or set SLACK_SIGNING_SECRET. '
                'Socket Mode needs no signing secret -- use run() instead if that is what you want.'
            )
        return AsyncSlackRequestHandler(self.app, path=path)

    async def start(self) -> None:
        """Connect over Socket Mode and serve until cancelled.

        Raises:
            ValueError: If no app token was configured.
        """
        if not self._app_token:
            raise ValueError(
                'app_token is required for Socket Mode; pass it or set SLACK_APP_TOKEN. '
                'The Events API needs no app token -- use http_app() instead if that is what you want.'
            )
        await AsyncSocketModeHandler(self.app, self._app_token).start_async()

    def run(self) -> None:
        """Start Socket Mode and block. The synchronous entry point for a script."""
        anyio.run(self.start)


def _find_chat(root: AbstractCapability[BotDepsT]) -> SlackChat[BotDepsT] | None:
    """The agent's `SlackChat`, so clicks and the bot token can reach it.

    Walks the whole capability tree rather than the top level, so a `SlackChat`
    inside a combined or wrapped capability is still found.
    """
    found: list[SlackChat[BotDepsT]] = []

    def visit(capability: AbstractCapability[BotDepsT]) -> AbstractCapability[BotDepsT]:
        if isinstance(capability, SlackChat):
            found.append(capability)
        return capability

    root.visit_and_replace(visit)
    if len(found) > 1:
        logger.warning(
            'The agent has %d SlackChat capabilities; button clicks are routed to the first. '
            'Use one SlackChat, or pass the same interactions= to each.',
            len(found),
        )
    return found[0] if found else None
