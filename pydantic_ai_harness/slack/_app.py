"""A ready-made Slack app that serves a Pydantic AI agent, over Socket Mode or HTTP."""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from time import monotonic
from typing import Generic, Protocol, TypeVar
from weakref import WeakValueDictionary

import anyio
from pydantic import AliasPath, BaseModel, Field, TypeAdapter, ValidationError
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability

from pydantic_ai_harness.slack._access import SlackAccess
from pydantic_ai_harness.slack._capability import Slack, SlackMCPAuthenticationError
from pydantic_ai_harness.slack._client import SlackClient
from pydantic_ai_harness.slack._context import SlackContext, SlackContextEntity, bind_slack_context
from pydantic_ai_harness.slack._interactions import PROMPT_ACTION_PREFIX
from pydantic_ai_harness.slack._store import ConversationStore, InMemoryConversationStore
from pydantic_ai_harness.slack._thread import SlackThread, bind_thread

try:
    from slack_bolt.adapter.asgi.async_handler import AsyncSlackRequestHandler
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_bolt.app.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'slack-bolt is required for SlackApp. Install it with: pip install "pydantic-ai-harness[slack]"'
    ) from _import_error

logger = logging.getLogger(__name__)

BotDepsT = TypeVar('BotDepsT')
"""The deps type the served agent takes. Its own variable because `BotDepsT` is
contravariant, and `SlackApp` builds deps as well as passing them."""


class _Ack(Protocol):
    async def __call__(self) -> object: ...  # pragma: no cover


def _string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


class _ActiveContextEntity(BaseModel):
    type: str
    value: str
    team_id: str | None = None


_ACTIVE_CONTEXT_ENTITY_ADAPTER = TypeAdapter(_ActiveContextEntity)
_OBJECT_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
_OBJECT_LIST_ADAPTER = TypeAdapter(list[object])


def _active_entities(event: Mapping[str, object]) -> tuple[SlackContextEntity, ...]:
    try:
        app_context = _OBJECT_MAPPING_ADAPTER.validate_python(event.get('app_context'))
        raw_entities = _OBJECT_LIST_ADAPTER.validate_python(app_context.get('entities'))
    except ValidationError:
        return ()
    entities: list[SlackContextEntity] = []
    for raw_entity in raw_entities:
        try:
            entity = _ACTIVE_CONTEXT_ENTITY_ADAPTER.validate_python(raw_entity)
        except ValidationError:
            continue
        entities.append(SlackContextEntity(entity.type, entity.value, entity.team_id))
    return tuple(entities)


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
MISSING_MCP_AUTH_REPLY = (
    'Slack workspace access is not connected for your user. '
    'Ask the app owner for the Slack authorization link, then retry.'
)
MAX_DELIVERY_CHARS = 3500

_EVENT_DEDUPLICATION_SECONDS = 600.0
"""Slack's standard retries finish after five minutes; retain ids for ten."""

_REMEMBERED_THREADS = 1000


class SlackApp(Generic[BotDepsT]):
    """Serve one Pydantic AI agent as a Slack bot.

    DMs and channel mentions start a run, history is kept per thread, and the
    agent's text output is posted back into the thread.

    Any agent works, whatever its `deps` type. The thread a run is answering is
    bound around the run rather than passed as `deps`, so
    [`Slack`][pydantic_ai_harness.slack.Slack] posts to the right place
    and the agent keeps the deps it already had.

    Button clicks route themselves: the app finds the `Slack` on the agent and
    hands approval clicks to it with nothing wired up by hand.

    Slack can reach the bot two ways, and only the last line differs:

    ```python {test="skip"}
    bot = SlackApp(agent)

    bot.run()                                   # Socket Mode, no public URL
    slack_asgi = bot.http_app()                 # Events API, complete ASGI app
    ```

    Socket Mode needs `SLACK_APP_TOKEN` and opens an outbound connection. The
    Events API needs `SLACK_SIGNING_SECRET` and a public HTTPS endpoint. This
    adapter expects a long-running process for both transports; FaaS deployments
    need Bolt's lazy-listener execution strategy.

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
        app: AsyncApp | None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        signing_secret: str | None = None,
        access: SlackAccess | None = None,
        install_url: str | None = None,
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
            app: Caller-configured Bolt app. Pass this for OAuth installations;
                Bolt then supplies the bot client and invoking user's token.
            bot_token: Bot token (`xoxb-`). Defaults to `SLACK_BOT_TOKEN`.
            app_token: App-level token (`xapp-`) with `connections:write`.
                Defaults to `SLACK_APP_TOKEN`. Only [`run`][pydantic_ai_harness.slack.SlackApp.run]
                needs it, so an Events API deployment can leave it unset.
            signing_secret: Secret Slack signs its HTTP requests with. Defaults to
                `SLACK_SIGNING_SECRET`. Only
                [`http_app`][pydantic_ai_harness.slack.SlackApp.http_app] needs
                it, so a Socket Mode deployment can leave it unset.
            access: Who may invoke the agent. Defaults to the comma-separated
                `SLACK_ALLOWED_USER_IDS`. Use `SlackAccess.workspace()` to opt
                into workspace-wide access.
            install_url: Public per-user OAuth URL shown when Slack MCP access
                is missing. Defaults to `SLACK_INSTALL_URL`.
            error_reply: Posted in the thread when a run raises.

        Raises:
            ValueError: If `bot_token` is missing, both `deps` and `deps_factory`
                were given, or no access policy was configured.
        """
        self._agent = agent
        if deps is not None and deps_factory is not None:
            raise ValueError('Pass deps or deps_factory, not both.')
        self._deps = deps
        self._deps_factory = deps_factory
        self._store = store if store is not None else InMemoryConversationStore()
        self._error_reply = error_reply
        self._install_url = install_url or os.environ.get('SLACK_INSTALL_URL')
        self._slack = _find_slack(agent.root_capability)
        self._caller_owned_app = app is not None

        bot = bot_token or os.environ.get('SLACK_BOT_TOKEN')
        if app is None and not bot:
            raise ValueError('bot_token is required when app is not passed; pass it or set SLACK_BOT_TOKEN')
        # Each transport needs one of these and neither needs both, so they are
        # checked where they are used rather than here.
        self._app_token = app_token or os.environ.get('SLACK_APP_TOKEN')
        self._signing_secret = signing_secret or os.environ.get('SLACK_SIGNING_SECRET')

        if access is None:
            configured = os.environ.get('SLACK_ALLOWED_USER_IDS', '')
            user_ids = [entry.strip() for entry in configured.split(',') if entry.strip()]
            if not user_ids:
                raise ValueError(
                    'Slack agent access is not configured. Set SLACK_ALLOWED_USER_IDS, pass '
                    'access=SlackAccess.users(...), or explicitly use SlackAccess.workspace().'
                )
            access = SlackAccess.users(*user_ids)
        self._access = access

        env_mcp_token = os.environ.get('SLACK_MCP_TOKEN')
        if (
            self._slack is not None
            and self._slack.tools.selected
            and not self._caller_owned_app
            and self._slack.mcp_token is None
            and env_mcp_token
            and (access.allowed_user_ids is None or len(access.allowed_user_ids) != 1)
        ):
            raise ValueError(
                'SLACK_MCP_TOKEN is a fixed user identity and requires exactly one allowed Slack user. '
                'Use per-user OAuth for multiple users, or pass mcp_token explicitly to opt into a shared identity.'
            )

        # Weak, so a thread's lock is collected once no turn holds it. A busy
        # workspace would otherwise accumulate one lock per thread forever.
        self._locks: WeakValueDictionary[str, anyio.Lock] = WeakValueDictionary()
        # Slack retries an event it thinks was not delivered, and a retried
        # mention would start a second run with the same side effects.
        self._handled_events: OrderedDict[str, float] = OrderedDict()
        self._engaged_threads: OrderedDict[str, None] = OrderedDict()
        # The capability posts through its own client. Handing it this token
        # means an agent set up for Slack does not need SLACK_BOT_TOKEN as well
        # when the bot was given its token directly.
        if self._slack is not None and bot is not None:
            self._slack.set_app_delivery_token(bot)
        self.app = app if app is not None else AsyncApp(token=bot, signing_secret=self._signing_secret)
        self._register_listeners()

    @property
    def access(self) -> SlackAccess:
        """The policy that decides who may invoke this Slack app."""
        return self._access

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
            enterprise_id=_string(context, 'enterprise_id'),
            user_token=_string(context, 'user_token'),
            event_id=_string(body, 'event_id'),
        )

    async def _on_direct_message(
        self,
        event: Mapping[str, object],
        client: AsyncWebClient,
        context: Mapping[str, object],
        body: Mapping[str, object],
    ) -> None:
        # A DM always invokes the agent. A channel reply does so only when stored
        # history proves the agent already joined that thread after one mention.
        if event.get('channel_type') in ('im', 'mpim'):
            await self._on_event(event, client, context, body)
            return
        channel_id = _string(event, 'channel')
        thread_ts = _string(event, 'thread_ts')
        if channel_id is None or thread_ts is None:
            return
        thread = SlackThread(channel_id=channel_id, thread_ts=thread_ts, team_id=_string(context, 'team_id'))
        if thread.key in self._engaged_threads or await self._store.load(thread.key):
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
        agent with no `Slack` that asks anything.
        """
        if self._slack is None:
            return False
        return self._slack.resolve_prompt(block_id=block_id, value=value, user_id=user_id)

    async def handle_message(
        self,
        event: Mapping[str, object],
        client: AsyncWebClient,
        *,
        bot_user_id: str | None = None,
        team_id: str | None = None,
        enterprise_id: str | None = None,
        user_token: str | None = None,
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
        if not self._access.allows(user_id):
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
        slack_context = SlackContext(
            channel_id=channel_id,
            thread_ts=thread.thread_ts or timestamp,
            message_ts=timestamp,
            user_id=user_id,
            team_id=thread.team_id,
            enterprise_id=enterprise_id,
            active_entities=_active_entities(event),
            user_token=user_token,
        )
        self._remember_engagement(thread.key)

        # One run at a time per thread, so a follow-up queues behind the run it is
        # adding to rather than racing it for the same history.
        lock = self._locks.setdefault(thread.key, anyio.Lock())
        async with lock:
            await self._run_turn(client, thread, slack_context, prompt)

    def _already_handled(self, event_id: str) -> bool:
        """True when this event has been delivered before. Records it either way."""
        now = monotonic()
        expires_before = now - _EVENT_DEDUPLICATION_SECONDS
        while self._handled_events:
            _, seen_at = next(iter(self._handled_events.items()))
            if seen_at > expires_before:
                break
            self._handled_events.popitem(last=False)
        if event_id in self._handled_events:
            return True
        self._handled_events[event_id] = now
        return False

    def _remember_engagement(self, thread_key: str) -> None:
        """Remember that this app owns replies in a thread, with bounded state."""
        self._engaged_threads[thread_key] = None
        self._engaged_threads.move_to_end(thread_key)
        if len(self._engaged_threads) > _REMEMBERED_THREADS:
            self._engaged_threads.popitem(last=False)

    async def _run_turn(
        self, client: SlackClient, thread: SlackThread, slack_context: SlackContext, prompt: str
    ) -> None:
        try:
            history = await self._store.load(thread.key)
            # Bound rather than passed as deps, so the agent's own deps are
            # untouched and `Slack` still knows where it is talking.
            with (
                bind_thread(thread),
                bind_slack_context(slack_context, client, allow_fixed_mcp_fallback=not self._caller_owned_app),
            ):
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
        except SlackMCPAuthenticationError:
            logger.exception('Slack MCP authentication is missing in %s', thread.key)
            reply = (
                f'Slack workspace access is not connected for your user. Authorize it at {self._install_url}, then retry.'
                if self._install_url is not None
                else MISSING_MCP_AUTH_REPLY
            )
            await self._post(client, thread, reply)
        except Exception:
            logger.exception('Slack agent run failed in %s', thread.key)
            try:
                await self._post(client, thread, self._error_reply)
            except Exception:
                logger.exception('Could not post the error reply in %s', thread.key)

    async def _post(self, client: SlackClient, thread: SlackThread, text: str) -> None:
        for start in range(0, len(text), MAX_DELIVERY_CHARS):
            await client.chat_postMessage(
                channel=thread.channel_id,
                thread_ts=thread.thread_ts,
                text=text[start : start + MAX_DELIVERY_CHARS],
            )

    def http_app(self, path: str = '/slack/events') -> AsyncSlackRequestHandler:
        """An ASGI app serving Slack's Events API, to mount on your own server.

        Mount it at `path`, and give Slack that same URL as the request URL. Bolt
        verifies the signature on every request and answers Slack's setup
        challenge, so there is nothing to write.

        Serve the returned handler as the ASGI application. It routes the event
        endpoint at `path` plus Bolt's OAuth install and redirect endpoints when
        the caller-configured app enables OAuth. If it is combined with another
        ASGI application, mount it at `/` after more specific routes so Bolt can
        still see the full request paths. Mounting it at `/slack/events` strips
        that prefix and makes Bolt return 404.

        ```python {test="skip"}
        slack_asgi = bot.http_app()
        # uvicorn my_module:slack_asgi
        ```

        Bolt acknowledges the request before running the listener in its normal
        long-running-server mode.

        Raises:
            ValueError: If no signing secret was configured.
        """
        if not self._signing_secret and not self._caller_owned_app:
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


def _find_slack(root: AbstractCapability[BotDepsT]) -> Slack[BotDepsT] | None:
    """The agent's `Slack`, so clicks and the bot token can reach it.

    Walks the whole capability tree rather than the top level, so a `Slack`
    inside a combined or wrapped capability is still found.
    """
    found: list[Slack[BotDepsT]] = []

    def visit(capability: AbstractCapability[BotDepsT]) -> AbstractCapability[BotDepsT]:
        if isinstance(capability, Slack):
            found.append(capability)
        return capability

    root.visit_and_replace(visit)
    if len(found) > 1:
        raise ValueError(
            'SlackApp requires exactly one Slack capability. Combine its tool selection and configuration in one Slack.'
        )
    return found[0] if found else None
