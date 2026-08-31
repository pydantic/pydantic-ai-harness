"""Connect a text-output Pydantic AI agent to messaging channels.

[`ChannelHost`][pydantic_ai_harness.channels.ChannelHost] translates inbound
messages into ordinary `Agent.run()` calls, so capabilities already configured
on the agent participate without a channel-specific wrapper.
"""

from pydantic_ai_harness.channels._host import ChannelHost
from pydantic_ai_harness.channels._types import (
    ChannelAdapter,
    ChannelError,
    ConversationStore,
    InboundMessage,
    InMemoryConversationStore,
    WebhookRequest,
    WebhookResponse,
)

__all__ = [
    'ChannelAdapter',
    'ChannelError',
    'ChannelHost',
    'ConversationStore',
    'InboundMessage',
    'InMemoryConversationStore',
    'WebhookRequest',
    'WebhookResponse',
]
