"""Haunt extraction capability that gives an agent honest web page reading and structured extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.haunt._toolset import HauntClient, HauntExtractToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = (
    'You have web extraction tools backed by the Haunt API. Use `read_page` to read a page as '
    'clean Markdown, and `extract_data` when you need specific fields, described in plain '
    'English, as structured JSON. These tools fail honestly: when a result reports '
    'access_denied, login_required, captcha_required, or not_found, the page really is '
    'unavailable -- report that instead of retrying the same URL or inventing the content.'
)


@dataclass
class HauntExtract(AbstractCapability[AgentDepsT]):
    """Web extraction for agents, backed by the [Haunt](https://hauntapi.com) API.

    Adds two tools: `read_page`, which returns a page as clean Markdown, and
    `extract_data`, which returns specific fields, described in plain English,
    as structured JSON. The differentiator is honest failure: a blocked,
    login-walled, captcha-guarded, or missing page comes back as an error code
    with a plain-words reason (`access_denied`, `login_required`,
    `captcha_required`, `not_found`) rather than fabricated content, so the
    agent can branch on the failure instead of hallucinating around it.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.haunt import HauntExtract

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[HauntExtract()])
    ```

    Authentication comes from the `HAUNT_API_KEY` environment variable by
    default; pass `client` to configure it explicitly.
    """

    max_text_chars: int = 50_000
    """Maximum characters of tool text returned per call (at least 1).

    Longer content is truncated head-first with a marker, so the model knows
    content was cut.
    """

    guidance: str | None = None
    """Custom extraction guidance for the system prompt.

    Leave as `None` for the default guidance (which explains the honest
    failure codes), or set `''` to contribute no instructions at all.
    """

    client: HauntClient | None = None
    """Haunt client to use; when `None`, an `HttpxHauntClient` is built from `HAUNT_API_KEY`.

    Any object satisfying the `HauntClient` protocol works: use it to pass an
    API key explicitly, point at a different base URL, or substitute a fake in
    tests.
    """

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_text_chars < 1:
            raise ValueError(f'max_text_chars must be at least 1, got {self.max_text_chars}')

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static extraction guidance: read or extract, and treat honest failures as final.

        A non-`None` `guidance` replaces the default; `''` disables
        instructions entirely.
        """
        if self.guidance is not None:
            return self.guidance or None
        return _INSTRUCTIONS

    def get_toolset(self) -> HauntExtractToolset[AgentDepsT]:
        """Build the toolset providing the `read_page` and `extract_data` tools."""
        return HauntExtractToolset[AgentDepsT](client=self.client, max_text_chars=self.max_text_chars)

    @classmethod
    def from_spec(cls, *, max_text_chars: int = 50_000, guidance: str | None = None) -> HauntExtract[AgentDepsT]:
        """Construct the capability from serializable spec options.

        The `client` field is not spec-serializable, so spec-loaded instances
        always build the default `HttpxHauntClient` from `HAUNT_API_KEY`.
        """
        return cls(max_text_chars=max_text_chars, guidance=guidance)
