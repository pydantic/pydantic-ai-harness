"""Haunt toolset -- gives agents page reading and structured extraction backed by the Haunt API."""

from __future__ import annotations

import functools
import json
from collections.abc import Awaitable, Callable
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar, cast

import httpx
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

HAUNT_BASE_URL = 'https://hauntapi.com'
"""Default base URL of the Haunt API."""

HONEST_FAILURE_CODES = frozenset({'access_denied', 'captcha_required', 'login_required', 'not_found'})
"""Error codes Haunt returns when a page is genuinely unavailable rather than mis-extracted.

These are reported to the model as tool text (so it can branch or report
honestly) instead of being raised, because retrying the same URL cannot
change the outcome within a run.
"""

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SelfT = TypeVar('_SelfT')


class HauntClient(Protocol):
    """The subset of the Haunt extraction API that `HauntExtractToolset` calls.

    Any object with this method can back the toolset. Pass one via
    `HauntExtract.client` to configure authentication or the base URL
    explicitly, or to substitute a fake in tests.
    """

    async def extract(self, url: str, prompt: str, *, response_format: str | None = None) -> dict[str, Any]:
        """Run one extraction and return the Haunt response body as a dict."""
        ...  # pragma: no cover


class HttpxHauntClient:
    """`HauntClient` backed by `httpx`, calling the hosted Haunt API.

    Auth failures raise `UserError` (configuration the model cannot correct);
    quota and rate-limit responses raise `ModelRetry` with the API's own
    message. Network failures propagate as `httpx.HTTPError` for the toolset
    to translate.
    """

    def __init__(self, api_key: str, *, base_url: str = HAUNT_BASE_URL, timeout: float = 120.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip('/'),
            headers={'X-API-Key': api_key, 'Content-Type': 'application/json'},
            timeout=timeout,
        )

    async def extract(self, url: str, prompt: str, *, response_format: str | None = None) -> dict[str, Any]:
        """POST `/v1/extract` and return the parsed response body."""
        body: dict[str, Any] = {'url': url, 'prompt': prompt}
        if response_format is not None:
            body['response_format'] = response_format
        response = await self._client.post('/v1/extract', json=body)
        if response.status_code == 401:
            raise UserError(
                'The Haunt API rejected the API key. Set the HAUNT_API_KEY environment variable '
                'to a valid key (free key: https://hauntapi.com/#signup), or pass a configured client.'
            )
        data: Any = response.json()
        if not isinstance(data, dict):  # pragma: no cover - contract violation guard
            raise ModelRetry(f'Haunt returned an unexpected response shape for {url!r}.')
        payload = cast('dict[str, Any]', data)
        if response.status_code == 429:
            message = payload.get('message') or payload.get('error') or 'Rate limit or quota exceeded'
            raise ModelRetry(f'Haunt request was rate limited: {message}')
        return payload


def _default_client() -> HauntClient:
    """Build an `HttpxHauntClient` from the `HAUNT_API_KEY` environment variable."""
    import os

    api_key = os.environ.get('HAUNT_API_KEY', '')
    if not api_key:
        raise UserError(
            'HauntExtract needs a Haunt API key: set the HAUNT_API_KEY environment variable, '
            'or pass a configured client, e.g. HauntExtract(client=HttpxHauntClient(api_key=...)). '
            'Free key: https://hauntapi.com/#signup'
        )
    return HttpxHauntClient(api_key)


def _recoverable(
    fn: Callable[Concatenate[_SelfT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_SelfT, _P], Awaitable[_R]]:
    """Convert transient network failures into `ModelRetry`.

    Only `ModelRetry` is fed back to the model as a retry prompt; any other
    exception aborts the run. Timeouts and connection errors are transient, so
    they become retries. `UserError` (bad API key) propagates: it is
    configuration, not something the model can fix. Honest page-level failures
    never raise at all -- they are returned as tool text so the model can
    branch on them (see `HONEST_FAILURE_CODES`).
    """

    @functools.wraps(fn)
    async def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await fn(self, *args, **kwargs)
        except httpx.HTTPError as error:
            raise ModelRetry(f'Haunt request failed: {error}') from error

    return wrapper


class HauntExtractToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web extraction tools backed by the Haunt API.

    `read_page` returns a page as clean Markdown; `extract_data` returns
    specific fields, described in plain English, as structured JSON. Both are
    honest about failure: when a page is blocked, behind a login, guarded by a
    captcha, or missing, the tool returns the error code and a plain-words
    reason instead of fabricated content, so the model can branch or report
    the failure instead of hallucinating around it.

    Each tool returns a `ToolReturn` whose `return_value` is the text the
    model sees and whose `metadata['error_code']` is `None` on success or the
    Haunt error code on an honest failure, so applications can branch without
    parsing the text.
    """

    def __init__(self, *, client: HauntClient | None, max_text_chars: int) -> None:
        super().__init__()
        self._client = client if client is not None else _default_client()
        self._max_text_chars = max_text_chars
        self.add_function(self.read_page, name='read_page')
        self.add_function(self.extract_data, name='extract_data')

    @_recoverable
    async def read_page(self, url: str) -> ToolReturn[str]:
        """Read a web page as clean Markdown.

        Use it to read an article, product page, or documentation page in
        full. If the page cannot be read (blocked, login wall, captcha, or
        not found), the result says so honestly with an error code instead of
        returning made-up content.

        Args:
            url: The URL of the page to read.

        Returns:
            The page content as Markdown, or an honest failure report.
        """
        payload = await self._client.extract(url, 'Return the readable page content.', response_format='markdown')
        failure = _failure_text(payload, url)
        if failure is not None:
            return failure
        markdown = _markdown_body(payload)
        if not markdown:
            raise ModelRetry(f'Haunt returned no content for {url!r}. Check the URL or try another page.')
        return ToolReturn(_truncate(markdown, self._max_text_chars), metadata={'error_code': None})

    @_recoverable
    async def extract_data(self, url: str, prompt: str) -> ToolReturn[str]:
        """Extract specific fields from a web page as structured JSON.

        Describe the fields you want in plain English (for example, 'the
        product name, price, and stock status'). If the page cannot be read
        (blocked, login wall, captcha, or not found), the result says so
        honestly with an error code instead of returning guessed values.

        Args:
            url: The URL of the page to extract from.
            prompt: Plain-English description of the fields to extract.

        Returns:
            The extracted fields as JSON, or an honest failure report.
        """
        payload = await self._client.extract(url, prompt)
        failure = _failure_text(payload, url)
        if failure is not None:
            return failure
        data = payload.get('data')
        if data is None:
            raise ModelRetry(f'Haunt returned no data for {url!r}. Check the URL, or rephrase the prompt.')
        body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)
        return ToolReturn(_truncate(body, self._max_text_chars), metadata={'error_code': None})


def _failure_text(payload: dict[str, Any], url: str) -> ToolReturn[str] | None:
    """Render an unsuccessful Haunt response as an honest failure result, or `None` on success.

    Failures are returned as tool text rather than raised: a blocked or
    missing page will not change within a run, so a `ModelRetry` would burn
    retries re-fetching it, and any other exception would abort the run. As
    text, the model can branch: try a different URL, report the failure, or
    ask the user.
    """
    if payload.get('success', False):
        return None
    error_code = payload.get('error_code') or 'extraction_failed'
    message = payload.get('message') or payload.get('error') or 'The page could not be extracted.'
    hint = (
        ' The page itself is unavailable; retrying the same URL will not help.'
        if error_code in HONEST_FAILURE_CODES
        else ''
    )
    return ToolReturn(
        f'Extraction failed for {url}: {error_code}. {message}{hint}',
        metadata={'error_code': error_code},
    )


def _markdown_body(payload: dict[str, Any]) -> str | None:
    """The Markdown text of a successful `response_format='markdown'` response.

    The API wraps Markdown output as `{'markdown': '...'}` in `data`; a bare
    string is accepted too so a client fake (or a future contract change)
    returning the text directly still works.
    """
    data = payload.get('data')
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        markdown = data.get('markdown')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(markdown, str):
            return markdown
    return None


def _truncate(text: str, max_chars: int) -> str:
    """Cap tool text at `max_chars`, keeping the head.

    A page's lead carries the substance, so the head is kept and the tail
    dropped, with a marker so the model knows content was cut.
    """
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}\n[... content truncated at {max_chars} characters]'
