"""Haunt toolset -- gives agents page reading and structured extraction backed by the Haunt API."""

from __future__ import annotations

import functools
import json
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Concatenate, ParamSpec, Protocol, TypeVar

import httpx
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import Self

# External-service assumptions, last verified 2026-07-29:
# - POST /v1/extract accepts url, prompt, and optional response_format.
# - X-API-Key is a supported authentication header.
# - Successful responses are JSON objects with a required boolean success field.
# Re-check the production schema at https://hauntapi.com/openapi.json and the
# user documentation at https://hauntapi.com/docs before changing this boundary.
HAUNT_BASE_URL = 'https://hauntapi.com'
"""Default base URL of the Haunt API."""

HONEST_FAILURE_CODES = frozenset({'access_denied', 'captcha_required', 'login_required', 'not_found'})
"""Terminal page-level error codes returned by Haunt.

These are reported to the model as tool text (so it can branch or report
the problem) instead of being raised because retrying the same URL during the
same run is unlikely to change the outcome.
"""

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SelfT = TypeVar('_SelfT')
_OBJECT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


class HauntClient(Protocol):
    """The subset of the Haunt extraction API that `HauntExtractToolset` calls.

    Any object with this method can back the toolset. Pass one via
    `HauntExtract.client` to configure authentication or the base URL
    explicitly, or to substitute a fake in tests.
    """

    async def extract(self, url: str, prompt: str, *, response_format: str | None = None) -> Mapping[str, object]:
        """Run one extraction and return the Haunt response body."""
        ...  # pragma: no cover


class HttpxHauntClient:
    """`HauntClient` backed by `httpx`, calling the hosted Haunt API.

    Auth failures raise `UserError` (configuration the model cannot correct);
    non-success HTTP responses raise `ModelRetry` with the status and API
    message. Network failures propagate as `httpx.HTTPError` for the toolset
    to translate. Redirects are not followed, so the API key is not forwarded
    to a redirect target.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = HAUNT_BASE_URL,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip('/'),
            headers={'X-API-Key': api_key, 'Content-Type': 'application/json'},
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
        )

    async def extract(self, url: str, prompt: str, *, response_format: str | None = None) -> Mapping[str, object]:
        """POST `/v1/extract` and return the parsed response body."""
        body: dict[str, object] = {'url': url, 'prompt': prompt}
        if response_format is not None:
            body['response_format'] = response_format
        response = await self._client.post('/v1/extract', json=body)
        if response.status_code in {401, 403}:
            raise UserError(
                'The Haunt API rejected the API key. Set the HAUNT_API_KEY environment variable '
                'to a valid key (free key: https://hauntapi.com/#signup), or pass a configured client. '
                f'HTTP status: {response.status_code}.'
            )
        try:
            payload = _OBJECT_ADAPTER.validate_json(response.content)
        except ValidationError as error:
            raise ModelRetry(
                f'Haunt returned invalid JSON for {url!r}. HTTP status: {response.status_code}.'
            ) from error
        if response.status_code == 429:
            message = _response_message(payload, fallback='Rate limit or quota exceeded')
            raise ModelRetry(f'Haunt request was rate limited: {message}')
        if not 200 <= response.status_code < 300:
            message = _response_message(payload, fallback='The API request did not succeed')
            raise ModelRetry(f'Haunt request failed with HTTP status {response.status_code}: {message}')
        if not isinstance(payload.get('success'), bool):
            raise ModelRetry(f'Haunt returned an invalid response for {url!r}: missing boolean success field.')
        return payload

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Enter the client context."""
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client context."""
        await self._client.__aexit__(exc_type, exc_value, traceback)


def _default_client() -> HttpxHauntClient:
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
    configuration, not something the model can fix. Terminal page-level
    failures are returned as tool text so the model can branch on them (see
    `HONEST_FAILURE_CODES`).
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
    specific fields, described in plain English, as structured JSON. When a
    page is blocked, behind a login, guarded by a captcha, or missing, the tool
    returns an error code and readable reason so the model can branch or report
    the failure.

    Each tool returns a `ToolReturn` whose `return_value` is the text the
    model sees and whose `metadata['error_code']` is `None` on success or the
    Haunt error code on a page-level failure, so applications can branch without
    parsing the text.
    """

    def __init__(self, *, client: HauntClient | None, max_text_chars: int) -> None:
        super().__init__()
        if client is None:
            owned_client = _default_client()
            self._owned_client: HttpxHauntClient | None = owned_client
            self._client: HauntClient = owned_client
        else:
            self._owned_client = None
            self._client = client
        self._max_text_chars = max_text_chars
        self.add_function(self.read_page, name='read_page')
        self.add_function(self.extract_data, name='extract_data')

    async def __aenter__(self) -> Self:
        """Enter the toolset context."""
        await super().__aenter__()
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        """Close a default HTTP client created by this toolset."""
        try:
            if self._owned_client is not None:
                await self._owned_client.aclose()
        finally:
            await super().__aexit__(*args)
        return None

    @_recoverable
    async def read_page(self, url: str) -> ToolReturn[str]:
        """Read a web page as clean Markdown.

        Use it to read an article, product page, or documentation page in
        full. If the page cannot be read (blocked, login wall, captcha, or
        not found), the result includes an error code and readable reason.

        Args:
            url: The URL of the page to read.

        Returns:
            The page content as Markdown, or a typed failure report.
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
        with an error code and readable reason.

        Args:
            url: The URL of the page to extract from.
            prompt: Plain-English description of the fields to extract.

        Returns:
            The extracted fields as JSON, or a typed failure report.
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


def _failure_text(payload: Mapping[str, object], url: str) -> ToolReturn[str] | None:
    """Render an unsuccessful Haunt response as a failure result, or `None` on success.

    Failures are returned as tool text rather than raised: a blocked or
    missing page will not change within a run, so a `ModelRetry` would burn
    retries re-fetching it, and any other exception would abort the run. As
    text, the model can branch: try a different URL, report the failure, or
    ask the user.
    """
    if payload.get('success') is True:
        return None
    raw_error_code = payload.get('error_code')
    error_code = raw_error_code if isinstance(raw_error_code, str) and raw_error_code else 'extraction_failed'
    message = _response_message(payload, fallback='The page could not be extracted.')
    hint = (
        ' The page itself is unavailable; retrying the same URL will not help.'
        if error_code in HONEST_FAILURE_CODES
        else ''
    )
    return ToolReturn(
        f'Extraction failed for {url}: {error_code}. {message}{hint}',
        metadata={'error_code': error_code},
    )


def _markdown_body(payload: Mapping[str, object]) -> str | None:
    """The Markdown text of a successful `response_format='markdown'` response.

    The API wraps Markdown output as `{'markdown': '...'}` in `data`; a bare
    string is accepted too so a client fake (or a future contract change)
    returning the text directly still works.
    """
    data = payload.get('data')
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        markdown = _OBJECT_ADAPTER.validate_python(data).get('markdown')
        if isinstance(markdown, str):
            return markdown
    return None


def _response_message(payload: Mapping[str, object], *, fallback: str) -> str:
    """Return the first non-empty API message field."""
    for field in ('message', 'error'):
        value = payload.get(field)
        if isinstance(value, str) and value:
            return value
    return fallback


def _truncate(text: str, max_chars: int) -> str:
    """Keep at most `max_chars` source characters, followed by a marker if cut.

    A page's lead carries the substance, so the head is kept and the tail
    dropped, with a marker so the model knows content was cut.
    """
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}\n[... content truncated at {max_chars} characters]'
